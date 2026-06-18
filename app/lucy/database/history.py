from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Dict, Literal, Any
import asyncio
import json
import logging
from datetime import datetime, timezone

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelMessagesTypeAdapter,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior

# Import our custom ModelFileResponse
from ..agents.common.models import ModelFileResponse

# LocalDatabase removed in Supabase-only mode
from .supabase_client import (
    fetch_lucy_messages,
    insert_lucy_message,
    ensure_lucy_conversation_exists,
)


# ---------------------------------------------------------------------
# Chat message conversion utilities
# ---------------------------------------------------------------------
class ChatMessage:
    role: Literal["user", "model", "model-files", "model-json"]
    timestamp: str
    content: str
    file: Optional[Dict[str, str]]  # Optional file attachment
    json: Optional[Dict[str, Any]]  # Optional structured JSON payload
    message_order: int


def to_chat_message(
    m: ModelMessage,
    supabase_client=None,
    message_order: int = 0,
    message_id: int = 0,
    user_feedback: Optional[bool] = None,
    attachments: Optional[list] = None,
) -> ChatMessage:
    logger = logging.getLogger(__name__)
    try:
        if isinstance(m, ModelRequest):
            user_part = next(
                (p for p in m.parts if isinstance(p, UserPromptPart)), None
            )
            if user_part is not None:
                ts = getattr(user_part, "timestamp", None) or getattr(
                    m, "timestamp", datetime.now(tz=timezone.utc)
                )

                # Re-sign stored attachment paths using the user-uploads bucket
                signed_attachments = None
                if attachments and supabase_client:
                    signed_attachments = []
                    for att in attachments:
                        storage_path = att.get("storage_path", "")
                        signed_url = ""
                        if storage_path:
                            try:
                                resp = supabase_client.storage.from_(
                                    "lucy-user-uploads"
                                ).create_signed_url(storage_path, expires_in=3600)
                                signed_url = (
                                    resp.get("signedURL", "")
                                    if isinstance(resp, dict)
                                    else str(resp)
                                )
                            except Exception:
                                pass
                        signed_attachments.append(
                            {
                                "file_name": att.get("file_name", ""),
                                "file_type": att.get("file_type", ""),
                                "signed_url": signed_url,
                            }
                        )

                return {
                    "role": "user",
                    "timestamp": ts.isoformat(),
                    "content": str(user_part.content),
                    "file": None,
                    "attachments": signed_attachments or [],
                    "message_order": message_order,
                    "message_id": message_id,
                    "user_feedback": user_feedback,
                }

        if isinstance(m, ModelResponse):
            text_parts = [p.content for p in m.parts if isinstance(p, TextPart)]
            if text_parts:
                content = "".join(text_parts)

                # Try to parse content as JSON and convert to file
                file_data = None
                json_payload = None
                try:
                    parsed_content = json.loads(content)

                    # Handle single object format: {"file_name": "...", "file_path": "...", "file_type": "..."}
                    if isinstance(parsed_content, dict) and any(
                        k in parsed_content
                        for k in ["file_name", "file_path", "file_type", "job_id"]
                    ):
                        file_data = parsed_content.copy()
                    # Handle JSON payload format: {"json_type": "...", "json_data": ...}
                    elif (
                        isinstance(parsed_content, dict)
                        and "json_type" in parsed_content
                        and "json_data" in parsed_content
                    ):
                        json_payload = {
                            "type": parsed_content.get("json_type"),
                            "data": parsed_content.get("json_data"),
                        }

                    # Generate signed URL if file data exists
                    if (
                        file_data
                        and supabase_client
                        and file_data.get("file_path")
                        and not file_data.get("job_id")
                    ):
                        signed_url_response = supabase_client.storage.from_(
                            "lucy-files"
                        ).create_signed_url(file_data["file_path"], expires_in=3600)
                        file_data["signed_url"] = (
                            signed_url_response.get("signedURL", "")
                            if isinstance(signed_url_response, dict)
                            else str(signed_url_response)
                        )
                    elif file_data:
                        file_data["signed_url"] = ""

                except (json.JSONDecodeError, TypeError, Exception):
                    parsed_content = None
                    file_data = None
                    json_payload = None

                # Return message with file data if available
                if parsed_content and file_data:
                    return {
                        "role": "model-files",
                        "timestamp": m.timestamp.isoformat(),
                        "content": "",
                        "file": file_data,
                        "json": None,
                        "message_order": message_order,
                        "message_id": message_id,
                        "user_feedback": user_feedback,
                    }
                # Return message with json payload if available
                elif parsed_content and json_payload:
                    return {
                        "role": "model-json",
                        "timestamp": m.timestamp.isoformat(),
                        "content": "",
                        "file": None,
                        "json": json_payload,
                        "message_order": message_order,
                        "message_id": message_id,
                        "user_feedback": user_feedback,
                    }
                else:
                    return {
                        "role": "model",
                        "timestamp": m.timestamp.isoformat(),
                        "content": content,
                        "file": None,
                        "json": None,
                        "message_order": message_order,
                        "message_id": message_id,
                        "user_feedback": user_feedback,
                    }

        raise UnexpectedModelBehavior(f"Unsupported message type: {type(m).__name__}")

    except Exception as e:
        logger.error(f"Error converting message: {e}")
        raise UnexpectedModelBehavior(str(e))


class HistoryStore(ABC):
    """
    Abstract message history store with common helpers for trimming and sanitizing history.
    """

    @abstractmethod
    async def get_messages(
        self, user_id: str, session_id: str
    ) -> List[tuple[ModelMessage, int]]:
        """Fetch deserialized messages with their order for a given user's session."""

    @abstractmethod
    async def add_messages(
        self, user_id: str, session_id: str, messages: bytes
    ) -> None:
        """Append serialized message list bytes for a given user's session."""

    @staticmethod
    def tail_history(
        msgs: List[ModelMessage], limit: Optional[int] = None
    ) -> List[ModelMessage]:
        if limit is None or not msgs:
            return msgs or []
        return msgs[-limit:] if len(msgs) > limit else msgs

    @staticmethod
    def strip_effective_prompt_header(text: str, header: str) -> str:
        """
        Remove the per-turn reminder header and any internal labels from text.

        Handles both ``User request:`` and ``User reply:`` framing so that
        only the raw user message is persisted into chat history.
        """
        if not isinstance(text, str):
            return text

        trimmed = text.lstrip()

        # Remove the reminder header if present at the start
        if trimmed.startswith(header):
            trimmed = trimmed[len(header) :].lstrip()

        lower = trimmed.lower()
        # Try both "user reply:" (short-message framing) and "user request:"
        for prefix in ("user reply:", "user request:"):
            idx = lower.find(prefix)
            if idx != -1:
                trimmed = trimmed[idx + len(prefix) :].lstrip()
                break

        return trimmed

    @staticmethod
    def sanitize_history(
        msgs: List[ModelMessage],
        *,
        override_user_text: Optional[str] = None,
        override_model_text: Optional[str] = None,
        user_text_mapper: Optional[Callable[[str], str]] = None,
        header: Optional[str] = None,
    ) -> List[ModelMessage]:
        """
        Remove tool/function call artifacts from history to avoid mismatched tool outputs.

        Keeps only:
        - ModelRequest that contain a UserPromptPart (drop tool/system requests)
        - ModelResponse with only TextPart parts; drops responses with no text
        - ModelFileResponse messages (kept as-is)
        """
        # If an explicit override is provided (e.g., delegated sub-agent output),
        # persist exactly one user prompt and one model response.
        if override_model_text is not None:
            user_text = override_user_text or ""
            return [
                ModelRequest(parts=[UserPromptPart(user_text)]),
                ModelResponse(parts=[TextPart(str(override_model_text))]),
            ]
        cleaned: List[ModelMessage] = []
        for m in msgs:
            if isinstance(m, ModelRequest):
                # Keep only user prompts; drop any requests without a user prompt part
                has_user_prompt = any(isinstance(p, UserPromptPart) for p in m.parts)
                if has_user_prompt:
                    if user_text_mapper is not None:
                        new_parts = []
                        for p in m.parts:
                            if isinstance(p, UserPromptPart) and isinstance(
                                p.content, str
                            ):
                                new_parts.append(
                                    UserPromptPart(user_text_mapper(p.content))
                                )
                            else:
                                new_parts.append(p)
                        cleaned.append(ModelRequest(parts=new_parts))
                    else:
                        cleaned.append(m)
            elif isinstance(m, ModelFileResponse):
                # Keep file messages as-is
                cleaned.append(m)
            elif isinstance(m, ModelResponse):
                # Keep all text parts (preserve order) so trailing Sources lines are not dropped
                text_parts = [p for p in m.parts if isinstance(p, TextPart)]
                if text_parts:
                    cleaned.append(
                        ModelResponse(parts=text_parts, timestamp=m.timestamp)
                    )
        return cleaned

    @staticmethod
    def _message_char_count(msg: ModelMessage) -> int:
        """
        Compute the character count of a message based on visible text only.

        - For ModelRequest: sum lengths of UserPromptPart contents
        - For ModelResponse: sum lengths of TextPart contents
        - Other message types contribute 0
        """
        if isinstance(msg, ModelRequest):
            return sum(
                len(p.content)
                for p in msg.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            )
        if isinstance(msg, ModelResponse):
            return sum(
                len(p.content)
                for p in msg.parts
                if isinstance(p, TextPart) and isinstance(p.content, str)
            )
        return 0

    @classmethod
    def trim_history_by_chars(
        cls, msgs: List[ModelMessage], max_chars: int
    ) -> List[ModelMessage]:
        """
        Return a tail-slice of msgs such that the total visible text length
        is <= max_chars. Drops messages from the start while preserving order.
        """
        if max_chars <= 0 or not msgs:
            return []

        # Fast path: compute total and early return if within budget
        total = sum(cls._message_char_count(m) for m in msgs)
        if total <= max_chars:
            return msgs

        trimmed: List[ModelMessage] = msgs[:]
        while trimmed and total > max_chars:
            head = trimmed.pop(0)
            total -= cls._message_char_count(head)
        return trimmed

    @classmethod
    def trim_history_centered_by_chars(
        cls, msgs: List[ModelMessage], max_chars: int
    ) -> List[ModelMessage]:
        """
        Return a contiguous window of msgs centered around the most recent
        user request (ModelRequest) such that total visible text length
        is <= max_chars. If there is no user request, center on the last message.

        Preserves message ordering within the selected window.
        """
        if max_chars <= 0 or not msgs:
            return []

        # Fast path
        total = sum(cls._message_char_count(m) for m in msgs)
        if total <= max_chars:
            return msgs

        # Find pivot: last user request
        pivot_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], ModelRequest):
                pivot_idx = i
                break
        if pivot_idx == -1:
            pivot_idx = len(msgs) - 1

        # Expand window around pivot
        left = right = pivot_idx
        window_total = cls._message_char_count(msgs[pivot_idx])
        # Greedy expansion to nearest neighbors while within budget
        step = 1
        while window_total < max_chars and (left > 0 or right < len(msgs) - 1):
            # Try to take one on the left, then one on the right, alternating
            took = False
            if (step % 2 == 1) and left > 0:
                next_len = cls._message_char_count(msgs[left - 1])
                if window_total + next_len <= max_chars:
                    left -= 1
                    window_total += next_len
                    took = True
            if (not took) and right < len(msgs) - 1:
                next_len = cls._message_char_count(msgs[right + 1])
                if window_total + next_len <= max_chars:
                    right += 1
                    window_total += next_len
                    took = True
            if not took:
                break
            step += 1

        return msgs[left : right + 1]

    @staticmethod
    def _trim_message_visible_text(msg: ModelMessage, max_chars: int) -> ModelMessage:
        """
        Return a copy of msg where visible text parts are truncated to max_chars
        total for that message. Non-text parts are preserved unmodified.
        """
        if max_chars <= 0:
            # Keep structure but zero-out visible text parts
            max_chars = 0

        remaining = max_chars

        if isinstance(msg, ModelRequest):
            new_parts = []
            for p in msg.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    take = max(0, remaining)
                    new_text = p.content[:take]
                    remaining -= len(new_text)
                    new_parts.append(UserPromptPart(new_text))
                else:
                    new_parts.append(p)
            return ModelRequest(parts=new_parts)

        if isinstance(msg, ModelResponse):
            new_parts = []
            for p in msg.parts:
                if isinstance(p, TextPart) and isinstance(p.content, str):
                    take = max(0, remaining)
                    new_text = p.content[:take]
                    remaining -= len(new_text)
                    new_parts.append(TextPart(new_text))
                else:
                    new_parts.append(p)
            return ModelResponse(parts=new_parts, timestamp=msg.timestamp)

        return msg

    @classmethod
    def trim_each_message_by_chars(
        cls, msgs: List[ModelMessage], per_message_max_chars: int
    ) -> List[ModelMessage]:
        """
        Trim each message's visible text independently to per_message_max_chars.
        Preserves list length and message ordering.
        """
        return [cls._trim_message_visible_text(m, per_message_max_chars) for m in msgs]

    async def get_recent_messages(
        self, user_id: str, session_id: str, limit: Optional[int] = None
    ) -> List[ModelMessage]:
        msgs_with_order = await self.get_messages(user_id, session_id)
        msgs = [msg[0] for msg in msgs_with_order]
        return self.tail_history(msgs, limit)

    async def get_recent_sanitized_messages(
        self, user_id: str, session_id: str, limit: Optional[int] = None
    ) -> List[ModelMessage]:
        recent = await self.get_recent_messages(user_id, session_id, limit)
        return self.sanitize_history(recent)


class SupabaseHistoryStore(HistoryStore):
    """
    History store backed by Supabase table public.lucy_messages.

    conversation_id is the same as session_id.
    Stores one row per visible message with columns: message, role, created_at, message_order.
    Messages are sorted by message_order for consistent chronological ordering.
    """

    async def get_messages(
        self, user_id: str, session_id: str
    ) -> List[tuple[ModelMessage, int, int, Optional[bool]]]:
        logger = logging.getLogger(__name__)
        # Fetch rows synchronously via client in a thread to avoid blocking the loop
        try:
            rows = await asyncio.to_thread(fetch_lucy_messages, user_id, session_id)
        except Exception as e:
            logger.error(
                "fetch_lucy_messages failed for user_id=%s session_id=%s: %s",
                user_id, session_id, e,
                exc_info=True,
            )
            return []

        # Create a list of tuples with (message, message_order, message_id, user_feedback, attachments)
        message_data: List[tuple[ModelMessage, int, int, Optional[bool], Optional[list]]] = []

        for r in rows:
            text = r.get("message") or ""
            role = (r.get("role") or "").lower()
            created = r.get("created_at")
            message_order = r.get("message_order", 0)
            message_id = r.get("id", 0)
            user_feedback = r.get("user_feedback")
            row_attachments = r.get("attachments") or None
            ts: datetime
            try:
                # Supabase returns RFC3339/ISO timestamps; ensure timezone-aware
                ts = (
                    datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if isinstance(created, str)
                    else datetime.now(tz=timezone.utc)
                )
            except Exception:
                ts = datetime.now(tz=timezone.utc)

            if role == "user":
                msg = ModelRequest(parts=[UserPromptPart(text)])
                message_data.append((msg, message_order, message_id, user_feedback, row_attachments))
            elif role == "model":
                msg = ModelResponse(parts=[TextPart(text)], timestamp=ts)
                message_data.append((msg, message_order, message_id, user_feedback, None))
            elif role == "model-files":
                msg = ModelResponse(parts=[TextPart(text)], timestamp=ts)
                message_data.append((msg, message_order, message_id, user_feedback, None))
            else:
                # Skip unknown roles
                continue

        # Sort by message_order for consistent chronological ordering
        message_data.sort(key=lambda x: x[1])  # message_order

        # Return messages with their order
        return message_data

    async def add_messages(
        self, user_id: str, session_id: str, messages: List[ModelMessage]
    ) -> dict[str, List[int]]:
        # Process messages directly (no need to deserialize from bytes)
        try:
            msg_list: List[ModelMessage] = messages
        except Exception:
            # If messages are not valid, do nothing; we only persist structured messages
            return {"user": [], "model": [], "file": []}

        user_message_ids: List[int] = []
        model_message_ids: List[int] = []
        file_message_ids: List[int] = []

        # Track saved messages to avoid duplicates by content only
        saved_messages: set[str] = set()  # content strings

        # Ensure parent conversation exists to satisfy FK
        try:
            await asyncio.to_thread(
                ensure_lucy_conversation_exists, user_id, session_id
            )
        except Exception:
            pass

        async def insert_message(content: str, role: str) -> Optional[int]:
            """Helper function to insert a single message and return its ID."""
            # Check if this message content was already saved
            if content in saved_messages:
                return None

            try:
                message_id = await asyncio.to_thread(
                    insert_lucy_message,
                    {
                        "user_id": user_id,
                        "conversation_id": session_id,
                        "message": content,
                        "role": role,
                    },
                )
                # Mark this message content as saved
                if message_id:
                    saved_messages.add(content)
                return message_id
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(
                    "insert_lucy_message failed for user_id=%s conversation_id=%s role=%s: %s",
                    user_id, session_id, role, e,
                    exc_info=True,
                )
                return None

        # Skip sanitization - messages are already sanitized when passed in
        for m in msg_list:
            if isinstance(m, ModelRequest):
                # Persist first user-visible text part
                for p in m.parts:
                    if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                        message_id = await insert_message(p.content, "user")
                        if message_id:
                            user_message_ids.append(message_id)
                        break
            elif isinstance(m, ModelFileResponse):
                # Handle file messages
                parts_text = "".join(
                    p.content for p in m.parts if isinstance(p, TextPart)
                )
                message_id = await insert_message(parts_text, "model-files")
                if message_id:
                    file_message_ids.append(message_id)
            elif isinstance(m, ModelResponse):
                # Handle regular model responses
                parts_text = "".join(
                    p.content for p in m.parts if isinstance(p, TextPart)
                )
                message_id = await insert_message(parts_text, "model")
                if message_id:
                    model_message_ids.append(message_id)

        return {
            "user": user_message_ids,
            "model": model_message_ids,
            "file": file_message_ids,
        }
