from __future__ import annotations

import json
import mimetypes
import os
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, AsyncGenerator

from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

# ✅ Restored critical imports
from lucy.utils.auth import verify_jwt, extract_user_id
from lucy.database import get_user_org_profiles as fetch_user_org_profiles

from lucy.agents.router_agent import RouterAgent
from lucy.agents.common.models import (
    UserOrgProfile,
    SaveFileOutput,
    ModelFileResponse,
    JSONOutput,
)
from lucy.agents import ChatDeps
from lucy.query_understanding.service import build_query_context
from lucy.database.history import HistoryStore, to_chat_message
from lucy.database.supabase_client import (
    get_conversation_last_agent,
    set_conversation_last_agent,
    get_full_brand,
    update_lucy_message_attachments,
)
from .common.deps import get_history_store, get_supabase_client
from .common.config import (
    DEFAULT_SESSION_ID,
    RECENT_MSGS_LIMIT,
    STREAM_DEBOUNCE,
    HISTORY_CHAR_BUDGET,
    PER_MESSAGE_CHAR_LIMIT,
    HEARTBEAT_INTERVAL_S,
    MAX_STREAM_DURATION_S,
)
from .common.config import DEFAULT_AGENT_NAME

router = APIRouter()

AGENT_DESCRIPTIONS: dict[str, str] = {
    "lucy": "Lucy",
    "keywords": "Market Analyst",
    "support": "Support Agent",
    "image": "Image Designer",
    "video": "Video Designer",
    "performance": "Performance Analyst",
    "campaign_planner": "Campaign Planner",
    "creative_director": "Creative Director",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

_STREAM_DONE = object()


async def _anext_or_done(aiter):
    """Get next item from async iterator, or _STREAM_DONE on StopAsyncIteration."""
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return _STREAM_DONE


def _drain_status_queue(queue: asyncio.Queue) -> list[bytes]:
    """Non-blocking drain of the ChatDeps status_queue.

    Returns serialised tool_status NDJSON chunks for every message currently
    in the queue.  Safe to call from the async streaming generator because
    asyncio.Queue is single-loop and get_nowait() never blocks.
    """
    chunks = []
    while not queue.empty():
        try:
            value = queue.get_nowait()
            chunks.append(
                json.dumps({"type": "tool_status", "value": value}).encode() + b"\n"
            )
        except asyncio.QueueEmpty:
            break
    return chunks


def _user_friendly_error(exc: Exception) -> str:
    """Map an internal exception to a message safe to show in the chat UI."""
    if isinstance(exc, UsageLimitExceeded):
        return "Lucy ran into a complexity limit on this request. Please try rephrasing or breaking your request into smaller steps."
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "Lucy is taking longer than expected. Please try again."
    if isinstance(exc, HTTPException):
        if exc.status_code in (401, 403):
            return "Your session may have expired. Please refresh and try again."
        return "Something went wrong. Please try again or rephrase your request."
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name or "rate_limit" in str(exc).lower():
        return "Lucy is experiencing high demand. Please try again in a moment."
    return "Something went wrong. Please try again or rephrase your request."


async def save_user_message(
    user_id: str,
    session_id: str,
    message: str,
    history_store: HistoryStore,
    attachments: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, List[int]]]:
    """Save user message to history store immediately.

    Args:
        attachments: Optional list of dicts with file_name, file_type, storage_path.

    Returns:
        Dictionary with message IDs by type, or None if saving failed.
    """
    try:
        user_message = ModelRequest(
            parts=[UserPromptPart(content=message.strip())],
        )
        sanitized_user_msg = history_store.sanitize_history([user_message])
        message_results = await history_store.add_messages(
            user_id, session_id, sanitized_user_msg
        )
        logger.debug("✅ User message saved to history immediately")

        # Persist attachment metadata on the saved message row
        if attachments and message_results:
            user_ids = message_results.get("user", [])
            if user_ids:
                await asyncio.to_thread(
                    update_lucy_message_attachments, user_ids[0], attachments
                )

        return message_results
    except Exception as e:
        logger.warning(f"Failed to save user message immediately: {e}")
        return None


def create_file_message(
    file: SaveFileOutput, signed_url: str, job_id: str
) -> ModelFileResponse:
    """Create a ModelFileResponse from a SaveFileOutput and signed URL."""
    file_metadata = {
        "file_name": file.file_name,
        "file_path": file.file_path,
        "file_type": file.file_type,
        "asset_id": getattr(file, "asset_id", None),
        "signed_url": signed_url,
        "job_id": job_id,
    }
    return ModelFileResponse(
        parts=[TextPart(content=json.dumps(file_metadata))],
        timestamp=datetime.now(tz=timezone.utc),
    )


def create_json_message(json_output: JSONOutput) -> ModelResponse:
    """Create a ModelResponse message from a JSONOutput payload."""
    payload = {
        "json_type": json_output.json_type,
        "json_data": json_output.json_data,
    }
    return ModelResponse(
        parts=[TextPart(content=json.dumps(payload))],
        timestamp=datetime.now(tz=timezone.utc),
    )


def get_signed_url_for_file(
    supabase_client, file_path: str, expires_in: int = 3600
) -> str:
    """Get a signed URL for a file from Supabase storage."""
    signed_url_response = supabase_client.storage.from_("lucy-files").create_signed_url(
        file_path, expires_in=expires_in
    )
    return signed_url_response.get("signedURL", "")


async def save_model_response(
    user_id: str,
    session_id: str,
    response_text: str,
    streamed_files: List[SaveFileOutput],
    streamed_jsons: List[JSONOutput],
    header: str,
    history_store: HistoryStore,
    supabase_client,
) -> Optional[Dict[str, List[int]]]:
    """Save model response and file messages to history store.

    Args:
        user_id: User identifier
        session_id: Session identifier
        response_text: The model's response text
        streamed_files: List of files generated during streaming
        streamed_jsons: List of JSON payloads generated during streaming
        header: Reminder header for sanitization
        history_store: History store instance
        supabase_client: Supabase client for generating signed URLs

    Returns:
        Dictionary with message IDs by type, or None if saving failed.
    """
    try:
        # Create model response message
        response_msg = ModelResponse(
            parts=[TextPart(content=response_text)],
            timestamp=datetime.now(tz=timezone.utc),
        )
        messages = [response_msg]

        # Add file messages if any files were generated
        if streamed_files:
            for f in streamed_files:
                # Async video jobs: never generate a signed URL while the job is pending.
                # For synchronous files (no job_id), generate from the stored path.
                signed_url = (
                    None
                    if f.job_id
                    else get_signed_url_for_file(supabase_client, f.file_path)
                )
                file_msg = create_file_message(f, signed_url, f.job_id)
                messages.append(file_msg)

        # Add JSON messages if any JSON payloads were generated
        if streamed_jsons:
            for j in streamed_jsons:
                messages.append(create_json_message(j))

        # Sanitize messages (with header stripping) and store
        sanitized = history_store.sanitize_history(
            messages,
            user_text_mapper=lambda s: history_store.strip_effective_prompt_header(
                s, header=header
            ),
        )
        message_results = await history_store.add_messages(
            user_id, session_id, sanitized
        )
        logger.debug("✅ Model response saved to history")
        return message_results
    except Exception as e:
        logger.warning(f"Failed to save model response: {e}")
        return None


async def load_user_profiles(user_id: str) -> List[UserOrgProfile]:
    """Load user and organization profiles for ChatDeps.

    Returns:
        List of UserOrgProfile objects, empty list on failure.
    """
    try:
        timeout_s = float(os.getenv("PROFILE_TOOL_TIMEOUT_SEC", "2.0"))
        rows = await asyncio.wait_for(
            asyncio.to_thread(fetch_user_org_profiles, user_id),
            timeout=timeout_s,
        )
        return [UserOrgProfile(**row) for row in rows]
    except Exception as e:
        logger.debug(f"Profile tool failed: {e}")
        return []


def _attachment_metadata(path: str) -> Dict[str, str]:
    """Convert a Supabase storage path into storable attachment metadata."""
    file_name = path.split("/")[-1] if path else ""
    file_type, _ = mimetypes.guess_type(file_name)
    return {
        "file_name": file_name,
        "file_type": file_type or "application/octet-stream",
        "storage_path": path,
    }


def _build_profile_summary(profiles: List[UserOrgProfile]) -> str:
    """Build a compact brand-context string from user/org profiles."""
    if not profiles:
        return ""
    p = profiles[0]
    parts: List[str] = []
    if p.company_name:
        frag = p.company_name
        if p.industry:
            frag += f" ({p.industry})"
        parts.append(frag)
    if p.brand_name and p.brand_name != (p.company_name or ""):
        parts.append(f"Brand: {p.brand_name}")
    if p.audience:
        parts.append(f"Audience: {p.audience}")
    if p.states:
        parts.append(f"Location: {p.states}")
    if p.tone_of_voice:
        parts.append(f"Tone: {p.tone_of_voice}")
    if p.description:
        parts.append(p.description[:120])
    return "; ".join(parts) if parts else ""


_SHORT_REPLY_WORD_LIMIT = 5


def _extract_last_followup(history: List[ModelMessage]) -> Optional[str]:
    """Return the last question or statement from the most recent assistant message.

    For short user replies (e.g. 'And?'), we need context whether Lucy asked a
    question or made a statement. If the last message has a '?', return the last
    question. Otherwise return the last sentence (ending with . or !) so the model
    can interpret continuation prompts like 'And?' correctly.
    """
    for msg in reversed(history or []):
        if not isinstance(msg, ModelResponse):
            continue
        text = ""
        for part in msg.parts:
            if isinstance(part, TextPart) and part.content:
                text = part.content
        if not text:
            continue
        # Prefer last question (sentence ending with '?')
        idx = text.rfind("?")
        if idx != -1:
            start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx)) + 1
            result = text[start : idx + 1].strip()
            if result:
                return result
        # No question: return last statement (sentence ending with . or !)
        for end_char in ("!", "."):
            idx = text.rfind(end_char)
            if idx != -1:
                start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx)) + 1
                result = text[start : idx + 1].strip()
                if result and len(result) > 10:
                    return result
        # Fallback: last ~120 chars if message has no sentence-ending punctuation
        fallback = text.strip()[-120:].strip() if text.strip() else ""
        if fallback and len(fallback) > 15:
            return fallback
        return None
    return None


def _build_effective_prompt(
    header: str,
    message: str,
    profile_summary: str,
    last_followup: Optional[str] = None,
) -> str:
    """Assemble the per-turn prompt injected as the user message.

    For short replies (< _SHORT_REPLY_WORD_LIMIT words) with a recoverable
    follow-up question, frame as a reply rather than a standalone request so
    the model correctly interprets intent.

    ``last_followup`` should be extracted from UNTRIMMED history so that the
    question is not lost to per-message character truncation.
    """
    sections: List[str] = [header]

    if profile_summary:
        sections.append(f"Brand context: {profile_summary}")

    word_count = len(message.strip().split())
    followup = last_followup if word_count < _SHORT_REPLY_WORD_LIMIT else None

    if followup:
        sections.append(f'Your last message to the user: "{followup}"')
        sections.append(f"User reply: {message.strip()}")
    else:
        sections.append(f"User request: {message.strip()}")

    return "\n\n".join(sections)


async def yield_message_ids(
    message_results: Optional[Dict[str, List[int]]],
    message_type_prefix: str = "",
) -> AsyncGenerator[bytes, None]:
    """Yield message IDs as JSON-encoded stream chunks.

    Args:
        message_results: Dictionary with message IDs by type
        message_type_prefix: Optional prefix for message type (e.g., "user_", "model_")
    """
    if not message_results:
        return

    for message_type, message_ids in message_results.items():
        type_key = (
            f"{message_type_prefix}{message_type}"
            if message_type_prefix
            else message_type
        )
        for message_id in message_ids:
            yield json.dumps(
                {"type": f"{type_key}_message_id", "value": message_id}
            ).encode() + b"\n"


# ---------------------------------------------------------------------
# Context block builder (Tier 1: lightweight, always in prompt)
# ---------------------------------------------------------------------


def _build_context_block(
    context: Optional["ChatContext"],
    brand_data: Optional[Dict[str, Any]],
) -> str:
    """Build a compact context string for prompt injection.

    Keeps total output to ~200-400 tokens. Only includes non-null fields.
    """
    if context is None and brand_data is None:
        return ""

    parts: List[str] = ["CONTEXT:"]

    # Page context
    if context:
        page = context.page_type
        section = context.page_section
        if page:
            label = f"Page: {page}"
            if section:
                label += f" > {section}"
            parts.append(label)

    # Brand summary (lightweight -- name, industry, tone only)
    if brand_data:
        brand_bits: List[str] = []
        if brand_data.get("brand_name"):
            brand_bits.append(f"Brand: {brand_data['brand_name']}")
        if brand_data.get("brand_industry"):
            brand_bits.append(f"Industry: {brand_data['brand_industry']}")
        if brand_data.get("brand_tone_of_voice"):
            brand_bits.append(f"Tone: {brand_data['brand_tone_of_voice']}")
        if brand_bits:
            parts.append(" | ".join(brand_bits))

    # Campaign draft state
    if context and context.campaign_draft:
        d = context.campaign_draft
        parts.append("The user is creating a new campaign:")
        if d.campaign_name:
            parts.append(f"  Name: {d.campaign_name}")
        draft_bits: List[str] = []
        if d.goal:
            draft_bits.append(f"Goal: {d.goal}")
        if d.campaign_channel:
            draft_bits.append(f"Channel: {d.campaign_channel}")
        if d.ad_platforms:
            draft_bits.append(f"Platforms: {', '.join(d.ad_platforms)}")
        if d.total_budget_cents is not None:
            draft_bits.append(f"Budget: ${d.total_budget_cents / 100:,.0f}")
        if draft_bits:
            parts.append("  " + " | ".join(draft_bits))
        if d.locations:
            loc_names = [
                loc.get("name", "unknown") for loc in d.locations if isinstance(loc, dict)
            ]
            if loc_names:
                parts.append(f"  Locations: {', '.join(loc_names)}")
        if d.current_step is not None:
            parts.append(f"  Step {d.current_step} of 4")

    # Campaign detail (viewing/editing existing campaign)
    if context and context.campaign_detail:
        cd = context.campaign_detail
        detail_bits: List[str] = []
        if cd.campaign_name:
            detail_bits.append(f'Viewing campaign "{cd.campaign_name}"')
        if cd.campaign_status:
            detail_bits.append(f"status: {cd.campaign_status}")
        if cd.campaign_channel:
            detail_bits.append(f"channel: {cd.campaign_channel}")
        if cd.ad_platforms:
            detail_bits.append(f"platforms: {', '.join(cd.ad_platforms)}")
        if cd.campaign_goal:
            detail_bits.append(f"goal: {cd.campaign_goal}")
        if detail_bits:
            parts.append(" | ".join(detail_bits))

    # Active filters and visible campaigns summary
    if context and context.active_filters:
        af = context.active_filters
        filter_bits: List[str] = []
        if af.status:
            filter_bits.append(f"status={af.status}")
        if af.platforms:
            filter_bits.append(f"platforms={', '.join(af.platforms)}")
        if filter_bits:
            parts.append(f"Filters: {', '.join(filter_bits)}")

    if context and context.visible_campaigns_summary:
        vs = context.visible_campaigns_summary
        summary = f"Viewing {vs.total} campaigns"
        if vs.by_status:
            breakdown = ", ".join(f"{v} {k}" for k, v in vs.by_status.items())
            summary += f" ({breakdown})"
        parts.append(summary)

    # Date range
    if context and context.date_range:
        dr = context.date_range
        if dr.get("start_date") and dr.get("end_date"):
            parts.append(f"Date range: {dr['start_date']} to {dr['end_date']}")

    if len(parts) <= 1:
        return ""
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Schema for POST /stream
# ---------------------------------------------------------------------


class Attachment(BaseModel):
    """Attachment model for file uploads."""

    path: str = Field(..., description="Storage path of the file")
    url: str = Field(..., description="Signed URL for accessing the file")


class ActiveFilters(BaseModel):
    status: Optional[str] = None
    platforms: Optional[List[str]] = None
    searchWord: Optional[str] = None


class CampaignsSummary(BaseModel):
    total: int = 0
    by_status: Optional[Dict[str, int]] = None


class CampaignDraft(BaseModel):
    goal: Optional[str] = None
    target: Optional[List[str]] = None
    campaign_channel: Optional[str] = None
    ad_platforms: Optional[List[str]] = None
    locations: Optional[List[Dict[str, Any]]] = None
    campaign_start: Optional[str] = None
    campaign_end: Optional[str] = None
    total_budget_cents: Optional[int] = None
    campaign_name: Optional[str] = None
    current_step: Optional[int] = None


class CampaignDetail(BaseModel):
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    campaign_status: Optional[str] = None
    campaign_channel: Optional[str] = None
    ad_platforms: Optional[List[str]] = None
    campaign_goal: Optional[str] = None


class ChatContext(BaseModel):
    date_range: Optional[Dict[str, str]] = None
    selected_campaigns: Optional[List[str]] = None
    brand_id: Optional[str] = None
    page_type: Optional[str] = None
    page_section: Optional[str] = None
    active_filters: Optional[ActiveFilters] = None
    visible_campaigns_summary: Optional[CampaignsSummary] = None
    campaign_draft: Optional[CampaignDraft] = None
    campaign_detail: Optional[CampaignDetail] = None


class StreamReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    agent: Optional[str] = None
    session_id: str = Field(default=DEFAULT_SESSION_ID)
    user_location: Optional[str] = Field(
        None,
        description="User's position/context on the website for context-aware responses",
    )
    request_type: Optional[Literal["image", "video"]] = Field(
        None, description="Type of request: image, video, or null"
    )
    request_params: Optional[Dict[str, Any]] = Field(
        None, description="Parameters for image/video generation"
    )
    attachments: Optional[List[Attachment]] = Field(
        None, description="Uploaded file attachments (images for modification)"
    )
    context: Optional[ChatContext] = Field(
        None, description="Frontend page context: brand, page type, filters, campaign state"
    )


# ---------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------
class ChatFileResponse(BaseModel):
    """File attachment model for chat messages, extending SaveFileOutput with signed URL."""

    file_name: str = Field(..., description="Name of the saved file")
    file_path: str = Field(..., description="Public URL or storage path of the file")
    file_type: str = Field(..., description="MIME type or file type")
    asset_id: Optional[str] = Field(
        None, description="Optional creative asset ID (public.creative_assets.asset_id)"
    )
    signed_url: str = Field(..., description="Signed URL for accessing the file")
    job_id: Optional[str] = Field(
        None, description="Optional job identifier associated with the file/task"
    )


class ChatJsonResponse(BaseModel):
    """Structured JSON payload stored alongside chat messages."""

    type: str = Field(..., description="Schema/type name for the JSON payload")
    data: Any = Field(..., description="JSON payload data")


class UserMessageAttachment(BaseModel):
    """Attachment on a user message (re-signed from stored storage_path)."""

    file_name: str = Field(..., description="Display file name")
    file_type: str = Field(..., description="MIME type, e.g. image/png or application/pdf")
    signed_url: Optional[str] = Field(None, description="Fresh signed URL for viewing")


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(..., description="The ID of the message")
    role: Literal["user", "model", "model-files", "model-json"] = Field(
        ..., description="The role of the message sender"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp when the message was created"
    )
    content: str = Field(..., description="The text content of the message")
    file: Optional[ChatFileResponse] = Field(
        None, description="Optional file attachment"
    )
    json_data: Optional[ChatJsonResponse] = Field(
        None, alias="json", description="Optional structured JSON payload"
    )
    message_order: int = Field(
        ..., description="Order of the message in the conversation"
    )
    user_feedback: Optional[bool] = Field(
        None,
        description="User's vote on this message (true=upvote, false=downvote, null=no vote)",
    )
    attachments: Optional[List[UserMessageAttachment]] = Field(
        None, description="File attachments on user messages"
    )


class PaginatedChatResponse(BaseModel):
    items: List[ChatMessageResponse] = Field(..., description="Array of chat messages")
    total: int = Field(..., description="Total number of messages")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")


# ---------------------------------------------------------------------
# GET / — Retrieve chat history
# ---------------------------------------------------------------------
@router.get("/", response_model=PaginatedChatResponse)
async def get_chat(
    session_id: str = DEFAULT_SESSION_ID,
    history_store: HistoryStore = Depends(get_history_store),
    payload: Dict[str, Any] = Depends(verify_jwt),
    page: Optional[int] = None,
    size: Optional[int] = None,
    supabase_client=Depends(get_supabase_client),
) -> PaginatedChatResponse:
    logger = logging.getLogger(__name__)

    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        user_id = extract_user_id(payload)
        msgs_with_order = await history_store.get_messages(user_id, session_id.strip())

        # Calculate pagination parameters
        page_num = page or 1
        page_size = size or 20

        # Sort by message_order descending (newest first) for pagination
        msgs_with_order.sort(key=lambda x: x[1], reverse=True)

        # Calculate slice bounds
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size

        # Slice messages before expensive conversion
        paginated_msgs = msgs_with_order[start_idx:end_idx]

        chat_msgs: List[ChatMessageResponse] = []
        for i, (m, message_order, message_id, user_feedback, row_attachments) in enumerate(
            paginated_msgs
        ):
            try:
                cm = to_chat_message(
                    m, supabase_client, message_order, message_id, user_feedback,
                    attachments=row_attachments,
                )

                # Convert file dict to ChatFileResponse if present
                if "file" in cm and cm["file"]:
                    cm["file"] = ChatFileResponse(**cm["file"])

                # Convert json dict to ChatJsonResponse if present
                if "json" in cm and cm["json"]:
                    cm["json"] = ChatJsonResponse(**cm["json"])

                # Convert attachments list to response models if present
                if cm.get("attachments"):
                    cm["attachments"] = [
                        UserMessageAttachment(**a) for a in cm["attachments"]
                    ]

                chat_msgs.append(ChatMessageResponse(**cm))
            except UnexpectedModelBehavior:
                logger.debug(f"Skipping unsupported message {i}: {type(m).__name__}")

        # Return paginated response
        total_messages = len(msgs_with_order)
        total_pages = (total_messages + page_size - 1) // page_size

        paginated_result = type(
            "PaginatedResult",
            (),
            {
                "items": chat_msgs,
                "total": total_messages,
                "page": page_num,
                "size": page_size,
                "pages": total_pages,
            },
        )()

        return PaginatedChatResponse(
            items=paginated_result.items,
            total=paginated_result.total,
            page=paginated_result.page,
            size=paginated_result.size,
            pages=paginated_result.pages,
        )

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


# ---------------------------------------------------------------------
# POST /stream — Main streaming chat endpoint
# ---------------------------------------------------------------------
@router.post("/stream", response_class=StreamingResponse)
async def chat_stream(
    req: StreamReq,
    payload: Dict[str, Any] = Depends(verify_jwt),
    history_store: HistoryStore = Depends(get_history_store),
    supabase_client=Depends(get_supabase_client),
) -> StreamingResponse:
    """Stream structured agent output (message + files) incrementally, and save both user and model messages."""
    if req.session_id and not req.session_id.strip():
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        user_id = extract_user_id(payload)
        if not user_id:
            raise ValueError("Missing user_id in JWT payload")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

    sid = req.session_id.strip()

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            store = history_store
            history = await store.get_recent_sanitized_messages(
                user_id, sid, RECENT_MSGS_LIMIT
            )
            last_followup = _extract_last_followup(history)
            history = HistoryStore.trim_each_message_by_chars(
                history, PER_MESSAGE_CHAR_LIMIT
            )
            history = HistoryStore.trim_history_by_chars(
                history, HISTORY_CHAR_BUDGET
            )

            yield json.dumps({"type": "status", "value": "started"}).encode() + b"\n"

            # Save user message + load context data all at once in parallel.
            # save_user_message has no dependency on profiles/brand/last_agent,
            # so there is no reason to await it sequentially first.
            attachment_meta = (
                [_attachment_metadata(att.path) for att in req.attachments]
                if req.attachments
                else None
            )
            brand_id = req.context.brand_id if req.context else None

            save_msg_task = asyncio.create_task(
                save_user_message(user_id, sid, req.message, store, attachments=attachment_meta)
            )
            profiles_task = asyncio.create_task(load_user_profiles(user_id))
            last_agent_task = asyncio.create_task(
                asyncio.to_thread(get_conversation_last_agent, user_id, sid)
            )
            brand_task = (
                asyncio.create_task(asyncio.to_thread(get_full_brand, brand_id))
                if brand_id
                else None
            )

            gather_tasks: list = [save_msg_task, profiles_task, last_agent_task]
            if brand_task:
                gather_tasks.append(brand_task)
            results = await asyncio.gather(*gather_tasks, return_exceptions=True)

            # save_user_message handles its own exceptions and returns None on failure,
            # so results[0] is always Optional[Dict] rather than an Exception in practice.
            user_message_results = results[0] if not isinstance(results[0], Exception) else None
            async for chunk in yield_message_ids(user_message_results, "user_"):
                yield chunk

            profiles_result = results[1]
            last_agent_result = results[2]
            brand_result = results[3] if brand_task else None

            if isinstance(profiles_result, Exception):
                raise profiles_result
            profiles = profiles_result
            if isinstance(last_agent_result, Exception):
                logger.debug(f"get_conversation_last_agent failed: {last_agent_result}")
                last_agent = None
            else:
                last_agent = last_agent_result

            brand_data: Optional[Dict[str, Any]] = None
            if isinstance(brand_result, Exception):
                logger.debug(f"get_full_brand failed: {brand_result}")
            elif brand_result is not None:
                # Verify the requested brand belongs to one of the user's orgs.
                # profiles are guaranteed loaded at this point (exception raised above if not).
                user_org_ids = {p.org_id for p in profiles if p.org_id}
                brand_org = brand_result.get("associated_organization_id")
                if brand_org and brand_org in user_org_ids:
                    brand_data = brand_result
                else:
                    logger.warning(
                        f"Brand {brand_id} org {brand_org!r} not in user's orgs {user_org_ids!r} — discarding"
                    )

            query_context = None
            try:
                query_context = await build_query_context(
                    query=req.message,
                    user_id=user_id,
                )
                logger.info(
                    "Query context created: {}",
                    query_context.intent,
                )
            except Exception as exc:
                logger.exception("Query understanding failed: {}", exc)
                query_context = None

            # Route to correct agent (skip router if agent is explicitly provided)
            if req.agent and req.agent != DEFAULT_AGENT_NAME:
                (
                    route,
                    selected_agent_class,
                    selected_agent,
                ) = RouterAgent.get_agent_for_name(req.agent)
                logger.info(f"🎯 Using explicit agent: {route}")
            else:
                # Reuse the already-fetched (and trimmed) history — last 6 messages
                # are enough for routing context, saving a Supabase round-trip.
                routing_history = list(history)[-6:]
                attachments = (
                    [att.model_dump() for att in req.attachments]
                    if req.attachments
                    else None
                )
                (
                    route,
                    selected_agent_class,
                    selected_agent,
                ) = await RouterAgent.route_request(
                    message=req.message,
                    routing_history=routing_history,
                    user_id=user_id,
                    request_type=req.request_type,
                    attachments=attachments,
                    last_agent=last_agent,
                )
                logger.info(f"🔄 Router: '{req.message[:60]}...' → {route} agent")

            # Persist sticky routing only for router-selected agents to avoid
            # over-anchoring after one-off explicit agent picks in the UI.
            if not (req.agent and req.agent != DEFAULT_AGENT_NAME):
                asyncio.create_task(
                    asyncio.to_thread(set_conversation_last_agent, user_id, sid, route)
                )

            header = selected_agent_class.reminder_header()
            context_block = _build_context_block(req.context, brand_data)
            if context_block:
                effective_prompt = f"{context_block}\n\n{header}\n\nUser request: {req.message.strip()}"
            else:
                effective_prompt = f"{header}\n\nUser request: {req.message.strip()}"

            yield json.dumps({
                "type": "status",
                "value": f"using {route} agent",
            }).encode() + b"\n"
            yield json.dumps({
                "type": "status",
                "value": "generating",
                "agent": route,
                "agent_description": AGENT_DESCRIPTIONS.get(route, route),
            }).encode() + b"\n"

            # Ensure agent tools that inspect `ctx.deps.message_history` can see the CURRENT user message.
            # When using short-reply framing, do NOT append the raw message — the framework adds
            # effective_prompt as the new user message. Appending raw "and?" would create two user
            # messages (raw + framed) and confuse the model.
            history_with_current: List[ModelMessage] = list(history or [])
            word_count = len(req.message.strip().split())
            use_framed = (
                last_followup is not None and word_count < _SHORT_REPLY_WORD_LIMIT
            )
            if not use_framed:
                try:
                    history_with_current.append(
                        ModelRequest(parts=[UserPromptPart(content=req.message.strip())])
                    )
                except Exception:
                    history_with_current = list(history or [])

            # Build deps once and reuse across pre_run_check and run_stream so
            # that any state stored on deps during pre_run_check is visible to
            # the agent's tools without a redundant LLM call.
            run_deps = ChatDeps(
                user_id=user_id,
                user_profiles=profiles,
                message_history=history_with_current,
                user_location=req.user_location,
                request_type=req.request_type,
                request_params=req.request_params,
                attachments=(
                    [att.model_dump() for att in req.attachments]
                    if req.attachments
                    else None
                ),
                context=req.context.model_dump() if req.context else None,
                brand_context=brand_data,
                query_context=query_context.model_dump() if query_context else None,
            )
            pre_result = await selected_agent_class.pre_run_check(run_deps)
            if pre_result is not None:
                yield json.dumps(
                    {"type": "status", "value": "generating"}
                ).encode() + b"\n"
                full_response_text = pre_result.message
                yield json.dumps(
                    {"type": "delta", "value": full_response_text}
                ).encode() + b"\n"

                message_results = await save_model_response(
                    user_id,
                    sid,
                    full_response_text,
                    [],
                    [],
                    header,
                    store,
                    supabase_client,
                )
                async for chunk in yield_message_ids(message_results):
                    yield chunk
                yield json.dumps({"type": "status", "value": "completed"}).encode() + b"\n"
                return

            full_response_text = ""
            streamed_files: List[SaveFileOutput] = []
            streamed_jsons: List[JSONOutput] = []

            # Run up to 2 passes: the first streams the model's reply; if that
            # reply included tool calls whose results were never used in a final
            # answer (pydantic-ai stops at the first text output), a second pass
            # resumes from the tool-return messages so the model can produce a
            # complete response informed by the tool results.
            followup_messages: Optional[List[ModelMessage]] = None
            for pass_num in range(2):
                prompt_arg = None if pass_num == 1 else effective_prompt
                history_arg = followup_messages if pass_num == 1 else history_with_current

                async with selected_agent.run_stream(
                    prompt_arg,
                    message_history=history_arg,
                    deps=run_deps,
                    usage_limits=UsageLimits(request_limit=25),
                ) as result:
                    if pass_num == 0:
                        yield json.dumps(
                            {"type": "status", "value": "generating"}
                        ).encode() + b"\n"

                    prev = ""
                    stream_iter = result.stream_output(debounce_by=STREAM_DEBOUNCE).__aiter__()
                    stream_start = time.monotonic()
                    while True:
                        if time.monotonic() - stream_start > MAX_STREAM_DURATION_S:
                            raise asyncio.TimeoutError()
                        next_item = asyncio.ensure_future(_anext_or_done(stream_iter))
                        while not next_item.done():
                            await asyncio.wait({next_item}, timeout=HEARTBEAT_INTERVAL_S)
                            for _chunk in _drain_status_queue(run_deps.status_queue):
                                yield _chunk
                            if not next_item.done():
                                yield json.dumps({"type": "heartbeat"}).encode() + b"\n"

                        try:
                            snapshot = next_item.result()
                        except ValidationError as e:
                            if full_response_text.strip():
                                logger.warning(
                                    "Tail ValidationError after partial stream content; preserving streamed content and completing: %s",
                                    e,
                                )
                                break
                            raise

                        if snapshot is _STREAM_DONE:
                            break
                        # Snapshot may be structured (BaseModel) or plain string
                        if hasattr(snapshot, "message"):
                            text = snapshot.message or ""
                        else:
                            text = str(snapshot)

                        # Compute incremental delta only for message text
                        delta = text[len(prev) :] if text.startswith(prev) else text
                        prev = text

                        if delta:
                            full_response_text += delta
                            yield json.dumps(
                                {"type": "delta", "value": delta}
                            ).encode() + b"\n"

                        for _chunk in _drain_status_queue(run_deps.status_queue):
                            yield _chunk

                        # Capture any structured files for later use
                        if hasattr(snapshot, "files") and snapshot.files:
                            streamed_files = snapshot.files

                        # Capture any structured JSON payloads for later persistence (store-only)
                        if hasattr(snapshot, "jsons") and snapshot.jsons:
                            streamed_jsons = snapshot.jsons

                    # After streaming, check whether the model emitted tool calls
                    # that were processed but never answered (pydantic-ai stops
                    # streaming at the first text/output, leaving a trailing
                    # ModelRequest of tool-return parts in all_messages()).
                    # If so, schedule a follow-up pass.
                    if pass_num == 0:
                        updated_messages = result.all_messages()
                        last_msg = updated_messages[-1] if updated_messages else None
                        if (
                            isinstance(last_msg, ModelRequest)
                            and len(updated_messages) > len(history_with_current) + 1
                        ):
                            logger.info("Tool results pending after first pass — scheduling follow-up call")
                            followup_messages = updated_messages
                        else:
                            # No follow-up needed; break after this pass
                            followup_messages = None

                if followup_messages is None:
                    break

            # Streaming snapshots rarely include jsons (models don't pass
            # through complex tool results in structured output).  The
            # output validator on the agent injects them into the final
            # validated output, so read it here as the authoritative source.
            if not streamed_jsons:
                try:
                    final_output = result.output
                    if hasattr(final_output, "jsons") and final_output.jsons:
                        streamed_jsons = final_output.jsons
                except Exception:
                    pass

            if not streamed_files:
                try:
                    final_output = result.output
                    if hasattr(final_output, "files") and final_output.files:
                        streamed_files = final_output.files
                except Exception:
                    pass

            # Stream file information.
            # Always include file_name so the frontend can render the pending card.
            # Async video jobs (job_id set) never get a signed URL here; the polling
            # endpoint (jobs_video_api) provides it once generation completes.
            for f in streamed_files or []:
                signed_url = None
                if not f.job_id:
                    try:
                        signed_url = get_signed_url_for_file(
                            supabase_client, f.file_path
                        )
                    except Exception:
                        signed_url = None
                yield json.dumps(
                    {
                        "type": "file",
                        "file_name": f.file_name,
                        "file_type": f.file_type,
                        "signed_url": signed_url,
                        "job_id": f.job_id,
                    }
                ).encode() + b"\n"
            for j in streamed_jsons or []:
                yield json.dumps(
                    {
                        "type": j.json_type,
                        "data": j.json_data,
                    }
                ).encode() + b"\n"

            # Guard: if the stream completed without producing any text, emit a
            # fallback so the user never sees a blank response.
            if not full_response_text.strip():
                full_response_text = (
                    "I'm sorry, I wasn't able to generate a response. "
                    "Could you try rephrasing your question?"
                )
                logger.warning("Empty response after stream completed -- using fallback message")
                yield json.dumps(
                    {"type": "delta", "value": full_response_text}
                ).encode() + b"\n"

            # ✅ Save model response (user message already saved)
            message_results = await save_model_response(
                user_id,
                sid,
                full_response_text,
                streamed_files,
                streamed_jsons,
                header,
                store,
                supabase_client,
            )
            async for chunk in yield_message_ids(message_results):
                yield chunk

            yield json.dumps({"type": "status", "value": "completed"}).encode() + b"\n"

        except UsageLimitExceeded as e:
            logger.warning(f"Usage limit exceeded in chat streaming (route={route}): {e}")
            yield json.dumps({"type": "status", "value": "error"}).encode() + b"\n"
            yield json.dumps({"type": "delta", "value": _user_friendly_error(e)}).encode() + b"\n"
        except Exception as e:
            logger.opt(exception=True).error(f"Error in chat streaming: {e}")
            yield json.dumps({"type": "status", "value": "error"}).encode() + b"\n"
            yield json.dumps({"type": "delta", "value": _user_friendly_error(e)}).encode() + b"\n"

    return StreamingResponse(stream(), media_type="text/plain")
