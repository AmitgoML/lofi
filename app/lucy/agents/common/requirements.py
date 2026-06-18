"""Shared requirements-gathering framework for conversational agents.

Provides:
- Shared history utilities (compact_history_for_llm, iter_message_texts, get_latest_user_text)
- RequirementsState: abstract base for typed field accumulators
- InterviewResult: standardized result from evaluate_requirements()
- evaluate_requirements(): async LLM-based extraction of structured values from conversation history
- run_requirements_precheck(): shared pre-run helper combining history extraction + evaluate_requirements

Usage pattern:
    1. Define a dataclass that extends RequirementsState with your agent's fields.
    2. Implement is_ready, missing_fields, and merge.
    3. Call evaluate_requirements() inside your interview tool, passing your schema and rules.
    4. Merge the returned InterviewResult.suggested into your state.
    5. Gate execution on state.is_ready.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from lucy.agents.common.models import ChatDeps

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.settings import ModelSettings


# ---------------------------------------------------------------------------
# History utilities
# ---------------------------------------------------------------------------


def iter_message_texts(msg: Any) -> List[str]:
    """Extract non-empty text strings from a pydantic-ai message's parts."""
    parts = getattr(msg, "parts", None) or []
    out: List[str] = []
    for p in parts:
        t = getattr(p, "content", None)
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


def compact_history_for_llm(
    message_history: Optional[List[Any]],
    *,
    max_messages: int = 30,
    max_chars: int = 8000,
) -> Tuple[str, bool]:
    """Render the recent conversation history as a plain-text string for LLM input.

    Returns (text, truncated) where truncated=True means the history exceeded
    max_chars and was trimmed from the front.
    """
    if not message_history:
        return "", False

    recent = message_history[-max_messages:]
    lines: List[str] = []

    for msg in recent:
        role = getattr(msg, "role", None) or (
            "assistant" if isinstance(msg, ModelResponse) else "user"
        )
        texts = iter_message_texts(msg)
        if not texts:
            continue
        lines.append(f"{role}: {texts[0]}")

    joined = "\n".join(lines).strip()
    if len(joined) <= max_chars:
        return joined, False

    return joined[-max_chars:], True


def get_latest_user_text(message_history: Optional[List[Any]]) -> str:
    """Return the most recent user-authored text from the message history."""
    if not message_history:
        return ""

    for msg in reversed(message_history):
        if getattr(msg, "role", None) == "user":
            texts = iter_message_texts(msg)
            if texts:
                return texts[0]

    # Fallback: last message that has any text
    for msg in reversed(message_history):
        texts = iter_message_texts(msg)
        if texts:
            return texts[0]

    return ""


# ---------------------------------------------------------------------------
# InterviewResult
# ---------------------------------------------------------------------------


@dataclass
class InterviewResult:
    """Standardized output from evaluate_requirements().

    Attributes:
        enough_context: True when all required fields have been collected.
        suggested: Extracted field values to merge into the agent's state.
        missing_fields: Names of required fields not yet provided.
        one_question: A single batched question to ask the user next (ends with '?').
        force_defaults: True when history is too long to parse; agent should fill defaults.
        reason: Human-readable note on why the result was reached.
    """

    enough_context: bool
    suggested: Dict[str, Any]
    missing_fields: List[str]
    one_question: str
    force_defaults: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# RequirementsState base class
# ---------------------------------------------------------------------------


class RequirementsState(ABC):
    """Abstract base for typed field accumulators used during requirements gathering.

    Subclasses define their own fields as dataclass attributes, then implement:
    - is_ready: True when all required fields are present.
    - missing_fields: Names of fields still needed.
    - merge(suggested): Overwrite fields where a non-None value was newly extracted.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when all required fields have been collected."""

    @property
    @abstractmethod
    def missing_fields(self) -> List[str]:
        """Return the names of required fields not yet provided."""

    @abstractmethod
    def merge(self, suggested: Dict[str, Any]) -> None:
        """Merge newly extracted values into this state.

        Convention: only overwrite fields where suggested[key] is not None.
        Never downgrade a field that already has a value back to None.
        """

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serialize state as a dict for inclusion in LLM context.

        Default implementation returns all non-private instance attributes.
        Override if you need custom serialization.
        """
        return {
            k: v
            for k, v in vars(self).items()
            if not k.startswith("_")
        }


# ---------------------------------------------------------------------------
# Core LLM-based extraction
# ---------------------------------------------------------------------------

# No default model — callers pass model per-request via Agent.run(model=...)
_requirements_agent = Agent(
    system_prompt="Return only valid JSON. No markdown. No extra keys.",
)


async def evaluate_requirements(
    *,
    history_text: str,
    schema: Dict[str, Any],
    rules: str,
    fallback_question: str,
    brand_context: Optional[str] = None,
    model: str,
    max_output_tokens: int = 400,
) -> InterviewResult:
    """Extract structured requirements from conversation history via an LLM call.

    Each agent supplies its own schema and rules so field definitions stay
    domain-specific. The framework handles the mechanics of calling the API,
    parsing the response, and building a consistent InterviewResult.

    Args:
        history_text: Compact conversation history string (from compact_history_for_llm).
        schema: JSON schema dict describing the expected output fields.
        rules: Extraction instructions for the LLM (what to look for, how to coerce, etc.).
        fallback_question: Question to return when extraction fails or history is empty.
        brand_context: Optional brand/org name to include for personalization.
        model: LLM model identifier to use for extraction.
        max_output_tokens: Token budget for the extraction call.

    Returns:
        InterviewResult with extracted values, readiness flag, and next question.
    """
    if not history_text.strip():
        return InterviewResult(
            enough_context=False,
            suggested={},
            missing_fields=[],
            one_question=fallback_question,
            force_defaults=False,
            reason="empty_history",
        )

    payload: Dict[str, Any] = {
        "rules": rules,
        "schema": schema,
        "history": history_text,
    }
    if brand_context:
        payload["brand_context"] = brand_context

    try:
        result = await _requirements_agent.run(
            json.dumps(payload),
            model=model,
            model_settings=ModelSettings(temperature=0.0, max_tokens=max_output_tokens),
        )
        text = (result.output or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text) if text else {}
    except Exception as e:
        logger.warning(f"evaluate_requirements extraction failed, using fallback: {e}")
        data = {}

    enough = bool(data.get("enough_context"))

    suggested = {k: v for k, v in data.items() if k not in {
        "enough_context", "missing_fields", "one_question", "notes", "reason"
    } and v is not None}

    missing = data.get("missing_fields")
    if not isinstance(missing, list):
        missing = []

    one_q = data.get("one_question")
    if not isinstance(one_q, str) or not one_q.strip().endswith("?"):
        one_q = fallback_question

    reason = data.get("notes") or data.get("reason") or ""
    if not isinstance(reason, str):
        reason = ""

    return InterviewResult(
        enough_context=enough,
        suggested=suggested,
        missing_fields=missing,
        one_question=one_q.strip(),
        force_defaults=False,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Pre-run check helper
# ---------------------------------------------------------------------------


async def run_requirements_precheck(
    *,
    deps: "ChatDeps",
    schema: Dict[str, Any],
    rules: str,
    fallback_question: str,
    model: str,
    max_messages: int = 30,
    max_chars: int = 8000,
) -> Tuple[Optional[InterviewResult], Optional[str]]:
    """Shared pre-run flow: extract history, short-circuit if empty, call evaluate_requirements.

    Returns (result, brand) where:
    - result is None when history is empty (caller should return PreRunResult with
      fallback_question directly), otherwise an InterviewResult from evaluate_requirements.
    - brand is the resolved brand/company name from the first user profile, or None.

    The caller is responsible for:
    - Returning PreRunResult(message=fallback_question) when result is None.
    - Merging result.suggested into their state object.
    - Setting agent-specific flags on deps when state.is_ready.
    """
    from lucy.agents.common.models import get_brand_name  # avoid circular import at module level

    message_history = getattr(deps, "message_history", None)
    history_text, _ = compact_history_for_llm(
        message_history, max_messages=max_messages, max_chars=max_chars,
    )

    if not history_text.strip():
        return None, None

    profiles = getattr(deps, "user_profiles", None) or []
    brand = get_brand_name(profiles)

    result = await evaluate_requirements(
        history_text=history_text,
        schema=schema,
        rules=rules,
        fallback_question=fallback_question,
        brand_context=brand,
        model=model,
    )

    return result, brand
