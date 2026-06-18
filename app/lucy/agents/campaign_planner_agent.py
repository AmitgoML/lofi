from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.base_agent import LofiAgent, PreRunResult
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import (
    ChatDeps,
    FileAgentOutput,
    JSONOutput,
    UserOrgProfile,
    get_brand_name,
)
from lucy.agents.common.requirements import (
    RequirementsState,
    InterviewResult,
    compact_history_for_llm,
    evaluate_requirements,
    run_requirements_precheck,
)
from lucy.agents.common.tools import register_user_org_profiles_tool
from lucy.agents.common.context_tools import register_brand_context_tool
from lucy.database import get_user_org_profiles


GoalType = Literal["conversions", "traffic", "awareness"]

DEFAULT_GOAL: GoalType = "conversions"
DEFAULT_DAILY_BUDGET_CENTS: int = 0
DEFAULT_TOTAL_BUDGET_CENTS: int = 0
DEFAULT_TARGET: List[str] = ["new"]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _default_campaign_name(
    profiles: Optional[List[UserOrgProfile]],
) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    brand = get_brand_name(profiles)
    prefix = f"Draft by Lucy - {brand}" if brand else "Draft by Lucy"
    return f"{prefix} ({ts})"


@dataclass
class CampaignPlanningState(RequirementsState):
    """Typed accumulator for campaign values collected during the interview phase.

    One instance is stored on ctx.deps.campaign_planning_state per request.
    Values are extracted from the full conversation history by evaluate_requirements
    and merged in on each call to campaign_interview_tool.
    """

    goal: Optional[GoalType] = None
    target: Optional[List[str]] = None
    budget_type: Optional[str] = None  # "per day" | "total"
    daily_budget_cents: Optional[int] = None
    total_budget_cents: Optional[int] = None
    campaign_channel: Optional[str] = None
    ad_platforms: Optional[List[str]] = None

    @property
    def is_ready(self) -> bool:
        if not all(
            [
                self.goal is not None,
                self.target is not None,
                self.budget_type is not None,
            ]
        ):
            return False
        # Only the budget matching the declared budget_type is required
        if self.budget_type == "per day":
            return self.daily_budget_cents is not None
        return self.total_budget_cents is not None

    @property
    def is_ready_to_draft(self) -> bool:
        """Alias for is_ready kept for backwards compatibility."""
        return self.is_ready

    @property
    def missing_fields(self) -> List[str]:
        missing = []
        if self.goal is None:
            missing.append("goal")
        if self.target is None:
            missing.append("target")
        if self.budget_type is None:
            # Ask for budget type and amount together
            missing.append("budget")
        elif self.budget_type == "per day" and self.daily_budget_cents is None:
            missing.append("daily_budget")
        elif self.budget_type == "total" and self.total_budget_cents is None:
            missing.append("total_budget")
        return missing

    def merge(self, suggested: Dict[str, Any]) -> None:
        """Overwrite only fields where a non-None value was newly extracted."""
        if suggested.get("goal") is not None:
            self.goal = suggested["goal"]
        if suggested.get("target") is not None:
            self.target = suggested["target"]
        if suggested.get("budget_type") is not None:
            self.budget_type = suggested["budget_type"]
        if suggested.get("daily_budget_cents") is not None:
            self.daily_budget_cents = suggested["daily_budget_cents"]
        if suggested.get("total_budget_cents") is not None:
            self.total_budget_cents = suggested["total_budget_cents"]
        if suggested.get("campaign_channel") is not None:
            self.campaign_channel = suggested["campaign_channel"]
        if suggested.get("ad_platforms") is not None:
            self.ad_platforms = suggested["ad_platforms"]


def _coerce_goal(val: Any) -> Optional[GoalType]:
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v in ("conversions", "traffic", "awareness"):
        return v  # type: ignore[return-value]
    return None


def _coerce_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        # Strip commas, currency symbols, whitespace, and common unit suffixes
        # so values like "$50", "5,000", "50.00", "$50/day" all parse correctly.
        s = val.strip().replace(",", "")
        s = re.sub(r"[^\d.\-]", "", s)
        if not s:
            return None
        try:
            return int(float(s))
        except (ValueError, OverflowError):
            return None
    return None


def _coerce_budget_cents(val: Any) -> Optional[int]:
    """Coerce a budget value to a positive integer in cents.

    If the raw value is a string containing '$', the LLM returned a dollar-sign
    expression (e.g. "$50/day") — strip non-digits and multiply by 100 to convert
    dollars to cents.  Plain numeric values are assumed to already be in cents per
    the LLM prompt instructions.
    Returns None for 0, negative, None, or unparseable values — 0 means "not
    provided by the user", not a $0 budget.
    """
    if val is None or isinstance(val, bool):
        return None

    has_dollar_sign = isinstance(val, str) and "$" in val

    if isinstance(val, str):
        s = val.strip().replace(",", "")
        s = re.sub(r"[^\d.\-]", "", s)
        if not s:
            return None
        try:
            numeric = int(float(s))
        except (ValueError, OverflowError):
            return None
    elif isinstance(val, int):
        numeric = val
    elif isinstance(val, float):
        numeric = int(val)
    else:
        return None

    if numeric <= 0:
        return None

    # LLM returned a dollar-sign string (e.g. "$50") — treat as dollars, convert to cents
    if has_dollar_sign:
        return numeric * 100

    return numeric


def _coerce_str_list(val: Any) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        out = [x.strip() for x in val if isinstance(x, str) and x.strip()]
        return out or None
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return None


_BUDGET_TYPE_ALIASES: Dict[str, str] = {
    "daily": "per day",
    "per_day": "per day",
    "per-day": "per day",
    "day": "per day",
    "lifetime": "total",
    "overall": "total",
    "lump sum": "total",
    "total budget": "total",
}


def _coerce_budget_type(val: Any) -> Optional[str]:
    """Coerce a budget type string to one of the canonical values: 'per day' or 'total'."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if v in ("per day", "total"):
        return v
    return _BUDGET_TYPE_ALIASES.get(v)


_CHANNEL_ALIASES: Dict[str, str] = {
    "social media": "social",
    "paid social": "social",
    "social ads": "social",
    "paid search": "search",
    "sem": "search",
    "ppc": "search",
    "google ads": "search",
    "connected tv": "ctv",
    "streaming tv": "ctv",
    "ott": "ctv",
    "audio": "digital_audio",
    "podcast": "digital_audio",
    "streaming audio": "digital_audio",
    "digital audio": "digital_audio",
    "google shopping": "shopping",
    "product listing": "shopping",
}


def _coerce_channel(val: Any) -> Optional[str]:
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    valid = {"social", "search", "display", "ctv", "digital_audio", "shopping"}
    if v in valid:
        return v
    # Try alias lookup before giving up
    return _CHANNEL_ALIASES.get(v)


def build_campaign_draft_json_data(
    *,
    user_id: str,
    org_id: str,
    goal: GoalType = DEFAULT_GOAL,
    daily_budget_cents: int = DEFAULT_DAILY_BUDGET_CENTS,
    total_budget_cents: int = DEFAULT_TOTAL_BUDGET_CENTS,
    budget_type: str = "per day",
    target: Optional[List[str]] = None,
    campaign_name: Optional[str] = None,
    now_iso: Optional[str] = None,
    campaign_channel: Optional[str] = None,
    ad_platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    now = now_iso or _now_iso()
    draft: Dict[str, Any] = {
        "user_id": user_id,
        "org_id": org_id,
        "is_paid": False,
        "campaign_status": "DRAFT",
        "campaign_version": 0,
        "goal": goal,
        "target": target or DEFAULT_TARGET,
        "campaign_start": now,
        "campaign_end": now,
        "daily_budget_cents": int(daily_budget_cents),
        "total_budget_cents": int(total_budget_cents),
        "budget_type": budget_type,
        "campaign_name": campaign_name or "Draft by Lucy",
        "locations": [],
        "creative_optimization": True,
        "audience_optimization": True,
        "budget_optimization": True,
        "automated_reporting": False,
    }
    if campaign_channel:
        draft["campaign_channel"] = campaign_channel
    if ad_platforms:
        draft["ad_platforms"] = ad_platforms
    return draft


class CampaignPlannerAgent(LofiAgent):
    REMINDER_HEADER = (
        "You are Lucy Campaign Planner — friendly, clear, action-oriented. "
        "Ask up to TWO questions per turn. Personalize using brand profile; "
        "if unavailable say you're using general best practices. Never narrate tool usage. "
        "If a draft exists, ask user to click Create to refine in UI. "
        "Use campaign_interview_tool before campaign_draft_tool. "
        "Use get_user_org_profiles_tool silently only when profiles have not been attempted yet."
    )

    @classmethod
    async def pre_run_check(cls, deps: ChatDeps) -> Optional[PreRunResult]:
        """Run the campaign interview before the orchestrator.

        If the conversation has no history, or all required fields are clearly
        missing, return the clarifying question directly — no orchestrator LLM
        call needed. When ready, cache computed state so campaign_interview_tool
        can skip its own evaluate_requirements() call.
        """
        result, _ = await run_requirements_precheck(
            deps=deps,
            schema=cls._EXTRACTION_SCHEMA,
            rules=cls._EXTRACTION_RULES,
            fallback_question=cls._FALLBACK_QUESTION,
            model=to_responses_model(Models.CAMPAIGN_PLANNER),
        )

        if result is None:
            return PreRunResult(message=cls._FALLBACK_QUESTION)

        coerced = {
            "goal": _coerce_goal(result.suggested.get("goal")),
            "budget_type": _coerce_budget_type(result.suggested.get("budget_type")),
            "daily_budget_cents": _coerce_budget_cents(result.suggested.get("daily_budget_cents")),
            "total_budget_cents": _coerce_budget_cents(result.suggested.get("total_budget_cents")),
            "target": _coerce_str_list(result.suggested.get("target")),
            "campaign_channel": _coerce_channel(result.suggested.get("campaign_channel")),
            "ad_platforms": _coerce_str_list(result.suggested.get("ad_platforms")),
        }
        result.suggested = {k: v for k, v in coerced.items() if v is not None}

        state = CampaignPlanningState()
        state.merge(result.suggested)

        if not state.is_ready:
            logger.info(
                f"Campaign Planner pre-check: not ready "
                f"(missing={state.missing_fields}), asking question"
            )
            return PreRunResult(message=result.one_question)

        # Cache computed state so campaign_interview_tool can skip its LLM call.
        deps.campaign_planning_state = state
        deps.planning_phase_complete = True
        deps.campaign_interview_used = True

        logger.info(
            f"Campaign Planner pre-check: ready "
            f"(goal={state.goal!r}), proceeding to agent"
        )
        return None

    @classmethod
    async def _ensure_profiles(cls, ctx: RunContext[ChatDeps]) -> List[UserOrgProfile]:
        # None means profiles were never attempted; [] means we already tried and found
        # nothing — don't hit the DB again in either the empty or the preloaded case.
        if ctx.deps.user_profiles is not None:
            return ctx.deps.user_profiles

        rows = await asyncio.to_thread(get_user_org_profiles, ctx.deps.user_id)
        profiles = [UserOrgProfile(**r) for r in rows]
        ctx.deps.user_profiles = profiles
        return profiles

    _EXTRACTION_RULES = (
        "Extract campaign planning values from the conversation. "
        "enough_context is true ONLY when ALL three required fields are non-null: "
        "goal, target, and budget_type + the relevant budget amount. "
        "campaign_channel and ad_platforms are optional — extract them if mentioned, "
        "but do NOT require them for enough_context=true. "
        "If ANY required field is missing or unclear, set enough_context=false. "
        "Extract whatever partial values have been mentioned even when not everything is present. "
        "BUDGET TYPE: Determine whether the user specified a daily budget or a total budget. "
        "Set budget_type to 'per day' if they mention a per-day amount (e.g. '$50/day', '$50 per day'). "
        "Set budget_type to 'total' if they mention a total/overall amount (e.g. '$5,000 total', '$2k budget'). "
        "If budget_type is 'per day', populate daily_budget_cents and leave total_budget_cents null. "
        "If budget_type is 'total', populate total_budget_cents and leave daily_budget_cents null. "
        "BUDGET AMOUNTS: Return budget values as integers in cents (multiply dollars by 100). "
        "IMPORTANT: $50/day = 5000 cents, NOT 50. $1,000 total = 100000 cents, NOT 1000. "
        "Never include dollar signs or units in numeric fields — return plain integers only. "
        "CHANNEL (optional): If mentioned, return exactly one of these canonical values: "
        "social, search, display, ctv, digital_audio, shopping. "
        "Map variants like 'social media' -> social, 'paid search' -> search, "
        "'SEM' -> search, 'connected TV' -> ctv, 'digital audio' -> digital_audio. "
        "PLATFORMS (optional): If mentioned, return each platform as a separate lowercase string "
        "(e.g. facebook, instagram, google, tiktok, bing, pinterest, youtube). "
        "Return strict JSON only."
    )

    _EXTRACTION_SCHEMA = {
        "enough_context": "boolean — true only if goal, target, and budget_type + relevant budget are all non-null (campaign_channel and ad_platforms are optional)",
        "goal": "conversions|traffic|awareness|null",
        "target": "list[str] (values: new, existing)|null",
        "budget_type": "per day|total|null",
        "daily_budget_cents": "int in cents if budget_type is 'per day' (e.g. $50/day = 5000)|null",
        "total_budget_cents": "int in cents if budget_type is 'total' (e.g. $1000 total = 100000)|null",
        "campaign_channel": "optional — social|search|display|ctv|digital_audio|shopping|null",
        "ad_platforms": "optional — list[str] lowercase platform names (e.g. facebook, instagram, google, tiktok, bing)|null",
        "missing_fields": "list[str] of required field names not yet provided (only goal, target, budget)",
        "one_question": "string ending with ? — ask for the next most important missing info, batch related fields",
        "notes": "string",
    }

    _FALLBACK_QUESTION = (
        "To get started, what's your campaign goal (awareness/traffic/conversions), "
        "which channel you'd like to run on (social/search/display/ctv), "
        "and what budget you have in mind (e.g. $50/day or $2,000 total)?"
    )

    @classmethod
    async def _evaluate_context(
        cls,
        *,
        ctx: RunContext[ChatDeps],
        max_messages: int,
        max_chars: int,
    ) -> InterviewResult:
        """Extract campaign planning fields from conversation history.

        Returns force_defaults=True when history is truncated (too long to parse);
        the caller should fill in defaults for any missing required fields.
        """
        history_text, truncated = compact_history_for_llm(
            ctx.deps.message_history,
            max_messages=max_messages,
            max_chars=max_chars,
        )

        if truncated:
            return InterviewResult(
                enough_context=True,
                suggested={},
                missing_fields=[],
                one_question="What campaign channel and ad platforms would you like to use?",
                force_defaults=True,
                reason="conversation_too_long",
            )

        profiles = ctx.deps.user_profiles or []
        brand = get_brand_name(profiles)

        result = await evaluate_requirements(
            history_text=history_text,
            schema=cls._EXTRACTION_SCHEMA,
            rules=cls._EXTRACTION_RULES,
            fallback_question=cls._FALLBACK_QUESTION,
            brand_context=brand,
            model=to_responses_model(Models.CAMPAIGN_PLANNER),
        )

        # Coerce raw LLM values into typed domain values
        coerced = {
            "goal": _coerce_goal(result.suggested.get("goal")),
            "budget_type": _coerce_budget_type(result.suggested.get("budget_type")),
            "daily_budget_cents": _coerce_budget_cents(result.suggested.get("daily_budget_cents")),
            "total_budget_cents": _coerce_budget_cents(result.suggested.get("total_budget_cents")),
            "target": _coerce_str_list(result.suggested.get("target")),
            "campaign_channel": _coerce_channel(result.suggested.get("campaign_channel")),
            "ad_platforms": _coerce_str_list(result.suggested.get("ad_platforms")),
        }
        result.suggested = {k: v for k, v in coerced.items() if v is not None}
        return result

    @classmethod
    def _select_tools(cls, deps: Any, tool_defs: List[Any]) -> List[Any]:
        """Pure gating logic for the agent's prepare_tools callback.

        Extracted as a classmethod so it can be unit-tested without constructing
        a real pydantic-ai Agent.

        Phase sequence:
          1. If profiles were never attempted (None), expose get_user_org_profiles_tool.
             A preloaded value — even an empty list [] — means chat.py already ran the
             lookup; re-exposing the tool prompts the model to narrate internal steps.
          2. Once profiles are known (not None), move to campaign_interview_tool.
          3. Once interview signals readiness, expose campaign_draft_tool.
          4. After draft is emitted, no tools (model produces final text output).
        """
        def _by_name(name: str) -> List[Any]:
            return [t for t in tool_defs if getattr(t, "name", "") == name]

        if getattr(deps, "draft_emitted", False):
            return []

        # Only offer the profile tool when profiles have never been attempted at all.
        if deps.user_profiles is None:
            return _by_name("get_user_org_profiles_tool")

        interview_used = bool(getattr(deps, "campaign_interview_used", False))
        planning_complete = bool(getattr(deps, "planning_phase_complete", False))

        if not interview_used:
            return _by_name("campaign_interview_tool")

        if planning_complete:
            return _by_name("campaign_draft_tool")

        # Interview ran but not enough info — model must produce clarifying output
        return []

    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.CAMPAIGN_PLANNER)

        async def _prepare_tools(ctx: RunContext[ChatDeps], tool_defs: List[Any]):
            """Enforce the phased 3-tool sequence per turn.

            Order:
              1. get_user_org_profiles_tool  (used silently when profiles not been attempted)
              2. campaign_interview_tool      (while planning phase is incomplete)
              3. campaign_draft_tool          (only once interview signals readiness)

            After campaign_draft_tool runs, remove all tools so the model produces final output.
            After campaign_interview_tool runs without readiness, remove tools so the model asks questions.
            """
            if ctx.deps.draft_emitted:
                return []

            if not ctx.deps.user_profiles_loaded:
                return [
                    t
                    for t in tool_defs
                    if getattr(t, "name", "") == "get_user_org_profiles_tool"
                ]

            interview_used = ctx.deps.campaign_interview_used
            planning_complete = ctx.deps.planning_phase_complete

            if not interview_used:
                return [
                    t
                    for t in tool_defs
                    if getattr(t, "name", "") == "campaign_interview_tool"
                ]

            if planning_complete:
                return [
                    t
                    for t in tool_defs
                    if getattr(t, "name", "") == "campaign_draft_tool"
                ]

            # Interview ran but not enough info — model must produce clarifying output
            return []

        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.3, max_tokens=1200),
            deps_type=ChatDeps,
            system_prompt=CAMPAIGN_PLANNER_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
            prepare_tools=_prepare_tools,
        )

        register_user_org_profiles_tool(agent)

        @agent.output_validator
        async def _ensure_draft_json_emitted(
            ctx: RunContext[ChatDeps],
            output: FileAgentOutput,
        ) -> FileAgentOutput:
            """Ensure the UI receives the draft JSON when the draft tool created one."""
            pending = ctx.deps.campaign_planner_pending_jsons
            if pending and (not getattr(output, "jsons", None)):
                output.jsons = pending

            # Campaign planner should not emit files — hallucinated `files` can crash streaming
            if getattr(output, "files", None):
                output.files = []
            return output

        @agent.tool
        async def campaign_interview_tool(
            ctx: RunContext[ChatDeps],
            json_type: str = Field("option_draft_campaign"),
            max_messages: int = Field(30),
            max_chars: int = Field(8000),
        ) -> Dict[str, Any]:
            """Phase 1 — Interview tool.

            Evaluates the conversation history to extract campaign planning values,
            accumulates them in CampaignPlanningState, and signals when all required
            fields have been collected (setting planning_phase_complete=True).

            Required fields: goal, target, daily_budget_cents, total_budget_cents,
            campaign_channel, ad_platforms.
            """
            ctx.deps.status_queue.put_nowait("Gathering campaign details")
            ctx.deps.campaign_interview_used = True

            profiles = await cls._ensure_profiles(ctx)

            # Guard against repeated calls within the same turn
            interview_calls = ctx.deps.campaign_interview_calls
            ctx.deps.campaign_interview_calls = interview_calls + 1
            if interview_calls >= 1:
                assistant_message = (
                    "I need a few more details to build a solid campaign draft. "
                    "Could you share your campaign goal, target audience, budgets, "
                    "and preferred channel and platform?"
                )
                return {
                    "message": assistant_message,
                    "files": [],
                    "jsons": [],
                    "enough_context": False,
                    "missing_fields": [],
                    "one_question": (
                        "Could you share your campaign goal, target audience, "
                        "budgets, and preferred channel and platform?"
                    ),
                    "assistant_message": assistant_message,
                    "reason": "interview_tool_called_multiple_times_in_same_turn",
                }

            ctx_eval = await cls._evaluate_context(
                ctx=ctx,
                max_messages=max_messages,
                max_chars=max_chars,
            )

            force_defaults = ctx_eval.force_defaults

            # Build / update planning state from extracted values
            state = CampaignPlanningState()
            state.merge(ctx_eval.suggested)

            if force_defaults:
                # Conversation too long — fill in defaults for anything missing
                if state.goal is None:
                    state.goal = DEFAULT_GOAL
                if state.target is None:
                    state.target = DEFAULT_TARGET
                if state.budget_type is None:
                    state.budget_type = "per day"
                if state.daily_budget_cents is None:
                    state.daily_budget_cents = DEFAULT_DAILY_BUDGET_CENTS
                if state.total_budget_cents is None:
                    state.total_budget_cents = DEFAULT_TOTAL_BUDGET_CENTS

            ctx.deps.campaign_planning_state = state

            if state.is_ready:
                ctx.deps.planning_phase_complete = True
                assistant_message = (
                    "I have everything I need to build a draft! "
                    "Let me put that together for you now."
                )
                return {
                    "message": assistant_message,
                    "files": [],
                    "jsons": [],
                    "enough_context": True,
                    "missing_fields": [],
                    "one_question": ctx_eval.one_question,
                    "assistant_message": assistant_message,
                    "reason": ctx_eval.reason,
                }

            missing = state.missing_fields
            assistant_message = (
                "To build a draft that makes sense for you, I still need a few details. "
                f"{ctx_eval.one_question}"
            )
            return {
                "message": assistant_message,
                "files": [],
                "jsons": [],
                "enough_context": False,
                "missing_fields": missing,
                "one_question": ctx_eval.one_question,
                "assistant_message": assistant_message,
                "reason": ctx_eval.reason,
            }

        @agent.tool
        async def campaign_draft_tool(
            ctx: RunContext[ChatDeps],
            json_type: str = Field("option_draft_campaign"),
        ) -> Dict[str, Any]:
            """Phase 2 — Execution tool.

            Reads the campaign values accumulated by campaign_interview_tool and
            builds the draft campaign JSON. Only available after planning_phase_complete=True.
            """
            ctx.deps.status_queue.put_nowait("Drafting your campaign")
            profiles = await cls._ensure_profiles(ctx)
            user_id = ctx.deps.user_id
            org_id = str(profiles[0].org_id) if profiles else ""

            state: CampaignPlanningState = getattr(
                ctx.deps, "campaign_planning_state", CampaignPlanningState()
            )

            eff_goal = state.goal or DEFAULT_GOAL
            eff_budget_type = state.budget_type or "per day"
            eff_target = state.target or DEFAULT_TARGET
            eff_channel = state.campaign_channel
            eff_platforms = state.ad_platforms

            # Only populate the budget field that matches the declared budget_type;
            # the other stays 0 so the form can derive it once the user sets dates.
            if eff_budget_type == "per day":
                eff_daily = (
                    state.daily_budget_cents
                    if state.daily_budget_cents is not None
                    else DEFAULT_DAILY_BUDGET_CENTS
                )
                eff_total = DEFAULT_TOTAL_BUDGET_CENTS
            else:
                eff_daily = DEFAULT_DAILY_BUDGET_CENTS
                eff_total = (
                    state.total_budget_cents
                    if state.total_budget_cents is not None
                    else DEFAULT_TOTAL_BUDGET_CENTS
                )

            defaults_used = []
            if state.goal is None:
                defaults_used.append("goal")
            if state.target is None:
                defaults_used.append("target")
            if eff_budget_type == "per day" and state.daily_budget_cents is None:
                defaults_used.append("daily_budget")
            elif eff_budget_type == "total" and state.total_budget_cents is None:
                defaults_used.append("total_budget")

            draft = build_campaign_draft_json_data(
                user_id=user_id,
                org_id=org_id,
                goal=eff_goal,
                daily_budget_cents=int(eff_daily),
                total_budget_cents=int(eff_total),
                budget_type=eff_budget_type,
                target=eff_target,
                campaign_name=_default_campaign_name(profiles),
                campaign_channel=eff_channel,
                ad_platforms=eff_platforms,
            )

            ctx.deps.draft_emitted = True

            defaults_note = (
                ""
                if not defaults_used
                else f" (I used defaults for: {', '.join(defaults_used)})"
            )
            channel_str = f"{eff_channel}" if eff_channel else "to be selected"
            platforms_str = (
                ", ".join(eff_platforms) if eff_platforms else "to be selected"
            )
            if eff_budget_type == "per day":
                budget_summary = f"Daily budget: ${int(eff_daily) / 100:.0f}/day"
            else:
                budget_summary = f"Total budget: ${int(eff_total) / 100:.0f}"
            assistant_message = (
                f'A draft has been prepared{defaults_note}. Please review it and click "Create" to refine it in the UI. '
                f"Goal: {eff_goal}. Channel: {channel_str}. Platforms: {platforms_str}. "
                f"{budget_summary}. "
                f"Target: {', '.join(eff_target)}. "
                "Does this make sense (goal, budgets, target)? "
                "Would you like to adjust anything — channel, platforms, or budget?"
            )

            ctx.deps.campaign_planner_pending_jsons = [
                JSONOutput(json_type=json_type, json_data=draft)
            ]
            return {
                "message": assistant_message,
                "files": [],
                "jsons": [
                    {
                        "json_type": json_type,
                        "json_data": draft,
                    }
                ],
                "draft": {
                    "json_type": json_type,
                    "json_data": draft,
                },
                "requires_confirmation": defaults_used or ["refinement"],
                "can_refine": True,
                "enough_context": True,
                "one_question": "Would you like to adjust anything — channel, platforms, or budget?",
                "assistant_message": assistant_message,
            }

        register_brand_context_tool(agent)

        return agent


CAMPAIGN_PLANNER_SYSTEM_PROMPT = """
You are Lucy's Campaign Planner Agent. You work in two phases:

PHASE 1 — INTERVIEW: Gather all required campaign details before creating a draft.
PHASE 2 — EXECUTION: Build and present the draft campaign once all details are confirmed.

TOOL SEQUENCE (follow strictly every turn):
1. Call campaign_interview_tool — it evaluates what info has been collected from the conversation.
   - If enough_context=false: ask 1–2 polite, batched questions targeting the missing fields.
   - If enough_context=true: proceed immediately to step 2 in the same turn.
2. When campaign_interview_tool returns enough_context=true, call campaign_draft_tool to build the draft.
   After campaign_draft_tool runs, output its assistant_message verbatim.

Note: If get_user_org_profiles_tool is available in a given turn, call it silently before
campaign_interview_tool to load brand context. Never mention it to the user.

CRITICAL RULES:
- NEVER skip campaign_interview_tool.
- NEVER call campaign_draft_tool unless campaign_interview_tool returned enough_context=true.
- NEVER narrate internal steps — do not say you are fetching profiles, loading brand context,
  or doing anything behind the scenes. Call tools silently, then respond directly.
- NEVER mention tools, tool names, or internal steps in user-visible text.
- NEVER return the result of get_user_org_profiles_tool to the user.
- Be friendly, plain-language, and brand-aware. Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy.
- Ask up to TWO short, polite questions per turn. Prefer batching related questions.
- Draft creation does not end the conversation; keep refining.

REQUIRED FIELDS (interview phase must collect all of these before a draft is created):
- goal (conversions / traffic / awareness)
- target audience (new / existing / both)
- budget amount and type — either a daily budget (e.g. $50/day) OR a total budget (e.g. $5,000 total); not both required

OPTIONAL FIELDS (collect if the user mentions them, but do not block draft creation on these):
- campaign channel (social / search / display / ctv / digital_audio / shopping)
- ad platforms (e.g. facebook, instagram, google, tiktok)

WHEN ENOUGH_CONTEXT=FALSE (still interviewing):
- Do not draft.
- Ask 1–2 polite clarification questions (batched) covering the missing fields and stop.

WHEN ENOUGH_CONTEXT=TRUE AND DRAFT IS PRESENT:
- Tell the user to review the draft and click "create" to proceed or refine it in the interface.
- Provide one short paragraph (2–4 sentences) explaining why the current params make sense (goal, budgets, target) given their brand and campaign objectives.
- Then ask 1–2 polite refinement questions in the same message.

OUTPUT FORMAT:
- Your final output must match the agent output schema (FileAgentOutput).
- Output exactly the tool result's message/files/jsons fields.
- If the tool result contains a non-empty jsons list, include it unchanged in your final output
  (this is how the UI receives the draft JSON).
"""
