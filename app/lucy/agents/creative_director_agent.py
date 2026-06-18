import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, WebSearchTool
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.base_agent import LofiAgent, PreRunResult
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.context_tools import register_brand_context_tool, register_creative_assets_tool
from lucy.agents.common.vision_tools import (
    register_analyze_creative_image_tool,
    register_audit_creative_library_tool,
)
from lucy.agents.common.models import (
    ChatDeps,
    FileAgentOutput,
    UserOrgProfile,
    get_brand_name,
)
from lucy.agents.common.requirements import (
    RequirementsState,
    InterviewResult,
    compact_history_for_llm,
    evaluate_requirements,
    get_latest_user_text,
    run_requirements_precheck,
)
from lucy.agents.common.tools import register_user_org_profiles_tool

TaskType = Literal[
    "ideation",
    "qa_and_coverage_audit",
    "brand_check",
    "variants",
    "web_search",
    "knowledge_search",
    "open",
]
TASK_TYPE_TUPLE_VALUES = get_args(TaskType)
TASK_TYPE_SET: set[TaskType] = set(TASK_TYPE_TUPLE_VALUES)

class RouteDecision(BaseModel):
    task_type: TaskType
    output_type: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    one_question: str = ""

_CD_ROUTER_SYSTEM_PROMPT = (
    "You are a router for a Creative Director agent.\n"
    "Choose the single best task_type for the user request, and output_type if task_type=ideation.\n"
    f"Task types: {', '.join(TASK_TYPE_TUPLE_VALUES)}.\n"
    "If user asks for new ideas/concepts/hooks/angles -> ideation.\n"
    "If user asks to review existing creative for issues and missing formats -> qa_and_coverage_audit.\n"
    "If user asks 'is this on brand' / brand consistency -> brand_check.\n"
    "If user asks for A/B tests, variations, versions -> variants.\n"
    "If user requests trends, competitors, market inspiration -> web_search.\n"
    "If user asks what assets exist, past performance, internal library -> knowledge_search.\n"
    "If unclear -> open.\n"
    "If ideation, infer output_type: concepts (default), image_briefs, video_briefs.\n"
    "Ask at most ONE question only if needed.\n"
)

_cd_router_agent: Optional[Agent] = None


def _get_cd_router_agent() -> Agent:
    """Return the Creative Director router agent, creating it on first call."""
    global _cd_router_agent
    if _cd_router_agent is None:
        _cd_router_agent = Agent(
            model=to_responses_model(Models.CREATIVE_DIRECTOR_ROUTER),
            model_settings=ModelSettings(temperature=0.0, max_tokens=140),
            system_prompt=_CD_ROUTER_SYSTEM_PROMPT,
            output_type=RouteDecision,
        )
    return _cd_router_agent

CREATIVE_DIRECTOR_ORCHESTRATOR_PROMPT = """
You are Lucy Creative Director (orchestrator).
You MUST call tools exactly once per turn, in strict order:
(1) creative_director_interview_tool  — evaluates whether the request is clear enough to proceed
(2) creative_director_router_tool     — only after interview signals ready
(3) creative_director_execute_tool    — only after routing is complete

After creative_director_interview_tool runs:
- If enough_context=false: call final_result with message set to EXACTLY the 'one_question' string from the tool result, verbatim. Do NOT paraphrase, summarize, or generate your own text.
- If enough_context=true: proceed immediately to creative_director_router_tool.

After creative_director_execute_tool returns, call final_result with message set to EXACTLY the 'message' string from the tool result, verbatim.
Do NOT return files. Do NOT add commentary. Do NOT paraphrase. Do NOT mention any tools. Do NOT generate your own summary.
Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.
"""

CREATIVE_DIRECTOR_IDEATION_PROMPT = """
Generate creative directions, big ideas, hooks, angles, and a clear asset plan (image and video briefs).
CREATIVE BAR:
- Very high creativity: surprising, distinctive, and scroll-stopping
- Must remain on-brand and on-strategy
- Prioritize clarity, memorability, and execution feasibility
OUTPUT RULES:
- Use structured headings and bullet points
- Be decisive and directional (not exploratory fluff)
OUTPUT GUIDELINES:
- If output_type is 'concepts':
  - Deliver big ideas, core hooks, strategic angles, and creative rationale
  - Frame ideas so the user can iterate and refine
- If output_type is 'image_briefs':
  - Provide shot list, composition, visual style, mood, lighting, and aspect ratios
  - Include prompt-ready guidance for image generation or production
- If output_type is 'video_briefs':
  - Provide storyboard beats, timing, pacing, transitions
  - Include voiceover, on-screen text, and platform-aware execution notes
"""

CREATIVE_DIRECTOR_AUDIT_PROMPT = """
You are a senior creative director performing a data-driven creative audit.
You have access to real creative assets and full brand specifications via tools.

WORKFLOW — follow this sequence:
1. Call get_brand_context to load brand identity, tone, values, dos/don'ts, and visual guidelines.
2. If the user asked for an audit of the full library (or no specific asset was named),
   call audit_creative_library. This fetches all assets, samples if needed, analyzes each
   image via vision, and returns a structured report. Pass any platform or focus context
   the user mentioned.
3. If the user named or shared a specific asset URL, call analyze_creative_image for that
   single asset instead of (or in addition to) the library audit.
4. Synthesize the tool results into a clear, structured audit report for the user.

AUDIT REPORT STRUCTURE:
- Overall health: average score, brand alignment distribution
- Top performers: what is working and why
- Weakest assets: specific issues and concrete improvement briefs
- Coverage gaps: missing formats, orientations, hooks, or platform-specific variants
- Immediate action items: ranked list of 3-5 specific things to fix or create next

RULES:
- Be specific and evidence-based — cite actual file names and scores from the tool output.
- If sampling was applied, state it clearly: "I analyzed X of Y assets due to the cost cap."
- Do not invent observations not supported by the tool results.
- Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy.
- Concrete, actionable recommendations only — no vague critique.
"""

CREATIVE_DIRECTOR_BRANDCHECK_PROMPT = """
Perform a brand consistency check.
TASKS:
- Evaluate whether provided creative assets are on-brand, slightly off-brand, or off-brand.
- Assess voice, tone, visual language, messaging, and emotional alignment.
- Identify specific issues with evidence tied to brand principles.
CLASSIFICATION:
- On-brand
- Slightly off-brand
- Off-brand
OUTPUT GUIDELINES:
- Clear classification with justification
- Specific fixes, rewrites, or visual adjustments
- Concrete examples of how to bring the asset back on-brand
"""

CREATIVE_DIRECTOR_VARIANTS_PROMPT = """
Generate high-quality A/B creative variants.
TASKS:
- Identify key creative levers: hook, angle, framing, tone, offer, social proof, layout, CTA, format.
- Generate multiple variants that are meaningfully different, not cosmetic tweaks.
- Ensure all variants remain aligned with brand, strategy, and platform norms.
OUTPUT GUIDELINES:
- Clearly label each variant
- Explain what lever is being tested and why
- Prioritize learnings and performance differentiation
"""

CREATIVE_DIRECTOR_WEB_SEARCH_PROMPT = """
Search the web, analyze patterns, and synthesize insights to inform creative strategy and execution.
TASKS:
- Trend & Inspiration Research: Scan the web, social platforms, ads libraries, and creative communities for emerging trends in design, branding, content, and marketing.
- Competitor & Market Analysis: Analyze competitors and relevant brands to understand positioning, messaging, visual language, formats, and content strategies.
- Strategic Translation: Turn research insights into clear creative guidance, ideas, and opportunities. Explain how trends and gaps can be applied strategically rather than copied.
IMPORTANT:
- You MUST use the available WebSearchTool to gather sources before making claims.
- If the tool is unavailable, say so and proceed with best-effort reasoning without inventing citations.
OUTPUT GUIDELINES:
- Clear, concise, and visually minded guidance
- Insight-driven summaries (avoid generic trend lists)
- Focus on what is working, why it works, and how to apply it creatively
- Use structured bullets and short sections
"""

CREATIVE_DIRECTOR_KNOWLEDGE_SEARCH_PROMPT = """
Search the internal knowledge base to inform creative decisions with historical context and available resources.
TASKS:
- Asset Inventory: Identify available creative assets, including images, videos, audio files.
- Concept Recall: Surface previous creative concepts, executions, and their outcomes.
- Brand Assets Check: Reference the brand asset library for approved visuals, messaging, and guidelines.
USAGE GUIDANCE:
- Use this knowledge to avoid duplication, identify proven patterns, and build on past wins.
- Flag constraints, gaps, or missing assets that impact future creative execution.
OUTPUT GUIDELINES:
- Clear, structured summaries (no raw data dumps)
- Focus on actionable insights and creative implications
- Highlight reusable assets, repeatable patterns, and learnings to inform next steps
"""

CREATIVE_DIRECTOR_OPEN_PROMPT = """
Handle open-ended or ambiguous requests.
GUIDANCE:
- Ask at most ONE clarifying question only if it materially changes the creative direction.
- If information is missing, make a strong, reasonable assumption and state it explicitly.
- Default to proposing a best-guess creative direction rather than waiting for input.
OUTPUT GUIDELINES:
- Lead with a clear recommended direction or solution.
- Explain assumptions briefly.
- Provide concrete next steps, options, or execution paths the user can act on immediately.
"""

SPECIALIST_OUTPUT_CONTRACT = """
OUTPUT MANDATORY REQUIREMENTS:
- Populate ONLY the `message` field with plain text
- Do NOT return files. `files` MUST be an empty list
- Avoid JSONs unless absolutely necessary; prefer prose/bullets in `message`
- Do NOT include raw JSON blobs in `message`.
- Do NOT narrate or announce what you are about to do. Never write phrases like
  "I'll pull your profile", "Let me search", "One sec", "Give me a moment",
  "I'll check...", or any similar preamble. Call tools silently, then produce
  your final output directly without any introduction.
"""

SPECIALIST_CONFIG: Dict[TaskType, dict] = {
    "ideation": {"temperature": 0.9, "max_output_tokens": 1800, "system": CREATIVE_DIRECTOR_IDEATION_PROMPT},
    "qa_and_coverage_audit": {"temperature": 0.3, "max_output_tokens": 4000, "system": CREATIVE_DIRECTOR_AUDIT_PROMPT},
    "brand_check": {"temperature": 0.35, "max_output_tokens": 1600, "system": CREATIVE_DIRECTOR_BRANDCHECK_PROMPT},
    "variants": {"temperature": 0.7, "max_output_tokens": 1600, "system": CREATIVE_DIRECTOR_VARIANTS_PROMPT},
    "web_search": {"temperature": 0.3, "max_output_tokens": 1000, "system": CREATIVE_DIRECTOR_WEB_SEARCH_PROMPT},
    "knowledge_search": {"temperature": 0.3, "max_output_tokens": 1000, "system": CREATIVE_DIRECTOR_KNOWLEDGE_SEARCH_PROMPT},
    "open": {"temperature": 0.4, "max_output_tokens": 900, "system": CREATIVE_DIRECTOR_OPEN_PROMPT},
}

@dataclass
class CreativeDirectorState(RequirementsState):
    """Typed accumulator for creative requirements gathered during the interview phase.

    One instance is stored on ctx.deps.creative_director_state per request.
    task_intent drives readiness: a clear, non-ambiguous intent means we can
    proceed directly to routing and specialist execution.
    """

    task_intent: Optional[str] = None
    """Identified task type: ideation / qa_and_coverage_audit / brand_check /
    variants / web_search / knowledge_search / open / unclear"""

    output_preference: Optional[str] = None
    """For ideation tasks: concepts / image_briefs / video_briefs"""

    target_description: Optional[str] = None
    """Brief description of the campaign, assets, or context being worked on"""

    @property
    def is_ready(self) -> bool:
        """Ready when intent is clearly identified (not null and not 'unclear')."""
        return bool(self.task_intent) and self.task_intent != "unclear"

    @property
    def missing_fields(self) -> List[str]:
        if not self.task_intent or self.task_intent == "unclear":
            return ["task_intent"]
        return []

    def merge(self, suggested: Dict[str, Any]) -> None:
        """Overwrite only fields where a non-None value was newly extracted."""
        if suggested.get("task_intent") is not None:
            self.task_intent = suggested["task_intent"]
        if suggested.get("output_preference") is not None:
            self.output_preference = suggested["output_preference"]
        if suggested.get("target_description") is not None:
            self.target_description = suggested["target_description"]

class CreativeDirectorAgent(LofiAgent):
    """Factory and metadata for the Creative Director Agent."""

    # Populated by create(); used by get_streaming_agent() outside the agent closure.
    _specialists: "dict[TaskType, Agent]" = {}

    REMINDER_HEADER = (
        "You are Lucy Creative Director — the creative brain of the system. "
        "Briefs, concepts, and QA only. Do NOT render images/videos/audio directly. "
        "Call tools silently — no announcement text before a tool call. "
        "Guard the brand; ensure coherent creative systems across channels."
    )

    _INTERVIEW_RULES = (
        "Determine the user's creative task intent from the conversation. "
        "enough_context is true when the task_intent is confidently identified as a specific task "
        "(not 'unclear'). Set enough_context=false only when the request is genuinely ambiguous. "
        "TASK INTENT: Choose exactly one of these values: "
        "ideation (new ideas/concepts/hooks/angles/briefs), "
        "qa_and_coverage_audit (review existing creatives for issues and gaps), "
        "brand_check (brand consistency or 'is this on brand'), "
        "variants (A/B tests, variations, alternative versions), "
        "web_search (trends, competitors, market/inspiration research), "
        "knowledge_search (internal assets, past performance, brand library), "
        "open (general creative question with clear intent but doesn't fit above), "
        "unclear (genuinely ambiguous — cannot determine what the user wants). "
        "If in doubt, lean toward setting a task_intent and enough_context=true rather than asking. "
        "Only set unclear when the request is too vague to route meaningfully. "
        "OUTPUT PREFERENCE: If task_intent=ideation, infer output_preference: "
        "concepts (default), image_briefs (if user asks for image/visual briefs), "
        "video_briefs (if user asks for video/storyboard briefs). "
        "TARGET DESCRIPTION: Extract a short description of what campaign, assets, or "
        "creative work is being discussed (e.g. 'summer sale campaign', 'Instagram ads'). "
        "ONE QUESTION: If enough_context=false, return ONE specific, actionable question "
        "that will clarify the task intent. Keep it short and friendly."
    )

    _INTERVIEW_SCHEMA = {
        "enough_context": "boolean — true when task_intent is clearly identified (not 'unclear')",
        "task_intent": (
            "ideation | qa_and_coverage_audit | brand_check | variants | "
            "web_search | knowledge_search | open | unclear | null"
        ),
        "output_preference": "concepts | image_briefs | video_briefs | null (only for ideation)",
        "target_description": "short string describing what is being worked on, or null",
        "missing_fields": "list[str] — ['task_intent'] if unclear, else []",
        "one_question": "string ending with ? — the single most useful clarifying question, or empty string if enough_context=true",
        "notes": "string",
    }

    _EMPTY_HISTORY_QUESTION = (
        "What would you like help with today? "
        "For example: ideation, a creative audit, brand check, "
        "A/B variants, trend research, or something else?"
    )

    _FALLBACK_QUESTION = (
        "What kind of creative work can I help you with? "
        "(e.g. new concepts, a creative audit, brand check, A/B variants, or trend research)"
    )

    @staticmethod
    def _tool_name(tool_def: Any) -> str:
        return (
            getattr(tool_def, "name", None)
            or getattr(getattr(tool_def, "defn", None), "name", None)
            or ""
        )

    @classmethod
    def _select_orchestrator_tools(
        cls,
        deps: Any,
        tool_defs: list,
    ) -> list:
        """Pure gating logic for the orchestrator's prepare_tools callback.

        Extracted as a class method so it can be unit-tested without
        constructing a real pydantic-ai Agent.
        """

        def _by_name(name: str) -> list:
            return [t for t in tool_defs if getattr(t, "name", "") == name]

        if getattr(deps, "creative_execute_used", False):
            return _by_name("final_result")

        if getattr(deps, "creative_route", None):
            return _by_name("creative_director_execute_tool")

        interview_used = getattr(deps, "creative_interview_used", False)
        interview_ready = getattr(deps, "creative_interview_ready", False)

        if interview_used and interview_ready:
            return _by_name("creative_director_router_tool")

        if interview_used and not interview_ready:
            return _by_name("final_result")

        return _by_name("creative_director_interview_tool")

    @classmethod
    def _filter_specialist_tool_defs(
        cls,
        *,
        task_type: TaskType,
        tool_defs: list[Any],
        has_preloaded_profiles: Optional[List[UserOrgProfile]],
    ) -> list[Any]:
        """Limit specialist tools to the minimum needed for the task.

        The chat endpoint always preloads user/org profiles into ``ctx.deps``
        before the agent runs.  ``has_preloaded_profiles`` is the raw list value
        (possibly an empty list ``[]``).  We only expose ``get_user_org_profiles_tool``
        when it is ``None`` (never attempted), not when it is ``[]`` (attempted but
        empty) — calling the tool again would waste a turn and prompt the model to
        narrate "I'll pull your profile..." before returning nothing new.
        """

        allowed = {"final_result"}
        # Only expose the profile tool when profiles haven't been attempted at
        # all (None). An empty list [] means we already tried and found nothing —
        # re-calling the tool would just waste a turn and cause the model to
        # narrate "I'll pull your profile..." before returning nothing useful.
        if has_preloaded_profiles is None:
            allowed.add("get_user_org_profiles_tool")

        if task_type == "web_search":
            allowed.update({"web_search", "web_search_preview", "browser.search"})

        if task_type == "qa_and_coverage_audit":
            # Full toolkit for real creative audits: brand specs, asset list,
            # single-image analysis, and the full library audit.
            allowed.update({
                "get_brand_context",
                "get_creative_assets",
                "analyze_creative_image",
                "audit_creative_library",
            })

        if task_type == "brand_check":
            # Brand check also benefits from brand specs and asset metadata.
            allowed.update({"get_brand_context", "get_creative_assets"})

        return [t for t in tool_defs if cls._tool_name(t) in allowed]

    @staticmethod
    def _build_profile_context(profiles: Optional[List[UserOrgProfile]]) -> str:
        if not profiles:
            return ""
        profile = profiles[0]
        parts = []
        brand = getattr(profile, "brand_name", None) or getattr(
            profile, "company_name", None
        )
        if brand:
            parts.append(f"brand={brand}")
        website = getattr(profile, "website_url", None)
        if website:
            parts.append(f"website={website}")
        industry = getattr(profile, "industry", None)
        if industry:
            parts.append(f"industry={industry}")
        audience = getattr(profile, "audience", None)
        if audience:
            parts.append(f"audience={audience}")
        positioning = getattr(profile, "positioning", None)
        if positioning:
            parts.append(f"positioning={positioning}")
        return ", ".join(parts)

    @classmethod
    def _build_specialist_input(cls, deps: ChatDeps, route: dict, task_type: TaskType) -> str:
        """Build the input string for a specialist agent from deps and route."""
        message_history = getattr(deps, "message_history", None)
        history_text, _ = compact_history_for_llm(message_history, max_messages=30, max_chars=8000)
        latest_user_message = get_latest_user_text(message_history).strip()
        profile_context = cls._build_profile_context(
            getattr(deps, "user_profiles", None)
        )

        sections = [
            f"ROUTE (authoritative): {json.dumps(route)}\n\n"
        ]
        if profile_context:
            sections.append(f"PROFILE CONTEXT:\n{profile_context}\n\n")
        sections.append(f"CONVERSATION CONTEXT:\n{history_text}\n\n")
        sections.append(f"USER REQUEST:\n{latest_user_message}")
        return "".join(sections).strip()

    @classmethod
    async def pre_run_check(cls, deps: ChatDeps) -> Optional[PreRunResult]:
        """Run the creative-director interview (and routing) before the specialist.

        If the user's request is too vague to route, return the clarifying
        question directly — no orchestrator LLM call needed.
        If ready, also runs routing and caches the route on deps so
        get_streaming_agent() can select the right specialist without an
        additional LLM round-trip.
        """
        result, _ = await run_requirements_precheck(
            deps=deps,
            schema=cls._INTERVIEW_SCHEMA,
            rules=cls._INTERVIEW_RULES,
            fallback_question=cls._FALLBACK_QUESTION,
            model=to_responses_model(Models.AGENT_FAST),
        )

        if result is None:
            return PreRunResult(message=cls._EMPTY_HISTORY_QUESTION)

        state = CreativeDirectorState()
        state.merge(result.suggested)

        if not state.task_intent and result.enough_context:
            state.task_intent = "ideation"

        if not state.is_ready:
            logger.info(
                f"Creative Director pre-check: not ready "
                f"(task_intent={state.task_intent!r}), asking question"
            )
            return PreRunResult(message=result.one_question)

        # Cache computed state so the interview tool can skip its own LLM call
        # (used by the orchestrator fallback path).
        deps.creative_director_state = state
        deps.creative_interview_ready = True
        deps.creative_interview_used = True

        logger.info(
            f"Creative Director pre-check: ready "
            f"(task_intent={state.task_intent!r}), running routing"
        )

        # Run routing eagerly so get_streaming_agent() can immediately select
        # the right specialist without another LLM call.
        try:
            route = await cls._route_from_context(deps=deps, max_messages=30, max_chars=8000)
        except Exception as e:
            logger.warning(f"Creative Director pre-check: routing failed, fallback to open. err={e}")
            route = cls._normalize_route({})

        deps.creative_route = route
        logger.info(
            f"Creative Director pre-check: routed to '{route.get('task_type')}' "
            f"(confidence={route.get('confidence', 0) * 100:.0f}%)"
        )

        return None

    @classmethod
    async def get_streaming_agent(
        cls, deps: ChatDeps
    ) -> Optional[Tuple[Agent, str]]:
        """Return the pre-selected specialist agent and its formatted input.

        Called by chat.py after pre_run_check succeeds. When a route is cached
        on deps (set by pre_run_check), we can skip the orchestrator entirely and
        stream the specialist directly — deterministic orchestration in Python.
        """
        route = getattr(deps, "creative_route", None)
        if not route:
            return None

        if not cls._specialists:
            # create() hasn't been called yet (shouldn't happen in production)
            return None

        task_type: TaskType = route.get("task_type", "open")
        specialist = cls._specialists.get(task_type) or cls._specialists.get("open")
        if not specialist:
            return None

        input_text = cls._build_specialist_input(deps, route, task_type)
        logger.info(
            f"Creative Director get_streaming_agent: streaming specialist "
            f"task_type='{task_type}' directly (bypassing orchestrator)"
        )
        return (specialist, input_text)

    @classmethod
    async def _route_from_context(
        cls,
        *,
        deps: ChatDeps,
        max_messages: int,
        max_chars: int,
    ) -> dict:
        history_text, _ = compact_history_for_llm(
            ctx.deps.message_history,
            max_messages=max_messages,
            max_chars=max_chars,
        )

        if not (history_text or "").strip():
            return {
                "task_type": "ideation",
                "output_type": "concepts",
                "confidence": 0.2,
                "one_question": "What’s the use case for these ideas, concepts, hooks, and/or angles?",
            }

        latest_user_message = get_latest_user_text(ctx.deps.message_history).strip()
        if not latest_user_message:
            latest_user_message = history_text.splitlines()[-1].strip()

        user_prompt = {
            "latest_user_message": latest_user_message,
            "conversation_context": history_text,
        }

        try:
            result = await _get_cd_router_agent().run(json.dumps(user_prompt))
            route_decision: RouteDecision = result.output
            data = route_decision.model_dump()
        except Exception as e:
            logger.warning(f"Routing failed; fallback to open. err={e}")
            data = {}

        return cls._normalize_route(data)

    @classmethod
    def _normalize_route(cls, data: dict) -> dict:
        """Single place to normalize + safety-check route dict."""

        task_type_raw = data.get("task_type")
        task_type: TaskType = task_type_raw if task_type_raw in TASK_TYPE_SET else "open"

        output_type = data.get("output_type") if isinstance(data.get("output_type"), str) else None
        confidence_raw = data.get("confidence")
        one_q = data.get("one_question") if isinstance(data.get("one_question"), str) else ""

        if task_type != "ideation":
            output_type = None

        if task_type == "ideation" and output_type not in {"concepts", "image_briefs", "video_briefs", None}:
            output_type = "concepts"

        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        one_q = one_q.strip()
        if one_q and not one_q.endswith("?"):
            one_q = ""
        return {
            "task_type": task_type,
            "output_type":output_type,
            "confidence": confidence,
            "one_question": one_q,
        }

    @staticmethod
    async def _specialist_web_search_tool(ctx: RunContext[ChatDeps], tool_defs: list[Any]):
        def tool_name(t: Any) -> str:
            return getattr(t, "name", None) or getattr(getattr(t, "defn", None), "name", None) or ""

        names = [tool_name(t) for t in tool_defs]
        # log once to learn what tools exist in your environment:
        if not ctx.deps._logged_web_tools:
            ctx.deps._logged_web_tools = True
            logger.info(f"[web_search specialist] available tools: {names}")

        allowed = {"web_search", "web_search_preview", "browser.search"}  # adjust based on log
        return [t for t in tool_defs if tool_name(t) in allowed]

    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.CREATIVE_DIRECTOR)
        """
        Returns a single Agent (the orchestrator) that enforces a three-step flow:
          1. creative_director_interview_tool — gates execution; asks clarifying question if intent is unclear
          2. creative_director_router_tool    — classifies task type once intent is clear
          3. creative_director_execute_tool   — dispatches to tuned specialist agent
        """
        logger.info(f"Creative Director starting with: '{model_name}'")

        async def _prepare_tools_orchestrator(ctx: RunContext[ChatDeps], tool_defs: list[Any]):
            # Step 3 done — no more tools; LLM produces final output
            if ctx.deps.creative_execute_used:
                return []

            if ctx.deps.creative_route:
                return [t for t in tool_defs if getattr(t, "name", "") == "creative_director_execute_tool"]

            interview_used = ctx.deps.creative_interview_used
            interview_ready = ctx.deps.creative_interview_ready

            # Step 1 done and ready — proceed to router
            if interview_used and interview_ready:
                return [t for t in tool_defs if getattr(t, "name", "") == "creative_director_router_tool"]

            # Step 1 done but not ready — force text output (ask clarifying question)
            if interview_used and not interview_ready:
                return []

            # Step 1 not yet run — run interview first
            return [t for t in tool_defs if getattr(t, "name", "") == "creative_director_interview_tool"]

        # 1) Orchestrator: cheap + deterministic (routing/glue only)
        # max_tokens is 400 to allow for clarifying questions from the interview phase
        orchestrator = Agent(
            model=to_responses_model(Models.AGENT_FAST),
            model_settings=ModelSettings(temperature=0.0, max_tokens=1200),
            deps_type=ChatDeps,
            system_prompt=CREATIVE_DIRECTOR_ORCHESTRATOR_PROMPT,
            output_type=FileAgentOutput,
            prepare_tools=_prepare_tools_orchestrator,
        )

        # 2) Specialist agents: tuned per use case
        specialists: dict[TaskType, Agent] = {}
        for task_type, cfg in SPECIALIST_CONFIG.items():
            spec_model = to_responses_model(model_name)

            async def _prepare_specialist_tools(
                ctx: RunContext[ChatDeps],
                tool_defs: list[Any],
                task_type: TaskType = task_type,
            ) -> list[Any]:
                return cls._filter_specialist_tool_defs(
                    task_type=task_type,
                    tool_defs=tool_defs,
                    has_preloaded_profiles=getattr(ctx.deps, "user_profiles", None),
                )

            agent = Agent(
                model=spec_model,
                model_settings=ModelSettings(
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_output_tokens"],
                ),
                deps_type=ChatDeps,
                system_prompt=cfg["system"] + "\n\n" + SPECIALIST_OUTPUT_CONTRACT,
                output_type=FileAgentOutput,
                prepare_tools=_prepare_specialist_tools,
                builtin_tools=(
                    tuple([WebSearchTool(search_context_size="medium")])
                    if task_type == "web_search"
                    else ()
                ),
            )
            register_user_org_profiles_tool(agent)
            # Register brand + asset context tools on all specialists so the
            # filter function can selectively expose them per task type.
            register_brand_context_tool(agent)
            register_creative_assets_tool(agent)
            # Vision tools are only used by qa_and_coverage_audit but are
            # registered on all specialists; the filter keeps them hidden for
            # other task types so they never appear to irrelevant specialists.
            register_analyze_creative_image_tool(agent)
            register_audit_creative_library_tool(agent)
            specialists[task_type] = agent

        # Cache specialists on the class so get_streaming_agent() can access them
        # without holding a closure reference.
        cls._specialists = specialists

        # Here we can build input based on task_type
        def _build_specialist_input(ctx: RunContext[ChatDeps], route: dict, task_type: TaskType) -> str:
            history_text, _ = compact_history_for_llm(ctx.deps.message_history, max_messages=30, max_chars=8000)
            latest_user_message = get_latest_user_text(ctx.deps.message_history).strip()

            # Route is authoritative context for the specialist; prevents re-routing.
            return (
                f"ROUTE (authoritative): {json.dumps(route)}\n\n"
                f"CONVERSATION CONTEXT:\n{history_text}\n\n"
                f"USER REQUEST:\n{latest_user_message}"
            ).strip()

        # Tool 1: interview (gates execution until request is clear)
        @orchestrator.tool
        async def creative_director_interview_tool(
            ctx: RunContext[ChatDeps],
            max_messages: int = 30,
            max_chars: int = 8000,
        ) -> dict:
            """Phase 0 — Requirements interview.

            Evaluates whether the user's request is clear enough to route and
            execute. Sets creative_interview_ready=True when task_intent is
            identified, which unblocks the router in subsequent tool slots.

            Returns enough_context and one_question so the orchestrator can ask
            a targeted clarifying question when the intent is ambiguous.
            """
            ctx.deps.status_queue.put_nowait("Gathering creative requirements")
            ctx.deps.creative_interview_used = True

            # Guard against repeated calls within the same turn
            calls = ctx.deps.creative_interview_calls
            ctx.deps.creative_interview_calls = calls + 1
            if calls >= 1:
                ctx.deps.creative_interview_ready = True
                return {
                    "enough_context": True,
                    "one_question": "",
                    "reason": "interview_tool_called_multiple_times_in_same_turn",
                }

            # If pre_run_check already evaluated and cached the state, skip the
            # redundant evaluate_requirements() call and return the cached result.
            cached_state = ctx.deps.creative_director_state
            if ctx.deps.creative_interview_ready and cached_state is not None:
                logger.info(
                    "Creative Director interview: using cached pre-run state "
                    f"(task_intent={cached_state.task_intent!r}), skipping LLM call"
                )
                return {
                    "enough_context": True,
                    "one_question": "",
                    "reason": "cached_from_pre_run_check",
                    "task_intent": cached_state.task_intent,
                    "output_preference": cached_state.output_preference,
                    "target_description": cached_state.target_description,
                }

            history_text, _ = compact_history_for_llm(
                ctx.deps.message_history,
                max_messages=max_messages,
                max_chars=max_chars,
            )

            # No history at all — ask an open question
            if not history_text.strip():
                ctx.deps.creative_interview_ready = False
                ctx.deps.creative_director_clarify_question = cls._EMPTY_HISTORY_QUESTION
                return {
                    "enough_context": False,
                    "one_question": cls._EMPTY_HISTORY_QUESTION,
                    "reason": "empty_history",
                }

            profiles = ctx.deps.user_profiles or []
            brand = get_brand_name(profiles)

            result = await evaluate_requirements(
                history_text=history_text,
                schema=cls._INTERVIEW_SCHEMA,
                rules=cls._INTERVIEW_RULES,
                fallback_question=cls._FALLBACK_QUESTION,
                brand_context=brand,
                model=to_responses_model(Models.AGENT_FAST),
            )

            # Build and store state
            state = CreativeDirectorState()
            state.merge(result.suggested)

            # Fallback: if the classification model said enough_context=True but
            # returned null for task_intent, default to 'ideation' rather than
            # blocking the flow. gpt-5-mini can confuse "do I know the task type?"
            # with "do I have all the brief details?" — null task_intent shouldn't
            # gate the pipeline when the LLM otherwise signalled readiness.
            # Do NOT apply when extraction itself failed (enough_context=False),
            # otherwise a JSON parse error silently skips the clarifying question.
            if not state.task_intent and result.enough_context:
                state.task_intent = "ideation"
                logger.info("Creative Director interview: task_intent was null but enough_context=True; defaulting to 'ideation'")

            ctx.deps.creative_director_state = state

            logger.info(
                f"Creative Director interview: task_intent={state.task_intent!r} "
                f"ready={state.is_ready} missing={state.missing_fields}"
            )

            ctx.deps.creative_interview_ready = state.is_ready
            if not state.is_ready and result.one_question:
                ctx.deps.creative_director_clarify_question = result.one_question
            return {
                "enough_context": state.is_ready,
                "one_question": result.one_question,
                "reason": result.reason,
                "task_intent": state.task_intent,
                "output_preference": state.output_preference,
                "target_description": state.target_description,
            }

        # Tool 3: router (stores decision on ctx.deps)
        @orchestrator.tool
        async def creative_director_router_tool(
            ctx: RunContext[ChatDeps],
            max_messages: int = 30,
            max_chars: int = 8000,
        ) -> dict:
            ctx.deps.status_queue.put_nowait("Selecting creative approach")
            calls = ctx.deps.creative_router_calls
            ctx.deps.creative_router_calls = calls + 1
            if calls >= 1:
                decision = {"task_type": "ideation", "output_type": "concepts", "confidence": 0.2, "one_question": ""}
                decision = cls._normalize_route(decision)
                ctx.deps.creative_route = decision
                return decision

            # pre_run_check already ran routing and cached the result — reuse it.
            cached_route = getattr(ctx.deps, "creative_route", None)
            if cached_route and isinstance(cached_route, dict):
                logger.info(
                    f"Creative Director router tool: using cached route "
                    f"'{cached_route.get('task_type')}' from pre_run_check"
                )
                return cached_route

            decision = await cls._route_from_context(deps=ctx.deps, max_messages=max_messages, max_chars=max_chars)
            ctx.deps.creative_route = decision
            
            logger.info(
                f"Creative Director routed to: '{decision['task_type']}' "
                f"with confidence: {decision.get('confidence') * 100:.2f}%"
            )
            
            return decision

        # Tool 4: executor (dispatches to specialist agent)
        @orchestrator.tool
        async def creative_director_execute_tool(ctx: RunContext[ChatDeps]) -> dict:
            ctx.deps.status_queue.put_nowait("Creating deliverable")
            ctx.deps.creative_execute_used = True

            route = ctx.deps.creative_route or {"task_type": "open", "output_type": None, "confidence": 0.2, "one_question": ""}
            route = cls._normalize_route(route if isinstance(route, dict) else {})

            task_type: TaskType = route.get("task_type", "open")
            one_q = (route.get("one_question") or "").strip()

            specialist = specialists.get(task_type) or specialists["open"]

            specialist_input = _build_specialist_input(ctx, route, task_type)

            # Run the tuned specialist. The audit path needs more requests
            # (brand context + library audit + optional single-image + final output);
            # other specialists are kept tight to prevent runaway tool loops.
            from pydantic_ai.usage import UsageLimits
            specialist_request_limit = 6 if task_type == "qa_and_coverage_audit" else 3
            specialist_result = await specialist.run(
                specialist_input,
                deps=ctx.deps,
                usage_limits=UsageLimits(request_limit=specialist_request_limit),
            )
            out = specialist_result.output

            if hasattr(out, "files"):
                out.files = []

            final_text = ""
            # Normalize to plain text for the orchestrator
            if hasattr(out, "message") and isinstance(out.message, str) and out.message.strip():
                final_text = out.message.strip()
            elif hasattr(out, "text") and isinstance(out.text, str) and out.text.strip():
                final_text = out.text.strip()

            # If message is empty, but jsons exist, fall back gracefully
            if not final_text and hasattr(out, "jsons") and out.jsons:
                try:
                    payload = out.jsons[0]
                    if hasattr(payload, "model_dump"):
                        payload = payload.model_dump()
                    final_text = json.dumps(payload, ensure_ascii=False, indent=2)
                except Exception:
                    final_text = str(out.jsons[0])

            # Final fallback — ensures final_text is always non-empty so the
            # cached FileAgentOutput is always valid.
            if not final_text:
                final_text = str(out).strip() or "Done."

            # If router asked a single clarifying question, append it
            if one_q:
                final_text = f"{final_text}\n\n{one_q}"

            # Cache the complete, validated output on ctx.deps so the
            # output_validator can inject it directly, bypassing any risk
            # of the orchestrator model reconstructing an empty FileAgentOutput.
            ctx.deps.creative_director_final_output = FileAgentOutput(
                message=final_text,
                files=[],
                jsons=getattr(out, "jsons", []) or [],
            )

            # Return a minimal payload so the orchestrator model isn't tempted
            # to copy-paste or paraphrase fields it doesn't need to touch.
            return {"message": "ok"}

        @orchestrator.output_validator
        async def _no_files(ctx: RunContext[ChatDeps], output: FileAgentOutput) -> FileAgentOutput:
            # If the execute tool cached the specialist's fully-validated output,
            # return it directly.  This eliminates the second schema hop: the
            # orchestrator model only needs to emit any valid final_result call;
            # the actual response content comes from the cached object.
            cached = getattr(ctx.deps, "creative_director_final_output", None)
            if cached is not None:
                return cached

            # For the clarify path: if the model produced an empty message but
            # a clarifying question was cached, inject it.
            if not (output.message or "").strip():
                clarify_q = getattr(ctx.deps, "creative_director_clarify_question", None)
                if clarify_q:
                    output.message = clarify_q

            if getattr(output, "files", None):
                output.files = []
            return output

        register_brand_context_tool(orchestrator)
        register_creative_assets_tool(orchestrator)

        return orchestrator