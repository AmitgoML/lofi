from loguru import logger
from typing import Literal, List, Optional, get_args
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.factory import build as _build_agent
from lucy.agents.common.model_config import Models
from lucy.agents.common.models import ChatDeps

RouterResult = Literal[
    "keywords",
    "support",
    "lucy",
    "image",
    "video",
    "performance",
    "campaign_planner",
    "creative_director",
]
ROUTER_TARGETS = get_args(RouterResult)

# Short affirmations that require context-aware routing
_AFFIRMATIONS = frozenset({
    "yes", "yeah", "yep", "yup", "yap", "yea", "ya", "yah",
    "sure", "ok", "okay", "right", "exactly", "correct",
    "absolutely", "definitely", "for sure", "totally", "bet", "cool",
    "go ahead", "go", "do it", "sounds good", "please",
    "works for me", "that works",
    "create it", "make it", "build it",
    "let's do it", "lets do it", "let's go", "lets go", "go for it",
})

_FOLLOW_UP_PREFIXES = (
    "what about", "how about", "can you also", "can you",
    "and ", "also ", "what if", "how would",
    "tell me more", "more about", "expand on",
    "why", "when", "where", "which",
)

_TOPIC_SHIFT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "image": ("generate an image", "create an image", "make an image",
              "generate a picture", "make a graphic", "generate a graphic",
              "image of", "picture of", "graphic of"),
    "video": ("generate a video", "create a video", "make a video",
              "make a clip", "make a reel", "video of", "clip of"),
    "keywords": ("keyword research", "find keywords", "keyword ideas",
                 "search terms", "trending searches", "share of search",
                 "keyword trends"),
    "support": ("how do i use lofi", "help with lofi", "set up in lofi",
                "lofi setup", "help me with lofi"),
    "performance": ("analyze my campaign", "my campaign data",
                    "my campaign performance", "how is my campaign",
                    "campaign performance", "analyze my"),
    "campaign_planner": ("plan a campaign", "campaign plan",
                         "go-to-market plan", "budget split",
                         "create a campaign", "build a campaign", "draft a campaign",
                         "draft me a campaign", "make a campaign", "generate a campaign"),
    "creative_director": ("write a brief", "creative brief", "image brief",
                          "video brief", "storyboard", "shot list",
                          "brief for", "creative concept"),
}

_THRESHOLD_SAME_AGENT = 0.70
_THRESHOLD_TO_LUCY = 0.80
_THRESHOLD_FROM_LUCY = 0.90
_THRESHOLD_SPECIALIST_SWITCH = 0.85


class RouterOutput(BaseModel):
    """Structured router output with route and confidence score."""
    route: RouterResult
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this routing decision (0-1)")


ROUTER_SYSTEM_PROMPT = """
You are Lucy's Router. Classify the user's request into exactly ONE of the following targets.

**lucy** — General marketing strategy, advice, and analytics (default / fallback)
- Digital advertising strategy, channel recommendations, competitive analysis
- General PPC/marketing metrics discussion without specific campaign data ("what's a good ROAS?", "explain CPC")
- Ad copy writing, creative ideation without a formal brief
- Industry benchmarks, market research, general advertising best practices
- File/document analysis
- When in doubt, route here — lucy is the safe default

**keywords** — Keyword research and discovery
- Finding/suggesting keywords to use: PPC keywords, Google Ads keywords, negative keywords, related keywords
- Seed keyword expansion, keyword ideas, keyword lists
- Search trends, trending searches, share of search, top/rising queries
- Location-scoped keyword research ("keywords for NJ", "keywords in Texas")
- NOT: conceptual questions about CPC/CTR metrics → lucy

**support** — Lofi platform help, product usage, troubleshooting
- How to use Lofi features, in-app navigation, setup guidance
- Billing, account management, plan limits
- Campaign setup and configuration inside Lofi ("set up campaign in Lofi", "launch in Lofi")
- Using specific keywords to create/launch a campaign inside the Lofi platform

**creative_director** — Creative strategy deliverables: briefs, concepts, reviews
- Campaign concepts, big ideas, hooks, angles, messaging direction, creative positioning
- Structured written deliverables: image briefs, video briefs, audio direction, prompt packs, shot lists, storyboards
- Reviewing/QA-ing existing creatives for brand alignment or performance potential
- Any request to WRITE a brief or concept — even if "image" or "video" is mentioned
- NOT: actually rendering/generating images or videos → image / video

**image** — AI image generation or modification
- Generating images, graphics, photos, illustrations, artwork ("create an image", "generate a picture", "make a graphic")
- Editing/modifying existing images ("change the lighting", "modify this image", "make it look like")
- NOT: writing a brief or description for an image → creative_director

**video** — AI video generation
- Generating videos, cinematic content, motion visuals, animations, reels ("create a video", "make a short clip")
- NOT: writing a video brief or treatment → creative_director

**performance** — Analysis of the user's OWN campaign data
- Analyzing the user's actual campaign performance: ROAS, CPA, CTR, conversion rates, spend efficiency
- Identifying trends, anomalies, or optimization opportunities in their campaign data
- "analyze my campaign", "show me my campaign data", "how is my campaign performing"
- NOT: general/conceptual questions about metrics → lucy

**campaign_planner** — Structured campaign planning (strategy + structure, not in-app setup)
- Planning or drafting a campaign: goal, budget, channel mix, timeline, measurement plan, creative plan
- "plan a campaign", "draft a campaign", "draft me a campaign", "create a campaign",
  "make a campaign", "build a campaign", "generate a campaign", "campaign brief",
  "go-to-market plan", "budget split"
- Use this whenever the user wants to produce a campaign — even if they say "draft", "create",
  "make", "build", or "generate" — because those all require a structured plan first
- NOT: how to set up a campaign inside Lofi → support
- NOT: general questions about channels, strategy, or industry best practices → lucy

## Priority rules (when multiple routes could match)
1. Written creative deliverable (brief, concept, treatment, storyboard) → creative_director, even if image/video are mentioned
2. Actual image generation/modification → image
3. Actual video generation → video
4. User's OWN campaign data analysis → performance (not lucy)
5. Lofi in-app setup or campaign launch → support (not campaign_planner)
6. Keyword discovery/research → keywords (not lucy)
7. General advice/strategy questions without explicit planning intent → lucy (not campaign_planner)

## Ambiguous examples
User: "What keywords should I consider for my campaign?" → keywords
User: "What's a good CPC for my industry?" → lucy (conceptual metric, not keyword research)
User: "What's a good channel for my industry?" → lucy (general advice, not campaign planning)
User: "Which channels work best for e-commerce?" → lucy (strategy advice, not a structured plan)
User: "Analyze my campaign's ROAS" → performance (own campaign data)
User: "Generate an image brief" → creative_director (written deliverable, not image generation)
User: "Create an image of a cityscape" → image (actual generation)
User: "Plan a campaign with a $500 budget on Facebook" → campaign_planner (explicit plan request with budget)
User: "Draft me a campaign" → campaign_planner (drafting = structured plan, not creative ideation)
User: "Create a campaign for my dating app" → campaign_planner (creating a campaign needs a plan)
User: "Set up my campaign in Lofi" → support

Return a JSON object with "route" (one of the valid targets) and "confidence" (0.0–1.0).

Never use emojis or emoticons under any circumstances.
Return ONLY the JSON object (no markdown, no prose).
"""


class RouterAgent(LofiAgent):
    """Factory and metadata for the Router agent."""

    REMINDER_HEADER = (
        "Classify into exactly one route. Return ONLY JSON with 'route' and 'confidence' (no markdown/prose): "
        "keywords | support | lucy | image | video | performance | campaign_planner | creative_director"
    )

    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        # Chat Completions (not Responses API) — the router is a stateless
        # classifier with no need for reasoning, web search, or built-in tools.
        model_name = model_name or Models.ROUTER
        logger.info(f"Creating router agent with model '{model_name}'")
        return Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.0, max_tokens=1000),
            deps_type=ChatDeps,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            output_type=RouterOutput,
            retries=1,
            output_retries=1,
        )

    @classmethod
    def _extract_last_assistant_text(cls, routing_history: List[ModelMessage]) -> str:
        """Walk routing history backwards and return the last assistant text."""
        for msg in reversed(routing_history):
            if isinstance(msg, ModelResponse):
                text = "".join(
                    p.content for p in msg.parts if isinstance(p, TextPart)
                )
                if text:
                    return text
        return ""

    @classmethod
    def _detect_affirmation_context(
        cls, message: str, routing_history: List[ModelMessage]
    ) -> Optional[str]:
        """
        Fallback for when last_agent is unknown: inspect the assistant's last
        message to guess which agent the affirmation refers to.
        Returns None if the message is not a short affirmation or context is unclear.
        """
        if message.strip().lower() not in _AFFIRMATIONS:
            return None
        last_text = cls._extract_last_assistant_text(routing_history)
        if not last_text:
            return None
        tail = last_text[-400:].lower()

        if any(kw in tail for kw in (
            "campaign draft", "campaign plan", "draft a campaign", "draft me a campaign",
            "build a campaign", "create a campaign", "create a campaign draft",
            "budget split", "channel mix", "measurement plan", "go-to-market",
        )):
            return "campaign_planner"
        if any(kw in tail for kw in (
            "generate a video", "create a video", "make a video",
            "video generation", "video content", "short clip", "reel",
        )):
            return "video"
        if any(kw in tail for kw in (
            "generate an image", "create an image", "make an image",
            "image generation", "generate a graphic", "create a graphic",
            "visual content", "generate a picture",
        )):
            return "image"
        if any(kw in tail for kw in (
            "campaign performance", "campaign metrics", "campaign data",
            "roas analysis", "cpa", "ctr", "analyze your campaign",
        )):
            return "performance"
        if any(kw in tail for kw in (
            "keyword", "keywords", "search term", "trending search", "search trend",
        )):
            return "keywords"
        if any(kw in tail for kw in (
            "help setting up", "using lofi", "in lofi", "with lofi", "through lofi",
            "set up", "configure", "implement", "walk you through",
            "specific part", "specific step", "any of these steps",
        )):
            return "support"
        return None

    @classmethod
    def _detect_continuation(
        cls, message: str, last_agent: str
    ) -> Optional[str]:
        """
        Fast pure-Python check: return *last_agent* when the message looks like
        a continuation of the current conversation, or None to invoke the router.
        """
        normalized = message.strip().lower()

        if normalized in _AFFIRMATIONS:
            return last_agent

        word_count = len(normalized.split())

        # Topic shift keywords override continuation regardless of length,
        # so "can you make an image ..." doesn't get absorbed by the
        # follow-up prefix "can you".
        for agent, keywords in _TOPIC_SHIFT_KEYWORDS.items():
            if agent == last_agent:
                continue
            if any(kw in normalized for kw in keywords):
                return None

        if any(normalized.startswith(p) for p in _FOLLOW_UP_PREFIXES):
            return last_agent

        if word_count < 10:
            return last_agent

        return None

    @classmethod
    def _resolve_sticky_route(
        cls, router_output: "RouterOutput", last_agent: str
    ) -> str:
        """
        Apply asymmetric confidence thresholds when a last_agent is known.

        Thresholds (from easiest to hardest switch):
          - Same agent as last_agent:        >= 0.70
          - Switch to lucy (safe fallback):  >= 0.80
          - Leave lucy for a specialist:     >= 0.80
          - Specialist-to-specialist switch:  >= 0.85
        Falls back to last_agent when confidence is too low.
        """
        route = router_output.route
        confidence = router_output.confidence

        if route == last_agent:
            threshold = _THRESHOLD_SAME_AGENT
            decision = route if confidence >= threshold else last_agent
            logger.info(
                f"Router sticky | llm_route={route} confidence={confidence:.2f} "
                f"threshold={threshold:.2f} last_agent={last_agent} decision={decision}"
            )
            return decision

        if route == "lucy":
            threshold = _THRESHOLD_TO_LUCY
            decision = route if confidence >= threshold else last_agent
            logger.info(
                f"Router sticky | llm_route={route} confidence={confidence:.2f} "
                f"threshold={threshold:.2f} last_agent={last_agent} decision={decision}"
            )
            return decision

        if last_agent == "lucy":
            threshold = _THRESHOLD_FROM_LUCY
            decision = route if confidence >= threshold else last_agent
            logger.info(
                f"Router sticky | llm_route={route} confidence={confidence:.2f} "
                f"threshold={threshold:.2f} last_agent={last_agent} decision={decision}"
            )
            return decision

        threshold = _THRESHOLD_SPECIALIST_SWITCH
        decision = route if confidence >= threshold else last_agent
        logger.info(
            f"Router sticky | llm_route={route} confidence={confidence:.2f} "
            f"threshold={threshold:.2f} last_agent={last_agent} decision={decision}"
        )
        return decision

    @classmethod
    async def route_request(
        cls,
        message: str,
        routing_history: List[ModelMessage],
        user_id: str = "routing",
        request_type: str = None,
        attachments: Optional[List[dict]] = None,
        last_agent: Optional[str] = None,
    ) -> tuple[str, LofiAgent, Agent]:
        """
        Route a user request to the appropriate agent.

        When *last_agent* is provided (sticky routing), the method tries to
        skip the LLM call entirely for continuations.  When the LLM is
        invoked, asymmetric confidence thresholds bias toward the current
        agent to avoid mid-conversation disruption.
        """
        # Short-circuit for explicit request types
        if request_type in ("image", "video"):
            route = request_type
            selected_agent_class, selected_agent_instance = _build_agent(route)
            logger.info(
                f"Router decision | route={route} confidence=1.0 method=explicit_request_type "
                f"last_agent={last_agent} threshold=None overridden=False"
            )
            return route, selected_agent_class, selected_agent_instance

        # --- Sticky routing: skip the LLM for continuations ---
        if last_agent and last_agent in ROUTER_TARGETS:
            continuation = cls._detect_continuation(message, last_agent)
            if continuation:
                selected_agent_class, selected_agent_instance = _build_agent(continuation)
                logger.info(
                    f"Router decision | route={continuation} confidence=1.0 method=continuation "
                    f"last_agent={last_agent} threshold=None overridden=False"
                )
                return continuation, selected_agent_class, selected_agent_instance
        else:
            # No last_agent — fall back to legacy affirmation heuristic
            affirmation_route = cls._detect_affirmation_context(
                message, routing_history
            )
            if affirmation_route:
                selected_agent_class, selected_agent_instance = _build_agent(affirmation_route)
                logger.info(
                    f"Router decision | route={affirmation_route} confidence=1.0 method=legacy_affirmation "
                    f"last_agent={last_agent} threshold=None overridden=False"
                )
                return affirmation_route, selected_agent_class, selected_agent_instance

        # --- Full router LLM call ---
        router: Agent = cls.create()

        try:
            router_header = cls.reminder_header()
            attachment_hint = (
                "\nNote: the user has attached file(s) alongside this message."
                if attachments and request_type not in ("image", "video")
                else ""
            )
            sticky_hint = (
                f"\nThe user is currently in a {last_agent} conversation. "
                "Only change route if the request clearly requires a different agent."
                if last_agent and last_agent in ROUTER_TARGETS
                else ""
            )
            router_effective_prompt = (
                f"{router_header}\n\nUser request: {message.strip()}"
                f"{attachment_hint}{sticky_hint}"
            )

            route_result = await router.run(
                router_effective_prompt,
                deps=ChatDeps(user_id=user_id),
            )

            router_output: RouterOutput = route_result.output

            if last_agent and last_agent in ROUTER_TARGETS:
                route = cls._resolve_sticky_route(router_output, last_agent)
                overridden = route != router_output.route
                logger.info(
                    f"Router decision | route={route} confidence={router_output.confidence:.2f} "
                    f"method=llm_sticky last_agent={last_agent} threshold=None overridden={overridden}"
                )
            else:
                route = router_output.route
                if route != "lucy" and router_output.confidence < 0.7:
                    logger.info(
                        f"Router low-confidence override | llm_route={router_output.route} "
                        f"confidence={router_output.confidence:.2f} threshold=0.70 → lucy"
                    )
                    route = "lucy"
                logger.info(
                    f"Router decision | route={route} confidence={router_output.confidence:.2f} "
                    f"method=llm last_agent={last_agent} threshold=None overridden={route != router_output.route}"
                )

            selected_agent_class, selected_agent_instance = _build_agent(route)
            return route, selected_agent_class, selected_agent_instance

        except Exception as e:
            route = last_agent if last_agent and last_agent in ROUTER_TARGETS else "lucy"
            logger.error(f"Router error: {e} | fallback_final={route}")
            logger.info(
                f"Router decision | route={route} confidence=0.0 method=error_fallback "
                f"last_agent={last_agent} threshold=None overridden=False"
            )
            selected_agent_class, selected_agent_instance = _build_agent(route)
            return route, selected_agent_class, selected_agent_instance

    @classmethod
    def get_agent_for_name(cls, agent_name: str) -> tuple[str, LofiAgent, Agent]:
        """Get agent instance for a given agent name, bypassing routing.

        Args:
            agent_name: Name of the agent to use (keywords, support, lucy, image, video,
                        performance, campaign_planner, creative_director)

        Returns:
            Tuple of (route_name, selected_agent_class, selected_agent_instance)
        """
        route = agent_name.lower()
        selected_agent_class, selected_agent_instance = _build_agent(route)
        return route, selected_agent_class, selected_agent_instance
