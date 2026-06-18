"""
AgentFactory — single boundary for creating agent class/instance pairs.

All code that needs to map a route name to a live pydantic-ai ``Agent``
should go through :func:`build` rather than duplicating the mapping inline.
This makes it easy to add caching, tracing, or model-override logic in one
place without touching every call-site.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from pydantic_ai import Agent

from lucy.agents.common.base_agent import LofiAgent

if TYPE_CHECKING:
    pass

# Route names that are valid targets.  Keep in sync with RouterAgent targets.
VALID_ROUTES: frozenset[str] = frozenset(
    [
        "keywords",
        "support",
        "lucy",
        "image",
        "video",
        "performance",
        "campaign_planner",
        "creative_director",
    ]
)


def _get_agent_class(route: str) -> type[LofiAgent]:
    """Return the agent *class* for a given route string.

    Uses deferred imports to avoid loading every agent module at import time
    and to prevent circular-import chains through ``router_agent``.
    Falls back to ``LucyAgent`` for unknown routes.
    """
    from lucy.agents.keywords_agent import KeywordsAgent
    from lucy.agents.support_agent import SupportAgent
    from lucy.agents.lucy_agent import LucyAgent
    from lucy.agents.image_agent import ImageAgent
    from lucy.agents.video_agent import VideoAgent
    from lucy.agents.performance_analyst_agent import PerformanceAnalystAgent
    from lucy.agents.campaign_planner_agent import CampaignPlannerAgent
    from lucy.agents.creative_director_agent import CreativeDirectorAgent

    _MAP: dict[str, type[LofiAgent]] = {
        "keywords":          KeywordsAgent,
        "support":           SupportAgent,
        "lucy":              LucyAgent,
        "image":             ImageAgent,
        "video":             VideoAgent,
        "performance":       PerformanceAnalystAgent,
        "campaign_planner":  CampaignPlannerAgent,
        "creative_director": CreativeDirectorAgent,
    }
    return _MAP.get(route, LucyAgent)


_cache: dict[tuple[str, str | None], tuple[type[LofiAgent], Agent]] = {}


def clear_cache() -> None:
    """Clear the agent instance cache. Useful for testing."""
    _cache.clear()


def build(
    route: str,
    model_name: Optional[str] = None,
) -> tuple[type[LofiAgent], Agent]:
    """Return a cached agent class/instance pair for *route*.

    pydantic-ai ``Agent`` objects are stateless (all per-request state
    lives on ``ctx.deps``), so they are safe to reuse across requests.
    The cache is keyed by ``(route, model_name)`` to support overrides.

    Args:
        route: One of the known route strings (e.g. ``"lucy"``, ``"image"``).
               Unknown routes fall back to ``LucyAgent``.
        model_name: Optional model override forwarded to the agent's
                    ``create()`` classmethod.  ``None`` uses the agent's
                    default from :data:`lucy.agents.common.model_config.Models`.

    Returns:
        A ``(agent_class, agent_instance)`` tuple.  The class is useful for
        calling classmethods such as ``pre_run_check`` and ``reminder_header``;
        the instance is the live ``pydantic_ai.Agent`` ready to run.
    """
    key = (route, model_name)
    if key not in _cache:
        agent_class = _get_agent_class(route)
        agent_instance: Agent = agent_class.create(model_name)
        _cache[key] = (agent_class, agent_instance)
    return _cache[key]
