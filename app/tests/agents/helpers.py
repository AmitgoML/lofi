"""Shared test utilities for agent tests.

Provides reusable setup helpers for:
- Router mocking (RouterAgent.create + _build_agent + RouterOutput)
- Mock context objects with ChatDeps
- Routing history construction
"""

from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

from lucy.agents.common.models import ChatDeps
from lucy.agents.router_agent import RouterAgent, RouterOutput


# ---------------------------------------------------------------------------
# Mock context helpers
# ---------------------------------------------------------------------------


def make_mock_ctx(request_params: Optional[dict] = None) -> Mock:
    """Return a Mock context whose .deps is a ChatDeps with the given request_params."""
    mock_ctx = Mock()
    mock_ctx.deps = ChatDeps(request_params=request_params)
    return mock_ctx


# ---------------------------------------------------------------------------
# Router mock helpers
# ---------------------------------------------------------------------------


@contextmanager
def mock_router(route: str, confidence: float, agent_class: Any, agent_instance: Any):
    """Context manager that patches RouterAgent.create and _build_agent.

    Yields the mock_router AsyncMock so callers can inspect calls if needed.

    Usage::

        async with mock_router("support", 0.95, SupportAgent, mock_agent):
            route, cls, inst = await RouterAgent.route_request(...)
    """
    with (
        patch.object(RouterAgent, "create") as mock_create,
        patch("lucy.agents.router_agent._build_agent") as mock_build,
    ):
        mock_router_run = AsyncMock()
        mock_router_run.run.return_value = AsyncMock(
            output=RouterOutput(route=route, confidence=confidence)
        )
        mock_create.return_value = mock_router_run
        mock_build.return_value = (agent_class, agent_instance)
        yield mock_router_run


# ---------------------------------------------------------------------------
# Routing history helpers
# ---------------------------------------------------------------------------


def make_assistant_history(text: str) -> list:
    """Build a minimal routing history list containing one assistant message."""
    from pydantic_ai.messages import ModelResponse, TextPart

    return [ModelResponse(parts=[TextPart(content=text)])]
