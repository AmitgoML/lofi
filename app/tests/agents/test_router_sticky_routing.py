"""Tests for sticky routing: continuation detection and asymmetric thresholds."""

import pytest
from unittest.mock import AsyncMock, patch

from lucy.agents.router_agent import (
    RouterAgent,
    RouterOutput,
    _THRESHOLD_SAME_AGENT,
    _THRESHOLD_TO_LUCY,
    _THRESHOLD_FROM_LUCY,
    _THRESHOLD_SPECIALIST_SWITCH,
)
from lucy.agents.image_agent import ImageAgent


# ---------------------------------------------------------------------------
# _detect_continuation tests
# ---------------------------------------------------------------------------


class TestDetectContinuation:

    def test_affirmation_returns_last_agent(self):
        assert RouterAgent._detect_continuation("yes", "campaign_planner") == "campaign_planner"
        assert RouterAgent._detect_continuation("sure", "keywords") == "keywords"
        assert RouterAgent._detect_continuation("go ahead", "image") == "image"

    def test_follow_up_prefix_returns_last_agent(self):
        assert RouterAgent._detect_continuation("what about the budget?", "campaign_planner") == "campaign_planner"
        assert RouterAgent._detect_continuation("how about Facebook?", "campaign_planner") == "campaign_planner"
        assert RouterAgent._detect_continuation("can you also add negative keywords?", "keywords") == "keywords"
        assert RouterAgent._detect_continuation("tell me more about that", "lucy") == "lucy"
        assert RouterAgent._detect_continuation("also include Instagram", "campaign_planner") == "campaign_planner"

    def test_short_message_without_shift_keywords_returns_last_agent(self):
        assert RouterAgent._detect_continuation("and the timeline?", "campaign_planner") == "campaign_planner"
        assert RouterAgent._detect_continuation("sounds great", "keywords") == "keywords"
        assert RouterAgent._detect_continuation("interesting", "performance") == "performance"
        assert RouterAgent._detect_continuation("what do you think?", "creative_director") == "creative_director"

    def test_message_with_shift_keywords_returns_none(self):
        """Message containing topic-shift keywords for a DIFFERENT agent should trigger the router."""
        assert RouterAgent._detect_continuation("generate an image of a sunset", "campaign_planner") is None
        assert RouterAgent._detect_continuation("can you generate an image of a sunset?", "campaign_planner") is None
        assert RouterAgent._detect_continuation("find keywords for my campaign", "performance") is None
        assert RouterAgent._detect_continuation("analyze my campaign data", "keywords") is None
        assert RouterAgent._detect_continuation("plan a campaign for me", "lucy") is None

    def test_long_message_with_shift_keywords_returns_none(self):
        """Topic shift keywords should override continuation even in 15+ word messages."""
        msg = "awesome can we make an image of a soccer player as well? inspired by maradona"
        assert RouterAgent._detect_continuation(msg, "lucy") is None
        assert RouterAgent._detect_continuation(msg, "video") is None

    def test_follow_up_prefix_with_shift_keywords_returns_none(self):
        """Topic shift keywords take priority over follow-up prefixes like 'can you'."""
        assert RouterAgent._detect_continuation(
            "can you generate an image of a soccer player for our campaign assets?", "campaign_planner"
        ) is None
        assert RouterAgent._detect_continuation(
            "can you create a video for our Instagram campaign with dramatic lighting?", "keywords"
        ) is None

    def test_short_message_with_same_agent_keywords_returns_last_agent(self):
        """Keywords matching the CURRENT agent should not trigger a topic shift."""
        assert RouterAgent._detect_continuation("find keywords for shoes", "keywords") == "keywords"
        assert RouterAgent._detect_continuation("plan a campaign with $500", "campaign_planner") == "campaign_planner"
        assert RouterAgent._detect_continuation("analyze my campaign performance", "performance") == "performance"

    def test_long_message_returns_none(self):
        """Messages with 10+ words should go through the router."""
        long_msg = "I want to create a completely new campaign strategy for my e-commerce brand targeting Gen Z consumers across multiple channels"
        assert RouterAgent._detect_continuation(long_msg, "keywords") is None

    def test_medium_message_10_words_returns_none(self):
        """Messages at 10 words are substantive enough to route."""
        msg = "I want to analyze my campaign performance data this quarter"
        assert RouterAgent._detect_continuation(msg, "keywords") is None

    def test_short_message_9_words_returns_last_agent(self):
        """Messages under 10 words still stay with current agent (no shift keywords)."""
        msg = "what about the timeline for our launch"
        assert RouterAgent._detect_continuation(msg, "campaign_planner") == "campaign_planner"

    def test_partial_shift_keywords_image_of(self):
        """Partial keyword 'image of' should trigger shift even in short messages."""
        assert RouterAgent._detect_continuation("make an image of a sunset", "campaign_planner") is None
        assert RouterAgent._detect_continuation("image of a soccer player", "performance") is None

    def test_partial_shift_keywords_video_of(self):
        assert RouterAgent._detect_continuation("video of our product launch", "keywords") is None

    def test_partial_shift_keywords_analyze_my(self):
        assert RouterAgent._detect_continuation("analyze my ad spend", "image") is None

    def test_partial_shift_keywords_create_campaign(self):
        assert RouterAgent._detect_continuation("create a campaign for summer", "performance") is None

    def test_partial_shift_keywords_brief_for(self):
        assert RouterAgent._detect_continuation("brief for our new product", "keywords") is None

    def test_empty_and_whitespace(self):
        assert RouterAgent._detect_continuation("", "lucy") == "lucy"
        assert RouterAgent._detect_continuation("   ", "lucy") == "lucy"


# ---------------------------------------------------------------------------
# _resolve_sticky_route tests
# ---------------------------------------------------------------------------


class TestResolveStickyRoute:

    def test_same_agent_accepted_at_low_confidence(self):
        output = RouterOutput(route="campaign_planner", confidence=0.72)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "campaign_planner"

    def test_same_agent_rejected_below_threshold(self):
        output = RouterOutput(route="campaign_planner", confidence=0.50)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "campaign_planner"

    def test_switch_to_lucy_accepted_at_moderate_confidence(self):
        output = RouterOutput(route="lucy", confidence=0.85)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "lucy"

    def test_switch_to_lucy_rejected_below_threshold(self):
        output = RouterOutput(route="lucy", confidence=0.75)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "campaign_planner"

    def test_switch_from_lucy_to_specialist_at_standard_threshold(self):
        output = RouterOutput(route="keywords", confidence=0.92)
        assert RouterAgent._resolve_sticky_route(output, "lucy") == "keywords"

    def test_switch_from_lucy_rejected_below_threshold(self):
        output = RouterOutput(route="keywords", confidence=0.85)
        assert RouterAgent._resolve_sticky_route(output, "lucy") == "lucy"

    def test_specialist_to_specialist_requires_high_confidence(self):
        output = RouterOutput(route="image", confidence=0.97)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "image"

    def test_specialist_to_specialist_rejected_below_threshold(self):
        output = RouterOutput(route="image", confidence=0.78)
        assert RouterAgent._resolve_sticky_route(output, "campaign_planner") == "campaign_planner"

    def test_boundary_values(self):
        """Thresholds are inclusive (>=)."""
        assert RouterAgent._resolve_sticky_route(
            RouterOutput(route="keywords", confidence=_THRESHOLD_SAME_AGENT), "keywords"
        ) == "keywords"
        assert RouterAgent._resolve_sticky_route(
            RouterOutput(route="lucy", confidence=_THRESHOLD_TO_LUCY), "keywords"
        ) == "lucy"
        assert RouterAgent._resolve_sticky_route(
            RouterOutput(route="keywords", confidence=_THRESHOLD_FROM_LUCY), "lucy"
        ) == "keywords"
        assert RouterAgent._resolve_sticky_route(
            RouterOutput(route="image", confidence=_THRESHOLD_SPECIALIST_SWITCH), "keywords"
        ) == "image"


# ---------------------------------------------------------------------------
# route_request integration tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestRouteRequestSticky:

    @pytest.mark.asyncio
    async def test_request_type_overrides_sticky(self):
        """Explicit request_type still takes priority over sticky routing."""
        with patch.object(ImageAgent, "create") as mock_create:
            mock_agent = AsyncMock()
            mock_create.return_value = mock_agent

            route, agent_class, _ = await RouterAgent.route_request(
                message="Create a sunset",
                routing_history=[],
                user_id="test-user",
                request_type="image",
                last_agent="campaign_planner",
            )

            assert route == "image"
            assert agent_class == ImageAgent
