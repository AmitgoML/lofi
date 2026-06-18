import pytest
from unittest.mock import AsyncMock, patch
from lucy.agents.router_agent import RouterAgent, RouterOutput, ROUTER_TARGETS
from lucy.agents.common import factory as agent_factory
from lucy.agents.image_agent import ImageAgent
from lucy.agents.video_agent import VideoAgent
from lucy.agents.support_agent import SupportAgent
from lucy.agents.keywords_agent import KeywordsAgent
from lucy.agents.lucy_agent import LucyAgent
from tests.agents.helpers import mock_router


class TestAgentFactory:
    """Tests for the AgentFactory route→class mapping."""

    def test_factory_image(self):
        assert agent_factory._get_agent_class("image") == ImageAgent

    def test_factory_video(self):
        assert agent_factory._get_agent_class("video") == VideoAgent

    def test_factory_support(self):
        assert agent_factory._get_agent_class("support") == SupportAgent

    def test_factory_keywords(self):
        assert agent_factory._get_agent_class("keywords") == KeywordsAgent

    def test_factory_lucy(self):
        assert agent_factory._get_agent_class("lucy") == LucyAgent

    def test_factory_unknown_defaults_to_lucy(self):
        assert agent_factory._get_agent_class("nonexistent") == LucyAgent

    def test_valid_routes_constant(self):
        """VALID_ROUTES must include all known route names."""
        expected = {"keywords", "support", "lucy", "image", "video",
                    "performance", "campaign_planner", "creative_director"}
        assert expected == agent_factory.VALID_ROUTES


class TestRouterAgent:
    """Test cases for RouterAgent routing functionality."""

    def test_create_uses_provided_model_name(self):
        """Router should use the provided model name when given."""
        with patch("lucy.agents.router_agent.Agent") as mock_agent_cls:
            RouterAgent.create("openai:gpt-5-mini")

            mock_agent_cls.assert_called_once()
            assert mock_agent_cls.call_args.kwargs["model"] == "openai:gpt-5-mini"

    def test_create_defaults_to_models_router(self):
        """Router should default to Models.ROUTER when no model_name is provided."""
        with (
            patch("lucy.agents.router_agent.Agent") as mock_agent_cls,
            patch("lucy.agents.router_agent.Models") as mock_models,
        ):
            mock_models.ROUTER = "openai:gpt-5-mini"

            RouterAgent.create()

            mock_agent_cls.assert_called_once()
            assert mock_agent_cls.call_args.kwargs["model"] == "openai:gpt-5-mini"

    def test_reminder_header_format(self):
        """Test that reminder header contains the valid routing targets."""
        header = RouterAgent.reminder_header()

        # All valid route targets must appear in the header
        assert "keywords" in header.lower()
        assert "support" in header.lower()
        assert "image" in header.lower()
        assert "video" in header.lower()
        assert "lucy" in header.lower()
        assert "performance" in header.lower()
        assert "campaign_planner" in header.lower()
        assert "creative_director" in header.lower()

    def test_router_targets_covers_all_routes(self):
        """ROUTER_TARGETS must cover all known route names."""
        expected = {"keywords", "support", "lucy", "image", "video",
                    "performance", "campaign_planner", "creative_director"}
        assert expected == set(ROUTER_TARGETS)

    @pytest.mark.asyncio
    async def test_route_request_with_image_request_type(self):
        """Test that request_type='image' short-circuits to image agent."""
        with patch.object(ImageAgent, "create") as mock_create:
            mock_agent = AsyncMock()
            mock_create.return_value = mock_agent

            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Create a sunset",
                routing_history=[],
                user_id="test-user",
                request_type="image",
            )

            assert route == "image"
            assert agent_class == ImageAgent
            assert agent_instance == mock_agent
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_request_with_none_request_type(self):
        """Test that request_type=None uses normal routing via LLM."""
        mock_support_agent = AsyncMock()
        with mock_router("support", 0.95, SupportAgent, mock_support_agent) as mock_r:
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Help me with billing",
                routing_history=[],
                user_id="test-user",
                request_type=None,
            )

            assert route == "support"
            assert agent_class == SupportAgent
            assert agent_instance == mock_support_agent
            mock_r.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_request_with_empty_request_type(self):
        """Test that empty request_type uses normal routing via LLM."""
        mock_keywords_agent = AsyncMock()
        with mock_router("keywords", 0.9, KeywordsAgent, mock_keywords_agent):
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="What keywords should I use?",
                routing_history=[],
                user_id="test-user",
                request_type="",
            )

            assert route == "keywords"
            assert agent_class == KeywordsAgent
            assert agent_instance == mock_keywords_agent

    @pytest.mark.asyncio
    async def test_route_request_low_confidence_falls_back_to_lucy(self):
        """Test that low-confidence specialist routes fall back to lucy."""
        with (
            patch.object(RouterAgent, "create") as mock_create,
        ):
            mock_router = AsyncMock()
            mock_router.run.return_value = AsyncMock(
                output=RouterOutput(route="campaign_planner", confidence=0.55)
            )
            mock_create.return_value = mock_router

            with patch.object(LucyAgent, "create") as mock_lucy_create:
                mock_lucy_agent = AsyncMock()
                mock_lucy_create.return_value = mock_lucy_agent

                route, agent_class, agent_instance = await RouterAgent.route_request(
                    message="What's a good channel for my industry?",
                    routing_history=[],
                    user_id="test-user",
                    request_type=None,
                )

                assert route == "lucy"
                assert agent_class == LucyAgent

    @pytest.mark.asyncio
    async def test_route_request_high_confidence_specialist_preserved(self):
        """Test that high-confidence specialist routes are kept."""
        from lucy.agents.campaign_planner_agent import CampaignPlannerAgent
        mock_cp_agent = AsyncMock()
        with mock_router("campaign_planner", 0.92, CampaignPlannerAgent, mock_cp_agent):
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Plan a campaign with a $5000 budget on Facebook and Instagram",
                routing_history=[],
                user_id="test-user",
                request_type=None,
            )

            assert route == "campaign_planner"
            assert agent_class == CampaignPlannerAgent

    @pytest.mark.asyncio
    async def test_route_request_low_confidence_lucy_not_overridden(self):
        """Test that low-confidence lucy routes stay as lucy (no double-fallback)."""
        mock_lucy_agent = AsyncMock()
        with mock_router("lucy", 0.5, LucyAgent, mock_lucy_agent):
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Tell me about marketing",
                routing_history=[],
                user_id="test-user",
                request_type=None,
            )

            assert route == "lucy"
            assert agent_class == LucyAgent

    @pytest.mark.asyncio
    async def test_route_request_error_handling(self):
        """Test that router errors fall back to lucy agent."""
        mock_lucy_agent = AsyncMock()
        with (
            patch.object(RouterAgent, "create") as mock_create,
            patch("lucy.agents.router_agent._build_agent") as mock_build,
        ):
            mock_router_run = AsyncMock()
            mock_router_run.run.side_effect = Exception("Router error")
            mock_create.return_value = mock_router_run
            mock_build.return_value = (LucyAgent, mock_lucy_agent)

            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Test message",
                routing_history=[],
                user_id="test-user",
                request_type=None,
            )

            assert route == "lucy"
            assert agent_class == LucyAgent
            assert agent_instance == mock_lucy_agent
