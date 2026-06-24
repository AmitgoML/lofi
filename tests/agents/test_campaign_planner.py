"""Unit tests for CampaignPlannerAgent's planning guard and replan-on-fail logic.

The orchestrator (route_from_campaign_planner) decides when this node runs;
these tests only cover what the node itself does once control reaches it.
"""

from unittest.mock import MagicMock

import pytest

from lofi.agents.campaign_planner import CampaignPlannerAgent
from lofi.schemas.common import QAStatus


@pytest.fixture
def agent() -> CampaignPlannerAgent:
    return CampaignPlannerAgent()


class TestRun:
    def test_does_nothing_without_a_brief(self, agent: CampaignPlannerAgent) -> None:
        state = {"performance_insights": object()}

        result = agent.run(state)

        assert "campaign_plan" not in result

    def test_does_nothing_without_performance_insights(self, agent: CampaignPlannerAgent) -> None:
        state = {"campaign_brief": object()}

        result = agent.run(state)

        assert "campaign_plan" not in result

    def test_plans_once_brief_and_insights_are_ready(self, agent: CampaignPlannerAgent) -> None:
        agent.plan = MagicMock(return_value=("plan", "creative-brief"))
        state = {"campaign_brief": "brief", "performance_insights": "insights"}

        result = agent.run(state)

        agent.plan.assert_called_once_with(campaign_brief="brief", performance_insights="insights")
        assert result["campaign_plan"] == "plan"
        assert result["creative_brief"] == "creative-brief"

    def test_does_not_replan_once_a_plan_exists_and_qa_has_not_failed(self, agent: CampaignPlannerAgent) -> None:
        agent.plan = MagicMock()
        state = {
            "campaign_brief": "brief",
            "performance_insights": "insights",
            "campaign_plan": "existing-plan",
            "creative_director_output": "existing-output",
        }

        result = agent.run(state)

        agent.plan.assert_not_called()
        assert result["campaign_plan"] == "existing-plan"
        assert result["creative_director_output"] == "existing-output"

    def test_replans_and_clears_stale_output_on_qa_fail(self, agent: CampaignPlannerAgent) -> None:
        agent.plan = MagicMock(return_value=("new-plan", "new-creative-brief"))
        qa_result = MagicMock(status=QAStatus.FAIL)
        state = {
            "campaign_brief": "brief",
            "performance_insights": "insights",
            "campaign_plan": "stale-plan",
            "creative_director_output": "stale-output",
            "qa_result": qa_result,
        }

        result = agent.run(state)

        agent.plan.assert_called_once()
        assert result["campaign_plan"] == "new-plan"
        assert "creative_director_output" not in result
        assert "qa_result" not in result

    def test_does_not_replan_when_qa_passed(self, agent: CampaignPlannerAgent) -> None:
        agent.plan = MagicMock()
        qa_result = MagicMock(status=QAStatus.PASS)
        state = {
            "campaign_brief": "brief",
            "performance_insights": "insights",
            "campaign_plan": "existing-plan",
            "qa_result": qa_result,
        }

        result = agent.run(state)

        agent.plan.assert_not_called()
        assert result["qa_result"] is qa_result
