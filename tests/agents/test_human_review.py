"""Unit tests for HumanReviewAgent.

run() calls interrupt(), which requires a running graph, so it's exercised
through a minimal single-node graph with a checkpointer (full coverage of
the approve/reject HTTP flow lives in tests/api/test_routes.py).
"""

from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from lofi.agents.human_review import HumanReviewAgent
from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import FinalCampaignProposal
from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, Location, Platform, QAStatus
from lofi.schemas.creative_director import TextAsset
from lofi.schemas.qa_agent import QAAgentOutput
from lofi.state.workflow_state import WorkflowState


def _sample_campaign_proposal() -> FinalCampaignProposal:
    return FinalCampaignProposal(
        organization_id="org-1",
        brand="Acme Coffee",
        campaign_plan=CampaignPlan(
            goal=CampaignGoal.AWARENESS,
            campaign_type="launch",
            objective="reach",
            audience=AudienceSpec(age_min=18, age_max=35),
            platforms=[Platform.META],
            locations=[Location(country="USA")],
            budget=BudgetSpec(total_budget=500.0),
            timing=CampaignTiming(start_date="2026-07-01"),
        ),
        creative_assets=[],
        copy_assets=TextAsset(headlines=["Wake up to Acme"], descriptions=["Best coffee in town"], cta="Order now"),
        qa_result=QAAgentOutput(
            status=QAStatus.PASS,
            budget_validation_passed=True,
            platform_compatibility_passed=True,
            creative_completeness_passed=True,
            required_fields_passed=True,
            policy_compliance_passed=True,
        ),
    )


@pytest.fixture
def supabase_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def agent(supabase_client: MagicMock) -> HumanReviewAgent:
    return HumanReviewAgent(supabase_client)


def _build_graph(agent: HumanReviewAgent):
    graph = StateGraph(WorkflowState)
    graph.add_node("review", agent.run)
    graph.set_entry_point("review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=MemorySaver())


class TestRun:
    def test_pauses_then_approves_and_persists(self, agent: HumanReviewAgent, supabase_client: MagicMock) -> None:
        supabase_client.save_campaign.return_value = "campaign-123"
        compiled = _build_graph(agent)
        config = {"configurable": {"thread_id": "t1"}}
        proposal = _sample_campaign_proposal()

        first = compiled.invoke({"campaign_proposal": proposal}, config=config)
        assert first["__interrupt__"][0].value["type"] == "human_review"
        assert first["__interrupt__"][0].value["campaign_proposal"]["brand"] == "Acme Coffee"

        second = compiled.invoke(Command(resume={"approved": True}), config=config)

        assert second["approved"] is True
        assert second["persisted_campaign_id"] == "campaign-123"
        supabase_client.save_campaign.assert_called_once()

    def test_pauses_then_rejects_without_persisting(self, agent: HumanReviewAgent, supabase_client: MagicMock) -> None:
        compiled = _build_graph(agent)
        config = {"configurable": {"thread_id": "t2"}}
        proposal = _sample_campaign_proposal()

        compiled.invoke({"campaign_proposal": proposal}, config=config)
        second = compiled.invoke(Command(resume={"approved": False}), config=config)

        assert second["approved"] is False
        assert "persisted_campaign_id" not in second
        supabase_client.save_campaign.assert_not_called()
