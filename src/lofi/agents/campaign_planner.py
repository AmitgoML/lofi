"""Campaign Planner Agent: the workflow's orchestrator.

It's the graph's entry/exit point (see workflow_graph.route_from_campaign_planner),
but this node's own job is narrower: do the actual planning once a brief and
Performance Analyst insights are available, and replan if QA comes back FAIL.
"""

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.common import QAStatus
from lofi.schemas.creative_director import CreativeBrief
from lofi.schemas.performance_analyst import PerformanceAnalystOutput
from lofi.state.workflow_state import WorkflowState


class CampaignPlannerAgent:
    """Produces a CampaignPlan and CreativeBrief from upstream context."""

    def run(self, state: WorkflowState) -> WorkflowState:
        if "campaign_brief" not in state or "performance_insights" not in state:
            # Not ready to plan yet - the orchestrator's router will send the
            # workflow to intake/performance_analyst first.
            return state

        qa_result = state.get("qa_result")
        needs_planning = "campaign_plan" not in state or (qa_result is not None and qa_result.status == QAStatus.FAIL)
        if not needs_planning:
            return state

        plan, creative_brief = self.plan(
            campaign_brief=state["campaign_brief"],
            performance_insights=state["performance_insights"],
        )
        state["campaign_plan"] = plan
        state["creative_brief"] = creative_brief
        # A (re)plan invalidates whatever Creative Director/QA produced against
        # the previous plan - clearing them lets the router regenerate both.
        state.pop("creative_director_output", None)
        state.pop("qa_result", None)
        return state

    def plan(
        self,
        campaign_brief: CampaignPlannerInput,
        performance_insights: PerformanceAnalystOutput,
    ) -> tuple[CampaignPlan, CreativeBrief]:
        raise NotImplementedError

    def _plan_goal(self, campaign_brief: CampaignPlannerInput) -> dict:
        raise NotImplementedError

    def _plan_audience(
        self, campaign_brief: CampaignPlannerInput, performance_insights: PerformanceAnalystOutput
    ) -> dict:
        raise NotImplementedError

    def _plan_platforms(self, performance_insights: PerformanceAnalystOutput) -> list[str]:
        raise NotImplementedError

    def _plan_locations(
        self, campaign_brief: CampaignPlannerInput, performance_insights: PerformanceAnalystOutput
    ) -> dict:
        raise NotImplementedError

    def _plan_budget(self, campaign_brief: CampaignPlannerInput, organization_max_budget: float) -> dict:
        raise NotImplementedError
