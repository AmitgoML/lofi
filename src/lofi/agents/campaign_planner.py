"""Campaign Planner Agent: turns the brief + insights into a concrete plan."""

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.creative_director import CreativeBrief
from lofi.schemas.performance_analyst import PerformanceAnalystOutput
from lofi.state.workflow_state import WorkflowState


class CampaignPlannerAgent:
    """Produces a CampaignPlan and CreativeBrief from upstream context."""

    def run(self, state: WorkflowState) -> WorkflowState:
        plan, creative_brief = self.plan(
            campaign_brief=state["campaign_brief"],
            performance_insights=state["performance_insights"],
        )
        state["campaign_plan"] = plan
        state["creative_brief"] = creative_brief
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
