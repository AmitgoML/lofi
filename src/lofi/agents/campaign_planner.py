"""Campaign Planner Agent: the workflow's orchestrator.

It's the graph's entry/exit point (see workflow_graph.route_from_campaign_planner),
but this node's own job is narrower: do the actual planning once a brief and
Performance Analyst insights are available, and replan if QA comes back FAIL.
"""

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.common import (
    AudienceSpec,
    BudgetSpec,
    CampaignGoal,
    CampaignTiming,
    Location,
    Platform,
    QAStatus,
)
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
        # No live planning logic yet (the _plan_* helpers below are still
        # unimplemented stubs) - serve the static sample instead of raising,
        # mirroring CreativeDirectorAgent.run() falling back to
        # produce_static_sample() in the same situation.
        return self.produce_static_sample()

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

    @staticmethod
    def produce_static_sample() -> tuple[CampaignPlan, CreativeBrief]:
        """Sample (CampaignPlan, CreativeBrief) pair for a Valvoline campaign.

        Stands in for plan() (still a NotImplementedError stub above)
        wherever a caller needs real CampaignPlanner-shaped output without
        live planning logic - e.g. local UI development or manual testing.
        The CreativeBrief here matches the goal/platform/audience that
        CreativeDirectorAgent.produce_static_sample()'s Valvoline assets
        were written for, so the two stay consistent end to end.
        """
        plan = CampaignPlan(
            goal=CampaignGoal.AWARENESS,
            campaign_type="always-on",
            objective="reach",
            audience=AudienceSpec(age_min=25, age_max=54, genders=["all"]),
            platforms=[Platform.META],
            locations=[Location(country="USA")],
            budget=BudgetSpec(
                total_budget=5000.0,
                daily_budget=200.0,
                currency="USD",
                platform_split={"meta": 5000.0},
            ),
            timing=CampaignTiming(start_date="2026-07-01", flight_duration_days=30),
        )
        creative_brief = CreativeBrief(
            goal=CampaignGoal.AWARENESS,
            audience=AudienceSpec(age_min=25, age_max=54, genders=["all"]),
            platforms=[Platform.META],
            offer="10% off your next oil change",
        )
        return plan, creative_brief
