"""Performance Analyst Agent: mines historical metrics for recommendations."""

from lofi.schemas.performance_analyst import PerformanceAnalystInput, PerformanceAnalystOutput
from lofi.state.workflow_state import WorkflowState


class PerformanceAnalystAgent:
    """Reads historical campaign metrics and produces a PerformanceAnalystOutput."""

    def run(self, state: WorkflowState) -> WorkflowState:
        brief = state["campaign_brief"]
        analyst_input = PerformanceAnalystInput(
            user_request=state["user_request"],
            brand=brief.brand,
            organization_id=brief.organization_id,
        )
        state["performance_insights"] = self.analyze(analyst_input)
        return state

    def analyze(self, analyst_input: PerformanceAnalystInput) -> PerformanceAnalystOutput:
        raise NotImplementedError

    def _read_platform_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def _read_location_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def _read_audience_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def _read_creative_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError
