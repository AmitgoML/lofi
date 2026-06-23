"""QA Agent: validates the assembled campaign before proposal assembly."""

from lofi.schemas.qa_agent import QAAgentInput, QAAgentOutput
from lofi.state.workflow_state import WorkflowState


class QAAgent:
    """Runs budget, platform, creative-completeness, and policy checks."""

    def run(self, state: WorkflowState) -> WorkflowState:
        creative_output = state["creative_director_output"]
        qa_input = QAAgentInput(
            campaign_plan=state["campaign_plan"],
            text_assets=creative_output.texts,
            assets=creative_output.assets,
            org_budget_maximum=state["organization_max_budget"],
        )
        state["qa_result"] = self.validate(qa_input)
        return state

    def validate(self, qa_input: QAAgentInput) -> QAAgentOutput:
        raise NotImplementedError

    def _check_budget(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_platform_compatibility(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_creative_completeness(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_policy_compliance(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError
