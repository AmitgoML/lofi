"""QA Agent: validates the assembled campaign before proposal assembly."""

from lofi.schemas.common import QAStatus
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
        # No live check logic yet (the _check_* helpers below are still
        # unimplemented stubs) - serve the static positive sample instead of
        # the bare `True` this returned before, mirroring CampaignPlannerAgent
        # and CreativeDirectorAgent falling back to produce_static_sample()
        # in the same situation.
        return self.produce_static_sample()

    def _check_budget(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_platform_compatibility(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_creative_completeness(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    def _check_policy_compliance(self, qa_input: QAAgentInput) -> bool:
        raise NotImplementedError

    @staticmethod
    def produce_static_sample() -> QAAgentOutput:
        """Static all-pass QAAgentOutput.

        Stands in for validate() (still backed by NotImplementedError _check_*
        stubs above) wherever a caller needs a real QAAgentOutput-shaped
        value without live check logic - e.g. local UI development or manual
        testing alongside CampaignPlannerAgent/CreativeDirectorAgent's own
        produce_static_sample() methods.
        """
        return QAAgentOutput(
            status=QAStatus.PASS,
            budget_validation_passed=True,
            platform_compatibility_passed=True,
            creative_completeness_passed=True,
            required_fields_passed=True,
            policy_compliance_passed=True,
            issues=[],
            return_to_campaign_planner=False,
        )
