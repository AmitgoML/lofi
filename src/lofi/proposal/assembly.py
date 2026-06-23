"""Campaign Proposal Assembly: combines all workflow outputs into a final proposal."""

from lofi.schemas.campaign_planner import FinalCampaignProposal
from lofi.state.workflow_state import WorkflowState


class CampaignProposalAssembler:
    """Combines plan, insights, creative assets, and QA results into a proposal."""

    def run(self, state: WorkflowState) -> WorkflowState:
        state["campaign_proposal"] = self.assemble(state)
        return state

    def assemble(self, state: WorkflowState) -> FinalCampaignProposal:
        creative_output = state["creative_director_output"]
        brief = state["campaign_brief"]
        return FinalCampaignProposal(
            organization_id=brief.organization_id,
            brand=brief.brand,
            campaign_plan=state["campaign_plan"],
            creative_assets=creative_output.assets,
            copy_assets=creative_output.texts,
            qa_result=state["qa_result"],
        )
