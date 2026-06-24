"""Human Review Agent: pauses the workflow for a person to approve or reject
the assembled campaign proposal, then persists it if approved.

Uses interrupt()/Command(resume=...) rather than a hand-rolled status field -
see api/routes.py, which reads the pause/resume state via graph.get_state().
"""

from langgraph.types import interrupt

from lofi.persistence.supabase_client import SupabaseClient
from lofi.state.workflow_state import WorkflowState


class HumanReviewAgent:
    """Surfaces the FinalCampaignProposal for human approval/rejection."""

    def __init__(self, supabase_client: SupabaseClient) -> None:
        self._supabase_client = supabase_client

    def run(self, state: WorkflowState) -> WorkflowState:
        proposal = state["campaign_proposal"]
        decision = interrupt({"type": "human_review", "campaign_proposal": proposal.model_dump(mode="json")})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

        state["approved"] = approved
        if approved:
            state["persisted_campaign_id"] = self._supabase_client.save_campaign(proposal.model_dump(mode="json"))
        return state
