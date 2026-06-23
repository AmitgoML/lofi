"""Entry point: compiles and invokes the campaign planning workflow graph."""

from lofi.graph.workflow_graph import build_campaign_workflow_graph
from lofi.intake.lucy_intake import LucyCampaignIntake
from lofi.schemas.intake import IntakeDraft
from lofi.state.workflow_state import WorkflowState


def run_campaign_workflow(user_request: str, organization_max_budget: float) -> WorkflowState:
    """Starts the workflow. If the request is missing required fields, the
    returned state's `intake_form_request` is set and the workflow stops
    there until `submit_intake_form` is called with the missing fields.
    """
    graph = build_campaign_workflow_graph().compile()
    initial_state: WorkflowState = {
        "user_request": user_request,
        "organization_max_budget": organization_max_budget,
    }
    return graph.invoke(initial_state)


def submit_intake_form(state: WorkflowState, submission: IntakeDraft) -> WorkflowState:
    """Resumes a paused workflow with the user's answers to the intake form."""
    draft = LucyCampaignIntake().apply_form_submission(state["intake_draft"], submission)
    state["intake_draft"] = draft
    graph = build_campaign_workflow_graph().compile()
    return graph.invoke(state)


if __name__ == "__main__":
    raise NotImplementedError("Wire up CLI/API input before running.")
