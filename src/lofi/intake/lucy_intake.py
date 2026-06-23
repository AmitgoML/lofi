"""Lucy Campaign Intake: extracts a structured brief from the user's request.

Every field but the raw request text is optional in what the user actually
says, so extraction produces an ``IntakeDraft`` rather than a full
``CampaignPlannerInput``. When any field is still missing after extraction,
``run`` stops the workflow short with an ``IntakeFormRequest`` naming what's
missing, instead of guessing. Once the caller collects the missing fields
(via whatever UI the form is rendered in) and resubmits them as an
``IntakeDraft``, ``apply_form_submission`` merges them into the draft and
the workflow can proceed; once nothing is missing, the draft is converted
into the validated ``CampaignPlannerInput`` the rest of the pipeline expects.
"""

from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.intake import IntakeDraft, IntakeField, IntakeFormRequest
from lofi.state.workflow_state import WorkflowState

ALL_FORM_FIELDS: tuple[IntakeField, ...] = tuple(IntakeField)


class LucyCampaignIntake:
    """Extracts a CampaignPlannerInput from a raw user campaign request."""

    def run(self, state: WorkflowState) -> WorkflowState:
        draft = state.get("intake_draft") or self.extract_brief(state["user_request"])
        missing = self.find_missing_fields(draft)

        state["intake_draft"] = draft
        if missing:
            state["intake_form_request"] = IntakeFormRequest(missing_fields=missing)
            return state

        state["intake_form_request"] = None
        state["campaign_brief"] = self.finalize_brief(draft)
        return state

    def extract_brief(self, user_request: str) -> IntakeDraft:
        """Parses the raw request into an IntakeDraft.

        Leaves any field unset (None) when the user didn't mention it,
        rather than guessing.
        """
        raise NotImplementedError

    def find_missing_fields(self, draft: IntakeDraft) -> list[IntakeField]:
        missing: list[IntakeField] = []
        for field in IntakeField:
            value = getattr(draft, field.value)
            if not value:
                missing.append(field)
        return missing

    def apply_form_submission(self, draft: IntakeDraft, submission: IntakeDraft) -> IntakeDraft:
        """Merges form answers into the draft, ready for `run` to re-check."""
        updates = submission.model_dump(exclude_none=True, exclude={"user_request"})
        return draft.model_copy(update=updates)

    def finalize_brief(self, draft: IntakeDraft) -> CampaignPlannerInput:
        """Converts a fully-filled-in draft into the validated planner input."""
        return CampaignPlannerInput(**draft.model_dump())
