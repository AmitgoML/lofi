"""Lucy Campaign Intake: extracts a structured brief from the user's request.

Split into two graph nodes rather than one. LangGraph replays a node's whole
function body from the top on every resume after an interrupt() inside it,
so any side effect that ran *before* the interrupt() in the same node would
re-run on every resume too - calling Bedrock again each time someone submits
the form. Keeping extraction in `extract`, a separate node that always
commits before `collect_missing_fields` ever runs, avoids that: Bedrock is
called exactly once.
"""

from langgraph.types import interrupt

from lofi.llm.bedrock_client import BedrockClient
from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.intake import ExtractedIntakeFields, Intent, IntakeDraft, IntakeField
from lofi.state.workflow_state import WorkflowState

EXTRACTION_PROMPT_TEMPLATE = """\
Extract the campaign fields mentioned in the user's request below.

Leave a field unset if the user did not mention it - do not guess or infer
values the user did not state.

Also classify the user's intent as one of:
- campaign_planning: the user wants a new campaign planned end-to-end
- performance_analysis: the user wants insights/recommendations from past
  campaign performance, not a new campaign plan
- creative_asset: the user wants creative assets (copy, image, video)
  produced, not a new campaign plan

User request: {user_request}"""

# Fields the user must supply before the corresponding intent's flow can
# proceed. campaign_planning needs the full brief; the other two only need
# enough to identify which brand's data/assets to work with.
REQUIRED_FIELDS_BY_INTENT: dict[Intent, list[IntakeField]] = {
    Intent.CAMPAIGN_PLANNING: list(IntakeField),
    Intent.PERFORMANCE_ANALYSIS: [IntakeField.BRAND],
    Intent.CREATIVE_ASSET: [IntakeField.BRAND],
}


class LucyCampaignIntake:
    """Extracts a CampaignPlannerInput from a raw user campaign request."""

    def __init__(self, bedrock_client: BedrockClient) -> None:
        self._bedrock_client = bedrock_client

    def extract(self, state: WorkflowState) -> WorkflowState:
        """Node 1: Bedrock extraction. Guarded so it's a no-op if re-entered
        (it shouldn't be, since the orchestrator stops routing here once
        intake_draft exists, but cheap to guard against either way)."""
        if "intake_draft" in state:
            return state
        draft = self.extract_brief(state["user_request"])
        state["intake_draft"] = draft.model_copy(update={"organization_id": state["organization_id"]})
        return state

    def collect_missing_fields(self, state: WorkflowState) -> WorkflowState:
        """Node 2: pauses via interrupt() once per round of missing fields
        until the draft is complete, then finalizes the brief."""
        draft = state["intake_draft"]
        missing = self.find_missing_fields(draft)
        while missing:
            submission = interrupt({"type": "intake_form", "missing_fields": [field.value for field in missing]})
            draft = self.apply_form_submission(draft, IntakeDraft.model_validate(submission))
            missing = self.find_missing_fields(draft)

        state["intake_draft"] = draft
        # Only campaign_planning runs the full chain that needs a complete
        # CampaignPlannerInput; the other intents only need intake_draft
        # itself (see route_from_campaign_planner).
        if draft.intent == Intent.CAMPAIGN_PLANNING:
            state["campaign_brief"] = self.finalize_brief(draft)
        return state

    def extract_brief(self, user_request: str) -> IntakeDraft:
        """Calls Bedrock to parse the raw request into an IntakeDraft.

        Leaves any field unset (None) when the user didn't mention it,
        rather than guessing. Falls back to campaign_planning if the model
        doesn't classify an intent.
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(user_request=user_request)
        extracted = self._bedrock_client.extract_structured(prompt, ExtractedIntakeFields)
        fields = extracted.model_dump(exclude={"intent"})
        return IntakeDraft(user_request=user_request, intent=extracted.intent or Intent.CAMPAIGN_PLANNING, **fields)

    def find_missing_fields(self, draft: IntakeDraft) -> list[IntakeField]:
        missing: list[IntakeField] = []
        for field in REQUIRED_FIELDS_BY_INTENT[draft.intent]:
            value = getattr(draft, field.value)
            if not value:
                missing.append(field)
        return missing

    def apply_form_submission(self, draft: IntakeDraft, submission: IntakeDraft) -> IntakeDraft:
        """Merges form answers into the draft, ready for collect_missing_fields to re-check.

        Uses model_validate rather than model_copy(update=...): the latter
        assigns nested fields (budget, locations, ...) without revalidating,
        leaving them as raw dicts instead of BudgetSpec/Location instances.

        Excludes intent: it's classified once during extraction, and the
        form submission doesn't carry it, so submission.intent would always
        be its default (campaign_planning) and clobber whatever extraction
        actually classified.
        """
        updates = submission.model_dump(exclude_none=True, exclude={"user_request", "intent"})
        return IntakeDraft.model_validate({**draft.model_dump(), **updates})

    def finalize_brief(self, draft: IntakeDraft) -> CampaignPlannerInput:
        """Converts a fully-filled-in draft into the validated planner input."""
        return CampaignPlannerInput(**draft.model_dump())
