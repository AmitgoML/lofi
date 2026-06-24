"""Request/response bodies for the campaign workflow API.

Thin wrappers around the domain schemas in :mod:`lofi.schemas` - the API
layer adds workflow_id/status, the domain schemas (FinalCampaignProposal,
IntakeFormRequest, IntakeDraft) carry the actual content unchanged.
"""

from typing import Optional

from pydantic import BaseModel, Field

from lofi.schemas.campaign_planner import FinalCampaignProposal
from lofi.schemas.intake import IntakeFormRequest
from lofi.state.workflow_state import WorkflowStatus


class StartCampaignRequest(BaseModel):
    user_request: str = Field(description="Raw natural-language campaign request from the user")
    organization_id: str = Field(description="Organization ID the brand/campaign belongs to")
    organization_max_budget: float = Field(description="Organization's maximum allowed budget")


class WorkflowResponse(BaseModel):
    workflow_id: str = Field(description="ID used to poll/resume this workflow")
    status: WorkflowStatus = Field(description="Current stage of the workflow")


class CampaignStatusResponse(WorkflowResponse):
    intake_form_request: Optional[IntakeFormRequest] = Field(
        default=None, description="Set when status is awaiting_intake_form"
    )
    campaign_proposal: Optional[FinalCampaignProposal] = Field(
        default=None, description="Set once the workflow reaches awaiting_review or later"
    )
    error: Optional[str] = Field(default=None, description="Set when status is failed")


class ApprovalResponse(WorkflowResponse):
    persisted_campaign_id: Optional[str] = Field(
        default=None, description="ID of the persisted campaigns row, set once approved"
    )
