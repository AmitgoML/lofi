"""Request/response bodies for the campaign workflow API.

Thin wrappers around the domain schemas in :mod:`lofi.schemas` - the API
layer adds workflow_id/status, the domain schemas (FinalCampaignProposal,
IntakeFormRequest, IntakeDraft) carry the actual content unchanged.
"""

from typing import Optional

from pydantic import BaseModel, Field

from lofi.schemas.campaign_planner import FinalCampaignProposal
from lofi.schemas.creative_director import CreativeDirectorOutput
from lofi.schemas.intake import Intent, IntakeFormRequest
from lofi.schemas.performance_analyst import PerformanceAnalystOutput
from lofi.state.workflow_state import WorkflowStatus


class StartCampaignRequest(BaseModel):
    user_request: str = Field(description="Raw natural-language campaign request from the user")
    organization_id: str = Field(description="Organization ID the brand/campaign belongs to")
    organization_max_budget: float = Field(description="Organization's maximum allowed budget")


class WorkflowResponse(BaseModel):
    workflow_id: str = Field(description="ID used to poll/resume this workflow")
    status: WorkflowStatus = Field(description="Current stage of the workflow")


class CampaignStatusResponse(WorkflowResponse):
    intent: Optional[Intent] = Field(
        default=None, description="What Lucy Intake classified the request as, once intake has run"
    )
    intake_form_request: Optional[IntakeFormRequest] = Field(
        default=None, description="Set when status is awaiting_intake_form"
    )
    performance_insights: Optional[PerformanceAnalystOutput] = Field(
        default=None, description="Set once the Performance Analyst has run (campaign_planning and performance_analysis intents)"
    )
    creative_director_output: Optional[CreativeDirectorOutput] = Field(
        default=None, description="Set once the Creative Director has run (campaign_planning and creative_asset intents)"
    )
    campaign_proposal: Optional[FinalCampaignProposal] = Field(
        default=None, description="Set once the workflow reaches awaiting_review or later"
    )
    error: Optional[str] = Field(default=None, description="Set when status is failed")


class ApprovalResponse(WorkflowResponse):
    persisted_campaign_id: Optional[str] = Field(
        default=None, description="ID of the persisted campaigns row, set once approved"
    )
