"""Input/output schemas for the Campaign Planner Agent.

Includes the end-to-end workflow input (from Lucy Campaign Intake) and the
top-level output (the assembled FinalCampaignProposal) handed to Review &
Approval and the Persistence Layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, Location, Platform
from lofi.schemas.creative_director import AssetRef, TextAsset
from lofi.schemas.qa_agent import QAAgentOutput


class CampaignPlannerInput(BaseModel):
    """What Lucy Campaign Intake extracts from the user request.

    locations, target_audience, and platforms are optional here because the
    user's raw request may not mention them; intake's form-completion step
    fills them in before this brief is handed to the Campaign Planner.
    """
    user_request: str = Field(description="Raw natural-language campaign request from the user")
    brand: str = Field(description="Brand the campaign is being run for")
    organization_id: str = Field(description="Organization ID the brand/campaign belongs to")
    goal: CampaignGoal = Field(description="Campaign goal as stated or inferred from the user request")
    budget: BudgetSpec = Field(description="Budget as stated or inferred from the user request")
    campaign_timing: CampaignTiming = Field(description="Campaign timing as stated or inferred from the user request")
    locations: Optional[List[Location]] = Field(
        default=None, description="Target locations as stated by the user, or filled in via the intake form"
    )
    target_audience: Optional[AudienceSpec] = Field(
        default=None, description="Target audience as stated by the user, or filled in via the intake form"
    )
    platforms: Optional[List[Platform]] = Field(
        default=None, description="Preferred platforms as stated by the user, or filled in via the intake form"
    )


class FinalCampaignProposal(BaseModel):
    """Assembled output handed to Review & Approval, then Persist Campaign."""
    organization_id: str = Field(description="Organization ID the proposal belongs to")
    brand: str = Field(description="Brand the proposal is for")
    campaign_plan: CampaignPlan = Field(description="Finalized campaign plan")
    creative_assets: List[AssetRef] = Field(description="Final set of image/video creative assets")
    copy_assets: TextAsset = Field(description="Final set of ad copy assets")
    qa_result: QAAgentOutput = Field(description="QA validation result for this proposal")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp the proposal was assembled"
    )
    requires_human_review: bool = Field(
        default=True, description="Whether this proposal must go through human Review & Approval before persisting"
    )


class CampaignPlannerOutput(BaseModel):
    """Top-level output of the whole workflow."""
    workflow_state_id: str = Field(description="ID of the Workflow State record this run is tied to")
    final_proposal: FinalCampaignProposal = Field(description="The assembled final campaign proposal for review")
    approved: Optional[bool] = Field(
        default=None, description="Set after the Review & Approval step; null while pending"
    )
    persisted_campaign_id: Optional[str] = Field(
        default=None,
        description="ID of the row written to the Supabase 'campaigns' table once persisted; null until then",
    )
