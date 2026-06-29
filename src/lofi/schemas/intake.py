"""Schemas for Lucy Campaign Intake's form-completion step.

Every field but the raw ``user_request`` is optional during extraction: the
user's message may not mention all of them. ``IntakeDraft`` holds whatever's
been extracted/filled in so far; once every field is set it's converted
into a full ``CampaignPlannerInput``. The same shape doubles as the payload
posted back when the user fills in a form for whatever fields are missing.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, Location, Platform


class Intent(str, Enum):
    """What the user's request is actually asking Lucy to do.

    Drives which agents the router sends the workflow through (see
    route_from_campaign_planner in graph/workflow_graph.py) - only
    CAMPAIGN_PLANNING runs the full chain; the other two run a single agent.
    """

    CAMPAIGN_PLANNING = "campaign_planning"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CREATIVE_ASSET = "creative_asset"


class IntakeField(str, Enum):
    """Fields a user can fill in via free text or the intake form.

    organization_id is deliberately excluded: it comes from the caller's
    auth/session context (see run_campaign_workflow), never from the user's
    free-text request or the intake form.
    """

    BRAND = "brand"
    GOAL = "goal"
    BUDGET = "budget"
    CAMPAIGN_TIMING = "campaign_timing"
    LOCATIONS = "locations"
    TARGET_AUDIENCE = "target_audience"
    PLATFORMS = "platforms"


class IntakeDraft(BaseModel):
    """Partial CampaignPlannerInput: every field but user_request may still be unset."""
    user_request: str = Field(description="Raw natural-language campaign request from the user")
    intent: Intent = Field(
        default=Intent.CAMPAIGN_PLANNING, description="What the user's request is asking Lucy to do"
    )
    brand: Optional[str] = Field(default=None, description="Brand the campaign is being run for")
    organization_id: Optional[str] = Field(
        default=None, description="Organization ID the brand/campaign belongs to"
    )
    goal: Optional[CampaignGoal] = Field(default=None, description="Campaign goal as stated or inferred")
    budget: Optional[BudgetSpec] = Field(default=None, description="Budget as stated or inferred")
    campaign_timing: Optional[CampaignTiming] = Field(
        default=None, description="Campaign timing as stated or inferred"
    )
    locations: Optional[List[Location]] = Field(
        default=None, description="Target locations as stated by the user, or filled in via the form"
    )
    target_audience: Optional[AudienceSpec] = Field(
        default=None, description="Target audience as stated by the user, or filled in via the form"
    )
    platforms: Optional[List[Platform]] = Field(
        default=None, description="Preferred platforms as stated by the user, or filled in via the form"
    )


class ExtractedIntakeFields(BaseModel):
    """Subset of IntakeDraft fields the LLM is asked to extract from free text.

    Excludes user_request (already known) and organization_id (sourced from
    the caller's auth/session context, never from the user's text).
    """

    intent: Optional[Intent] = Field(
        default=None, description="What the user's request is asking Lucy to do"
    )
    brand: Optional[str] = Field(default=None, description="Brand the campaign is being run for")
    goal: Optional[CampaignGoal] = Field(default=None, description="Campaign goal as stated or inferred")
    budget: Optional[BudgetSpec] = Field(default=None, description="Budget as stated or inferred")
    campaign_timing: Optional[CampaignTiming] = Field(
        default=None, description="Campaign timing as stated or inferred"
    )
    locations: Optional[List[Location]] = Field(
        default=None, description="Target locations as stated by the user"
    )
    target_audience: Optional[AudienceSpec] = Field(
        default=None, description="Target audience as stated by the user"
    )
    platforms: Optional[List[Platform]] = Field(
        default=None, description="Preferred platforms as stated by the user"
    )


class IntakeFormRequest(BaseModel):
    """Returned when the user's request is missing fields needed to proceed."""
    missing_fields: List[IntakeField] = Field(
        description="Fields the user must fill in via a form before planning can proceed"
    )
