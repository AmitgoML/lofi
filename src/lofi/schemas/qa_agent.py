"""Input/output schemas for the QA Agent."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.common import QAStatus
from lofi.schemas.creative_director import AssetRef, TextAsset


class QAAgentInput(BaseModel):
    campaign_plan: CampaignPlan = Field(description="Finalized campaign plan to validate")
    text_assets: TextAsset = Field(description="Generated copy assets to validate for completeness/compliance")
    assets: List[AssetRef] = Field(
        default_factory=list, description="Generated assets"
    )
    org_budget_maximum: float = Field(description="Organization's maximum allowed budget, used for budget validation")


class QAIssue(BaseModel):
    check: str = Field(description="Name of the QA check that raised the issue, e.g. 'budget_validation'")
    severity: str = Field(description="Severity of the issue: 'blocker' or 'warning'")
    message: str = Field(description="Human-readable description of the issue")


class QAAgentOutput(BaseModel):
    status: QAStatus = Field(description="Overall QA result for the campaign")
    budget_validation_passed: bool = Field(description="Whether the campaign budget passed validation")
    platform_compatibility_passed: bool = Field(description="Whether assets/settings are compatible with chosen platforms")
    creative_completeness_passed: bool = Field(description="Whether all required creative assets are present")
    required_fields_passed: bool = Field(description="Whether all required campaign fields are populated")
    policy_compliance_passed: bool = Field(description="Whether copy/creative complies with platform and brand policy")
    issues: List[QAIssue] = Field(
        default_factory=list, description="List of issues found during QA, if any"
    )
    return_to_campaign_planner: bool = Field(
        default=False, description="True if status is FAIL and the workflow should loop back to the Campaign Planner"
    )
