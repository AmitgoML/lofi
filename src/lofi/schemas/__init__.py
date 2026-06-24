"""Pydantic schemas for the Lofi Campaign Planner agents.

Field sets are derived directly from lofi_campaign_execution_flow_v2.md.
Split by agent/module for use in :mod:`lofi.agents`; re-exported here for
convenience (``from lofi.schemas import CampaignPlan``).
"""

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import (
    CampaignPlannerInput,
    CampaignPlannerOutput,
    FinalCampaignProposal,
)
from lofi.schemas.common import (
    AudienceSpec,
    BudgetSpec,
    CampaignGoal,
    CampaignTiming,
    CreativeFormat,
    Location,
    Platform,
    QAStatus,
)
from lofi.schemas.intake import IntakeDraft, IntakeField, IntakeFormRequest
from lofi.schemas.creative_director import (
    AssetDecision,
    AssetRef,
    BrandGuidelines,
    CreativeBrief,
    CreativeDirectorInput,
    CreativeDirectorOutput,
    ExistingAssetRef,
    TextAsset,
)
from lofi.schemas.performance_analyst import (
    AudienceRecommendation,
    CreativeRecommendation,
    LocationRecommendation,
    PerformanceAnalystInput,
    PerformanceAnalystOutput,
    PlatformRecommendation,
)
from lofi.schemas.qa_agent import QAAgentInput, QAAgentOutput, QAIssue

__all__ = [
    "Platform",
    "CampaignGoal",
    "CreativeFormat",
    "QAStatus",
    "Location",
    "AudienceSpec",
    "BudgetSpec",
    "CampaignTiming",
    "PerformanceAnalystInput",
    "PerformanceAnalystOutput",
    "PlatformRecommendation",
    "AudienceRecommendation",
    "LocationRecommendation",
    "CreativeRecommendation",
    "CreativeBrief",
    "BrandGuidelines",
    "ExistingAssetRef",
    "CreativeDirectorInput",
    "AssetDecision",
    "TextAsset",
    "AssetRef",
    "CreativeDirectorOutput",
    "CampaignPlan",
    "QAAgentInput",
    "QAIssue",
    "QAAgentOutput",
    "CampaignPlannerInput",
    "FinalCampaignProposal",
    "CampaignPlannerOutput",
    "IntakeDraft",
    "IntakeField",
    "IntakeFormRequest",
]
