"""Input/output schemas for the Creative Director Agent and its sub-agents
(Copywriter, Image Generator, Video Generator)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from lofi.schemas.common import AudienceSpec, CampaignGoal, CreativeFormat, Platform
from lofi.schemas.performance_analyst import PerformanceAnalystOutput


class CreativeBrief(BaseModel):
    """Produced upstream by Campaign Planner; consumed here."""
    goal: CampaignGoal = Field(description="Campaign goal driving the creative strategy")
    audience: AudienceSpec = Field(description="Target audience the creative needs to speak to")
    platforms: List[Platform] = Field(description="Platforms the creative assets need to support")
    offer: Optional[str] = Field(default=None, description="Specific offer or promotion to feature in the creative")


class BrandGuidelines(BaseModel):
    brand: str = Field(description="Brand these guidelines apply to")
    tone_of_voice: Optional[str] = Field(default=None, description="Description of the brand's tone of voice")
    color_palette: List[str] = Field(
        default_factory=list, description="Approved brand colors, e.g. hex codes"
    )
    logo_usage_rules: Optional[str] = Field(default=None, description="Rules governing how/where the logo may be used")
    restricted_claims: List[str] = Field(
        default_factory=list, description="Claims or phrases the brand prohibits in ad copy/creative"
    )


class ExistingAssetRef(BaseModel):
    asset_id: str = Field(description="Unique identifier of the existing creative asset")
    s3_url: str = Field(description="S3 URL where the asset is stored")
    creative_format: CreativeFormat = Field(description="Format of the existing asset")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags/metadata for the asset")
    last_used_campaign_id: Optional[str] = Field(
        default=None, description="ID of the most recent campaign this asset was used in, if any"
    )
    performance_score: Optional[float] = Field(
        default=None, description="Historical performance score for this asset, if available"
    )


class CreativeDirectorInput(BaseModel):
    creative_brief: CreativeBrief = Field(description="Creative brief produced by the Campaign Planner")
    existing_campaign_assets: List[ExistingAssetRef] = Field(
        default_factory=list, description="Assets already attached to this or related active campaigns"
    )
    brand_guidelines: BrandGuidelines = Field(description="Brand guidelines to constrain creative decisions")
    performance_analysis: PerformanceAnalystOutput


class AssetDecision(BaseModel):
    creative_format: CreativeFormat = Field(description="Creative format this decision applies to")
    action: str = Field(description="Decision outcome: 'reuse' an existing asset or 'generate' a new one")
    reused_asset_id: Optional[str] = Field(
        default=None, description="ID of the existing asset being reused, if action is 'reuse'"
    )
    generation_brief: Optional[str] = Field(
        default=None,
        description="Generation instructions passed to the Image/Video Generator agent, if action is 'generate'",
    )


class TextAsset(BaseModel):
    headlines: List[str] = Field(description="Generated ad headlines")
    descriptions: List[str] = Field(description="Generated ad body descriptions")
    cta: str = Field(description="Generated call-to-action text")
    hooks: List[str] = Field(default_factory=list, description="Generated attention-grabbing hooks/openers")
    keywords: List[str] = Field(default_factory=list, description="Generated keywords, e.g. for search platforms")
    long_headlines: List[str] = Field(default_factory=list, description="Generated long-form headline variants")


class AssetRef(BaseModel):
    asset_url: str = Field(description="S3 URL of the generated creative asset")
    creative_format: CreativeFormat = Field(description="Format of the generated asset")
    platform: Platform = Field(description="Platform this asset was generated for")


class CreativeDirectorOutput(BaseModel):
    asset_decisions: List[AssetDecision] = Field(
        description="Per-format reuse-vs-generate decisions made by the Creative Director"
    )
    best_creative_format: CreativeFormat = Field(description="Primary creative format selected for the campaign")
    best_messaging_angle: str = Field(description="Primary messaging angle selected for the campaign")
    assets: List[AssetRef]
    texts: TextAsset
