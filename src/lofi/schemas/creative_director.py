"""Input/output schemas for the Creative Director Agent and its sub-agents
(Copywriter, Image Generator, Video Generator).

DB-level brand data lives in lofi.persistence.models.BrandRow — not here.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from lofi.persistence.models import BrandRow
from lofi.schemas.common import AudienceSpec, CampaignGoal, CreativeFormat, Platform
from lofi.schemas.performance_analyst import PerformanceAnalystOutput


class CreativeBrief(BaseModel):
    """Produced upstream by Campaign Planner; consumed here."""

    goal: CampaignGoal = Field(description="Campaign goal driving the creative strategy")
    audience: AudienceSpec = Field(description="Target audience the creative needs to speak to")
    platforms: List[Platform] = Field(description="Platforms the creative assets need to support")
    offer: Optional[str] = Field(default=None, description="Specific offer or promotion to feature")


class BrandGuidelines(BaseModel):
    """Condensed brand rules — kept for VideoGeneratorAgent compatibility."""

    brand: str
    tone_of_voice: Optional[str] = None
    color_palette: List[str] = Field(default_factory=list)
    logo_usage_rules: Optional[str] = None
    restricted_claims: List[str] = Field(default_factory=list)


class ExistingAssetRef(BaseModel):
    asset_id: str
    s3_url: str
    creative_format: CreativeFormat
    tags: List[str] = Field(default_factory=list)
    last_used_campaign_id: Optional[str] = None
    performance_score: Optional[float] = None


class CreativeDirectorInput(BaseModel):
    creative_brief: CreativeBrief
    existing_campaign_assets: List[ExistingAssetRef] = Field(default_factory=list)
    brand_data: BrandRow = Field(description="Full brand record from the brands table")
    performance_analysis: PerformanceAnalystOutput


class AssetDecision(BaseModel):
    creative_format: CreativeFormat
    action: str = Field(description="'reuse' an existing asset or 'generate' a new one")
    reused_asset_id: Optional[str] = None
    generation_brief: Optional[str] = None


class TextAsset(BaseModel):
    headlines: List[str]
    descriptions: List[str]
    cta: str
    hooks: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    long_headlines: List[str] = Field(default_factory=list)


class AssetRef(BaseModel):
    asset_url: str = Field(description="S3 URL of the generated creative asset")
    creative_format: CreativeFormat
    platform: Platform


class RecommendedAsset(BaseModel):
    """Slot 1: best-performing historical asset surfaced by the Performance Analyst."""

    asset_id: str
    creative_format: CreativeFormat
    historical_engagement_rate: Optional[float] = None


class ABVariant(BaseModel):
    """One side of an A/B image creative test."""

    variant_label: str = Field(description="'A' or 'B'")
    image_prompt: str = Field(description="Detailed Titan-compatible image generation prompt")
    negative_prompt: Optional[str] = Field(default=None, description="Elements to exclude from the image")
    rationale: str = Field(description="What hypothesis this variant tests and how it differs from the other")


class CreativeDirectorOutput(BaseModel):
    # Slot 1 — historical best (no generation needed, just the asset_id reference)
    recommended_assets: List[RecommendedAsset] = Field(
        description="Best-performing historical assets from PerformanceAnalyst, one per format"
    )
    # Slot 2 — brand-hero generated image per platform
    variant_a: List[AssetRef] = Field(
        description="Variant A: brand-forward (product hero, dominant brand colors, prominent logo)"
    )
    # Slot 3 — lifestyle/emotion generated image per platform
    variant_b: List[AssetRef] = Field(
        description="Variant B: lifestyle/context (aspirational scene, subtle branding, emotional hook)"
    )
    # Strategy metadata
    best_creative_format: CreativeFormat
    best_messaging_angle: str
    variant_a_rationale: str
    variant_b_rationale: str
    asset_decisions: List[AssetDecision]
    texts: TextAsset


class CreativeStrategy(BaseModel):
    """Structured output from Claude's creative strategy call — internal to the director."""

    best_creative_format: CreativeFormat
    best_messaging_angle: str
    asset_decisions: List[AssetDecision]
    variant_a: ABVariant = Field(description="Brand-hero variant for A/B test")
    variant_b: ABVariant = Field(description="Lifestyle/context variant for A/B test")
    rationale: str = Field(description="Overall strategic rationale")
