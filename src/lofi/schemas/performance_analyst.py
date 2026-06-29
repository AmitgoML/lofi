"""Input/output schemas for the Performance Analyst Agent."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from lofi.schemas.common import CreativeFormat, Location, Platform


class PerformanceAnalystInput(BaseModel):
    user_request: str = Field(description="Raw user campaign request, as captured by Lucy Campaign Intake")
    brand: str = Field(description="Brand the campaign is being run for")
    organization_id: str = Field(description="Organization ID the brand/campaign belongs to")
    lookback_days: int = Field(
        default=90, description="Number of days of historical data to analyze when generating recommendations"
    )


class ConfidenceLevel(str, Enum):
    """How much historical evidence backs a recommendation's numbers."""

    HIGH = "high"
    MEDIUM = "medium"
    DIRECTIONAL = "directional"


class MetricAnomaly(BaseModel):
    """A single day where a metric deviated sharply from its rolling history."""

    date: str = Field(description="ISO date (YYYY-MM-DD) the anomaly occurred on")
    metric: str = Field(description="Metric key that deviated, e.g. 'roas', 'cost'")
    value: float = Field(description="Actual value of the metric on this date")
    rolling_mean: float = Field(description="7-day rolling mean the value is compared against")
    deviation_pct: Optional[float] = Field(
        default=None, description="Percent deviation from the rolling mean, or null if the mean was zero"
    )
    direction: Literal["spike", "drop"] = Field(description="Whether the value was above or below the rolling mean")


class TrendDelta(BaseModel):
    """Week-over-week change for one metric, comparing two trailing periods."""

    metric: str = Field(description="Metric key this trend is for, e.g. 'roas', 'ctr'")
    last_period_avg: float = Field(description="Average value over the most recent period")
    prior_period_avg: float = Field(description="Average value over the period before that")
    change_pct: Optional[float] = Field(
        default=None, description="Percent change from prior to last period, or null if the prior average was zero"
    )


class PlatformRecommendation(BaseModel):
    platform: Platform = Field(description="Recommended ad platform")
    historical_roas: Optional[float] = Field(default=None, description="Historical return on ad spend for this platform")
    historical_ctr: Optional[float] = Field(default=None, description="Historical click-through rate for this platform")
    historical_cpa: Optional[float] = Field(default=None, description="Historical cost per acquisition for this platform")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.DIRECTIONAL, description="How much historical evidence backs this recommendation"
    )
    anomalies: List[MetricAnomaly] = Field(
        default_factory=list, description="Notable single-day deviations found in this platform's history"
    )
    trend: List[TrendDelta] = Field(
        default_factory=list, description="Week-over-week trend for this platform's key metrics"
    )


class AudienceRecommendation(BaseModel):
    audience_segment: str = Field(description="Name or ID of the recommended audience segment")
    historical_performance_score: Optional[float] = Field(
        default=None, description="Composite historical performance score for this segment"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.DIRECTIONAL, description="How much historical evidence backs this recommendation"
    )


class LocationRecommendation(BaseModel):
    location: Location = Field(description="Recommended target location")
    historical_performance_score: Optional[float] = Field(
        default=None, description="Composite historical performance score for this location"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.DIRECTIONAL, description="How much historical evidence backs this recommendation"
    )
    anomalies: List[MetricAnomaly] = Field(
        default_factory=list, description="Notable single-day deviations found in this location's history"
    )
    trend: List[TrendDelta] = Field(
        default_factory=list, description="Week-over-week trend for this location's key metrics"
    )


class CreativeRecommendation(BaseModel):
    creative_format: CreativeFormat = Field(description="Recommended creative format")
    historical_engagement_rate: Optional[float] = Field(
        default=None, description="Historical engagement rate observed for this creative format"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.DIRECTIONAL, description="How much historical evidence backs this recommendation"
    )


class PerformanceAnalystOutput(BaseModel):
    platform_recommendations: List[PlatformRecommendation] = Field(
        description="Ranked list of recommended platforms"
    )
    audience_recommendations: List[AudienceRecommendation] = Field(
        description="Ranked list of recommended audience segments"
    )
    location_recommendations: List[LocationRecommendation] = Field(
        description="Ranked list of recommended target locations"
    )
    creative_recommendations: List[CreativeRecommendation] = Field(
        description="Ranked list of recommended creative formats"
    )
    narrative_summary: Optional[str] = Field(
        default=None,
        description=(
            "Optional LLM-generated prose summary of the recommendations above. Purely "
            "descriptive - never a source of new numbers or recommendations, and safe to omit."
        ),
    )


class NarrativeSummary(BaseModel):
    """Structured-extraction target for the narrative-generation LLM call."""

    summary: str = Field(
        description=(
            "A 2-3 sentence narrative synthesizing the structured performance data provided. "
            "Must reference only numbers and findings already present in that data."
        )
    )
