"""Input/output schemas for the Performance Analyst Agent."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from lofi.schemas.common import CreativeFormat, Location, Platform


class PerformanceAnalystInput(BaseModel):
    user_request: str = Field(description="Raw user campaign request, as captured by Lucy Campaign Intake")
    brand: str = Field(description="Brand the campaign is being run for")
    organization_id: str = Field(description="Organization ID the brand/campaign belongs to")
    lookback_days: int = Field(
        default=90, description="Number of days of historical data to analyze when generating recommendations"
    )


class PlatformRecommendation(BaseModel):
    platform: Platform = Field(description="Recommended ad platform")
    historical_roas: Optional[float] = Field(default=None, description="Historical return on ad spend for this platform")
    historical_ctr: Optional[float] = Field(default=None, description="Historical click-through rate for this platform")
    historical_cpa: Optional[float] = Field(default=None, description="Historical cost per acquisition for this platform")


class AudienceRecommendation(BaseModel):
    audience_segment: str = Field(description="Name or ID of the recommended audience segment")
    historical_performance_score: Optional[float] = Field(
        default=None, description="Composite historical performance score for this segment"
    )


class LocationRecommendation(BaseModel):
    location: Location = Field(description="Recommended target location")
    historical_performance_score: Optional[float] = Field(
        default=None, description="Composite historical performance score for this location"
    )


class CreativeRecommendation(BaseModel):
    creative_format: CreativeFormat = Field(description="Recommended creative format")
    historical_engagement_rate: Optional[float] = Field(
        default=None, description="Historical engagement rate observed for this creative format"
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
