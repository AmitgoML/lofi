"""Shared enums and value types used across multiple agent schemas."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Platform(str, Enum):
    META = "meta"
    GOOGLE = "google"
    TIKTOK = "tiktok"
    SPOTIFY = "spotify"


class CampaignGoal(str, Enum):
    AWARENESS = "awareness"
    TRAFFIC = "traffic"
    CONVERSIONS = "conversions"
    ENGAGEMENT = "engagement"
    LEAD_GEN = "lead_gen"
    APP_INSTALLS = "app_installs"


class CreativeFormat(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class QAStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class Location(BaseModel):
    city: Optional[str] = Field(default=None, description="City name, if the targeting is city-level")
    state: Optional[str] = Field(default=None, description="State/province name, if applicable")
    country: str = Field(description="Country name or ISO code being targeted")
    radius_km: Optional[float] = Field(
        default=None, description="Radius strategy in km around the city/point, if applicable"
    )


class AudienceSpec(BaseModel):
    age_min: int = Field(ge=13, description="Minimum age of the target audience")
    age_max: int = Field(ge=13, description="Maximum age of the target audience")
    genders: List[str] = Field(
        default_factory=lambda: ["all"], description="Target genders, e.g. ['male', 'female', 'all']"
    )


class BudgetSpec(BaseModel):
    total_budget: float = Field(description="Total budget for the campaign")
    daily_budget: Optional[float] = Field(default=None, description="Daily spend cap, if set")
    currency: str = Field(default="USD", description="ISO currency code for the budget figures")
    platform_split: Optional[dict[str, float]] = Field(
        default=None, description="Mapping of platform name -> allocated budget amount"
    )


class CampaignTiming(BaseModel):
    start_date: date = Field(description="Campaign flight start date")
    end_date: Optional[date] = Field(default=None, description="Campaign flight end date, if fixed")
    flight_duration_days: Optional[int] = Field(
        default=None, description="Total flight length in days, if duration-based rather than fixed end date"
    )
