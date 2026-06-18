from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

QueryIntent = Literal[
    "campaign_performance",
    "campaign_comparison",
    "brand_analysis",
    "budget_analysis",
    "creative_analysis",
    "general_query",
]

DEFAULT_INTENT: QueryIntent = "general_query"


class ExtractedDetails(BaseModel):
    brand: Optional[str] = None
    platform: Optional[str] = None
    campaign: Optional[str] = None
    budget: Optional[float] = None
    date_range: Optional[str] = None
    additional_details: Dict[str, Any] = Field(default_factory=dict)


class QueryContext(BaseModel):
    intent: str
    extracted_details: ExtractedDetails
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)
