"""Supabase table row models.

Each class here mirrors a DB table's row shape. These are used by SupabaseClient
and passed to agents — keeping DB concerns separate from agent input/output schemas.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BrandRow(BaseModel):
    """Mirrors the `brands` table row returned from Supabase."""

    brand_id: str
    brand_name: str
    brand_description: Optional[str] = None
    brand_audiences: List[Dict[str, Any]] = Field(default_factory=list)
    product_descriptions: List[Dict[str, Any]] = Field(default_factory=list)

    # Visual identity
    brand_primary_color: Optional[str] = None
    brand_secondary_color: Optional[str] = None
    brand_logos: List[str] = Field(default_factory=list)
    brand_reference_images: List[str] = Field(default_factory=list)
    brand_imagery_style: Optional[str] = None
    brand_design_elements: Optional[str] = None
    brand_heading_font: Optional[str] = None
    brand_body_font: Optional[str] = None
    brand_logo_usage_rules: Optional[str] = None

    # Brand voice + copy guidelines
    brand_tone_of_voice: Optional[str] = None
    brand_copywriting_tone: Optional[str] = None
    brand_core_values: Optional[str] = None
    brand_messaging_pillars: Optional[str] = None
    brand_tagline: Optional[str] = None
    brand_positioning: Optional[str] = None
    brand_dos_and_donts: Optional[str] = None
    brand_keyword_blacklist: List[str] = Field(default_factory=list)
    brand_competitors: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)

    # Campaign configuration
    brand_goal_config: Optional[Dict[str, Any]] = None
