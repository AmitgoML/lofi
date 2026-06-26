"""Creative director agent prompts — versioned."""

from __future__ import annotations

from lofi.persistence.models import BrandRow
from lofi.schemas.creative_director import CreativeBrief
from lofi.schemas.performance_analyst import PerformanceAnalystOutput


def creative_strategy_v1(
    brief: CreativeBrief,
    brand: BrandRow,
    perf: PerformanceAnalystOutput,
    existing_assets: list[dict],
) -> str:
    """v1 — A/B creative strategy prompt.

    Asks Claude to produce two deliberately differentiated image variants:
    - Variant A: Brand Hero — product/brand front and center, structured composition
    - Variant B: Lifestyle/Context — aspirational scene, subtle branding
    """
    top_platforms = [r.platform.value for r in perf.platform_recommendations[:3]]
    top_formats = [r.creative_format.value for r in perf.creative_recommendations[:3]]
    top_audiences = [r.audience_segment for r in perf.audience_recommendations[:2]]

    products = "\n".join(
        f"  - {p}" for p in brand.product_descriptions[:3]
    ) if brand.product_descriptions else "  - (none provided)"

    audience_profiles = "\n".join(
        f"  - {a}" for a in brand.brand_audiences[:3]
    ) if brand.brand_audiences else "  - (none provided)"

    existing_str = str(existing_assets) if existing_assets else "None"

    return f"""You are the Creative Director at a top-tier digital advertising agency.
Your task: produce a full A/B creative strategy for the campaign below.

The strategy must include TWO deliberately different image variants (A and B) that
can be tested against each other in a real paid media A/B experiment.

## Campaign Brief
- Goal: {brief.goal.value}
- Platforms: {[p.value for p in brief.platforms]}
- Offer: {brief.offer or "No specific offer"}
- Audience: ages {brief.audience.age_min}–{brief.audience.age_max}, genders: {brief.audience.genders}

## Brand Identity
- Name: {brand.brand_name}
- Description: {brand.brand_description or "N/A"}
- Tagline: {brand.brand_tagline or "N/A"}
- Positioning: {brand.brand_positioning or "N/A"}
- Primary color: {brand.brand_primary_color or "N/A"}
- Secondary color: {brand.brand_secondary_color or "N/A"}
- Imagery style: {brand.brand_imagery_style or "N/A"}
- Design elements: {brand.brand_design_elements or "N/A"}
- Heading font: {brand.brand_heading_font or "N/A"}
- Body font: {brand.brand_body_font or "N/A"}
- Logo usage rules: {brand.brand_logo_usage_rules or "N/A"}
- Dos and don'ts: {brand.brand_dos_and_donts or "N/A"}
- Prohibited keywords: {brand.brand_keyword_blacklist or []}

## Products / Services
{products}

## Target Audience Profiles
{audience_profiles}

## Historical Performance Insights
- Top performing platforms: {top_platforms}
- Top performing creative formats: {top_formats}
- Top performing audience segments: {top_audiences}

## Existing Assets Available for Reuse
{existing_str}

---
## Strategy Output Requirements

**Variant A — "Brand Hero":**
Produce a high-converting, brand-forward creative.
- Product or service is the unmistakable hero of the image
- Brand's primary color palette dominates the composition
- Logo placed prominently per brand guidelines (typically top-left or top-right corner)
- Clean, structured layout with clear visual hierarchy
- Direct, conversion-focused — designed to drive immediate action
- Image prompt must include: specific color values, logo placement instruction, product framing,
  lighting direction, background treatment, and any brand design elements

**Variant B — "Lifestyle / Context":**
Produce an emotion-led, lifestyle creative for the same campaign.
- Product/service shown in real-world usage or aspirational context
- Scene-first composition — the brand colors appear as accents, not dominant blocks
- Logo placed subtly (bottom-left, small watermark style)
- Less "ad-like" — immersive, storytelling-driven, builds brand affinity
- Designed for upper-funnel awareness and emotional recall
- Image prompt must include: scene description, emotional mood, human element (if relevant),
  ambient lighting, how the product appears naturally in the scene

**Both variants must:**
- Be clearly distinct enough to test different hypotheses
- Be technically valid Titan Image Generator v2 prompts (descriptive, comma-separated visual
  attributes — no instructions, no text, no logos in the prompt itself)
- Respect the brand's prohibited keywords and dos/don'ts

Provide asset decisions (reuse vs generate) for each relevant format,
the best overall creative format and messaging angle, and a strategic rationale.
"""
