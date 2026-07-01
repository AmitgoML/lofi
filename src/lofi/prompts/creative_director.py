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

    logos_str = ", ".join(brand.brand_logos) if brand.brand_logos else "N/A"
    ref_images_str = ", ".join(brand.brand_reference_images) if brand.brand_reference_images else "N/A"
    goal_config_str = str(brand.brand_goal_config) if brand.brand_goal_config else "N/A"

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
- Core values: {brand.brand_core_values or "N/A"}
- Logo usage rules: {brand.brand_logo_usage_rules or "N/A"}
- Logo assets: {logos_str}
- Reference images (visual style benchmarks): {ref_images_str}
- Dos and don'ts: {brand.brand_dos_and_donts or "N/A"}
- Prohibited keywords: {brand.brand_keyword_blacklist or []}
- Brand goal config: {goal_config_str}

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

The image_prompt you write for each variant IS the direct input to an image generation model.
It must be a self-contained, richly descriptive prompt that embeds all relevant brand visual
details inline — the image model receives only this string, nothing else.

Every image_prompt must explicitly incorporate:
- Brand colors: use the exact values (primary: {brand.brand_primary_color or "brand primary"}, secondary: {brand.brand_secondary_color or "brand secondary"}) as color descriptors in the scene
- Imagery style: reflect "{brand.brand_imagery_style or "brand imagery style"}" in the visual treatment
- Design elements: reference "{brand.brand_design_elements or "brand design elements"}" in the composition
- Font style: evoke the feel of heading font "{brand.brand_heading_font or "N/A"}" and body font "{brand.brand_body_font or "N/A"}" in any typographic mood description
- Logo placement: follow the rule "{brand.brand_logo_usage_rules or "standard placement"}" and describe exactly where the logo sits in the scene
- Product/service: reference the specific product or service from the product descriptions above
- Audience: the visual scene should resonate with the target audience profiles above
- Goal alignment: the visual mood must serve the campaign goal "{brief.goal.value}" and brand goal config above
- Dos and don'ts: strictly follow "{brand.brand_dos_and_donts or "N/A"}"

**Variant A — "Brand Hero":**
Produce a high-converting, brand-forward creative.
- Product or service is the unmistakable hero of the image
- Brand's primary color ({brand.brand_primary_color or "N/A"}) dominates the composition
- Logo placed prominently per brand guidelines
- Clean, structured layout with clear visual hierarchy
- Direct, conversion-focused — designed to drive immediate action

**Variant B — "Lifestyle / Context":**
Produce an emotion-led, lifestyle creative for the same campaign.
- Product/service shown in real-world usage or aspirational context
- Scene-first composition — brand colors appear as accents, not dominant blocks
- Logo placed subtly (bottom-left, small watermark style)
- Less "ad-like" — immersive, storytelling-driven, builds brand affinity
- Designed for upper-funnel awareness and emotional recall

**Both variants must:**
- Be clearly distinct enough to test different hypotheses
- Be technically valid image generation prompts (descriptive, comma-separated visual
  attributes — no instructions, no text overlay, no explicit logo graphics in the prompt)
- Respect the brand's prohibited keywords and dos/don'ts

Provide asset decisions (reuse vs generate) for each relevant format,
the best overall creative format and messaging angle, and a strategic rationale.
"""
