"""Vision analysis tools for the Creative Director's qa_and_coverage_audit specialist.

- analyze_creative_image: analyze a single image via the vision model
- audit_creative_library: batch-analyze an org's creative library with cost-controlled sampling
"""

from __future__ import annotations

import asyncio
import base64
import math
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from pydantic_ai import Agent, RunContext

from lucy.agents.common.model_config import Models
from lucy.agents.common.models import ChatDeps

# ---------------------------------------------------------------------------
# Cost / rate constants
# ---------------------------------------------------------------------------

# gpt-4o-mini vision: ~$0.000213 per 1K input tokens at 2025 pricing.
# A 512px image tile uses ~170 tokens; worst-case high-res = 2041 tokens.
# We budget $0.03 per image (generous upper bound including output tokens).
_COST_PER_IMAGE_USD = 0.03
_COST_PER_VIDEO_USD = 0.10  # video analysis via representative frames is costlier
_AUDIT_BUDGET_USD = 2.00
_MAX_IMAGES_IN_BUDGET = int(_AUDIT_BUDGET_USD / _COST_PER_IMAGE_USD)   # ~66
_MAX_VIDEOS_IN_BUDGET = int(_AUDIT_BUDGET_USD / _COST_PER_VIDEO_USD)   # ~20
_BATCH_CONCURRENCY = 5  # parallel vision calls per batch

# Maximum image dimension before we resize the data URL to keep token usage low.
_MAX_IMAGE_BYTES = 512 * 1024  # 512 KB; larger images are low-detail mode only

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _download_image_b64(url: str, timeout: float = 15.0) -> Optional[str]:
    """Download an image from ``url`` and return a base64-encoded data URL.

    Returns ``None`` on failure so callers can skip bad URLs gracefully.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            data = base64.b64encode(resp.content).decode("utf-8")
            return f"data:{content_type};base64,{data}"
    except Exception as exc:
        logger.warning(f"vision_tools: failed to download {url!r}: {exc}")
        return None


def _build_brand_summary(brand: Optional[Dict[str, Any]]) -> str:
    if not brand:
        return "No brand context available."
    parts = []
    for key, label in [
        ("brand_name", "Brand"),
        ("brand_tone_of_voice", "Tone of voice"),
        ("brand_core_values", "Core values"),
        ("brand_messaging_pillars", "Messaging pillars"),
        ("brand_imagery_style", "Imagery style"),
        ("brand_primary_color", "Primary color"),
        ("brand_dos_and_donts", "Dos and don'ts"),
        ("brand_keyword_blacklist", "Keyword blacklist"),
    ]:
        v = brand.get(key)
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(parts) if parts else "No brand details available."


async def _analyze_single_image(
    *,
    image_url: str,
    file_name: str,
    asset_type: str,
    brand_summary: str,
    platform_hint: Optional[str],
    analysis_focus: Optional[str],
) -> Dict[str, Any]:
    """Call the vision model on one image and return a structured result dict."""
    from openai import AsyncOpenAI

    model_name = Models.CREATIVE_VISION
    # Strip "openai:" prefix that pydantic-ai uses; the OpenAI SDK wants just the name.
    bare_model = model_name.split(":", 1)[-1] if ":" in model_name else model_name

    data_url = await _download_image_b64(image_url)
    if not data_url:
        return {
            "file_name": file_name,
            "error": "Could not download image",
            "skipped": True,
        }

    focus_line = f"\nAnalysis focus: {analysis_focus}" if analysis_focus else ""
    platform_line = f"\nTarget platform: {platform_hint}" if platform_hint else ""

    system = (
        "You are a senior creative director reviewing advertising assets. "
        "Evaluate the image objectively and concisely. "
        "Respond in valid JSON only — no markdown fences, no extra text."
    )
    user_text = (
        f"Review this creative asset.\n"
        f"File: {file_name} (type: {asset_type}){platform_line}{focus_line}\n\n"
        f"Brand context:\n{brand_summary}\n\n"
        "Return a JSON object with these keys:\n"
        "  brand_alignment: 'strong' | 'moderate' | 'weak' | 'off-brand'\n"
        "  platform_fit: 'excellent' | 'good' | 'poor' | 'unknown'\n"
        "  strengths: [list of up to 3 short strings]\n"
        "  weaknesses: [list of up to 3 short strings]\n"
        "  recommendation: one-sentence actionable recommendation\n"
        "  overall_score: integer 1-10\n"
    )

    client = AsyncOpenAI()
    try:
        response = await client.chat.completions.create(
            model=bare_model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        import json
        result = json.loads(raw)
        result["file_name"] = file_name
        result["asset_type"] = asset_type
        return result
    except Exception as exc:
        logger.warning(f"vision_tools: analysis failed for {file_name!r}: {exc}")
        return {
            "file_name": file_name,
            "asset_type": asset_type,
            "error": str(exc),
            "skipped": True,
        }


def _sample_assets(
    assets: List[Dict[str, Any]],
    max_images: int,
    max_videos: int,
) -> tuple[List[Dict[str, Any]], int, int]:
    """Apply budget-driven sampling to the asset list.

    Strategy: keep recency (assets are already ordered newest-first) while
    ensuring type diversity — mix images and videos proportionally up to caps.

    Returns (sampled_list, total_images_in_library, total_videos_in_library).
    """
    images = [a for a in assets if a.get("asset_type") == "image"]
    videos = [a for a in assets if a.get("asset_type") == "video"]
    others = [a for a in assets if a.get("asset_type") not in ("image", "video")]

    sampled = images[:max_images] + videos[:max_videos] + others[:5]
    return sampled, len(images), len(videos)


# ---------------------------------------------------------------------------
# Public tool registration functions
# ---------------------------------------------------------------------------

def register_analyze_creative_image_tool(agent: Agent) -> None:
    """Register ``analyze_creative_image`` on the given agent."""

    @agent.tool
    async def analyze_creative_image(
        ctx: RunContext[ChatDeps],
        image_url: str,
        file_name: str = "unknown",
        asset_type: str = "image",
        platform_hint: Optional[str] = None,
        analysis_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a single creative image using a vision model.

        Downloads the image from ``image_url`` (use the signed_url from
        get_creative_assets), evaluates it against the brand context loaded
        in this session, and returns a structured assessment.

        Args:
            image_url: Signed URL of the image to analyze.
            file_name: Human-readable asset name (for attribution in the report).
            asset_type: 'image' or 'video' (video is treated as a static thumbnail).
            platform_hint: Optional target platform (e.g. 'Facebook', 'Google Display').
            analysis_focus: Optional free-text focus for the review (e.g. 'CTA clarity').

        Returns a dict with keys: brand_alignment, platform_fit, strengths,
        weaknesses, recommendation, overall_score.
        """
        ctx.deps.status_queue.put_nowait("Analyzing image")
        brand = ctx.deps.brand_context
        brand_summary = _build_brand_summary(brand)

        result = await _analyze_single_image(
            image_url=image_url,
            file_name=file_name,
            asset_type=asset_type,
            brand_summary=brand_summary,
            platform_hint=platform_hint,
            analysis_focus=analysis_focus,
        )
        return result


def register_audit_creative_library_tool(agent: Agent) -> None:
    """Register ``audit_creative_library`` on the given agent."""

    @agent.tool
    async def audit_creative_library(
        ctx: RunContext[ChatDeps],
        asset_type_filter: Optional[str] = None,
        analysis_focus: Optional[str] = None,
        platform_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Audit the full creative library for brand alignment, quality, and coverage gaps.

        Fetches all creative assets for the organization, applies cost-controlled
        sampling (budget cap: $2.00 ≈ 66 images or 20 videos), runs parallel vision
        analysis, and returns an aggregated report.

        The tool is transparent about sampling: if not all assets can be analyzed
        within the budget, it will say so in the report.

        Args:
            asset_type_filter: Optional — 'image', 'video', or omit for all types.
            analysis_focus: Optional — a specific dimension to focus on
                (e.g. 'CTA clarity', 'brand tone', 'platform fit').
            platform_hint: Optional — target platform context
                (e.g. 'Facebook Ads', 'Google Display').

        Returns a structured audit report with per-asset results and summary insights.
        """
        ctx.deps.status_queue.put_nowait("Auditing creative library")
        brand = ctx.deps.brand_context
        if not brand:
            return {
                "available": False,
                "error": "No brand context loaded. Cannot determine organization for asset lookup.",
            }

        org_id = brand.get("associated_organization_id")
        if not org_id:
            return {
                "available": False,
                "error": "No organization ID in brand context.",
            }

        brand_summary = _build_brand_summary(brand)

        # Step 1: fetch full asset list (up to 200 to understand library size)
        from lucy.database.creative_assets_client import list_creative_assets

        all_assets = await asyncio.to_thread(
            list_creative_assets,
            org_id=org_id,
            asset_type=asset_type_filter,
            limit=200,
        )

        if not all_assets:
            return {
                "available": True,
                "total_assets": 0,
                "analyzed": 0,
                "sampled": False,
                "message": "No creative assets found in the library.",
                "results": [],
                "summary": {},
            }

        # Step 2: apply budget-driven sampling
        sampled, total_images, total_videos = _sample_assets(
            all_assets,
            max_images=_MAX_IMAGES_IN_BUDGET,
            max_videos=_MAX_VIDEOS_IN_BUDGET,
        )

        total_library = len(all_assets)
        was_sampled = len(sampled) < total_library
        sampling_note = ""
        if was_sampled:
            sampling_note = (
                f"Analyzed {len(sampled)} of {total_library} assets "
                f"(sampled by recency and type diversity to stay within the $2.00 cost budget). "
                f"Library contains {total_images} images and {total_videos} videos."
            )

        # Step 3: run vision analysis in batches
        analyzable = [a for a in sampled if a.get("signed_url") and a.get("asset_type") in ("image",)]
        skipped_no_url = [a for a in sampled if not a.get("signed_url")]
        non_image = [a for a in sampled if a.get("asset_type") not in ("image",) and a.get("signed_url")]

        results: List[Dict[str, Any]] = []

        # Process images in batches
        for i in range(0, len(analyzable), _BATCH_CONCURRENCY):
            batch = analyzable[i : i + _BATCH_CONCURRENCY]
            batch_tasks = [
                _analyze_single_image(
                    image_url=asset["signed_url"],
                    file_name=asset.get("file_name", f"asset_{j}"),
                    asset_type=asset.get("asset_type", "image"),
                    brand_summary=brand_summary,
                    platform_hint=platform_hint,
                    analysis_focus=analysis_focus,
                )
                for j, asset in enumerate(batch, start=i)
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for r in batch_results:
                if isinstance(r, Exception):
                    results.append({"error": str(r), "skipped": True})
                else:
                    results.append(r)

        # Note non-image assets separately (video thumbnails not yet supported)
        for asset in non_image:
            results.append({
                "file_name": asset.get("file_name", "unknown"),
                "asset_type": asset.get("asset_type", "video"),
                "skipped": True,
                "skip_reason": "Video vision analysis not yet supported; manual review recommended.",
            })

        for asset in skipped_no_url:
            results.append({
                "file_name": asset.get("file_name", "unknown"),
                "asset_type": asset.get("asset_type", "unknown"),
                "skipped": True,
                "skip_reason": "No signed URL available.",
            })

        # Step 4: aggregate summary
        scored = [r for r in results if not r.get("skipped") and "overall_score" in r]
        if scored:
            avg_score = round(sum(r["overall_score"] for r in scored) / len(scored), 1)
            brand_dist: Dict[str, int] = {}
            for r in scored:
                ba = r.get("brand_alignment", "unknown")
                brand_dist[ba] = brand_dist.get(ba, 0) + 1
            weak_assets = sorted(
                [r for r in scored if r.get("overall_score", 10) <= 5],
                key=lambda x: x.get("overall_score", 10),
            )[:5]
            top_assets = sorted(
                [r for r in scored if r.get("overall_score", 0) >= 8],
                key=lambda x: -x.get("overall_score", 0),
            )[:5]
        else:
            avg_score = None
            brand_dist = {}
            weak_assets = []
            top_assets = []

        summary: Dict[str, Any] = {
            "average_score": avg_score,
            "brand_alignment_distribution": brand_dist,
            "top_performers": [{"file_name": a["file_name"], "score": a["overall_score"]} for a in top_assets],
            "weakest_assets": [{"file_name": a["file_name"], "score": a["overall_score"], "recommendation": a.get("recommendation")} for a in weak_assets],
            "total_library_size": total_library,
            "total_analyzed": len(scored),
            "total_skipped": len(results) - len(scored),
        }
        if sampling_note:
            summary["sampling_note"] = sampling_note

        return {
            "available": True,
            "sampled": was_sampled,
            "sampling_note": sampling_note,
            "results": results,
            "summary": summary,
        }
