from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from lucy.agents.performance_analyst_agent import (
    _fetch_and_process_metrics,
    _fetch_user_campaigns,
)
from lucy.database.creative_assets_client import list_creative_assets
from lucy.database.supabase_client import get_full_brand, get_user_org_profiles
from lucy.query_understanding.models import ExtractedDetails

INTENT_REQUIRED_DATA: dict[str, list[str]] = {
    "campaign_performance": ["campaigns", "campaign_metrics"],
    "campaign_comparison": ["campaigns", "campaign_metrics"],
    "brand_analysis": ["brand"],
    "budget_analysis": ["campaigns", "campaign_metrics"],
    "creative_analysis": ["creative_assets"],
    "general_query": [],
}

_CAMPAIGN_CONTEXT_COLUMNS = (
    "campaign_id, campaign_name, campaign_status, goal, "
    "budget_type, daily_budget_cents, total_budget_cents"
)

_CAMPAIGN_CONTEXT_FIELDS = (
    "campaign_id",
    "campaign_name",
    "campaign_status",
    "goal",
    "budget_type",
    "daily_budget_cents",
    "total_budget_cents",
)

_CAMPAIGN_METRIC_INTENTS = frozenset(
    {"campaign_performance", "campaign_comparison", "budget_analysis"}
)


async def determine_required_data(
    intent: str,
    details: ExtractedDetails,
) -> list[str]:
    """Return the data keys needed to answer a query for the given intent."""
    required = list(INTENT_REQUIRED_DATA.get(intent, []))
    logger.info(
        "Determined required data for intent {} (brand={}, campaign={}): {}",
        intent,
        details.brand,
        details.campaign,
        required,
    )
    return required


def _project_campaign(row: Dict[str, Any]) -> Dict[str, Any]:
    return {field: row.get(field) for field in _CAMPAIGN_CONTEXT_FIELDS}


def _campaign_name_matches(campaign_name: str, hint: str) -> bool:
    campaign_name_lower = campaign_name.lower()
    hint_lower = hint.lower()
    if hint_lower in campaign_name_lower or campaign_name_lower in hint_lower:
        return True
    return any(
        word in campaign_name_lower for word in hint_lower.split() if len(word) > 3
    )


def _filter_campaigns_by_name(
    campaigns: List[Dict[str, Any]],
    name_hint: str,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for campaign in campaigns:
        campaign_name = campaign.get("campaign_name", "")
        if campaign_name and _campaign_name_matches(campaign_name, name_hint):
            matched.append(campaign)
    return matched


def _resolve_campaign_id(
    campaigns: List[Dict[str, Any]],
    name_hint: str,
) -> Optional[str]:
    matched = _filter_campaigns_by_name(campaigns, name_hint)
    if len(matched) == 1:
        return matched[0].get("campaign_id")
    return None


def _select_org_profile(
    profiles: List[Dict[str, Any]],
    brand_hint: Optional[str],
) -> Dict[str, Any]:
    if not brand_hint:
        return profiles[0]

    hint_lower = brand_hint.lower()
    for profile in profiles:
        for field in ("brand_name", "company_name"):
            value = profile.get(field) or ""
            value_lower = value.lower()
            if hint_lower in value_lower or value_lower in hint_lower:
                return profile
    return profiles[0]


def _record_count(value: Any) -> int:
    if isinstance(value, dict) and "error" in value:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        if "count" in value:
            try:
                return int(value["count"])
            except (TypeError, ValueError):
                pass
        data = value.get("data")
        if isinstance(data, list):
            return len(data)
        if "org_profile" in value:
            return 1
    return 0


async def _fetch_campaigns(
    user_id: str,
    details: ExtractedDetails,
) -> Any:
    try:
        rows = await _fetch_user_campaigns(user_id, _CAMPAIGN_CONTEXT_COLUMNS)
        if details.campaign:
            rows = _filter_campaigns_by_name(rows, details.campaign)
        return [_project_campaign(row) for row in rows]
    except Exception as exc:
        logger.error("Campaign retrieval failed for user {}: {}", user_id, exc)
        return {"error": str(exc)}


async def _fetch_campaign_metrics(
    user_id: str,
    details: ExtractedDetails,
    intent: str,
) -> Any:
    if intent not in _CAMPAIGN_METRIC_INTENTS:
        return {}

    try:
        campaign_id: Optional[str] = None
        if details.campaign:
            campaigns = await _fetch_user_campaigns(
                user_id, "campaign_id, campaign_name"
            )
            campaign_id = _resolve_campaign_id(campaigns, details.campaign)

        return await _fetch_and_process_metrics(
            user_id=user_id,
            campaign_id=campaign_id,
            requested_metric_key="roas",
            generate_charts=False,
        )
    except Exception as exc:
        logger.error("Campaign metrics retrieval failed for user {}: {}", user_id, exc)
        return {"error": str(exc)}


async def _fetch_brand(
    user_id: str,
    details: ExtractedDetails,
) -> Any:
    try:
        profiles = await asyncio.to_thread(get_user_org_profiles, user_id)
        if not profiles:
            return {"error": "No organization profiles found for user"}

        org_profile = _select_org_profile(profiles, details.brand)
        brand_id = org_profile.get("brand_id")
        brand_profile: Optional[Dict[str, Any]] = None
        if brand_id:
            brand_profile = await asyncio.to_thread(get_full_brand, brand_id)

        return {
            "org_profile": org_profile,
            "brand_profile": brand_profile,
        }
    except Exception as exc:
        logger.error("Brand retrieval failed for user {}: {}", user_id, exc)
        return {"error": str(exc)}


async def _fetch_creative_assets(
    user_id: str,
    details: ExtractedDetails,
) -> Any:
    try:
        profiles = await asyncio.to_thread(get_user_org_profiles, user_id)
        if not profiles:
            return {"error": "No organization profiles found for user"}

        org_profile = _select_org_profile(profiles, details.brand)
        org_id = org_profile.get("org_id")
        if not org_id:
            return {"error": "No organization ID found for user"}

        return await asyncio.to_thread(
            list_creative_assets,
            org_id=org_id,
            latest_only=True,
        )
    except Exception as exc:
        logger.error("Creative assets retrieval failed for user {}: {}", user_id, exc)
        return {"error": str(exc)}


async def _retrieve_data_key(
    key: str,
    user_id: str,
    details: ExtractedDetails,
    intent: str,
) -> Any:
    if key == "campaigns":
        return await _fetch_campaigns(user_id, details)
    if key == "campaign_metrics":
        return await _fetch_campaign_metrics(user_id, details, intent)
    if key == "brand":
        return await _fetch_brand(user_id, details)
    if key == "creative_assets":
        return await _fetch_creative_assets(user_id, details)
    logger.warning("No retrieval handler for data key: {}", key)
    return []


async def retrieve_context(
    user_id: str,
    intent: str,
    details: ExtractedDetails,
) -> Dict[str, Any]:
    """Fetch contextual data for a classified query."""
    logger.info(
        "Starting context retrieval for intent {} (user_id={}, details={})",
        intent,
        user_id,
        details.model_dump(),
    )

    required_data = await determine_required_data(intent, details)
    context: Dict[str, Any] = {"required_data": required_data}
    records_retrieved: Dict[str, int] = {}

    logger.info("Context retrieval intent={}, required_data={}", intent, required_data)

    if not required_data:
        logger.info(
            "No data retrieval required for intent {}; records_retrieved={}",
            intent,
            records_retrieved,
        )
        return context

    for key in required_data:
        try:
            result = await _retrieve_data_key(key, user_id, details, intent)
            context[key] = result
            records_retrieved[key] = _record_count(result)
            if isinstance(result, dict) and "error" in result:
                logger.error(
                    "Context retrieval returned error for {} (intent={}): {}",
                    key,
                    intent,
                    result["error"],
                )
            else:
                logger.info(
                    "Context retrieval succeeded for {} (intent={}, count={})",
                    key,
                    intent,
                    records_retrieved[key],
                )
        except Exception as exc:
            logger.exception(
                "Context retrieval failed for {} (intent={}): {}",
                key,
                intent,
                exc,
            )
            context[key] = {"error": str(exc)}
            records_retrieved[key] = 0

    logger.info(
        "Context retrieval complete for intent={}, required_data={}, records_retrieved={}",
        intent,
        required_data,
        records_retrieved,
    )
    return context
