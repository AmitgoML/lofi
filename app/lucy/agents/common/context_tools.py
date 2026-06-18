"""Shared context tools for Lucy agents.

- get_brand_context: exposes full brand specs (Tier 2 deep context)
- get_creative_assets: queries creative_assets table metadata + signed URLs
- get_ad_accounts: user's connected advertising platform accounts
- get_login_history: user's recent login history
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext

from lucy.agents.common.models import ChatDeps
from lucy.database.supabase_client import get_client


def _stringify(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            return ", ".join(v)
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def register_brand_context_tool(agent: Agent) -> None:
    """Register get_brand_context on the given agent."""

    @agent.tool
    async def get_brand_context(ctx: RunContext[ChatDeps]) -> Dict[str, Any]:
        """Fetch full brand specifications: identity, tone, values, audiences,
        competitors, products, locations, messaging, keyword blacklist,
        dos/donts, visuals, and design guidelines.

        Use this tool when you need detailed brand information for tasks like
        writing copy, building creative briefs, making strategy recommendations,
        or ensuring brand compliance. You do NOT need this for simple questions
        that don't require brand-specific knowledge.

        Large list fields (brand_locations, brand_customers_lists) are summarised
        as entry counts to keep the model context size manageable.
        If you need the full locations list, it is available in brand_locations.
        """
        ctx.deps.status_queue.put_nowait("Loading brand context")
        brand = ctx.deps.brand_context
        if not brand:
            return {"available": False, "message": "No brand context loaded for this session."}

        # Fields that can contain thousands of entries — summarise with a count
        # instead of dumping the full payload into the model context.
        LARGE_JSONB_KEYS = {"brand_locations", "brand_customers_lists"}

        result: Dict[str, Any] = {"available": True}
        for key, value in brand.items():
            if key in LARGE_JSONB_KEYS and isinstance(value, (list, dict)):
                    count = len(value)
                    result[key] = f"[{count} entries — use brand_locations field for full data]"
            else:
                result[key] = _stringify(value)
        return result


def register_creative_assets_tool(agent: Agent) -> None:
    """Register get_creative_assets on the given agent."""

    @agent.tool
    async def get_creative_assets(
        ctx: RunContext[ChatDeps],
        asset_type: Optional[str] = Field(
            default=None,
            description='Filter by asset type: "image", "video", or "audio". Omit for all types.',
        ),
        limit: int = Field(
            default=20,
            description="Maximum number of assets to return (default 20, max 50).",
        ),
    ) -> Dict[str, Any]:
        """Query creative assets for the user's organization.

        Returns metadata (name, type, source, created date) and signed URLs
        for each asset. Use this for creative audits, reviewing existing assets,
        or understanding what creative material is available.
        """
        ctx.deps.status_queue.put_nowait("Fetching creative assets")
        brand = ctx.deps.brand_context
        if not brand:
            return {"available": False, "assets": [], "message": "No brand context -- cannot determine organization."}

        org_id = brand.get("associated_organization_id")
        if not org_id:
            return {"available": False, "assets": [], "message": "No organization ID in brand context."}

        from lucy.database.creative_assets_client import list_creative_assets

        clamped_limit = min(max(limit, 1), 50)

        assets = await asyncio.to_thread(
            list_creative_assets,
            org_id=org_id,
            asset_type=asset_type,
            limit=clamped_limit,
        )

        return {
            "available": True,
            "total_returned": len(assets),
            "assets": assets,
        }


def register_ad_accounts_tool(agent: Agent) -> None:
    """Register get_ad_accounts on the given agent."""

    @agent.tool
    async def get_ad_accounts(ctx: RunContext[ChatDeps]) -> Dict[str, Any]:
        """Fetch the user's connected advertising platform accounts (Google Ads, Meta, etc.).

        Returns account IDs, platforms, statuses, and whether each account is
        the primary one. Use this to understand which ad platforms the user has
        connected before giving platform-specific advice.
        """
        ctx.deps.status_queue.put_nowait("Loading ad accounts")
        try:
            sb = await asyncio.to_thread(get_client)
            res = await asyncio.to_thread(
                lambda: sb.table("ad_accounts")
                .select("account_id,platform,status,is_primary,created_at")
                .eq("user_id", ctx.deps.user_id)
                .execute()
            )
            rows = getattr(res, "data", None) or []
            return {"accounts": rows, "total": len(rows)}
        except Exception as exc:
            logger.warning(f"get_ad_accounts failed: {exc}")
            return {"accounts": [], "total": 0, "error": "Could not load ad accounts."}


def register_login_history_tool(agent: Agent) -> None:
    """Register get_login_history on the given agent."""

    @agent.tool
    async def get_login_history(
        ctx: RunContext[ChatDeps],
        limit: int = Field(
            default=10,
            description="Maximum number of login entries to return (default 10, max 50).",
        ),
    ) -> Dict[str, Any]:
        """Fetch the user's recent login history (device, location, timestamp).

        Use this when the user asks about their account activity, recent logins,
        or security-related questions about their account access.
        """
        ctx.deps.status_queue.put_nowait("Loading login history")
        try:
            sb = await asyncio.to_thread(get_client)
            clamped = min(max(limit, 1), 50)
            res = await asyncio.to_thread(
                lambda: sb.table("login_history")
                .select("login_id,user_id,last_login_at,ip_address,user_agent,location")
                .eq("user_id", ctx.deps.user_id)
                .order("last_login_at", desc=True)
                .limit(clamped)
                .execute()
            )
            rows = getattr(res, "data", None) or []
            return {"logins": rows, "total": len(rows)}
        except Exception as exc:
            logger.warning(f"get_login_history failed: {exc}")
            return {"logins": [], "total": 0, "error": "Could not load login history."}
