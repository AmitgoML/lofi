"""Supabase persistence: campaigns and reference metric tables.

Workflow-state persistence (pause/resume across requests) is the LangGraph
checkpointer's job now, not this client's - see api/app.py.
"""

from datetime import date, timedelta

from supabase import Client, create_client

from lofi.config.settings import Settings


class SupabaseClient:
    """Reads/writes campaigns, campaign_*_metrics, and workflow state tables."""

    def __init__(self, settings: Settings, client: Client | None = None) -> None:
        self._client = client or create_client(settings.supabase_url, settings.supabase_key)

    def get_platform_metrics(self, organization_id: str, lookback_days: int = 90) -> list[dict]:
        since = (date.today() - timedelta(days=lookback_days)).isoformat()
        response = (
            self._client.table("campaign_platform_metrics")
            .select("*")
            .eq("organization_id", organization_id)
            .gte("date", since)
            .execute()
        )
        return response.data

    def get_location_metrics(self, organization_id: str, lookback_days: int = 90) -> list[dict]:
        # No FK exists between campaign_location_metrics.place_id and
        # brand_locations (which has no place_id column at all), so this can't
        # embed brand_locations - it's place_id-only until that mapping exists.
        since = (date.today() - timedelta(days=lookback_days)).isoformat()
        response = (
            self._client.table("campaign_location_metrics")
            .select("*")
            .eq("organization_id", organization_id)
            .gte("date", since)
            .execute()
        )
        return response.data

    def get_audience_metrics(self, organization_id: str) -> list[dict]:
        # campaign_audience_metrics carries its own organization_id column, and
        # there's no FK to campaigns to embed through anyway (its only FK is
        # source_campaign_metric_id -> campaign_metrics).
        response = (
            self._client.table("campaign_audience_metrics")
            .select("*")
            .eq("organization_id", organization_id)
            .execute()
        )
        return response.data

    def get_creative_metrics(self, organization_id: str) -> list[dict]:
        # campaign_creative_metrics carries its own organization_id column, but
        # creative_id has no FK to creative_assets.asset_id for PostgREST to
        # embed through, so asset_type is joined manually below instead.
        response = (
            self._client.table("campaign_creative_metrics")
            .select("*")
            .eq("organization_id", organization_id)
            .execute()
        )
        rows = response.data
        creative_ids = list({row["creative_id"] for row in rows if row.get("creative_id")})
        if not creative_ids:
            return rows

        assets_response = (
            self._client.table("creative_assets").select("asset_id, asset_type").in_("asset_id", creative_ids).execute()
        )
        asset_types = {asset["asset_id"]: asset["asset_type"] for asset in assets_response.data}
        for row in rows:
            row["asset_type"] = asset_types.get(row.get("creative_id"))
        return rows

    def save_campaign(self, campaign_proposal: dict) -> str:
        plan = campaign_proposal["campaign_plan"]
        audience = plan["audience"]
        budget = plan["budget"]
        row = {
            "organization_id": campaign_proposal["organization_id"],
            "goal": plan["goal"],
            "campaign_type": plan["campaign_type"],
            # campaign_channel has no corresponding field in CampaignPlan; left
            # for whatever downstream process classifies channel from platforms.
            "ad_platforms": plan["platforms"],
            "locations": plan["locations"],
            "age_ranges": [audience["age_min"], audience["age_max"]],
            "gender": audience["genders"],
            "interests": [],
            "daily_budget_cents": round(budget["daily_budget"] * 100) if budget.get("daily_budget") else None,
            "total_budget_cents": round(budget["total_budget"] * 100),
        }
        response = self._client.table("campaigns").insert(row).execute()
        return response.data[0]["id"]
