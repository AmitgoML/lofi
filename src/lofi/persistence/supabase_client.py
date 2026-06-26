"""Supabase persistence: campaigns and reference metric tables.

Workflow-state persistence (pause/resume across requests) is the LangGraph
checkpointer's job now, not this client's - see api/app.py.
"""

from datetime import date, timedelta

from supabase import Client, create_client

from lofi.config.settings import Settings
from lofi.persistence.models import BrandRow


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
        since = (date.today() - timedelta(days=lookback_days)).isoformat()
        response = (
            self._client.table("campaign_location_metrics")
            .select("*, brand_locations(name, address, timezone)")
            .eq("organization_id", organization_id)
            .gte("date", since)
            .execute()
        )
        return response.data

    def get_audience_metrics(self, organization_id: str) -> list[dict]:
        # campaign_audience_metrics carries no organization_id of its own; it's scoped
        # to the org via the owning campaign.
        response = (
            self._client.table("campaign_audience_metrics")
            .select("*, campaigns!inner(organization_id)")
            .eq("campaigns.organization_id", organization_id)
            .execute()
        )
        return response.data

    def get_creative_metrics(self, organization_id: str) -> list[dict]:
        # campaign_creative_metrics carries no organization_id of its own; it's scoped
        # to the org via the owning creative asset.
        response = (
            self._client.table("campaign_creative_metrics")
            .select("*, creative_assets!inner(org_id, asset_type)")
            .eq("creative_assets.org_id", organization_id)
            .execute()
        )
        return response.data

    def get_brand(self, brand_id: str) -> BrandRow:
        response = (
            self._client.table("brands")
            .select(
                "brand_id, brand_name, brand_description, brand_audiences, product_descriptions,"
                " brand_primary_color, brand_secondary_color, brand_logos, brand_reference_images,"
                " brand_imagery_style, brand_design_elements, brand_heading_font, brand_body_font,"
                " brand_logo_usage_rules, brand_dos_and_donts, brand_goal_config,"
                " brand_tone_of_voice, brand_copywriting_tone, brand_keyword_blacklist,"
                " brand_core_values, brand_messaging_pillars, brand_tagline,"
                " brand_positioning, brand_competitors"
            )
            .eq("brand_id", brand_id)
            .single()
            .execute()
        )
        return BrandRow(**{k: v for k, v in response.data.items() if v is not None or k in ("brand_id", "brand_name")})

    def get_brand_by_name(self, brand_name: str, organization_id: str) -> dict | None:
        response = (
            self._client.table("brands")
            .select("brand_id, brand_name")
            .eq("brand_name", brand_name)
            .eq("associated_organization_id", organization_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

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
