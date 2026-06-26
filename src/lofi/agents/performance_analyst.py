"""Performance Analyst Agent: mines historical metrics for recommendations."""

from collections import defaultdict
from typing import Callable, Optional, TypeVar

from lofi.persistence.supabase_client import SupabaseClient
from lofi.schemas.common import CreativeFormat, Location, Platform
from lofi.schemas.performance_analyst import (
    AudienceRecommendation,
    CreativeRecommendation,
    LocationRecommendation,
    PerformanceAnalystInput,
    PerformanceAnalystOutput,
    PlatformRecommendation,
)
from lofi.state.workflow_state import WorkflowState

TOP_N = 5

T = TypeVar("T")


def _weighted_average(rows: list[dict], value_key: str, weight_key: str) -> Optional[float]:
    """Weighted average of value_key over rows, weighted by weight_key.

    Rows with a null value or non-positive weight are excluded so a handful of
    zero-spend/zero-impression rows don't drag a real metric to None/0.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key) or 0
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _group_by(rows: list[dict], key_fn: Callable[[dict], Optional[T]]) -> dict[T, list[dict]]:
    groups: dict[T, list[dict]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is not None:
            groups[key].append(row)
    return groups


def _composite_score(roas: Optional[float], ctr: Optional[float]) -> float:
    """Ranks by historical ROAS first; falls back to CTR when ROAS isn't available.

    ROAS is the more direct signal of spend efficiency, but rows/groups with no
    recorded conversions still have a CTR worth ranking on rather than dropping.
    """
    if roas is not None:
        return roas
    return ctr if ctr is not None else float("-inf")


def _location_from_metrics_row(row: dict) -> Location:
    """Best-effort Location from a brand_locations join.

    brand_locations only stores a free-text `address`, not separate
    city/state/country fields, so country is parsed as the last comma-separated
    segment of the address; this is approximate, not authoritative geocoding.
    """
    brand_location = row.get("brand_locations") or {}
    address = brand_location.get("address") or ""
    segments = [segment.strip() for segment in address.split(",") if segment.strip()]
    return Location(
        city=brand_location.get("name"),
        country=segments[-1] if segments else "unknown",
    )


class PerformanceAnalystAgent:
    """Reads historical campaign metrics and produces a PerformanceAnalystOutput."""

    def __init__(self, supabase_client: SupabaseClient) -> None:
        self._supabase_client = supabase_client

    def run(self, state: WorkflowState) -> WorkflowState:
        brief = state["campaign_brief"]
        analyst_input = PerformanceAnalystInput(
            user_request=state["user_request"],
            brand=brief.brand,
            organization_id=brief.organization_id,
        )
        output = self.analyze(analyst_input)
        brand_row = self._supabase_client.get_brand_by_name(brief.brand, brief.organization_id)
        if brand_row:
            output = output.model_copy(update={"brand_id": brand_row["brand_id"]})
        state["performance_insights"] = output
        return state

    def analyze(self, analyst_input: PerformanceAnalystInput) -> PerformanceAnalystOutput:
        platform_rows = self._read_platform_metrics(analyst_input.organization_id, analyst_input.lookback_days)
        location_rows = self._read_location_metrics(analyst_input.organization_id, analyst_input.lookback_days)
        audience_rows = self._read_audience_metrics(analyst_input.organization_id)
        creative_rows = self._read_creative_metrics(analyst_input.organization_id)

        return PerformanceAnalystOutput(
            platform_recommendations=self._rank_platforms(platform_rows),
            location_recommendations=self._rank_locations(location_rows),
            audience_recommendations=self._rank_audiences(audience_rows),
            creative_recommendations=self._rank_creatives(creative_rows),
        )

    def _read_platform_metrics(self, organization_id: str, lookback_days: int) -> list[dict]:
        return self._supabase_client.get_platform_metrics(organization_id, lookback_days)

    def _read_location_metrics(self, organization_id: str, lookback_days: int) -> list[dict]:
        return self._supabase_client.get_location_metrics(organization_id, lookback_days)

    def _read_audience_metrics(self, organization_id: str) -> list[dict]:
        return self._supabase_client.get_audience_metrics(organization_id)

    def _read_creative_metrics(self, organization_id: str) -> list[dict]:
        return self._supabase_client.get_creative_metrics(organization_id)

    def _rank_platforms(self, rows: list[dict]) -> list[PlatformRecommendation]:
        def key_fn(row: dict) -> Optional[Platform]:
            try:
                return Platform(row.get("platform"))
            except ValueError:
                return None

        groups = _group_by(rows, key_fn)
        recommendations = []
        for platform, group_rows in groups.items():
            roas = _weighted_average(group_rows, "roas", "cost")
            ctr = _weighted_average(group_rows, "ctr", "impressions")
            cpa = _weighted_average(group_rows, "cpa", "conversions")
            recommendations.append(
                (
                    _composite_score(roas, ctr),
                    PlatformRecommendation(
                        platform=platform, historical_roas=roas, historical_ctr=ctr, historical_cpa=cpa
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]

    def _rank_locations(self, rows: list[dict]) -> list[LocationRecommendation]:
        groups = _group_by(rows, lambda row: row.get("place_id"))
        recommendations = []
        for group_rows in groups.values():
            roas = _weighted_average(group_rows, "roas", "cost")
            ctr = _weighted_average(group_rows, "ctr", "impressions")
            score = _composite_score(roas, ctr)
            recommendations.append(
                (
                    score,
                    LocationRecommendation(
                        location=_location_from_metrics_row(group_rows[0]),
                        historical_performance_score=score if score != float("-inf") else None,
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]

    def _rank_audiences(self, rows: list[dict]) -> list[AudienceRecommendation]:
        def key_fn(row: dict) -> Optional[str]:
            segment_type, segment = row.get("segment_type"), row.get("segment")
            return f"{segment_type}:{segment}" if segment_type and segment else None

        groups = _group_by(rows, key_fn)
        recommendations = []
        for segment_key, group_rows in groups.items():
            roas = _weighted_average(group_rows, "roas", "cost")
            ctr = _weighted_average(group_rows, "ctr", "impressions")
            score = _composite_score(roas, ctr)
            recommendations.append(
                (
                    score,
                    AudienceRecommendation(
                        audience_segment=segment_key,
                        historical_performance_score=score if score != float("-inf") else None,
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]

    def _rank_creatives(self, rows: list[dict]) -> list[CreativeRecommendation]:
        def key_fn(row: dict) -> Optional[CreativeFormat]:
            asset_type = (row.get("creative_assets") or {}).get("asset_type")
            try:
                return CreativeFormat(asset_type)
            except ValueError:
                return None

        groups = _group_by(rows, key_fn)
        recommendations = []
        for creative_format, group_rows in groups.items():
            engagement_rate = _weighted_average(group_rows, "ctr", "impressions")
            best_row = max(group_rows, key=lambda r: r.get("ctr") or 0.0)
            asset_id = best_row.get("asset_id")
            recommendations.append(
                (
                    engagement_rate if engagement_rate is not None else float("-inf"),
                    CreativeRecommendation(
                        creative_format=creative_format,
                        historical_engagement_rate=engagement_rate,
                        asset_id=asset_id,
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]
