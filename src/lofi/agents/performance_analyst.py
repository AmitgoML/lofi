"""Performance Analyst Agent: mines historical metrics for recommendations.

Signal computation (confidence labeling, anomaly detection, week-over-week
trend) lives in this file alongside the ranking it feeds, rather than in a
separate module - it's only ever called from the _rank_* methods below, so
splitting it out added an indirection without an independent caller to
justify it. It's still plain, dependency-free functions, kept above the
class so each one stays unit-testable on its own.
"""

import logging
from collections import defaultdict
from typing import Callable, Optional, TypeVar

from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.supabase_client import SupabaseClient
from lofi.schemas.common import CreativeFormat, Location, Platform
from lofi.schemas.performance_analyst import (
    AudienceRecommendation,
    ConfidenceLevel,
    CreativeRecommendation,
    LocationRecommendation,
    MetricAnomaly,
    NarrativeSummary,
    PerformanceAnalystInput,
    PerformanceAnalystOutput,
    PlatformRecommendation,
    TrendDelta,
)
from lofi.state.workflow_state import WorkflowState

logger = logging.getLogger(__name__)

TOP_N = 5

T = TypeVar("T")

# Confidence thresholds, modeled on the budget-optimization calibration the
# pasted reference implementation describes (>=30 days & >=$500 spend = high,
# etc). Spend is only factored in when the caller has a confirmed "cost"
# column for that table; otherwise confidence falls back to sample size alone.
HIGH_CONFIDENCE_MIN_SAMPLE = 30
HIGH_CONFIDENCE_MIN_SPEND = 500.0
MEDIUM_CONFIDENCE_MIN_SAMPLE = 14
MEDIUM_CONFIDENCE_MIN_SPEND = 100.0

ANOMALY_WINDOW = 7
ANOMALY_THRESHOLD_STD = 2.0
TREND_PERIOD_DAYS = 7

DEFAULT_ANOMALY_METRICS: tuple[str, ...] = ("cost", "roas", "ctr", "cpa")
DEFAULT_TREND_METRICS: tuple[str, ...] = ("roas", "ctr", "cpa", "cost")

# campaign_platform_metrics.platform stores raw ad-platform names rather than
# the normalized Platform enum - "facebook" is Meta's ad platform, so it maps
# to Platform.META; the rest already match the enum values directly.
RAW_PLATFORM_TO_PLATFORM: dict[str, Platform] = {"facebook": Platform.META}


def confidence_label(sample_size: int, total_spend: Optional[float] = None) -> ConfidenceLevel:
    """Label how much historical evidence backs a group's aggregated metrics.

    `sample_size` is a distinct-day count for tables with a confirmed date
    column, or a raw row count otherwise - either way, "more is better".
    `total_spend` should be omitted (None) for tables where a "cost" column
    isn't confirmed, rather than guessed.
    """
    if total_spend is not None:
        if sample_size >= HIGH_CONFIDENCE_MIN_SAMPLE and total_spend >= HIGH_CONFIDENCE_MIN_SPEND:
            return ConfidenceLevel.HIGH
        if sample_size >= MEDIUM_CONFIDENCE_MIN_SAMPLE or total_spend >= MEDIUM_CONFIDENCE_MIN_SPEND:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.DIRECTIONAL

    if sample_size >= HIGH_CONFIDENCE_MIN_SAMPLE:
        return ConfidenceLevel.HIGH
    if sample_size >= MEDIUM_CONFIDENCE_MIN_SAMPLE:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.DIRECTIONAL


def _sorted_dated_rows(rows: list[dict], date_key: str) -> list[dict]:
    dated = [row for row in rows if row.get(date_key)]
    return sorted(dated, key=lambda row: row[date_key])


def compute_anomalies(
    rows: list[dict],
    metric_keys: tuple[str, ...] = DEFAULT_ANOMALY_METRICS,
    date_key: str = "date",
    window: int = ANOMALY_WINDOW,
    threshold_std: float = ANOMALY_THRESHOLD_STD,
    top_n: int = 10,
) -> list[MetricAnomaly]:
    """Flag days where a metric deviates more than `threshold_std` rolling
    standard deviations from its trailing `window`-day mean.

    Returns [] if there isn't a confirmed date on the rows, or not enough
    history to fill at least one rolling window.
    """
    sorted_rows = _sorted_dated_rows(rows, date_key)
    if len(sorted_rows) <= window:
        return []

    anomalies: list[MetricAnomaly] = []
    for metric in metric_keys:
        values = [float(row.get(metric, 0) or 0) for row in sorted_rows]
        for i in range(window, len(values)):
            history = values[i - window : i]
            mean = sum(history) / window
            variance = sum((v - mean) ** 2 for v in history) / window
            std = variance**0.5
            current = values[i]
            if std > 0 and abs(current - mean) > threshold_std * std:
                deviation_pct = round((current - mean) / mean * 100, 1) if mean != 0 else None
                anomalies.append(
                    MetricAnomaly(
                        date=sorted_rows[i][date_key],
                        metric=metric,
                        value=round(current, 4),
                        rolling_mean=round(mean, 4),
                        deviation_pct=deviation_pct,
                        direction="spike" if current > mean else "drop",
                    )
                )

    anomalies.sort(key=lambda anomaly: abs(anomaly.deviation_pct or 0), reverse=True)
    return anomalies[:top_n]


def compute_trend(
    rows: list[dict],
    metric_keys: tuple[str, ...] = DEFAULT_TREND_METRICS,
    date_key: str = "date",
    period_days: int = TREND_PERIOD_DAYS,
) -> list[TrendDelta]:
    """Compare the most recent `period_days` of history to the period before it.

    Returns [] if there isn't a confirmed date on the rows, or fewer than
    2 * period_days of dated history to compare.
    """
    sorted_rows = _sorted_dated_rows(rows, date_key)
    if len(sorted_rows) < period_days * 2:
        return []

    last_period = sorted_rows[-period_days:]
    prior_period = sorted_rows[-period_days * 2 : -period_days]

    def _avg(period_rows: list[dict], metric: str) -> float:
        values = [float(row.get(metric, 0) or 0) for row in period_rows]
        return sum(values) / len(values) if values else 0.0

    deltas: list[TrendDelta] = []
    for metric in metric_keys:
        last_avg = _avg(last_period, metric)
        prior_avg = _avg(prior_period, metric)
        change_pct = round((last_avg - prior_avg) / prior_avg * 100, 1) if prior_avg else None
        deltas.append(
            TrendDelta(
                metric=metric,
                last_period_avg=round(last_avg, 4),
                prior_period_avg=round(prior_avg, 4),
                change_pct=change_pct,
            )
        )
    return deltas


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


def _distinct_date_count(rows: list[dict]) -> int:
    """Number of distinct dates among rows that have one.

    0 when no row in the group carries a date - the caller should fall back
    to len(rows) in that case rather than treat 0 as a real sample size.
    """
    return len({row["date"] for row in rows if row.get("date")})


def _total_spend(rows: list[dict]) -> float:
    return sum(float(row.get("cost") or 0) for row in rows)


def _location_from_metrics_row(row: dict) -> Location:
    """Location from a campaign_location_metrics row alone.

    There's no FK from campaign_location_metrics.place_id to brand_locations
    (which has no place_id column), so no brand_locations join is available
    here - city/country aren't resolvable, just the raw place_id.
    """
    return Location(city=row.get("place_id"), country="unknown")


class PerformanceAnalystAgent:
    """Reads historical campaign metrics and produces a PerformanceAnalystOutput."""

    def __init__(self, supabase_client: SupabaseClient, bedrock_client: Optional[BedrockClient] = None) -> None:
        self._supabase_client = supabase_client
        # Optional: only the final narrative_summary depends on this. Every
        # ranking/signal computed above it stays deterministic regardless of
        # whether this is wired up.
        self._bedrock_client = bedrock_client

    def run(self, state: WorkflowState) -> WorkflowState:
        # Reads brand from intake_draft rather than campaign_brief: the
        # latter only exists for the campaign_planning intent, and this
        # agent also runs standalone for the performance_analysis intent
        # (see route_from_campaign_planner), which never builds one.
        draft = state["intake_draft"]
        analyst_input = PerformanceAnalystInput(
            user_request=state["user_request"],
            brand=draft.brand,
            organization_id=state["organization_id"],
        )
        state["performance_insights"] = self.analyze(analyst_input)
        return state

    def analyze(self, analyst_input: PerformanceAnalystInput) -> PerformanceAnalystOutput:
        platform_rows = self._read_platform_metrics(analyst_input.organization_id, analyst_input.lookback_days)
        location_rows = self._read_location_metrics(analyst_input.organization_id, analyst_input.lookback_days)
        audience_rows = self._read_audience_metrics(analyst_input.organization_id)
        creative_rows = self._read_creative_metrics(analyst_input.organization_id)

        output = PerformanceAnalystOutput(
            platform_recommendations=self._rank_platforms(platform_rows),
            location_recommendations=self._rank_locations(location_rows),
            audience_recommendations=self._rank_audiences(audience_rows),
            creative_recommendations=self._rank_creatives(creative_rows),
        )
        output.narrative_summary = self._generate_narrative_summary(output, analyst_input.brand)
        return output

    def _generate_narrative_summary(self, output: PerformanceAnalystOutput, brand: str) -> Optional[str]:
        """Narrate the already-computed output. Purely cosmetic: every number
        and recommendation above this point is final before this is called,
        and this call is never allowed to add to or change them - it can only
        fail closed (return None) without affecting the rest of the output.
        """
        if self._bedrock_client is None:
            return None

        prompt = (
            f"You are summarizing historical ad performance data for the brand '{brand}' "
            "ahead of planning a new campaign. Using ONLY the structured data below, write "
            "a 2-3 sentence narrative highlighting the most notable trend, anomaly, or "
            "confidence caveat. Do not invent numbers, and do not add recommendations beyond "
            "what is already represented in the data.\n\n"
            f"{output.model_dump_json(exclude={'narrative_summary'})}"
        )
        try:
            return self._bedrock_client.extract_structured(prompt, NarrativeSummary).summary
        except Exception as exc:
            logger.warning(f"Narrative summary generation failed (non-fatal): {exc}")
            return None

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
            raw = row.get("platform")
            if raw in RAW_PLATFORM_TO_PLATFORM:
                return RAW_PLATFORM_TO_PLATFORM[raw]
            try:
                return Platform(raw)
            except ValueError:
                return None

        groups = _group_by(rows, key_fn)
        recommendations = []
        # campaign_platform_metrics has a confirmed "date" column (it's
        # queried with .gte("date", ...) in SupabaseClient), so unlike
        # audience/creative below, this group can support anomaly detection
        # and week-over-week trend, not just a single aggregated score.
        for platform, group_rows in groups.items():
            roas = _weighted_average(group_rows, "roas", "cost")
            ctr = _weighted_average(group_rows, "ctr", "impressions")
            cpa = _weighted_average(group_rows, "cpa", "conversions")
            sample_size = _distinct_date_count(group_rows) or len(group_rows)
            recommendations.append(
                (
                    _composite_score(roas, ctr),
                    PlatformRecommendation(
                        platform=platform,
                        historical_roas=roas,
                        historical_ctr=ctr,
                        historical_cpa=cpa,
                        confidence=confidence_label(sample_size, _total_spend(group_rows)),
                        anomalies=compute_anomalies(group_rows),
                        trend=compute_trend(group_rows),
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]

    def _rank_locations(self, rows: list[dict]) -> list[LocationRecommendation]:
        groups = _group_by(rows, lambda row: row.get("place_id"))
        recommendations = []
        # Same reasoning as _rank_platforms: campaign_location_metrics also
        # has a confirmed "date" column, so anomaly/trend apply here too.
        for group_rows in groups.values():
            roas = _weighted_average(group_rows, "roas", "cost")
            ctr = _weighted_average(group_rows, "ctr", "impressions")
            score = _composite_score(roas, ctr)
            sample_size = _distinct_date_count(group_rows) or len(group_rows)
            recommendations.append(
                (
                    score,
                    LocationRecommendation(
                        location=_location_from_metrics_row(group_rows[0]),
                        historical_performance_score=score if score != float("-inf") else None,
                        confidence=confidence_label(sample_size, _total_spend(group_rows)),
                        anomalies=compute_anomalies(group_rows),
                        trend=compute_trend(group_rows),
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
            # campaign_audience_metrics has no confirmed date column, so
            # confidence here uses a row count rather than a true day count.
            recommendations.append(
                (
                    score,
                    AudienceRecommendation(
                        audience_segment=segment_key,
                        historical_performance_score=score if score != float("-inf") else None,
                        confidence=confidence_label(len(group_rows), _total_spend(group_rows)),
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]

    def _rank_creatives(self, rows: list[dict]) -> list[CreativeRecommendation]:
        def key_fn(row: dict) -> Optional[CreativeFormat]:
            asset_type = row.get("asset_type")
            try:
                return CreativeFormat(asset_type)
            except ValueError:
                return None

        groups = _group_by(rows, key_fn)
        recommendations = []
        for creative_format, group_rows in groups.items():
            engagement_rate = _weighted_average(group_rows, "ctr", "impressions")
            # campaign_creative_metrics has no confirmed date or cost column,
            # so confidence here is row count alone - no spend factor.
            recommendations.append(
                (
                    engagement_rate if engagement_rate is not None else float("-inf"),
                    CreativeRecommendation(
                        creative_format=creative_format,
                        historical_engagement_rate=engagement_rate,
                        confidence=confidence_label(len(group_rows)),
                    ),
                )
            )
        recommendations.sort(key=lambda pair: pair[0], reverse=True)
        return [recommendation for _, recommendation in recommendations[:TOP_N]]
