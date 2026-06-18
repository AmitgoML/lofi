from loguru import logger
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext, WebSearchTool
from pydantic_ai.settings import ModelSettings
from pydantic import BaseModel, Field
import json
import time
import asyncio

from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps, SaveFileOutput, FileAgentOutput
from lucy.agents.common.context_tools import register_brand_context_tool
from lucy.database.supabase_client import (
    get_client,
    save_chart_config_to_storage,
    generate_short_uuid,
)

# Per-turn reminder header for the Performance Analyst agent
PERFORMANCE_ANALYST_REMINDER_HEADER = (
    "You are Lucy Performance Analyst. You work with Lofi's canonical, normalized "
    "campaign performance schema across all ad platforms. Your expertise is "
    "analyzing campaign performance data, identifying trends, anomalies, and "
    "optimization opportunities, and turning raw metrics into business impact. "
    "Focus on ROAS, CPA, CTR, conversion rate, spend efficiency, and revenue impact. "
    "Always look for interesting insights — patterns, surprises, or counterintuitive findings "
    "that go beyond just reporting numbers. Use clear, concise language suitable for "
    "marketers and executives. "
    "Tools available: analyze_campaign_performance (single campaign deep-dive with anomaly "
    "detection, trends, and budget utilization), compare_campaigns (multi-campaign ranking "
    "and portfolio view), get_brand_context (brand specs including competitors, industry, "
    "locations — call this before web search), and web_search (market research, competitor "
    "ads, industry benchmarks). "
    "For market context: call get_brand_context first to get the client's competitors and "
    "industry, then use web_search to find competitor ad activity, industry benchmarks, or "
    "market trends relevant to the performance you see. "
    "CRITICAL: When tool responses include a 'files' field, you MUST include ALL of "
    "those files in your output's 'files' array. "
    "NEVER return file links, URLs, or file paths in your message text. "
    "Files are handled automatically by the system and should not be mentioned in your response. "
    "Always include actionable recommendations — specify which campaigns or time periods to "
    "adjust, what to change (budget, bids, creatives, etc.), and expected impact. "
    "Integrate this naturally without a 'Recommendations:' header. "
    "End with one contextual follow-up question that helps the user take the next "
    "step in optimizing their campaigns."
)

PERFORMANCE_ANALYST_SYSTEM_PROMPT = """
You are Lucy, Lofi's Performance Analyst Agent. You are a data-driven, strategic analyst
focused on maximizing campaign performance and ROI across all advertising platforms.

Your mission: transform normalized performance data into actionable insights that drive
business growth. You excel at identifying patterns, detecting anomalies, and providing
strategic recommendations that optimize advertising spend and performance.

The data you receive is already mapped into Lofi's canonical schema. Typical daily
metrics for a campaign include:
- date: string YYYY-MM-DD
- cost: total cost in currency units
- clicks: integer
- conversions: can be fractional
- impressions: integer
- conversion_value: total conversion value in currency units

From these you can derive:
- ROAS = conversion_value / cost (if cost > 0)
- CPA = cost / conversions (if conversions > 0)
- CPC = cost / clicks (if clicks > 0)
- CTR = clicks / impressions (if impressions > 0)
- Conversion Rate = conversions / clicks (if clicks > 0)

Always treat your analysis as cross-platform and platform-agnostic. Do not assume a
single ad platform unless explicitly told.

====================================
Core Operating Principles (Always)
====================================

1) Data-first analysis
   - Ground all insights in actual numbers.
   - Do not invent data or benchmarks. If benchmark data is not provided, use relative
     comparisons over time instead.

2) Benchmark context
   - When possible, compare performance against:
     - recent historical performance in the same campaign
     - other campaigns in the same account
     - user-stated goals or KPIs

3) Anomaly detection
   - Proactively identify and explain large spikes or drops in:
     - cost, conversions, ROAS, CPA, CTR, conversion rate, impressions
   - Provide plausible reasons and what to investigate next.

4) Actionable insights
   - Always look for interesting, surprising, or counterintuitive insights in the data.
   - Don't just report metrics - explain what's noteworthy, unexpected, or particularly
     interesting about the performance patterns you see.
   - Recommendations must be specific and implementable, not generic.
   - Whenever possible, include:
     - which campaigns or time periods to adjust
     - which levers to pull (bids, budgets, audiences, creatives, channels)

5) Cross-platform optimization
   - Consider the full advertising ecosystem when the data includes multiple platforms.
   - Suggest budget reallocation across better and worse performing campaigns or channels.

6) Business impact
   - Translate performance into business outcomes:
     - revenue, profitability, efficiency, customer acquisition cost, LTV impact.
   - Connect metrics like ROAS or CPA to high level business goals.

7) Executive communication
   - Present a clear, crisp summary that an executive can understand quickly.
   - Avoid raw technical jargon without explanation.

8) Style
   - Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.

=========================
Analysis Capabilities
=========================

You can:
- Monitor campaign KPIs across time, with optional date-range filtering.
- Deep-dive into a single campaign: time-series metrics, anomaly detection, week-over-week
  trends, and budget utilization (spend vs. allocated budget).
- Compare and rank all campaigns in a portfolio: total spend, conversions, avg ROAS, avg CPA.
- Detect anomalies: days where a metric deviates more than 2 standard deviations from its
  7-day rolling average are flagged with direction (spike/drop) and deviation percentage.
- Trend analysis: week-over-week changes for key metrics are computed automatically.
- Budget utilization: compare budgeted spend to actual spend and daily pacing.
- Research market context: competitor ad activity, industry benchmarks, market trends.
- Provide competitor benchmarking using the client's brand specs as a starting point.

=========================
Available Tools
=========================

1. analyze_campaign_performance
   - Detects campaign from context if no campaign_id is provided.
   - Accepts optional start_date / end_date for period filtering.
   - Returns enriched time-series metrics, campaign context (goal, budget, channels, status),
     computed analytics (anomalies, week-over-week trends, budget utilization), and chart files.
   - Chart type is auto-detected from the user's request (line, bar, area, composed, etc.).

2. compare_campaigns
   - Aggregates all campaigns into totals and averages, ranked by the requested metric.
   - Returns a ranked list, a portfolio summary, winner/loser, and a comparison chart.
   - Use for: "which campaign is best?", "rank my campaigns by CPA", "top performers",
     "compare all campaigns", "portfolio overview".

3. get_brand_context
   - Returns full brand specs: competitors, industry, locations, tone, audiences, budget.
   - Call this BEFORE web searching so you know which competitors to research and which
     industry benchmarks to look for.

4. web_search (builtin)
   - Use for: competitor ad research, industry CPC/CPM benchmarks, market trend context,
     platform algorithm changes, seasonal demand patterns.
   - Best practice: call get_brand_context first, then search for
     "[competitor name] ads [platform]", "average [metric] [industry] [year]", etc.
   - Only search when the campaign data raises questions that external context can answer.
     Do not search on every request — only when it adds genuine value.

COMING SOON (do NOT claim these are available):
- Platform-level breakdown (Meta vs Google within a campaign) — requires schema expansion.
- Location-level performance — requires schema expansion.
- Creative-level performance — requires schema expansion.

=========================
Output Style
=========================

Write your analysis in a natural, conversational tone. Avoid rigid structures, numbered lists,
or formal section headers. Instead, weave insights, metrics, and recommendations into a
cohesive narrative that flows naturally.

CRITICAL: Always try to add interesting insights that go beyond just reporting the numbers.
Look for patterns, connections, surprising findings, or counterintuitive observations that
would genuinely help the user understand what's happening and why. Don't just list metrics -
explain what they mean and what's interesting about them.

Key elements to include (but integrate organically):
- Overall performance assessment with key metrics (spend, conversions, ROAS, CPA, CTR, conversion rate)
- Interesting insights and observations with concrete numbers, dates, and relative changes
- Any anomalies, spikes, or trends you notice - and what might be causing them
- A separate unnamed paragraph with actionable recommendations (specific campaigns/time periods,
  what to change, and expected impact) - integrate this naturally without a "Recommendations:"
  header or label
- A natural follow-up question at the end

Write as if you're having a conversation with a colleague - be direct, insightful, and helpful
without being overly formal or structured. Focus on what's genuinely interesting or surprising
in the data, not just what's expected.

=========================
Visualization and Files
=========================

- The visualization layer is separate. You will not create charts directly.
- Tools may return chart configuration files or data files.
- These files are used by the frontend to render charts.

The chart type is automatically selected based on the user's request. The following
chart types are supported — the system selects the best one from context:

- line (default): time-series trend for a single metric over dates (e.g. ROAS, CTR, CPA)
- bar: single-campaign metric comparison by time period
- area: cumulative or volume metrics (e.g. impressions, spend curves)
- pie: distribution breakdowns where parts sum to a whole (e.g. "budget split by platform")
- donut: same as pie with a hole in the center
- stacked_bar: multi-campaign or multi-platform side-by-side stacked comparison
- composed: two metrics overlaid on the same chart (e.g. spend as bars + CTR as a line)
- radar: performance overview across multiple KPIs at once (spider/radar chart)
- scatter: correlation between two metrics (e.g. cost vs conversions)

You do not need to specify the chart type yourself. The system detects the best type
from the user's words. Simply call analyze_campaign_performance as normal.

CRITICAL file handling rule:
- When a tool response includes a `files` field, you MUST copy ALL of those file
  descriptors into your final FileAgentOutput.files list.
- Never drop, filter, or alter file entries, except to rephrase any human readable labels.
- NEVER return file links, URLs, signed URLs, file paths, or any file references in your
  message text. Files are handled automatically by the frontend system.
- The rest of your analysis should refer to the data conceptually, not to implementation
  details of those files.
- Do not mention charts, files, or visualizations in your text - they will be displayed
  automatically by the system.

Communication style:
- Be concise but comprehensive.
- Use data to support every claim.
- Use clear, business friendly language.
- Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.
- Include numbers, percentages, and date ranges where possible.
- Always look for and highlight interesting insights - what's surprising, noteworthy,
  or counterintuitive in the data. Don't just report metrics, explain what makes
  them interesting or significant.
- Always end with one contextual follow up question that helps the user take the
  next step in optimizing their campaigns. When appropriate, propose specifying a
  campaign name and a specific metric (e.g., ROAS, CPA, CTR, conversion rate) for
  deeper analysis.
"""

# Default metric for charts when no specific metric was requested
DEFAULT_CHART_METRIC_KEY = "roas"

# Mapping from metric key to human readable label
METRIC_LABELS: Dict[str, str] = {
    "cost": "Cost",
    "clicks": "Clicks",
    "conversions": "Conversions",
    "impressions": "Impressions",
    "conversion_value": "Conversion Value",
    "roas": "ROAS",
    "cpa": "CPA",
    "cpc": "CPC",
    "ctr": "CTR",
    "conversion_rate": "Conversion Rate",
}


def _build_chart_config(
    metric_key: Optional[str],
    chart_type: str = "line",
    enriched_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the full chart configuration dict for the requested metric and chart type.

    Returns a dict with keys consumed by ``save_chart_config_to_storage``:
    ``series``, and optionally ``name_key``, ``value_key``, ``inner_radius``.

    Handles all Recharts-supported chart types:
    - line / bar / area: standard time-series, single metric
    - pie / donut: distribution breakdown — data must be category-keyed (name + value)
    - stacked_bar: multi-series bars stacked per group; secondary metric alongside primary
    - composed: mixed bar (spend) + line (secondary metric like CTR)
    - radar: all KPIs for a single campaign as a spider chart
    - scatter: two-metric correlation (x=cost, y=conversions)
    """
    key = (metric_key or DEFAULT_CHART_METRIC_KEY).lower()
    if key not in METRIC_LABELS:
        key = DEFAULT_CHART_METRIC_KEY
    label = METRIC_LABELS[key]

    # --- Simple time-series types ---
    if chart_type in ("line", "bar", "area"):
        return {
            "series": [
                {
                    "id": key,
                    "label": label,
                    "type": chart_type,
                    "dataKey": key,
                }
            ],
        }

    # --- Pie / Donut: aggregate total per date → single summary slice ---
    # Re-aggregate enriched_metrics into a summary for a meaningful pie.
    # We produce a per-period aggregation; if metrics are daily we sum them
    # into a single "Total" slice per campaign, which is most useful when
    # showing budget splits across multiple campaigns.
    if chart_type in ("pie", "donut"):
        return {
            "series": [
                {
                    "id": "s1",
                    "label": label,
                    "type": "pie",
                    "dataKey": "value",
                }
            ],
            "name_key": "name",
            "value_key": "value",
            "inner_radius": 60 if chart_type == "donut" else None,
        }

    # --- Stacked bar: primary metric + cost stacked together ---
    # Use the requested metric as first stack and "cost" as second (unless key IS cost).
    secondary_key = "cost" if key != "cost" else "clicks"
    secondary_label = METRIC_LABELS.get(secondary_key, secondary_key.title())
    return_series = [
        {
            "id": key,
            "label": label,
            "type": "bar",
            "dataKey": key,
            "stack": "group1",
        }
    ]
    if chart_type == "stacked_bar":
        return_series.append(
            {
                "id": secondary_key,
                "label": secondary_label,
                "type": "bar",
                "dataKey": secondary_key,
                "stack": "group1",
            }
        )
        return {"series": return_series}

    # --- Composed: spend as bars + requested metric as line ---
    if chart_type == "composed":
        composed_series: List[Dict[str, Any]] = [
            {
                "id": "cost",
                "label": "Spend",
                "type": "bar",
                "dataKey": "cost",
            }
        ]
        if key != "cost":
            composed_series.append(
                {
                    "id": key,
                    "label": label,
                    "type": "line",
                    "dataKey": key,
                }
            )
        return {"series": composed_series}

    # --- Radar: all KPIs as spokes; each date becomes a spoke label ---
    if chart_type == "radar":
        radar_metrics = ["roas", "ctr", "conversion_rate", "cpc", "cpa"]
        return {
            "series": [
                {
                    "id": m,
                    "label": METRIC_LABELS[m],
                    "type": "radar",
                    "dataKey": m,
                }
                for m in radar_metrics
            ],
        }

    # --- Scatter: cost (x-axis) vs requested metric (y-axis) ---
    if chart_type == "scatter":
        return {
            "series": [
                {
                    "id": "s1",
                    "label": f"Cost vs {label}",
                    "type": "scatter",
                    "dataKey": key,
                }
            ],
        }

    # Fallback: line
    return {
        "series": [
            {
                "id": key,
                "label": label,
                "type": "line",
                "dataKey": key,
            }
        ],
    }


def _aggregate_metrics_for_pie(
    enriched_metrics: List[Dict[str, Any]],
    metric_key: str,
    campaign_name: Optional[str],
) -> List[Dict[str, Any]]:
    """Aggregate a time-series metric list into a single-row pie-ready format.

    For a single campaign this produces one slice, which is most useful when
    multiple campaigns are charted together. Returns a list with one dict:
    ``[{"name": campaign_name, "value": total_metric}]``.
    """
    total = sum(float(m.get(metric_key, 0) or 0) for m in enriched_metrics)
    return [{"name": campaign_name or "Campaign", "value": round(total, 4)}]

def _compute_derived_metrics(metric: Dict[str, Any]) -> Dict[str, float]:
    """Compute canonical derived KPIs for a single daily metric row."""
    cost = float(metric.get("cost", 0) or 0)
    clicks = float(metric.get("clicks", 0) or 0)
    conversions = float(metric.get("conversions", 0) or 0)
    impressions = float(metric.get("impressions", 0) or 0)
    conversion_value = float(metric.get("conversion_value", 0) or 0)

    roas = conversion_value / cost if cost > 0 else 0.0
    cpa = cost / conversions if conversions > 0 else 0.0
    cpc = cost / clicks if clicks > 0 else 0.0
    ctr = clicks / impressions if impressions > 0 else 0.0
    conversion_rate = conversions / clicks if clicks > 0 else 0.0

    return {
        "cost": cost,
        "clicks": clicks,
        "conversions": conversions,
        "impressions": impressions,
        "conversion_value": conversion_value,
        "roas": roas,
        "cpa": cpa,
        "cpc": cpc,
        "ctr": ctr,
        "conversion_rate": conversion_rate,
    }


def _metrics_failure(error: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": error,
        "data": [],
        "chart_configs": [],
        "files": [],
    }


async def _fetch_user_campaigns(
    user_id: str,
    columns: str,
    campaign_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sb = await asyncio.to_thread(get_client)
    query = sb.table("campaigns").select(columns).eq("user_id", user_id)
    if campaign_id:
        query = query.eq("campaign_id", campaign_id)
    result = await asyncio.to_thread(query.execute)
    return getattr(result, "data", None) or []


class CampaignExtractionResult(BaseModel):
    detected_names: List[str] = Field(default_factory=list)
    requested_metric_key: str = "roas"
    requested_metric_raw: Optional[str] = None
    chart_type: str = "line"
    compare_intent: bool = False
    reasoning: str = ""


_campaign_extractor: Optional[Agent] = None


def _get_campaign_extractor() -> Agent:
    """Return the campaign-extraction agent, creating it on first call."""
    global _campaign_extractor
    if _campaign_extractor is None:
        _campaign_extractor = Agent(
            model=to_responses_model(Models.PERF_EXTRACTION),
            model_settings=ModelSettings(temperature=0.2, max_tokens=400),
            system_prompt=(
                "You are a helpful assistant that extracts campaign names, "
                "requested performance metrics, and the most appropriate chart "
                "type from user messages about advertising campaign performance."
            ),
            output_type=CampaignExtractionResult,
        )
    return _campaign_extractor


def _build_context_text(
    user_message: Optional[str], message_history: Optional[List[Any]]
) -> str:
    context_parts: List[str] = []
    if user_message:
        context_parts.append(f"Current user message: {user_message}")

    message_history = message_history or []
    recent_messages = message_history[-5:]

    if recent_messages:
        context_parts.append("\nRecent conversation:")
        for msg in recent_messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                context_parts.append(f"{role}: {content[:200]}")

    return "\n".join(context_parts) if context_parts else (user_message or "")


def _match_detected_to_campaigns(
    detected_names: List[str], available_campaigns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    matched_campaigns: List[Dict[str, Any]] = []
    for campaign in available_campaigns:
        campaign_name = campaign.get("campaign_name", "")
        if not campaign_name:
            continue

        campaign_name_lower = campaign_name.lower()
        campaign_id = campaign.get("campaign_id")

        for detected in detected_names:
            detected_lower = detected.lower()
            if (
                detected_lower in campaign_name_lower
                or campaign_name_lower in detected_lower
                or any(
                    word in campaign_name_lower
                    for word in detected_lower.split()
                    if len(word) > 3
                )
            ):
                matched_campaigns.append(
                    {"campaign_id": campaign_id, "campaign_name": campaign_name}
                )
                break
    return matched_campaigns


async def _save_chart_config_for_campaign(
    enriched_metrics: List[Dict[str, Any]],
    user_id: str,
    campaign_id_val: Optional[str],
    campaign_name: Optional[str],
    metric_key: Optional[str],
    chart_type: str = "line",
) -> Optional[Dict[str, Any]]:
    """Save a chart configuration for a single campaign and a single metric.

    Supports all Recharts chart types via ``chart_type``. For pie/donut the
    time-series data is aggregated into a single-slice summary first.
    """
    if not enriched_metrics:
        return None

    try:
        timestamp = int(time.time() * 1000)
        short_uuid = generate_short_uuid()
        if campaign_id_val:
            file_name = f"campaign_{campaign_id_val}_canonical_chart_{timestamp}_{short_uuid}.json"
        else:
            file_name = f"campaign_canonical_chart_{timestamp}_{short_uuid}.json"

        cfg = _build_chart_config(
            metric_key=metric_key,
            chart_type=chart_type,
            enriched_metrics=enriched_metrics,
        )

        # Pie/donut charts need pre-aggregated data rather than raw time-series
        key = (metric_key or DEFAULT_CHART_METRIC_KEY).lower()
        if key not in METRIC_LABELS:
            key = DEFAULT_CHART_METRIC_KEY

        if chart_type in ("pie", "donut"):
            chart_data = _aggregate_metrics_for_pie(enriched_metrics, key, campaign_name)
            x_key = "name"
        else:
            chart_data = enriched_metrics
            x_key = "date"

        chart_result = await asyncio.to_thread(
            save_chart_config_to_storage,
            x_key=x_key,
            series=cfg["series"],
            data=chart_data,
            user_id=user_id,
            file_name=file_name,
            title=campaign_name or "Campaign Performance",
            name_key=cfg.get("name_key"),
            value_key=cfg.get("value_key"),
            inner_radius=cfg.get("inner_radius"),
        )

        if not chart_result.get("success"):
            return None

        signed_url = chart_result.get("signed_url")
        if isinstance(signed_url, dict):
            signed_url = (
                signed_url.get("signedURL") or signed_url.get("signed_url") or ""
            )

        file_path = chart_result.get("file_path", "")

        return {
            "chart_config": {
                "campaign_id": campaign_id_val,
                "campaign_name": campaign_name,
                "chart_file": file_name,
                "chart_url": signed_url or "",
                "chart_path": file_path,
                "metric_key": (metric_key or DEFAULT_CHART_METRIC_KEY),
            },
            "chart_file": SaveFileOutput(
                file_name=file_name,
                file_path=file_path,
                file_type="recharts_data",
            ),
        }
    except Exception as e:
        logger.warning(
            f"Failed to generate chart config for campaign {campaign_id_val}: {e}"
        )
        return None


async def _detect_campaign_and_metric(
    user_id: str,
    available_campaigns: List[Dict[str, Any]],
    context_text: str,
) -> Dict[str, Any]:
    """
    Detect campaign name and requested metric from user context using GPT-4-mini.

    Returns a dictionary with:
    - suggested_campaign_id: Optional[str]
    - suggested_campaign_name: Optional[str]
    - requested_metric_key: str (normalized, defaults to "roas")
    - requested_metric_raw: Optional[str]
    - matched_campaigns: List[Dict[str, Any]]
    """
    if not available_campaigns:
        return {
            "suggested_campaign_id": None,
            "suggested_campaign_name": None,
            "requested_metric_key": DEFAULT_CHART_METRIC_KEY,
            "requested_metric_raw": None,
            "matched_campaigns": [],
        }

    campaign_names_list = [
        c.get("campaign_name", "")
        for c in available_campaigns
        if c.get("campaign_name")
    ]
    campaigns_text = "\n".join(f"- {name}" for name in campaign_names_list)

    prompt = f"""Analyze the following user message and conversation context to identify which campaign(s) the user is referring to and which performance metric they care about most.

Available campaigns for this user:
{campaigns_text}

User context:
{context_text}

You must:
1) Extract campaign names or references from the user's message. The user might refer to campaigns by:
   - Exact name match
   - Partial name (for example, "summer" if campaign is "Summer Sale Campaign")
   - Descriptive terms (for example, "Q4 campaign", "holiday campaign")
   - Relative terms (for example, "my campaign", "the campaign", "that campaign")

2) Determine whether the user asked to focus on a specific metric. Supported metrics and their canonical keys:
   - "roas"
   - "cpa"
   - "cpc"
   - "ctr"
   - "conversion_rate"
   - "cost"
   - "clicks"
   - "conversions"
   - "conversion_value"
   - "impressions"

If the user explicitly asks to focus on one of these metrics (for example, "show me CPA trend", "graph CTR", "ROAS curve"), set requested_metric_key to the corresponding key. If the user does not clearly request a specific metric, default requested_metric_key to "roas".

3) Determine the most appropriate chart type for the request. Use these rules:
   - "line" (default): time-series trend for a single metric over dates (e.g. "show ROAS over time", "CTR trend")
   - "bar": single-campaign metric comparison or when user asks for a bar chart
   - "area": cumulative spend or impression curves
   - "pie": distribution or breakdown by category where parts sum to a whole (e.g. "budget split by platform", "spend breakdown", "share of spend")
   - "donut": same as pie but user wants a donut/ring style (e.g. "donut chart of budget")
   - "stacked_bar": comparing multiple campaigns or platforms side by side on the same axis (e.g. "compare campaigns", "spend by platform over time", "stacked")
   - "composed": when the user wants two different metrics shown together (e.g. "spend and CTR", "cost vs ROAS", "overlay")
   - "radar": comparing multiple KPIs for a single campaign at a glance (e.g. "performance overview", "radar chart", "spider chart")
   - "scatter": correlation between two metrics (e.g. "spend vs conversions", "cost vs clicks scatter")
   Default to "line" if nothing specific is requested.

4) Determine if the user is asking for a portfolio comparison across ALL campaigns, not a
   single specific campaign. Comparison signals include:
   - "compare campaigns", "rank campaigns", "top performer", "best campaign",
   - "worst campaign", "which campaign", "all campaigns", "portfolio overview",
   - "how do my campaigns compare", "benchmark across campaigns"
   Set "compare_intent": true in the response if any of these apply (or similar phrasing).
   When compare_intent is true, set "detected_names" to [].

Return a JSON object with:
- "detected_names": array of campaign name strings extracted from the context (can be partial or descriptive). Empty array if compare_intent is true.
- "requested_metric_key": one of the canonical keys listed above, or "roas" by default if nothing specific is requested
- "requested_metric_raw": the exact phrase from the user that refers to the metric, or null if none
- "chart_type": one of: line, bar, area, pie, donut, stacked_bar, composed, radar, scatter
- "compare_intent": true if the user wants to compare/rank all campaigns, false otherwise
- "reasoning": brief explanation of why these names, metric, and chart type were detected

Always return valid JSON only, with no markdown fences.

Example response:
{{
  "detected_names": ["Summer Sale Campaign"],
  "requested_metric_key": "roas",
  "requested_metric_raw": "ROAS performance",
  "chart_type": "line",
  "compare_intent": false,
  "reasoning": "User mentioned 'summer campaign' which matches 'Summer Sale Campaign' and asked about ROAS performance over time."
}}"""

    try:
        result = await _get_campaign_extractor().run(prompt)
        extraction: CampaignExtractionResult = result.output

        detected_names = extraction.detected_names
        # Normalize metric key against known metrics, falling back to default
        requested_metric_key = extraction.requested_metric_key.lower().strip()
        if requested_metric_key not in METRIC_LABELS:
            requested_metric_key = DEFAULT_CHART_METRIC_KEY
        requested_metric_raw = extraction.requested_metric_raw

        # Normalize chart type
        chart_type = (extraction.chart_type or "line").lower().strip()
        valid_chart_types = {
            "line", "bar", "area", "pie", "donut",
            "stacked_bar", "composed", "radar", "scatter",
        }
        if chart_type not in valid_chart_types:
            chart_type = "line"

        matched_campaigns = _match_detected_to_campaigns(
            detected_names, available_campaigns
        )

        suggested_campaign_id = (
            matched_campaigns[0].get("campaign_id") if matched_campaigns else None
        )
        suggested_campaign_name = (
            matched_campaigns[0].get("campaign_name") if matched_campaigns else None
        )

        logger.info(f"Detected campaign ID: {suggested_campaign_id}")
        logger.info(f"Detected campaign name: {suggested_campaign_name}")
        logger.info(f"Requested metric key: {requested_metric_key}")
        logger.info(f"Detected chart type: {chart_type}")

        return {
            "suggested_campaign_id": suggested_campaign_id,
            "suggested_campaign_name": suggested_campaign_name,
            "requested_metric_key": requested_metric_key,
            "requested_metric_raw": requested_metric_raw,
            "chart_type": chart_type,
            "compare_intent": extraction.compare_intent,
            "matched_campaigns": matched_campaigns,
        }
    except Exception as e:
        logger.error(f"Error detecting campaign and metric: {e}")
        return {
            "suggested_campaign_id": None,
            "suggested_campaign_name": None,
            "requested_metric_key": DEFAULT_CHART_METRIC_KEY,
            "requested_metric_raw": None,
            "chart_type": "line",
            "matched_campaigns": [],
        }


def _compute_campaign_analytics(
    enriched_metrics: List[Dict[str, Any]],
    budget_cents: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute higher-level analytics for one campaign's time-series.

    Returns a dict with:
    - anomalies: days with a metric >2 std-devs from its 7-day rolling mean
    - trends: week-over-week % changes for key metrics
    - budget_utilization: spend vs budget (if budget_cents provided)
    """
    if not enriched_metrics:
        return {"anomalies": [], "trends": {}, "budget_utilization": None}

    # Sort by date ascending
    sorted_metrics = sorted(enriched_metrics, key=lambda m: m.get("date", ""))

    # --- Anomaly detection ---
    anomaly_metrics = ["cost", "clicks", "conversions", "roas", "cpa", "ctr"]
    anomalies: List[Dict[str, Any]] = []
    window = 7

    for key in anomaly_metrics:
        values = [float(m.get(key, 0) or 0) for m in sorted_metrics]
        for i in range(window, len(values)):
            roll = values[i - window : i]
            mean = sum(roll) / len(roll)
            variance = sum((v - mean) ** 2 for v in roll) / len(roll)
            std = variance ** 0.5
            current = values[i]
            if std > 0 and abs(current - mean) > 2 * std:
                deviation_pct = round((current - mean) / mean * 100, 1) if mean != 0 else None
                anomalies.append(
                    {
                        "date": sorted_metrics[i].get("date"),
                        "metric": key,
                        "value": round(current, 4),
                        "rolling_mean": round(mean, 4),
                        "deviation_pct": deviation_pct,
                        "direction": "spike" if current > mean else "drop",
                    }
                )

    # Keep only the most significant anomalies (top 10 by abs deviation)
    anomalies = sorted(
        anomalies,
        key=lambda a: abs(a.get("deviation_pct") or 0),
        reverse=True,
    )[:10]

    # --- Week-over-week trends ---
    trends: Dict[str, Any] = {}
    if len(sorted_metrics) >= 14:
        last7 = sorted_metrics[-7:]
        prev7 = sorted_metrics[-14:-7]

        def _avg(rows: List[Dict], key: str) -> float:
            vals = [float(r.get(key, 0) or 0) for r in rows]
            return sum(vals) / len(vals) if vals else 0.0

        for key in ["roas", "cpa", "ctr", "cost", "conversions"]:
            last_avg = _avg(last7, key)
            prev_avg = _avg(prev7, key)
            if prev_avg != 0:
                wow_pct = round((last_avg - prev_avg) / prev_avg * 100, 1)
            else:
                wow_pct = None
            trends[f"{key}_last7_avg"] = round(last_avg, 4)
            trends[f"{key}_prev7_avg"] = round(prev_avg, 4)
            trends[f"{key}_wow_pct"] = wow_pct

    # --- Budget utilization ---
    budget_utilization: Optional[Dict[str, Any]] = None
    if budget_cents and budget_cents > 0:
        total_spend = sum(float(m.get("cost", 0) or 0) for m in enriched_metrics)
        budget_usd = budget_cents / 100
        num_days = len(sorted_metrics)
        daily_avg_spend = total_spend / num_days if num_days > 0 else 0
        # For daily budgets the reference spend is budget_usd * num_days
        expected_spend = budget_usd * num_days
        utilization_pct = round(total_spend / expected_spend * 100, 1) if expected_spend > 0 else None
        budget_utilization = {
            "budget_usd": round(budget_usd, 2),
            "total_spend_usd": round(total_spend, 2),
            "utilization_pct": utilization_pct,
            "days_with_data": num_days,
            "daily_avg_spend_usd": round(daily_avg_spend, 2),
        }

    return {
        "anomalies": anomalies,
        "trends": trends,
        "budget_utilization": budget_utilization,
    }


_CAMPAIGN_SUMMARY_COLUMNS = (
    "campaign_id, campaign_name, campaign_status, goal, campaign_type, "
    "budget_type, daily_budget_cents, total_budget_cents, campaign_start, campaign_end, "
    "campaign_metrics"
)


def _resolve_budget_cents(row: Dict[str, Any]) -> Optional[int]:
    """Return the relevant budget in cents based on budget_type.

    - budget_type == 'per day' -> daily_budget_cents
    - budget_type == 'total'   -> total_budget_cents
    - fallback: whichever is non-zero
    """
    budget_type = (row.get("budget_type") or "").lower()
    daily = row.get("daily_budget_cents")
    total = row.get("total_budget_cents")

    if budget_type == "per day" and daily:
        return int(daily)
    if budget_type == "total" and total:
        return int(total)
    # fallback: prefer whichever exists
    return int(daily or total) if (daily or total) else None


def _extract_campaign_context(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return human-readable campaign context fields (non-metrics) from a row."""
    platforms_raw = row.get("platforms") or {}
    channels: List[str] = []
    if isinstance(platforms_raw, dict):
        channels = platforms_raw.get("channels") or []
    elif isinstance(platforms_raw, list):
        channels = platforms_raw

    daily = row.get("daily_budget_cents")
    total = row.get("total_budget_cents")

    return {
        "campaign_status": row.get("campaign_status"),
        "goal": row.get("goal"),
        "campaign_type": row.get("campaign_type"),
        "budget_type": row.get("budget_type"),
        "daily_budget_usd": round(daily / 100, 2) if daily else None,
        "total_budget_usd": round(total / 100, 2) if total else None,
        "campaign_start": row.get("campaign_start"),
        "campaign_end": row.get("campaign_end"),
        "channels": channels,
    }


def _filter_metrics_by_date(
    metrics: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    """Filter a campaign_metrics list to the requested date window.

    Both ``start_date`` and ``end_date`` are inclusive ISO-8601 date strings
    (e.g. "2025-01-01"). Missing bounds are treated as open-ended.
    """
    if not start_date and not end_date:
        return metrics
    result = []
    for m in metrics:
        d = m.get("date", "")
        if not d:
            continue
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        result.append(m)
    return result


async def _fetch_and_process_metrics(
    user_id: str,
    campaign_id: Optional[str],
    requested_metric_key: str,
    generate_charts: bool = True,
    chart_type: str = "line",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch campaign metrics and process them into enriched data with charts.

    Args:
        user_id: User ID to fetch campaigns for
        campaign_id: Optional campaign ID to filter by
        requested_metric_key: Metric key to use for charting
        generate_charts: Whether to generate chart configurations.
        chart_type: Recharts chart type string.
        start_date: Optional ISO date string (inclusive) to filter metrics.
        end_date: Optional ISO date string (inclusive) to filter metrics.

    Returns a dictionary with:
    - success: bool
    - data: List[Dict[str, Any]] (campaign data with metrics + context)
    - count: int
    - chart_configs: List[Dict[str, Any]]
    - files: List[Dict[str, Any]]
    - error: Optional[str] (only if success is False)
    """
    # Use full columns for a single-campaign deep-dive; summary columns for list/compare.
    # The campaign_id filter is pushed to the DB so we never pull the full table.
    if campaign_id:
        campaigns = await _fetch_user_campaigns(user_id, "*", campaign_id=campaign_id)
    else:
        campaigns = await _fetch_user_campaigns(user_id, _CAMPAIGN_SUMMARY_COLUMNS)

    parsed_data: List[Dict[str, Any]] = []
    chart_configs: List[Dict[str, Any]] = []
    chart_files: List[SaveFileOutput] = []

    for row in campaigns:
        raw_metrics = row.get("campaign_metrics") or []

        if isinstance(raw_metrics, str):
            try:
                raw_metrics = json.loads(raw_metrics)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse campaign_metrics JSON for campaign "
                    f"{row.get('campaign_id')}"
                )
                raw_metrics = []

        metrics_list = raw_metrics if isinstance(raw_metrics, list) else []

        # Apply date-range filter before enrichment
        metrics_list = _filter_metrics_by_date(metrics_list, start_date, end_date)

        enriched_metrics: List[Dict[str, Any]] = [
            {
                "date": metric.get("date", ""),
                **_compute_derived_metrics(metric),
            }
            for metric in metrics_list
        ]

        campaign_id_val = row.get("campaign_id")
        campaign_name = row.get("campaign_name")
        campaign_ctx = _extract_campaign_context(row)

        # Compute analytics inline
        analytics = _compute_campaign_analytics(enriched_metrics, _resolve_budget_cents(row))

        parsed_data.append(
            {
                "campaign_id": campaign_id_val,
                "campaign_name": campaign_name,
                "context": campaign_ctx,
                "metrics": enriched_metrics,
                "analytics": analytics,
            }
        )

        # Only generate charts if generate_charts is True (i.e., specific campaign requested)
        if generate_charts:
            chart_payload = await _save_chart_config_for_campaign(
                enriched_metrics=enriched_metrics,
                user_id=user_id,
                campaign_id_val=campaign_id_val,
                campaign_name=campaign_name,
                metric_key=requested_metric_key,
                chart_type=chart_type,
            )

            if chart_payload:
                chart_configs.append(chart_payload["chart_config"])
                chart_files.append(chart_payload["chart_file"])

    campaigns_with_metrics = [c for c in parsed_data if c.get("metrics")]

    if not campaigns_with_metrics:
        return _metrics_failure(
            (
                "No campaign metrics available yet. "
                "Please run campaigns or wait for data to accumulate."
            )
        )

    return {
        "success": True,
        "data": campaigns_with_metrics,
        "count": len(campaigns_with_metrics),
        "chart_configs": chart_configs,
        "files": [f.model_dump() for f in chart_files],
    }


def register_campaign_analysis_tool(agent: Agent) -> None:
    """Register the unified campaign analysis tool on the given agent."""

    @agent.tool
    async def analyze_campaign_performance(
        ctx: RunContext[ChatDeps],
        campaign_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional campaign ID to filter by. If not provided, the tool will "
                "attempt to detect the campaign from user context, or return all "
                "campaigns for the current user."
            ),
        ),
        user_message: Optional[str] = Field(
            default=None,
            description=(
                "Optional explicit user message to analyze for campaign detection. "
                "If not provided, the tool will use the conversation history from context."
            ),
        ),
        start_date: Optional[str] = Field(
            default=None,
            description=(
                "Optional start date for metric filtering, ISO format YYYY-MM-DD (inclusive). "
                "Use when the user specifies a date range, e.g. 'last 30 days', 'since January'."
            ),
        ),
        end_date: Optional[str] = Field(
            default=None,
            description=(
                "Optional end date for metric filtering, ISO format YYYY-MM-DD (inclusive)."
            ),
        ),
    ) -> Dict[str, Any]:
        """Analyze campaign performance with enriched context, analytics, and charts.

        This tool automatically:
        1. Detects which campaign the user is referring to (if campaign_id is not provided)
        2. Extracts the requested metric and chart type from context
        3. Filters metrics to the specified date range (if provided)
        4. Fetches full campaign context: status, goal, budget, channels, start/end dates
        5. Computes analytics: anomaly detection, week-over-week trends, budget utilization
        6. Generates chart configurations for frontend visualization

        Returns enriched metrics, campaign context, computed analytics, and chart files.
        """
        ctx.deps.status_queue.put_nowait("Analyzing campaign performance")
        try:
            user_id = ctx.deps.user_id
            if not user_id:
                return _metrics_failure("User ID not available")

            requested_metric_key = DEFAULT_CHART_METRIC_KEY
            resolved_campaign_id = campaign_id
            detected_chart_type = "line"

            # Pull date range from frontend context if not explicitly passed
            resolved_start = start_date
            resolved_end = end_date
            if not resolved_start and not resolved_end:
                chat_context = getattr(ctx.deps, "context", None) or {}
                date_range = chat_context.get("date_range") or {}
                if date_range:
                    raw_start = date_range.get("start_date") or date_range.get("startDate")
                    raw_end = date_range.get("end_date") or date_range.get("endDate")
                    if raw_start:
                        resolved_start = str(raw_start)[:10]
                    if raw_end:
                        resolved_end = str(raw_end)[:10]

            # If no campaign_id provided, attempt to detect from context
            if not resolved_campaign_id:
                message_history = ctx.deps.message_history
                context_text = _build_context_text(user_message, message_history)

                if context_text.strip():
                    available_campaigns = await _fetch_user_campaigns(
                        user_id, "campaign_id, campaign_name"
                    )

                    if available_campaigns:
                        detection_result = await _detect_campaign_and_metric(
                            user_id=user_id,
                            available_campaigns=available_campaigns,
                            context_text=context_text,
                        )
                        resolved_campaign_id = detection_result["suggested_campaign_id"]
                        requested_metric_key = detection_result["requested_metric_key"]
                        detected_chart_type = detection_result.get("chart_type", "line")

                        logger.info(
                            f"Campaign detection: ID={resolved_campaign_id}, "
                            f"metric={requested_metric_key}, "
                            f"chart_type={detected_chart_type}"
                        )

            # Normalize metric key
            if not isinstance(requested_metric_key, str):
                requested_metric_key = DEFAULT_CHART_METRIC_KEY
            requested_metric_key = requested_metric_key.lower()
            if requested_metric_key not in METRIC_LABELS:
                requested_metric_key = DEFAULT_CHART_METRIC_KEY

            should_generate_charts = resolved_campaign_id is not None

            result = await _fetch_and_process_metrics(
                user_id=user_id,
                campaign_id=resolved_campaign_id,
                requested_metric_key=requested_metric_key,
                generate_charts=should_generate_charts,
                chart_type=detected_chart_type,
                start_date=resolved_start,
                end_date=resolved_end,
            )

            if not campaign_id and resolved_campaign_id:
                result["detected_campaign_id"] = resolved_campaign_id
                result["requested_metric_key"] = requested_metric_key
                result["chart_type"] = detected_chart_type

            if resolved_start or resolved_end:
                result["date_range_applied"] = {
                    "start_date": resolved_start,
                    "end_date": resolved_end,
                }

            return result

        except Exception as e:
            logger.error(f"Error analyzing campaign performance: {e}")
            return _metrics_failure(str(e))


def register_compare_campaigns_tool(agent: Agent) -> None:
    """Register a multi-campaign comparison tool on the given agent."""

    @agent.tool
    async def compare_campaigns(
        ctx: RunContext[ChatDeps],
        metric_key: Optional[str] = Field(
            default=None,
            description=(
                "Metric to rank/compare campaigns by. One of: roas, cpa, cpc, ctr, "
                "conversion_rate, cost, clicks, conversions, impressions, conversion_value. "
                "Defaults to roas."
            ),
        ),
        start_date: Optional[str] = Field(
            default=None,
            description="Optional start date YYYY-MM-DD to filter metrics.",
        ),
        end_date: Optional[str] = Field(
            default=None,
            description="Optional end date YYYY-MM-DD to filter metrics.",
        ),
        chart_type: Optional[str] = Field(
            default=None,
            description=(
                "Chart type for the comparison visualization. "
                "One of: bar, stacked_bar, pie, donut. Defaults to bar."
            ),
        ),
    ) -> Dict[str, Any]:
        """Compare all campaigns for the user, ranked by the requested metric.

        Aggregates each campaign's metrics into totals and averages, then ranks them.
        Returns:
        - ranked_campaigns: list sorted best-to-worst by the requested metric
        - summary: aggregate stats across all campaigns
        - files: chart file for frontend visualization
        - winner / loser: top and bottom performing campaign names

        Use this tool when the user asks to compare campaigns, find top/bottom performers,
        or see a ranking across their campaign portfolio.
        """
        ctx.deps.status_queue.put_nowait("Comparing campaigns")
        try:
            user_id = getattr(ctx.deps, "user_id", None)
            if not user_id:
                return _metrics_failure("User ID not available")

            # Normalize metric key
            key = (metric_key or DEFAULT_CHART_METRIC_KEY).lower()
            if key not in METRIC_LABELS:
                key = DEFAULT_CHART_METRIC_KEY

            # Pull date range from frontend context if not provided
            resolved_start = start_date
            resolved_end = end_date
            if not resolved_start and not resolved_end:
                chat_context = getattr(ctx.deps, "context", None) or {}
                date_range = chat_context.get("date_range") or {}
                if date_range:
                    raw_start = date_range.get("start_date") or date_range.get("startDate")
                    raw_end = date_range.get("end_date") or date_range.get("endDate")
                    if raw_start:
                        resolved_start = str(raw_start)[:10]
                    if raw_end:
                        resolved_end = str(raw_end)[:10]

            resolved_chart_type = (chart_type or "bar").lower()
            valid_comparison_charts = {"bar", "stacked_bar", "pie", "donut"}
            if resolved_chart_type not in valid_comparison_charts:
                resolved_chart_type = "bar"

            campaigns = await _fetch_user_campaigns(user_id, _CAMPAIGN_SUMMARY_COLUMNS)

            ranked: List[Dict[str, Any]] = []
            chart_data_rows: List[Dict[str, Any]] = []

            for row in campaigns:
                raw_metrics = row.get("campaign_metrics") or []
                if isinstance(raw_metrics, str):
                    try:
                        raw_metrics = json.loads(raw_metrics)
                    except Exception:
                        raw_metrics = []

                metrics_list = raw_metrics if isinstance(raw_metrics, list) else []
                metrics_list = _filter_metrics_by_date(metrics_list, resolved_start, resolved_end)

                if not metrics_list:
                    continue

                enriched = [
                    {"date": m.get("date", ""), **_compute_derived_metrics(m)}
                    for m in metrics_list
                ]

                # Aggregate totals and averages
                n = len(enriched)
                totals: Dict[str, float] = {}
                for agg_key in ["cost", "clicks", "conversions", "impressions", "conversion_value"]:
                    totals[f"total_{agg_key}"] = round(
                        sum(float(m.get(agg_key, 0) or 0) for m in enriched), 2
                    )
                # Derived metrics: average over the period
                for agg_key in ["roas", "cpa", "cpc", "ctr", "conversion_rate"]:
                    vals = [float(m.get(agg_key, 0) or 0) for m in enriched]
                    totals[f"avg_{agg_key}"] = round(sum(vals) / n, 4) if n else 0.0

                campaign_ctx = _extract_campaign_context(row)
                comparison_key = f"avg_{key}" if key in ("roas", "cpa", "cpc", "ctr", "conversion_rate") else f"total_{key}"
                sort_value = totals.get(comparison_key, 0.0)

                ranked.append(
                    {
                        "campaign_id": row.get("campaign_id"),
                        "campaign_name": row.get("campaign_name"),
                        "context": campaign_ctx,
                        "totals": totals,
                        "days_with_data": n,
                        "sort_value": sort_value,
                    }
                )
                chart_data_rows.append(
                    {
                        "name": row.get("campaign_name") or row.get("campaign_id"),
                        "value": round(sort_value, 4),
                        key: round(sort_value, 4),
                    }
                )

            if not ranked:
                return _metrics_failure("No campaign data available for comparison.")

            # For CPA: lower is better — reverse sort for those
            lower_is_better = key in ("cpa", "cpc", "cost")
            ranked.sort(key=lambda r: r["sort_value"], reverse=not lower_is_better)
            chart_data_rows.sort(
                key=lambda r: r["value"], reverse=not lower_is_better
            )

            # Build chart
            cfg = _build_chart_config(
                metric_key=key,
                chart_type=resolved_chart_type,
            )
            timestamp = int(time.time() * 1000)
            short_uuid = generate_short_uuid()
            file_name = f"campaign_comparison_{timestamp}_{short_uuid}.json"

            chart_result = await asyncio.to_thread(
                save_chart_config_to_storage,
                x_key="name",
                series=cfg["series"],
                data=chart_data_rows,
                user_id=user_id,
                file_name=file_name,
                title=f"Campaign Comparison — {METRIC_LABELS.get(key, key)}",
                name_key=cfg.get("name_key"),
                value_key=cfg.get("value_key"),
                inner_radius=cfg.get("inner_radius"),
            )

            chart_files = []
            if chart_result.get("success"):
                file_path = chart_result.get("file_path", "")
                chart_files.append(
                    SaveFileOutput(
                        file_name=file_name,
                        file_path=file_path,
                        file_type="recharts_data",
                    ).model_dump()
                )

            # Portfolio summary
            all_spend = sum(r["totals"].get("total_cost", 0) for r in ranked)
            all_conversions = sum(r["totals"].get("total_conversions", 0) for r in ranked)
            portfolio_roas = (
                sum(r["totals"].get("total_conversion_value", 0) for r in ranked) / all_spend
                if all_spend > 0 else 0.0
            )

            return {
                "success": True,
                "ranked_campaigns": ranked,
                "ranked_by": key,
                "lower_is_better": lower_is_better,
                "winner": ranked[0]["campaign_name"] if ranked else None,
                "loser": ranked[-1]["campaign_name"] if len(ranked) > 1 else None,
                "count": len(ranked),
                "portfolio_summary": {
                    "total_spend_usd": round(all_spend, 2),
                    "total_conversions": round(all_conversions, 2),
                    "portfolio_roas": round(portfolio_roas, 4),
                },
                "date_range_applied": {
                    "start_date": resolved_start,
                    "end_date": resolved_end,
                },
                "files": chart_files,
            }

        except Exception as e:
            logger.error(f"Error comparing campaigns: {e}")
            return _metrics_failure(str(e))


class PerformanceAnalystAgent(LofiAgent):
    """Factory and metadata for the Performance Analyst agent."""

    REMINDER_HEADER = PERFORMANCE_ANALYST_REMINDER_HEADER

    @classmethod
    def create(
        cls, model_name: Optional[str] = None, web_tool_type: str = "web_search"
    ) -> Agent:
        model_name = to_responses_model(model_name or Models.PERFORMANCE)
        logger.info(f"Creating Performance Analyst agent with model '{model_name}'")
        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.2, max_tokens=2500),
            deps_type=ChatDeps,
            system_prompt=PERFORMANCE_ANALYST_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
            builtin_tools=(WebSearchTool(search_context_size="medium"),),
        )
        register_campaign_analysis_tool(agent)
        register_compare_campaigns_tool(agent)
        register_brand_context_tool(agent)

        return agent
