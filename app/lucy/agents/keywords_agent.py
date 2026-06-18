import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

import certifi
import pandas as pd
from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps, FileAgentOutput


_US_STATES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

_NAME_TO_CODE: dict[str, str] = {v.upper(): k for k, v in _US_STATES.items()}


def _normalize_region_name(raw: str) -> str:
    """Convert a US state code or name to its full name (title case)."""
    if not raw:
        return ""
    s = raw.strip()
    m = re.search(r"([A-Za-z]{2})\b$", s)
    code = m.group(1).upper() if m else s.upper()
    if code in _US_STATES:
        return _US_STATES[code]
    s = re.sub(r"^US-", "", s, flags=re.IGNORECASE).strip()
    return s.title()


def _state_code_for(raw: Optional[str]) -> Optional[str]:
    """Convert a US state name or abbreviation to its 2-letter code."""
    if not raw:
        return None
    s = raw.strip()
    m = re.search(r"([A-Za-z]{2})\b$", s)
    if m:
        return m.group(1).upper()
    return _NAME_TO_CODE.get(s.upper())


class KeywordsAgent(LofiAgent):
    """Factory and metadata for the Keywords agent."""

    REMINDER_HEADER = (
        "You are Lucy Keywords — crisp, data-driven keyword analyst. "
        "Two sections: 'Keywords analysis' (numbers) and 'Recommendations' (actions). "
        "Call tools silently — no announcement text before a tool call. "
        "End with one contextual follow-up question."
    )

    @staticmethod
    def _tool_name(tool_def: Any) -> str:
        return (
            getattr(tool_def, "name", None)
            or getattr(getattr(tool_def, "defn", None), "name", None)
            or ""
        )

    @classmethod
    def _select_tools(cls, deps: Any, tool_defs: list[Any]) -> list[Any]:
        if getattr(deps, "keywords_tool_used", False):
            allowed = {"final_result"}
        else:
            allowed = {"keyword_trends_exec_tool"}
        return [t for t in tool_defs if cls._tool_name(t) in allowed]

    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.KEYWORDS)
        logger.info(f"Creating keywords agent with model '{model_name}'")

        async def _prepare_tools(ctx: RunContext[ChatDeps], tool_defs: list[Any]) -> list[Any]:
            return cls._select_tools(ctx.deps, tool_defs)

        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.2, max_tokens=1200),
            deps_type=ChatDeps,
            system_prompt=KEYWORDS_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
        )
        return agent


KEYWORDS_SYSTEM_PROMPT = """
You are Lucy's Keyword Trends Agent. Your job is to produce fast, high-signal keyword insights as clear text.

Operating rules:
- Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.
- Pass ISO-3166-1 alpha-2 code or empty for worldwide as `geo` parameter 
- Then write a concise answer with TWO sections using these exact headers (strictly):
  - "Keywords analysis" — numeric snapshot (seed value and % change; top regions 3–5 and region-specific value/rank if requested; share-of-search leaders with %; counts of rising/related/suggested).
  - "Recommendations" — 2–4 brand/industry-tailored actions (targeting, messaging, timing, channels). Use `user_profiles` context when present.

Inputs you may infer or receive:
- Extract seed keywords from the user message.
- If the user mentions a US location (state code like "NJ" or state/city name like "New Jersey"), set `region_filter` to that location.
- Use sensible defaults when unspecified: timeframe "today 3-m", geo "US", include_related=true, mode="quick", top_n=10.

### Follow-up
End with one short follow-up question that advances the user's current goal.
The question should be informed by what the user is trying to accomplish — not a generic upsell.
Mention Lofi only when the natural next step involves a Lofi feature (e.g. creating a campaign, launching an ad).
Never conflate Lofi (the platform) with the user's brand or company.
"""


def create_keywords_agent(model_name: str = Models.KEYWORDS) -> Agent:
    agent = KeywordsAgent.create(model_name)

    @agent.tool
    async def keyword_trends_exec_tool(
        ctx: RunContext[ChatDeps],
        keywords: List[str] = Field(description="Keywords to analyze (1-5)"),
        timeframe: str = Field(default="today 3-m", description="Pytrends timeframe"),
        geo: str = Field(
            default="US", description="ISO-3166-1 alpha-2 code or empty for worldwide"
        ),
        include_related: bool = Field(
            default=True, description="Include related queries/topics"
        ),
        mode: str = Field(default="quick", description="quick or full"),
        top_n: int = Field(default=10, description="Top items to return in quick mode"),
        region_filter: Optional[str] = Field(
            default=None,
            description="Optional region/city/state filter, e.g., 'NJ' or 'New Jersey'",
        ),
    ) -> Dict[str, Any]:
        """
        Execute Google Trends lookups and return a structured JSON payload.
        """
        logger.info(
            f"Keywords exec: keywords={keywords}, timeframe='{timeframe}', geo='{geo}', mode='{mode}', top_n={top_n}, region_filter={region_filter}"
        )

        @agent.output_validator
        async def _inject_cached_output(
            ctx: RunContext[ChatDeps], output: FileAgentOutput
        ) -> FileAgentOutput:
            cached = getattr(ctx.deps, "keywords_final_message", None)
            if cached:
                output.message = cached
            output.files = []
            output.jsons = []
            return output

        @agent.tool
        async def keyword_trends_exec_tool(
            ctx: RunContext[ChatDeps],
            keywords: Annotated[List[str], Field(description="Keywords to analyze (1-5)")],
            timeframe: Annotated[str, Field(description="Pytrends timeframe")] = "today 3-m",
            geo: Annotated[str, Field(description="ISO-3166-1 alpha-2 code or empty for worldwide")] = "US",
            include_related: Annotated[bool, Field(description="Include related queries/topics")] = True,
            mode: Annotated[str, Field(description="quick or full")] = "quick",
            top_n: Annotated[int, Field(description="Top items to return in quick mode")] = 10,
            region_filter: Annotated[Optional[str], Field(description="Optional region/city/state filter, e.g., 'NJ' or 'New Jersey'")] = None,
        ) -> str:
            """
            Execute Google Trends lookups and return a structured analysis.
            """
            ctx.deps.status_queue.put_nowait("Researching keyword trends")
            ctx.deps.keywords_tool_used = True

            def _cache(report: str) -> str:
                ctx.deps.keywords_final_message = report
                return "ok"

            logger.info(
                f"Keywords exec: keywords={keywords}, timeframe='{timeframe}', geo='{geo}', mode='{mode}', top_n={top_n}, region_filter={region_filter}"
            )

            if not keywords:
                # Always return both sections, even if input is incomplete
                lines: List[str] = []
                lines.append("Keywords analysis")
                lines.append("No keywords provided. Add 1–3 seed terms to analyze.")
                lines.append("\nRecommendations")
                lines.append(
                    "- Provide at least one seed keyword (e.g., your category or hero product)."
                )
                lines.append(
                    "- Optionally include a location to tailor results (e.g., 'NJ')."
                )
                return _cache("\n".join(lines))

            # Bound sizes for speed
            if len(keywords) > 5:
                keywords = keywords[:5]
            seed = keywords[0]

            try:
                from pytrends.request import TrendReq  # type: ignore
            except Exception as e:
                logger.error(f"pytrends import failed: {e}")
                lines: List[str] = []
                lines.append("Keywords analysis")
                lines.append(
                    "Service temporarily unavailable. Unable to fetch Google Trends data."
                )
                lines.append("\nRecommendations")
                lines.append("- Try again shortly or adjust timeframe (e.g., 'today 3-m').")
                lines.append(
                    "- Share 1–3 seed terms and an optional location (e.g., 'IPA in NJ')."
                )
                return _cache("\n".join(lines))

            timeout_s = (
                float(os.getenv("PYTRENDS_TOOL_TIMEOUT_SEC", "10.0"))
                if mode == "quick"
                else float(os.getenv("PYTRENDS_TOOL_TIMEOUT_SEC", "20.0"))
            )

            def _safe_int(v: Any) -> int:
                try:
                    return int(v)
                except Exception:
                    return 0

            def _share_of_search(
                latest_row: pd.Series, keys: List[str]
            ) -> List[Dict[str, float]]:
                total = float(sum(latest_row.get(k, 0) for k in keys)) or 1.0
                return [
                    {"keyword": k, "share": float(latest_row.get(k, 0)) / total}
                    for k in keys
                ]

            def fast_flow() -> Dict[str, Any]:
                api_key = os.getenv("ZYTE_API_KEY")
                verify_path = certifi.where()
                requests_args = {"verify": verify_path}
                if api_key:
                    zyte = f"http://{api_key}:@api.zyte.com:8011"
                    requests_args["proxies"] = {"http": zyte, "https": zyte}

                py = TrendReq(hl="en-US", tz=0, requests_args=requests_args)

                out: Dict[str, Any] = {
                    "quick": {
                        "seed": str(seed),
                        "top_rising": [],
                        "top_related": [],
                        "suggested_keywords": [],
                        "share_of_search_now": [],
                        "summary": "",
                        "seed_interest_now": 0,
                        "seed_interest_recent_avg": 0,
                        "seed_interest_change": 0,
                        "seed_interest_change_percent": 0.0,
                        "top_regions": [],
                        "top_keywords_now": [],
                        "regions_total": 0,
                        "region_filter": region_filter or "",
                        "region_filter_match": "",
                        "region_filter_value": None,
                        "region_filter_rank": None,
                        "region_filter_percentile": None,
                        "num_top_rising": 0,
                        "num_top_related": 0,
                        "suggested_keywords_count": 0,
                    },
                    "diagnostics": {"geo": geo, "timeframe": timeframe, "notes": []},
                }

                # 1) Build for seed only
                try:
                    # Use state-level geo if region_filter is a US state; else fallback to provided geo/US
                    state_code = _state_code_for(region_filter)
                    geo_for_payload = f"US-{state_code}" if state_code else (geo or "US")
                    py.build_payload([seed], timeframe=timeframe, geo=geo_for_payload)
                except Exception as e:
                    out["diagnostics"]["notes"].append(f"build_payload_error: {e}")
                    return out

                # 1a) Seed interest snapshot and top regions
                try:
                    iot_seed = py.interest_over_time()
                    if iot_seed is not None and not iot_seed.empty:
                        if "isPartial" in iot_seed.columns:
                            iot_seed = iot_seed.drop(columns=["isPartial"])
                        # current value and short moving average (last up to 8 points)
                        latest_row = iot_seed.iloc[-1]
                        out["quick"]["seed_interest_now"] = int(
                            _safe_int(latest_row.get(seed, 0))
                        )
                        window = min(8, len(iot_seed))
                        recent_avg = (
                            float(iot_seed[seed].tail(window).astype(float).mean())
                            if seed in iot_seed.columns
                            else 0.0
                        )
                        now_val = float(out["quick"]["seed_interest_now"])
                        out["quick"]["seed_interest_recent_avg"] = int(recent_avg)
                        out["quick"]["seed_interest_change"] = int(now_val - recent_avg)
                        try:
                            out["quick"]["seed_interest_change_percent"] = round(
                                ((now_val - recent_avg) / (recent_avg or 1.0)) * 100.0, 1
                            )
                        except Exception:
                            out["quick"]["seed_interest_change_percent"] = 0.0
                except Exception as e:
                    out["diagnostics"]["notes"].append(f"seed_interest_error: {e}")

                try:
                    # Build US view to compute state-level ranking and top regions
                    py.build_payload([seed], timeframe=timeframe, geo="US")
                    ibr_seed = py.interest_by_region(resolution="REGION")
                    if (
                        ibr_seed is not None
                        and not ibr_seed.empty
                        and seed in ibr_seed.columns
                    ):
                        out["quick"]["regions_total"] = int(len(ibr_seed))
                        ibr_sorted = ibr_seed.sort_values(by=seed, ascending=False).head(5)
                        top_regions: List[Dict[str, int]] = []
                        for idx, row in ibr_sorted.iterrows():
                            try:
                                top_regions.append(
                                    {
                                        "geo": str(idx),
                                        "value": int(_safe_int(row.get(seed, 0))),
                                    }
                                )
                            except Exception:
                                continue
                        out["quick"]["top_regions"] = top_regions
                        # Specific region filter lookup
                        if region_filter:
                            target = _normalize_region_name(region_filter)
                            try:
                                full_sorted = ibr_seed.sort_values(by=seed, ascending=False)
                                if target in full_sorted.index:
                                    val = int(_safe_int(full_sorted.loc[target, seed]))
                                    out["quick"]["region_filter_match"] = str(target)
                                    out["quick"]["region_filter_value"] = val
                                    rank = int(list(full_sorted.index).index(target) + 1)
                                    out["quick"]["region_filter_rank"] = rank
                                    n = max(1, len(full_sorted))
                                    out["quick"]["region_filter_percentile"] = round(
                                        100.0 * (1.0 - (rank - 1) / n), 1
                                    )
                                else:
                                    # Best-effort fuzzy contains match
                                    candidates = [
                                        name
                                        for name in full_sorted.index
                                        if str(name).lower() == target.lower()
                                        or target.lower() in str(name).lower()
                                    ]
                                    if candidates:
                                        best = candidates[0]
                                        val = int(_safe_int(full_sorted.loc[best, seed]))
                                        out["quick"]["region_filter_match"] = str(best)
                                        out["quick"]["region_filter_value"] = val
                                        rank = int(list(full_sorted.index).index(best) + 1)
                                        out["quick"]["region_filter_rank"] = rank
                                        n = max(1, len(full_sorted))
                                        out["quick"]["region_filter_percentile"] = round(
                                            100.0 * (1.0 - (rank - 1) / n), 1
                                        )
                                    else:
                                        out["diagnostics"]["notes"].append(
                                            f"region_not_found:{region_filter}"
                                        )
                            except Exception as e:
                                out["diagnostics"]["notes"].append(
                                    f"region_lookup_error:{e}"
                                )
                except Exception as e:
                    out["diagnostics"]["notes"].append(f"seed_regions_error: {e}")

                # 2) Related queries for seed
                try:
                    rq = py.related_queries() or {}
                    block = rq.get(seed) or {}
                    rising_df = block.get("rising")
                    top_df = block.get("top")
                    if rising_df is not None and not rising_df.empty:
                        rows = [
                            {"query": str(q), "value": _safe_int(v)}
                            for q, v in zip(rising_df["query"], rising_df["value"])
                        ]
                        out["quick"]["top_rising"] = rows[:top_n]
                        out["quick"]["num_top_rising"] = len(out["quick"]["top_rising"])
                    if top_df is not None and not top_df.empty:
                        rows = [
                            {"query": str(q), "value": _safe_int(v)}
                            for q, v in zip(top_df["query"], top_df["value"])
                        ]
                        out["quick"]["top_related"] = rows[:top_n]
                        out["quick"]["num_top_related"] = len(out["quick"]["top_related"])
                except Exception as e:
                    out["diagnostics"]["notes"].append(f"related_error: {e}")

                # Suggested keywords = first 5 from rising then top, dedup
                seen = set()
                suggested: List[str] = []
                for row in out["quick"]["top_rising"] + out["quick"]["top_related"]:
                    q = row["query"]
                    if q.lower() not in seen and q.lower() != str(seed).lower():
                        suggested.append(q)
                        seen.add(q.lower())
                    if len(suggested) >= 5:
                        break
                out["quick"]["suggested_keywords"] = suggested
                out["quick"]["suggested_keywords_count"] = len(suggested)

                # 3) Optional second quick call to get a share-of-search snapshot
                if suggested:
                    try:
                        focus = suggested[:5]
                        py.build_payload(focus, timeframe=timeframe, geo=geo or "")
                        iot = py.interest_over_time()
                        if iot is not None and not iot.empty:
                            if "isPartial" in iot.columns:
                                iot = iot.drop(columns=["isPartial"])
                            latest = iot.iloc[-1]
                            share_items = sorted(
                                _share_of_search(latest, focus),
                                key=lambda r: r["share"],
                                reverse=True,
                            )
                            for item in share_items:
                                try:
                                    item["share_percent"] = round(
                                        float(item.get("share", 0.0)) * 100.0, 1
                                    )
                                except Exception:
                                    item["share_percent"] = 0.0
                            out["quick"]["share_of_search_now"] = share_items
                            out["quick"]["top_keywords_now"] = [
                                r["keyword"] for r in share_items[:3]
                            ]
                    except Exception as e:
                        out["diagnostics"]["notes"].append(f"sos_error: {e}")

                # 4) Numeric-rich 1–2 line summary
                try:
                    leaders = out["quick"]["share_of_search_now"][:3]
                    leader_str = (
                        ", ".join(
                            [
                                f"{r['keyword']} {round(float(r.get('share_percent', 0.0)), 0):.0f}%"
                                for r in leaders
                            ]
                        )
                        if leaders
                        else ""
                    )
                    change_pct = round(
                        float(out["quick"].get("seed_interest_change_percent", 0.0)), 1
                    )
                    parts: List[str] = []
                    parts.append(
                        f"now {seed}: {out['quick']['seed_interest_now']} ({'+' if change_pct>=0 else ''}{change_pct}%)"
                    )
                    if out["quick"].get("region_filter_value") is not None:
                        rf = out["quick"]
                        parts.append(
                            f"{rf.get('region_filter_match')}: {rf.get('region_filter_value')} (rank {rf.get('region_filter_rank')} of {rf.get('regions_total')}, ~{rf.get('region_filter_percentile')}th pct)"
                        )
                    if leader_str:
                        parts.append(f"top now: {leader_str}")
                    parts.append(
                        f"{out['quick']['num_top_rising']} rising, {out['quick']['num_top_related']} related"
                    )
                    out["quick"]["summary"] = "; ".join([p for p in parts if p])
                except Exception:
                    out["quick"]["summary"] = f"Snapshot created for {seed}"

                # 5) Structured view (stable schema)
                try:
                    out["quick"]["structured"] = {
                        "overview": {
                            "seed": str(seed),
                            "geo": geo,
                            "timeframe": timeframe,
                            "mode": mode,
                        },
                        "metrics": {
                            "now": out["quick"].get("seed_interest_now"),
                            "recent_avg": out["quick"].get("seed_interest_recent_avg"),
                            "change": out["quick"].get("seed_interest_change"),
                            "change_percent": out["quick"].get(
                                "seed_interest_change_percent"
                            ),
                        },
                        "region": {
                            "regions_total": out["quick"].get("regions_total", 0),
                            "filter": out["quick"].get("region_filter"),
                            "match": out["quick"].get("region_filter_match"),
                            "value": out["quick"].get("region_filter_value"),
                            "rank": out["quick"].get("region_filter_rank"),
                            "percentile": out["quick"].get("region_filter_percentile"),
                            "top_regions": out["quick"].get("top_regions", []),
                        },
                        "share_of_search": {
                            "items": out["quick"].get("share_of_search_now", []),
                            "top_keywords": out["quick"].get("top_keywords_now", []),
                        },
                        "lists": {
                            "top_rising": out["quick"].get("top_rising", []),
                            "top_rising_count": out["quick"].get("num_top_rising", 0),
                            "top_related": out["quick"].get("top_related", []),
                            "top_related_count": out["quick"].get("num_top_related", 0),
                        },
                        "suggested": {
                            "keywords": out["quick"].get("suggested_keywords", []),
                            "count": out["quick"].get("suggested_keywords_count", 0),
                        },
                    }
                except Exception as e:
                    out["diagnostics"]["notes"].append(f"structured_build_error:{e}")

                return out

            try:
                if mode != "quick":
                    logger.info(
                        "Full mode not implemented; returning quick snapshot with diagnostics note"
                    )
                data = await asyncio.wait_for(
                    asyncio.to_thread(fast_flow), timeout=timeout_s
                )
            except Exception as e:
                logger.error(f"keywords_agent exec error: {e}")
                # Graceful two-section fallback
                lines: List[str] = []
                lines.append("Keywords analysis")
                lines.append("No snapshot available due to a temporary error.")
                lines.append("\nRecommendations")
                lines.append(
                    "- Retry with a single seed (e.g., 'IPA') and timeframe 'today 3-m'."
                )
                lines.append("- Add a region (e.g., 'NJ') for localized insight.")
                return _cache("\n".join(lines))

            # Build final plain-text response with two sections
            try:
                q = data.get("quick", {}) if isinstance(data, dict) else {}
                seed = q.get("seed") or (keywords[0] if keywords else "")
                change_pct = q.get("seed_interest_change_percent")
                now_val = q.get("seed_interest_now")
                regions = q.get("top_regions", []) or []
                leaders = q.get("share_of_search_now", []) or []
                rising_count = q.get("num_top_rising", 0)
                related_count = q.get("num_top_related", 0)
                suggested_count = q.get("suggested_keywords_count", 0)
                rf = q.get("region_filter_match")
                rf_val = q.get("region_filter_value")
                rf_rank = q.get("region_filter_rank")
                rf_total = q.get("regions_total")
                rf_pct = q.get("region_filter_percentile")

                lines: List[str] = []
                # Section 1: Keyword analysis
                lines.append("Keywords analysis")
                line1 = (
                    f"{seed}: {now_val} ("
                    + (
                        f"+{change_pct}%"
                        if isinstance(change_pct, (int, float))
                        and change_pct is not None
                        and change_pct >= 0
                        else f"{change_pct}%"
                    )
                    + ")."
                )
                lines.append(line1)
                if rf and rf_val is not None:
                    lines.append(
                        f"{rf}: {rf_val} (rank {rf_rank} of {rf_total}, ~{rf_pct}th pct)."
                    )
                if regions:
                    top_regions_str = ", ".join(
                        [f"{r.get('geo')} {r.get('value')}" for r in regions[:3]]
                    )
                    lines.append(f"Top regions: {top_regions_str}.")
                if leaders:
                    leaders_str = ", ".join(
                        [
                            f"{r.get('keyword')} {r.get('share_percent')}%"
                            for r in leaders[:3]
                        ]
                    )
                    lines.append(f"Share-of-search leaders: {leaders_str}.")
                lines.append(
                    f"Rising: {rising_count}, Related: {related_count}, Suggested: {suggested_count}."
                )

                # Section 2: Recommendations
                lines.append("\nRecommendations")
                recs: List[str] = []
                if rf and isinstance(rf_val, int) and rf_val > 0:
                    recs.append(
                        f"Prioritize {rf} in geo targeting; extend to top regions if budget allows."
                    )
                else:
                    recs.append(
                        "Focus spend where interest is highest; expand as performance holds."
                    )
                if leaders:
                    recs.append(
                        f"Lean into demand for {leaders[0].get('keyword')} while testing adjacent terms from rising queries."
                    )
                else:
                    recs.append(
                        "Test 2–3 adjacent keyword themes; keep one control and two variants."
                    )
                recs.append(
                    "Run a short learn phase (7–14 days) and promote top performer across your primary channels."
                )
                if q.get("suggested_keywords"):
                    sk = ", ".join(q.get("suggested_keywords")[:3])
                    recs.append(f"Next tests: {sk}.")
                for r in recs:
                    lines.append(f"- {r}")

                return _cache("\n".join(lines))
            except Exception:
                # Always return two sections on failure
                lines: List[str] = []
                lines.append("Keyword analysis")
                lines.append("No snapshot available due to a rendering error.")
                lines.append("\nRecommendations")
                lines.append("- Retry with clear seeds and timeframe; keep seeds to 1–3.")
                return _cache("\n".join(lines))

        return agent
