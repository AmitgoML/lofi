"""Tests for the campaign planner extraction / coercion pipeline.

These tests verify that realistic LLM outputs for budget, channel, and
platform values correctly survive the coerce → merge → draft pipeline.
"""
import pytest

from lucy.agents.campaign_planner_agent import (
    CampaignPlanningState,
    _coerce_budget_cents,
    _coerce_budget_type,
    _coerce_channel,
    _coerce_int,
    _coerce_str_list,
    build_campaign_draft_json_data,
)


# ---------------------------------------------------------------------------
# _coerce_int — general integer parsing (unchanged helper)
# ---------------------------------------------------------------------------


class TestCoerceInt:
    def test_plain_integer(self):
        assert _coerce_int(5000) == 5000

    def test_float_truncates(self):
        assert _coerce_int(50.9) == 50

    def test_digit_string(self):
        assert _coerce_int("5000") == 5000

    def test_dollar_sign_stripped(self):
        # _coerce_int strips $ but does NOT multiply — use _coerce_budget_cents for budgets
        assert _coerce_int("$50") == 50

    def test_comma_separated(self):
        assert _coerce_int("5,000") == 5000

    def test_zero(self):
        assert _coerce_int(0) == 0

    def test_none_returns_none(self):
        assert _coerce_int(None) is None

    def test_bool_returns_none(self):
        assert _coerce_int(True) is None
        assert _coerce_int(False) is None

    def test_empty_string_returns_none(self):
        assert _coerce_int("") is None

    def test_non_numeric_string_returns_none(self):
        assert _coerce_int("unknown") is None


# ---------------------------------------------------------------------------
# _coerce_budget_cents — budget-specific coercion
# ---------------------------------------------------------------------------


class TestCoerceBudgetCents:
    def test_plain_cents_integer(self):
        assert _coerce_budget_cents(5000) == 5000

    def test_plain_cents_string(self):
        assert _coerce_budget_cents("5000") == 5000

    def test_dollar_sign_string_multiplied_to_cents(self):
        # LLM returns "$50" meaning $50 → should become 5000 cents
        assert _coerce_budget_cents("$50") == 5000

    def test_dollar_sign_with_decimal_multiplied(self):
        assert _coerce_budget_cents("$50.00") == 5000

    def test_dollar_sign_comma_formatted(self):
        # "$1,000" → 1000 * 100 = 100000 cents
        assert _coerce_budget_cents("$1,000") == 100000

    def test_dollar_sign_per_day_suffix(self):
        # "$50/day" — strip non-digits, detect $, multiply
        assert _coerce_budget_cents("$50/day") == 5000

    def test_zero_returns_none(self):
        # 0 means "not provided by the user"
        assert _coerce_budget_cents(0) is None

    def test_zero_string_returns_none(self):
        assert _coerce_budget_cents("0") is None

    def test_negative_returns_none(self):
        assert _coerce_budget_cents(-100) is None

    def test_none_returns_none(self):
        assert _coerce_budget_cents(None) is None

    def test_bool_returns_none(self):
        assert _coerce_budget_cents(True) is None
        assert _coerce_budget_cents(False) is None

    def test_empty_string_returns_none(self):
        assert _coerce_budget_cents("") is None

    def test_non_numeric_string_returns_none(self):
        assert _coerce_budget_cents("unknown") is None

    def test_float_truncates(self):
        assert _coerce_budget_cents(5000.9) == 5000

    def test_large_cents_value_unchanged(self):
        assert _coerce_budget_cents(100000) == 100000


# ---------------------------------------------------------------------------
# _coerce_budget_type — budget type coercion
# ---------------------------------------------------------------------------


class TestCoerceBudgetType:
    @pytest.mark.parametrize("value", ["per day", "Per Day", "PER DAY"])
    def test_canonical_per_day_case_insensitive(self, value):
        assert _coerce_budget_type(value) == "per day"

    @pytest.mark.parametrize("value", ["total", "Total", "TOTAL"])
    def test_canonical_total_case_insensitive(self, value):
        assert _coerce_budget_type(value) == "total"

    @pytest.mark.parametrize("alias,expected", [
        ("daily", "per day"),
        ("per_day", "per day"),
        ("per-day", "per day"),
        ("day", "per day"),
        ("lifetime", "total"),
        ("overall", "total"),
        ("lump sum", "total"),
        ("total budget", "total"),
    ])
    def test_aliases(self, alias, expected):
        assert _coerce_budget_type(alias) == expected

    def test_unknown_returns_none(self):
        assert _coerce_budget_type("monthly") is None

    def test_none_returns_none(self):
        assert _coerce_budget_type(None) is None

    def test_non_string_returns_none(self):
        assert _coerce_budget_type(123) is None


# ---------------------------------------------------------------------------
# _coerce_channel — canonical name + alias mapping
# ---------------------------------------------------------------------------


class TestCoerceChannel:
    @pytest.mark.parametrize("value", ["social", "Social", "SOCIAL"])
    def test_canonical_social_case_insensitive(self, value):
        assert _coerce_channel(value) == "social"

    @pytest.mark.parametrize("value", ["search", "Search"])
    def test_canonical_search(self, value):
        assert _coerce_channel(value) == "search"

    @pytest.mark.parametrize("value", ["display", "ctv", "shopping"])
    def test_canonical_other_channels(self, value):
        assert _coerce_channel(value) == value

    def test_canonical_digital_audio_underscore(self):
        # Frontend enum uses underscore
        assert _coerce_channel("digital_audio") == "digital_audio"

    def test_digital_audio_space_variant_maps_to_underscore(self):
        assert _coerce_channel("digital audio") == "digital_audio"

    @pytest.mark.parametrize("alias,expected", [
        ("social media", "social"),
        ("paid social", "social"),
        ("social ads", "social"),
        ("paid search", "search"),
        ("sem", "search"),
        ("ppc", "search"),
        ("google ads", "search"),
        ("connected tv", "ctv"),
        ("streaming tv", "ctv"),
        ("ott", "ctv"),
        ("audio", "digital_audio"),
        ("podcast", "digital_audio"),
        ("streaming audio", "digital_audio"),
        ("google shopping", "shopping"),
        ("product listing", "shopping"),
    ])
    def test_aliases_map_to_canonical(self, alias, expected):
        assert _coerce_channel(alias) == expected

    def test_unknown_value_returns_none(self):
        assert _coerce_channel("email marketing") is None

    def test_none_returns_none(self):
        assert _coerce_channel(None) is None

    def test_non_string_returns_none(self):
        assert _coerce_channel(123) is None


# ---------------------------------------------------------------------------
# End-to-end: extraction result → CampaignPlanningState → draft JSON
# ---------------------------------------------------------------------------


class TestExtractionToDraftPipeline:
    """Simulate what happens after _openai_enough_context returns values and
    they are coerced then merged into CampaignPlanningState."""

    def _simulate_coerce_and_merge(self, raw: dict) -> CampaignPlanningState:
        """Mirrors the coercion block in campaign_interview_tool."""
        suggested = {
            "goal": raw.get("goal"),
            "budget_type": _coerce_budget_type(raw.get("budget_type")),
            "daily_budget_cents": _coerce_budget_cents(raw.get("daily_budget_cents")),
            "total_budget_cents": _coerce_budget_cents(raw.get("total_budget_cents")),
            "target": _coerce_str_list(raw.get("target")),
            "campaign_channel": _coerce_channel(raw.get("campaign_channel")),
            "ad_platforms": _coerce_str_list(raw.get("ad_platforms")),
        }
        state = CampaignPlanningState()
        state.merge({k: v for k, v in suggested.items() if v is not None})
        return state

    def test_realistic_per_day_campaign(self):
        """LLM returns cents correctly with per-day budget type."""
        raw = {
            "goal": "conversions",
            "target": ["new"],
            "budget_type": "per day",
            "daily_budget_cents": 5000,
            "campaign_channel": "social media",  # alias
            "ad_platforms": ["facebook", "instagram"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.is_ready_to_draft
        assert state.campaign_channel == "social"
        assert state.budget_type == "per day"
        assert state.daily_budget_cents == 5000
        assert state.total_budget_cents is None

    def test_realistic_total_budget_campaign(self):
        """LLM returns a total budget type with correct cents."""
        raw = {
            "goal": "awareness",
            "target": ["new"],
            "budget_type": "total",
            "total_budget_cents": 500000,
            "campaign_channel": "search",
            "ad_platforms": ["google"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.is_ready_to_draft
        assert state.budget_type == "total"
        assert state.total_budget_cents == 500000
        assert state.daily_budget_cents is None

    def test_dollar_sign_per_day_budget_converted_to_cents(self):
        """LLM returns dollar-sign strings — _coerce_budget_cents multiplies by 100."""
        raw = {
            "goal": "awareness",
            "target": ["new", "existing"],
            "budget_type": "per day",
            "daily_budget_cents": "$50",   # dollar-sign string → 5000 cents
            "campaign_channel": "search",
            "ad_platforms": ["google"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.is_ready_to_draft
        assert state.daily_budget_cents == 5000

    def test_dollar_sign_total_budget_converted_to_cents(self):
        """LLM returns dollar-sign total budget string."""
        raw = {
            "goal": "traffic",
            "target": ["new"],
            "budget_type": "total",
            "total_budget_cents": "$1,000",  # → 100000 cents
            "campaign_channel": "display",
            "ad_platforms": ["google"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.is_ready_to_draft
        assert state.total_budget_cents == 100000

    def test_zero_budget_treated_as_not_provided(self):
        """LLM returning 0 for a budget field means it wasn't specified."""
        raw = {
            "goal": "conversions",
            "target": ["new"],
            "budget_type": "per day",
            "daily_budget_cents": 0,   # treated as "not provided"
            "total_budget_cents": 0,
            "campaign_channel": "social",
            "ad_platforms": ["facebook"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.daily_budget_cents is None
        assert state.total_budget_cents is None
        assert not state.is_ready_to_draft
        assert "daily_budget" in state.missing_fields

    def test_paid_search_channel_alias(self):
        raw = {
            "goal": "traffic",
            "target": ["new"],
            "budget_type": "per day",
            "daily_budget_cents": 2000,
            "campaign_channel": "paid search",
            "ad_platforms": ["google", "bing"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.campaign_channel == "search"
        assert state.is_ready_to_draft

    def test_draft_json_populated_from_per_day_state(self):
        """Verify per-day budget state reaches build_campaign_draft_json_data correctly."""
        state = CampaignPlanningState(
            goal="conversions",
            target=["new"],
            budget_type="per day",
            daily_budget_cents=5000,
            campaign_channel="social",
            ad_platforms=["facebook", "instagram"],
        )

        draft = build_campaign_draft_json_data(
            user_id="u1",
            org_id="o1",
            goal=state.goal,
            daily_budget_cents=state.daily_budget_cents,
            total_budget_cents=0,
            budget_type=state.budget_type,
            target=state.target,
            campaign_channel=state.campaign_channel,
            ad_platforms=state.ad_platforms,
        )

        assert draft["goal"] == "conversions"
        assert draft["daily_budget_cents"] == 5000
        assert draft["total_budget_cents"] == 0
        assert draft["budget_type"] == "per day"
        assert draft["campaign_channel"] == "social"
        assert draft["ad_platforms"] == ["facebook", "instagram"]
        assert draft["target"] == ["new"]
        assert draft["campaign_status"] == "DRAFT"

    def test_draft_json_populated_from_total_budget_state(self):
        """Verify total budget state reaches build_campaign_draft_json_data correctly."""
        state = CampaignPlanningState(
            goal="awareness",
            target=["new"],
            budget_type="total",
            total_budget_cents=500000,
            campaign_channel="search",
            ad_platforms=["google"],
        )

        draft = build_campaign_draft_json_data(
            user_id="u1",
            org_id="o1",
            goal=state.goal,
            daily_budget_cents=0,
            total_budget_cents=state.total_budget_cents,
            budget_type=state.budget_type,
            target=state.target,
            campaign_channel=state.campaign_channel,
            ad_platforms=state.ad_platforms,
        )

        assert draft["total_budget_cents"] == 500000
        assert draft["daily_budget_cents"] == 0
        assert draft["budget_type"] == "total"

    def test_partial_extraction_preserves_existing_state(self):
        """merge() must not overwrite non-None fields with None from a later turn."""
        state = CampaignPlanningState(goal="traffic", campaign_channel="social")
        state.merge({"goal": None, "budget_type": "per day", "daily_budget_cents": 3000})
        assert state.goal == "traffic"
        assert state.campaign_channel == "social"
        assert state.budget_type == "per day"
        assert state.daily_budget_cents == 3000

    def test_missing_channel_does_not_block_draft(self):
        """If channel is an unrecognised alias, core planning state can still be ready."""
        raw = {
            "goal": "conversions",
            "target": ["new"],
            "budget_type": "per day",
            "daily_budget_cents": 5000,
            "campaign_channel": "email marketing",  # unknown
            "ad_platforms": ["mailchimp"],
        }
        state = self._simulate_coerce_and_merge(raw)
        assert state.campaign_channel is None
        assert state.is_ready_to_draft
        assert "campaign_channel" not in state.missing_fields
