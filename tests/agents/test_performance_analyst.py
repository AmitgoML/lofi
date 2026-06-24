"""Unit tests for PerformanceAnalystAgent."""

from unittest.mock import MagicMock

import pytest

from lofi.agents.performance_analyst import PerformanceAnalystAgent
from lofi.schemas.common import BudgetSpec, CampaignGoal, CampaignTiming
from lofi.schemas.campaign_planner import CampaignPlannerInput
from lofi.schemas.performance_analyst import PerformanceAnalystInput


@pytest.fixture
def supabase_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def agent(supabase_client: MagicMock) -> PerformanceAnalystAgent:
    return PerformanceAnalystAgent(supabase_client)


@pytest.fixture
def analyst_input() -> PerformanceAnalystInput:
    return PerformanceAnalystInput(user_request="Run a campaign", brand="Acme", organization_id="org-1")


class TestRankPlatforms:
    def test_ranks_by_roas_descending(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {"platform": "meta", "roas": 4.0, "cost": 100, "ctr": 0.02, "impressions": 1000, "cpa": 10, "conversions": 10},
            {"platform": "google", "roas": 2.0, "cost": 100, "ctr": 0.03, "impressions": 1000, "cpa": 8, "conversions": 12},
        ]

        recommendations = agent._rank_platforms(rows)

        assert [r.platform.value for r in recommendations] == ["meta", "google"]
        assert recommendations[0].historical_roas == 4.0

    def test_cost_weights_roas_across_multiple_rows(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {"platform": "meta", "roas": 1.0, "cost": 900, "ctr": 0.01, "impressions": 1000},
            {"platform": "meta", "roas": 10.0, "cost": 100, "ctr": 0.01, "impressions": 1000},
        ]

        recommendations = agent._rank_platforms(rows)

        # weighted: (1*900 + 10*100) / 1000 = 1.9
        assert recommendations[0].historical_roas == pytest.approx(1.9)

    def test_unknown_platform_is_skipped(self, agent: PerformanceAnalystAgent) -> None:
        rows = [{"platform": "unknown-platform", "roas": 5.0, "cost": 100}]

        recommendations = agent._rank_platforms(rows)

        assert recommendations == []

    def test_empty_rows_returns_empty_list(self, agent: PerformanceAnalystAgent) -> None:
        assert agent._rank_platforms([]) == []


class TestRankLocations:
    def test_builds_location_from_joined_brand_location(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {
                "place_id": "place-1",
                "roas": 3.0,
                "cost": 100,
                "ctr": 0.02,
                "impressions": 1000,
                "brand_locations": {"name": "Downtown Store", "address": "123 Main St, Springfield, USA"},
            }
        ]

        recommendations = agent._rank_locations(rows)

        assert recommendations[0].location.city == "Downtown Store"
        assert recommendations[0].location.country == "USA"
        assert recommendations[0].historical_performance_score == 3.0

    def test_missing_address_falls_back_to_unknown_country(self, agent: PerformanceAnalystAgent) -> None:
        rows = [{"place_id": "place-1", "roas": 1.0, "cost": 100, "brand_locations": {}}]

        recommendations = agent._rank_locations(rows)

        assert recommendations[0].location.country == "unknown"


class TestRankAudiences:
    def test_combines_segment_type_and_segment(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {"segment_type": "age_range", "segment": "25-34", "roas": 2.0, "cost": 100, "ctr": 0.05, "impressions": 1000}
        ]

        recommendations = agent._rank_audiences(rows)

        assert recommendations[0].audience_segment == "age_range:25-34"

    def test_falls_back_to_ctr_when_no_conversions(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {"segment_type": "device", "segment": "mobile", "roas": None, "cost": 0, "ctr": 0.08, "impressions": 1000}
        ]

        recommendations = agent._rank_audiences(rows)

        assert recommendations[0].historical_performance_score == pytest.approx(0.08)


class TestRankCreatives:
    def test_groups_by_asset_type_via_join(self, agent: PerformanceAnalystAgent) -> None:
        rows = [
            {"creative_id": "c1", "ctr": 0.04, "impressions": 1000, "creative_assets": {"asset_type": "video"}},
            {"creative_id": "c2", "ctr": 0.01, "impressions": 1000, "creative_assets": {"asset_type": "image"}},
        ]

        recommendations = agent._rank_creatives(rows)

        assert recommendations[0].creative_format.value == "video"
        assert recommendations[0].historical_engagement_rate == 0.04

    def test_unrecognized_asset_type_is_skipped(self, agent: PerformanceAnalystAgent) -> None:
        rows = [{"creative_id": "c1", "ctr": 0.04, "impressions": 1000, "creative_assets": {"asset_type": "carousel"}}]

        assert agent._rank_creatives(rows) == []


class TestAnalyze:
    def test_reads_all_four_metric_sources_and_assembles_output(
        self, agent: PerformanceAnalystAgent, supabase_client: MagicMock, analyst_input: PerformanceAnalystInput
    ) -> None:
        supabase_client.get_platform_metrics.return_value = [
            {"platform": "meta", "roas": 3.0, "cost": 100, "ctr": 0.02, "impressions": 1000, "cpa": 5, "conversions": 5}
        ]
        supabase_client.get_location_metrics.return_value = []
        supabase_client.get_audience_metrics.return_value = []
        supabase_client.get_creative_metrics.return_value = []

        output = agent.analyze(analyst_input)

        supabase_client.get_platform_metrics.assert_called_once_with(analyst_input.organization_id, analyst_input.lookback_days)
        supabase_client.get_audience_metrics.assert_called_once_with(analyst_input.organization_id)
        assert len(output.platform_recommendations) == 1
        assert output.location_recommendations == []


class TestRun:
    def test_writes_performance_insights_into_state(
        self, agent: PerformanceAnalystAgent, supabase_client: MagicMock
    ) -> None:
        supabase_client.get_platform_metrics.return_value = []
        supabase_client.get_location_metrics.return_value = []
        supabase_client.get_audience_metrics.return_value = []
        supabase_client.get_creative_metrics.return_value = []

        brief = CampaignPlannerInput(
            user_request="Run a campaign",
            brand="Acme",
            organization_id="org-1",
            goal=CampaignGoal.AWARENESS,
            budget=BudgetSpec(total_budget=1000.0),
            campaign_timing=CampaignTiming(start_date="2026-07-01"),
        )
        state = {"user_request": "Run a campaign", "campaign_brief": brief}

        result_state = agent.run(state)

        assert "performance_insights" in result_state
        assert result_state["performance_insights"].platform_recommendations == []
