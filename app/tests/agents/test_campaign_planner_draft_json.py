from lucy.agents.campaign_planner_agent import build_campaign_draft_json_data


def test_build_campaign_draft_json_data_defaults():
    data = build_campaign_draft_json_data(user_id="u1", org_id="o1")

    assert data["user_id"] == "u1"
    assert data["org_id"] == "o1"
    assert data["campaign_status"] == "DRAFT"
    assert data["campaign_version"] == 0
    assert data["is_paid"] is False

    # Required-ish fields with safe defaults
    assert data["goal"] == "conversions"
    assert data["daily_budget_cents"] == 0
    assert data["total_budget_cents"] == 0
    assert data["budget_type"] == "per day"

    # Not required right now
    assert data["locations"] == []

    # Channel and platforms are optional — absent when not provided
    assert "campaign_channel" not in data
    assert "ad_platforms" not in data


def test_build_campaign_draft_json_data_per_day_budget():
    data = build_campaign_draft_json_data(
        user_id="u1",
        org_id="o1",
        goal="traffic",
        daily_budget_cents=5000,
        total_budget_cents=0,
        budget_type="per day",
        target=["new", "existing"],
        campaign_name="Draft by Lucy (test)",
        now_iso="2026-01-01T00:00:00+00:00",
    )

    assert data["goal"] == "traffic"
    assert data["daily_budget_cents"] == 5000
    assert data["total_budget_cents"] == 0
    assert data["budget_type"] == "per day"
    assert data["target"] == ["new", "existing"]
    assert data["campaign_name"] == "Draft by Lucy (test)"
    assert data["campaign_start"] == "2026-01-01T00:00:00+00:00"
    assert data["campaign_end"] == "2026-01-01T00:00:00+00:00"


def test_build_campaign_draft_json_data_total_budget():
    data = build_campaign_draft_json_data(
        user_id="u1",
        org_id="o1",
        goal="awareness",
        daily_budget_cents=0,
        total_budget_cents=500000,
        budget_type="total",
        target=["new"],
        campaign_name="Draft by Lucy (total)",
        now_iso="2026-01-01T00:00:00+00:00",
    )

    assert data["total_budget_cents"] == 500000
    assert data["daily_budget_cents"] == 0
    assert data["budget_type"] == "total"


def test_build_campaign_draft_json_data_with_channel_and_platforms():
    data = build_campaign_draft_json_data(
        user_id="u1",
        org_id="o1",
        goal="awareness",
        daily_budget_cents=500,
        total_budget_cents=0,
        budget_type="per day",
        target=["new"],
        campaign_channel="social",
        ad_platforms=["facebook", "instagram"],
    )

    assert data["goal"] == "awareness"
    assert data["campaign_channel"] == "social"
    assert data["ad_platforms"] == ["facebook", "instagram"]


def test_build_campaign_draft_json_data_channel_without_platforms():
    data = build_campaign_draft_json_data(
        user_id="u1",
        org_id="o1",
        campaign_channel="search",
    )

    assert data["campaign_channel"] == "search"
    assert "ad_platforms" not in data


def test_build_campaign_draft_json_data_platforms_without_channel():
    data = build_campaign_draft_json_data(
        user_id="u1",
        org_id="o1",
        ad_platforms=["google"],
    )

    assert data["ad_platforms"] == ["google"]
    assert "campaign_channel" not in data
