from lucy.agents.campaign_planner_agent import (
    CAMPAIGN_PLANNER_SYSTEM_PROMPT,
    CampaignPlannerAgent,
    CampaignPlanningState,
)
from lucy.agents.common.models import ChatDeps, UserOrgProfile


# ---------------------------------------------------------------------------
# Profile-tool gating regression tests
# Mirrors the pattern already in test_creative_director_agent.py.
# Uses _select_tools directly so no real Agent needs to be instantiated.
# ---------------------------------------------------------------------------

class _FakeTool:
    def __init__(self, name: str):
        self.name = name


_ALL_TOOLS = [
    _FakeTool("get_user_org_profiles_tool"),
    _FakeTool("campaign_interview_tool"),
    _FakeTool("campaign_draft_tool"),
]


def _deps(**kwargs) -> ChatDeps:
    """Build a ChatDeps with typed fields set directly."""
    deps = ChatDeps(user_id="test-user")
    for k, v in kwargs.items():
        setattr(deps, k, v)
    return deps


def test_profile_tool_skipped_when_profiles_preloaded():
    """Non-empty profiles list → profile tool must NOT be exposed.

    chat.py always preloads user_profiles before the agent runs, so this is
    the normal production path. The key fix: we now check user_profiles is None
    rather than user_profiles_loaded, so the tool is never unnecessarily offered.
    """
    deps = _deps(user_profiles=[UserOrgProfile(brand_name="Acme")])
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    names = [t.name for t in result]
    assert "get_user_org_profiles_tool" not in names
    assert "campaign_interview_tool" in names


def test_profile_tool_skipped_when_profiles_empty_list():
    """Empty list [] means already attempted — profile tool must NOT be re-exposed.

    This is the key regression: the old code used user_profiles_loaded (which
    chat.py never sets), so [] was treated as "not loaded" and the tool was offered,
    prompting the model to narrate 'Let me pull up your profile first...'
    """
    deps = _deps(user_profiles=[])
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    names = [t.name for t in result]
    assert "get_user_org_profiles_tool" not in names
    assert "campaign_interview_tool" in names


def test_profile_tool_exposed_when_profiles_none():
    """None means profiles were never attempted — tool should be offered."""
    deps = _deps(user_profiles=None)
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    names = [t.name for t in result]
    assert "get_user_org_profiles_tool" in names
    assert "campaign_interview_tool" not in names


def test_after_draft_all_tools_removed():
    """Once a draft is emitted, no tools should be offered."""
    deps = _deps(user_profiles=[], draft_emitted=True)
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    assert result == []


def test_interview_tool_offered_after_profiles_known():
    """With profiles preloaded and no interview yet, expose campaign_interview_tool."""
    deps = _deps(user_profiles=[UserOrgProfile(brand_name="Acme")])
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    names = [t.name for t in result]
    assert names == ["campaign_interview_tool"]


def test_draft_tool_offered_when_planning_complete():
    """After interview signals readiness, expose campaign_draft_tool."""
    deps = _deps(
        user_profiles=[],
        campaign_interview_used=True,
        planning_phase_complete=True,
    )
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    names = [t.name for t in result]
    assert names == ["campaign_draft_tool"]


def test_no_tools_when_interview_ran_but_not_ready():
    """If interview ran but not enough info, no tools exposed — model asks questions."""
    deps = _deps(
        user_profiles=[],
        campaign_interview_used=True,
        planning_phase_complete=False,
    )
    result = CampaignPlannerAgent._select_tools(deps, _ALL_TOOLS)
    assert result == []


def test_campaign_planner_minimum_info_threshold_in_prompt():
    """
    Asserts the *public contract* (prompted behavior) rather than internal helpers.
    The agent should require goal + budget + target + channel + platform before drafting,
    and explain draft params in plain language when presenting the draft.
    """
    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    assert "goal" in prompt
    assert "budget" in prompt
    assert "target" in prompt
    assert "channel" in prompt
    assert "platform" in prompt
    # Draft explanation guidance
    assert "make sense (goal, budgets, target)" in prompt
    assert "refinement question" in prompt or "refinement questions" in prompt


def test_campaign_planning_state_is_ready_to_draft_per_day():
    state = CampaignPlanningState(
        goal="conversions",
        target=["new"],
        budget_type="per day",
        daily_budget_cents=500,
        campaign_channel="social",
        ad_platforms=["facebook"],
    )
    assert state.is_ready_to_draft is True
    assert state.missing_fields == []


def test_campaign_planning_state_is_ready_to_draft_total_budget():
    state = CampaignPlanningState(
        goal="awareness",
        target=["new"],
        budget_type="total",
        total_budget_cents=100000,
        campaign_channel="search",
        ad_platforms=["google"],
    )
    assert state.is_ready_to_draft is True
    assert state.missing_fields == []


def test_campaign_planning_state_ready_missing_channel():
    state = CampaignPlanningState(
        goal="traffic",
        target=["new"],
        budget_type="per day",
        daily_budget_cents=200,
        campaign_channel=None,
        ad_platforms=["google"],
    )
    assert state.is_ready_to_draft is True
    assert "campaign_channel" not in state.missing_fields


def test_campaign_planning_state_ready_missing_platforms():
    state = CampaignPlanningState(
        goal="awareness",
        target=["existing"],
        budget_type="per day",
        daily_budget_cents=100,
        campaign_channel="search",
        ad_platforms=None,
    )
    assert state.is_ready_to_draft is True
    assert "ad_platforms" not in state.missing_fields


def test_campaign_planning_state_not_ready_missing_budget_type():
    """When budget_type is unknown, 'budget' should appear in missing_fields."""
    state = CampaignPlanningState(
        goal="conversions",
        target=["new"],
        budget_type=None,
        daily_budget_cents=None,
        total_budget_cents=None,
        campaign_channel="social",
        ad_platforms=["instagram"],
    )
    assert state.is_ready_to_draft is False
    assert "budget" in state.missing_fields


def test_campaign_planning_state_not_ready_missing_per_day_amount():
    """budget_type='per day' but no daily amount provided."""
    state = CampaignPlanningState(
        goal="conversions",
        target=["new"],
        budget_type="per day",
        daily_budget_cents=None,
        campaign_channel="social",
        ad_platforms=["instagram"],
    )
    assert state.is_ready_to_draft is False
    assert "daily_budget" in state.missing_fields


def test_campaign_planning_state_not_ready_missing_total_amount():
    """budget_type='total' but no total amount provided."""
    state = CampaignPlanningState(
        goal="conversions",
        target=["new"],
        budget_type="total",
        total_budget_cents=None,
        campaign_channel="social",
        ad_platforms=["instagram"],
    )
    assert state.is_ready_to_draft is False
    assert "total_budget" in state.missing_fields


def test_campaign_planning_state_per_day_ready_does_not_require_total():
    """Per-day budget type should be ready without total_budget_cents set."""
    state = CampaignPlanningState(
        goal="traffic",
        target=["new"],
        budget_type="per day",
        daily_budget_cents=5000,
        total_budget_cents=None,
        campaign_channel="social",
        ad_platforms=["facebook"],
    )
    assert state.is_ready_to_draft is True


def test_campaign_planning_state_total_budget_ready_does_not_require_daily():
    """Total budget type should be ready without daily_budget_cents set."""
    state = CampaignPlanningState(
        goal="traffic",
        target=["new"],
        budget_type="total",
        daily_budget_cents=None,
        total_budget_cents=200000,
        campaign_channel="search",
        ad_platforms=["google"],
    )
    assert state.is_ready_to_draft is True


def test_campaign_planning_state_merge_updates_only_non_none():
    state = CampaignPlanningState(goal="traffic", target=["new"])
    state.merge(
        {
            "goal": None,  # should NOT overwrite existing
            "budget_type": "per day",
            "daily_budget_cents": 300,
            "campaign_channel": "social",
            "ad_platforms": ["facebook", "instagram"],
        }
    )
    # Existing goal should be preserved since incoming value is None
    assert state.goal == "traffic"
    assert state.budget_type == "per day"
    assert state.daily_budget_cents == 300
    assert state.campaign_channel == "social"
    assert state.ad_platforms == ["facebook", "instagram"]
    assert state.is_ready_to_draft is True


def test_campaign_planning_state_missing_fields_empty_state():
    state = CampaignPlanningState()
    missing = state.missing_fields
    assert "goal" in missing
    assert "target" in missing
    assert "budget" in missing  # unknown budget type → single "budget" entry
    assert "campaign_channel" not in missing
    assert "ad_platforms" not in missing
    assert len(missing) == 3
