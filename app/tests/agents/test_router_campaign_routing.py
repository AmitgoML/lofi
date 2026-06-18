"""Tests for router campaign planning routing rules."""

from lucy.agents.router_agent import RouterAgent, ROUTER_SYSTEM_PROMPT
from tests.agents.helpers import make_assistant_history


# ---------------------------------------------------------------------------
# Prompt / contract tests
# ---------------------------------------------------------------------------

def test_router_prompt_contains_campaign_planner_route():
    """System prompt must describe the campaign_planner route."""
    prompt = ROUTER_SYSTEM_PROMPT.lower()
    assert "campaign_planner" in prompt
    assert "campaign planning" in prompt or "plan a campaign" in prompt


def test_router_prompt_contains_campaign_planner_exclusions():
    """System prompt must distinguish campaign_planner from support."""
    prompt = ROUTER_SYSTEM_PROMPT.lower()
    # campaign_planner excludes in-app setup (support)
    assert "support" in prompt
    assert "in-app" in prompt or "in lofi" in prompt or "inside lofi" in prompt


def test_router_reminder_header_includes_campaign_planner_in_targets():
    """Reminder header must list campaign_planner as a valid routing target."""
    header = RouterAgent.REMINDER_HEADER
    assert "campaign_planner" in header


def test_get_agent_class_for_route_campaign_planner():
    from lucy.agents.campaign_planner_agent import CampaignPlannerAgent
    from lucy.agents.common import factory as agent_factory
    cls = agent_factory._get_agent_class("campaign_planner")
    assert cls is CampaignPlannerAgent


def test_get_agent_for_name_campaign_planner():
    from lucy.agents.campaign_planner_agent import CampaignPlannerAgent
    route, agent_class, _ = RouterAgent.get_agent_for_name("campaign_planner")
    assert route == "campaign_planner"
    assert agent_class is CampaignPlannerAgent


# ---------------------------------------------------------------------------
# _detect_affirmation_context tests (Python-based context routing)
# ---------------------------------------------------------------------------

def test_affirmation_non_affirmation_returns_none():
    """Non-affirmation messages should not be intercepted."""
    history = make_assistant_history("Would you like help setting up in Lofi?")
    result = RouterAgent._detect_affirmation_context("tell me about campaigns", history)
    assert result is None


def test_affirmation_no_history_returns_none():
    """Affirmation with empty history should not be intercepted."""
    result = RouterAgent._detect_affirmation_context("yes", [])
    assert result is None


def test_affirmation_campaign_planner_draft_offer():
    """'yes' after assistant offers to build a campaign draft → campaign_planner."""
    history = make_assistant_history(
        "Based on your goal and budget I can build a campaign draft for you. Want me to go ahead?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "campaign_planner"


def test_affirmation_support_lofi_setup():
    """'yes' after assistant asks about setting up in Lofi → support."""
    history = make_assistant_history(
        "Would you like help setting up a social media campaign using Lofi?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "support"


def test_affirmation_support_walk_you_through():
    """'sure' after assistant offers to walk through in Lofi → support."""
    history = make_assistant_history(
        "Shall I walk you through the campaign setup step by step?"
    )
    result = RouterAgent._detect_affirmation_context("sure", history)
    assert result == "support"


def test_affirmation_image_generation():
    """'yes' after assistant asks about image generation → image."""
    history = make_assistant_history(
        "Would you like me to generate an image for your campaign?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "image"


def test_affirmation_video_generation():
    """'yes' after assistant asks about video generation → video."""
    history = make_assistant_history(
        "Would you like to create a video for your social media campaign?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "video"


def test_affirmation_performance():
    """'yes' after assistant asks about campaign performance → performance."""
    history = make_assistant_history(
        "Would you like me to analyze your campaign performance and ROAS?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "performance"


def test_affirmation_keywords():
    """'yes' after assistant asks about keywords → keywords."""
    history = make_assistant_history(
        "Would you like me to find more keywords for your campaign?"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result == "keywords"


def test_affirmation_case_insensitive():
    """Affirmations should be matched case-insensitively."""
    history = make_assistant_history(
        "Would you like help setting up your campaign in Lofi?"
    )
    for affirmation in ("YES", "Yes", "YEP", "OK", "Sure"):
        result = RouterAgent._detect_affirmation_context(affirmation, history)
        assert result == "support", f"Expected support for '{affirmation}'"


def test_affirmation_go_ahead():
    """'go ahead' is a valid affirmation."""
    history = make_assistant_history(
        "I can build a campaign draft for you — want me to go ahead?"
    )
    result = RouterAgent._detect_affirmation_context("go ahead", history)
    assert result == "campaign_planner"


def test_affirmation_unclear_context_returns_none():
    """If the last assistant message is generic, return None and let the LLM decide."""
    history = make_assistant_history(
        "That's a great question about your marketing strategy!"
    )
    result = RouterAgent._detect_affirmation_context("yes", history)
    assert result is None


# ---------------------------------------------------------------------------
# _detect_continuation topic-shift tests for campaign_planner phrases
# ---------------------------------------------------------------------------

def test_topic_shift_draft_me_a_campaign_triggers_campaign_planner():
    """`draft me a campaign` must break continuation and route to campaign_planner."""
    result = RouterAgent._detect_continuation("draft me a campaign", last_agent="lucy")
    # topic-shift keyword present → should NOT return last_agent (None triggers full router)
    assert result is None


def test_topic_shift_create_a_campaign_triggers_campaign_planner():
    """`create a campaign` must break continuation."""
    result = RouterAgent._detect_continuation("create a campaign for dating app", last_agent="lucy")
    assert result is None


def test_topic_shift_build_a_campaign():
    """`build a campaign` must break continuation."""
    result = RouterAgent._detect_continuation("build a campaign with a $500 budget", last_agent="support")
    assert result is None


def test_router_prompt_includes_draft_campaign_example():
    """Router system prompt should show 'draft me a campaign' → campaign_planner."""
    prompt = ROUTER_SYSTEM_PROMPT.lower()
    assert "draft me a campaign" in prompt or "draft a campaign" in prompt


def test_router_prompt_includes_generate_campaign():
    """Router system prompt should cover 'generate a campaign' → campaign_planner."""
    prompt = ROUTER_SYSTEM_PROMPT.lower()
    assert "generate a campaign" in prompt or "generate" in prompt


# ---------------------------------------------------------------------------
# Topic-shift keywords: make and generate a campaign
# ---------------------------------------------------------------------------

def test_topic_shift_make_a_campaign():
    """`make a campaign` must break continuation regardless of last_agent."""
    result = RouterAgent._detect_continuation("make a campaign for our summer sale", last_agent="lucy")
    assert result is None


def test_topic_shift_generate_a_campaign():
    """`generate a campaign` must break continuation and trigger the router."""
    result = RouterAgent._detect_continuation("generate a campaign for my app", last_agent="lucy")
    assert result is None


def test_topic_shift_generate_campaign_from_support():
    """`generate a campaign` also breaks sticky support threads."""
    result = RouterAgent._detect_continuation("generate a campaign for my app", last_agent="support")
    assert result is None


# ---------------------------------------------------------------------------
# Lucy follow-up sticky routing regression
# When Lucy says "let me generate a campaign for you" and the user says "ok",
# the affirmation should route to campaign_planner, not stay on lucy.
# ---------------------------------------------------------------------------

def test_affirmation_after_lucy_campaign_offer_routes_to_planner():
    """'ok let's generate a campaign' → campaign_planner.

    This is the exact flow observed in the bug report:
    - Lucy: 'Maybe we should make a campaign'
    - User: 'ok let's generate a campaign'
    The second message contains a topic-shift keyword, so continuation is broken
    and the router should map it to campaign_planner.
    """
    result = RouterAgent._detect_continuation("ok let's generate a campaign", last_agent="lucy")
    # "generate a campaign" is a topic-shift keyword → None (send to LLM router)
    assert result is None


def test_affirmation_after_lucy_campaign_offer_legacy_affirmation():
    """'ok let's generate a campaign' is NOT a pure affirmation, so legacy path skips it."""
    history = make_assistant_history(
        "I can build a campaign draft for you — want me to go ahead?"
    )
    # Not a short affirmation, so _detect_affirmation_context returns None
    result = RouterAgent._detect_affirmation_context("ok let's generate a campaign", history)
    assert result is None  # not a short affirmation; full LLM router decides
