from lucy.agents.campaign_planner_agent import (
    CampaignPlannerAgent,
    CAMPAIGN_PLANNER_SYSTEM_PROMPT,
)


def test_campaign_planner_mentions_click_create_and_questions():
    header = CampaignPlannerAgent.reminder_header().lower()
    assert "click" in header
    assert "create" in header
    assert "up to two" in header
    # Profile tool must NOT be in the mandatory tool-sequence list in the header —
    # it is now only offered when profiles have not been attempted yet.
    assert "(1) get_user_org_profiles_tool" not in header
    assert "campaign_interview_tool" in header
    assert "campaign_draft_tool" in header
    # Anti-narration rule must be explicit in the header
    assert "never narrate" in header or "narrate" in header

    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    # Profile tool may still be mentioned in a conditional note but must not be step 1
    assert "campaign_interview_tool" in prompt
    assert "campaign_draft_tool" in prompt
    assert 'click "create"' in prompt
    assert "ask up to two" in prompt
    assert "one short paragraph" in prompt
    # Anti-narration instruction must be present
    assert "never narrate" in prompt or "do not say you are fetching" in prompt or "call tools silently" in prompt


def test_campaign_planner_required_fields_documented():
    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    assert "goal" in prompt
    assert "target" in prompt
    assert "budget" in prompt
    assert "channel" in prompt
    assert "platform" in prompt


def test_campaign_planner_tool_sequence_instructions():
    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    # Interview tool must run before draft tool
    interview_pos = prompt.index("campaign_interview_tool")
    draft_pos = prompt.index("campaign_draft_tool")
    assert interview_pos < draft_pos
