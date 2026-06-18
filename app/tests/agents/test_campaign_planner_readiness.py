from lucy.agents.campaign_planner_agent import (
    CAMPAIGN_PLANNER_SYSTEM_PROMPT,
    CampaignPlannerAgent,
)


def test_campaign_planner_public_contract_tool_order():
    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    assert "campaign_interview_tool" in prompt
    assert "campaign_draft_tool" in prompt
    # Profile tool is mentioned conditionally (not as mandatory step 1).
    # Confirm interview comes before draft in the sequence.
    interview_pos = prompt.index("campaign_interview_tool")
    draft_pos = prompt.index("campaign_draft_tool")
    assert interview_pos < draft_pos
    # Reminder header must NOT list profile tool as a mandatory first step
    header = CampaignPlannerAgent.reminder_header().lower()
    assert "(1) get_user_org_profiles_tool" not in header
    assert "campaign_interview_tool" in header
    assert "campaign_draft_tool" in header


def test_campaign_planner_public_contract_personalization_and_fallback():
    # Personalization guidance is in the header (agent reminders).
    header = CampaignPlannerAgent.reminder_header().lower()
    assert "personalize" in header or "personalization" in header or "brand context" in header
    assert "general best practices" in header


def test_campaign_planner_public_contract_tone_and_ui_instructions():
    header = CampaignPlannerAgent.reminder_header().lower()
    assert "ask up to two" in header
    assert "click" in header
    assert "create" in header


def test_campaign_planner_two_phase_workflow_described():
    prompt = CAMPAIGN_PLANNER_SYSTEM_PROMPT.lower()
    assert "phase 1" in prompt
    assert "phase 2" in prompt
    assert "interview" in prompt
    assert "execution" in prompt
