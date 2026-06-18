"""Tests for LOFI_NAVIGATION_BLOCK scoping.

The navigation block teaches agents to embed internal Lofi app links in their
responses. It must only be present in the system prompts of agents that answer
general or operational questions (Lucy, Support). Specialist agents that produce
structured outputs (Campaign Planner, Creative Director, Performance Analyst)
should not include it, because navigation hints can corrupt their tightly-defined
output formats.
"""
from __future__ import annotations

import pytest

from lucy.agents.common.navigation import LOFI_NAVIGATION_BLOCK


# ---------------------------------------------------------------------------
# Agents expected to include navigation prompting
# ---------------------------------------------------------------------------

class TestNavigationIncludedInGeneralAgents:
    def test_lucy_system_prompt_includes_navigation_block(self):
        from lucy.agents.lucy_agent import LUCY_SYSTEM_PROMPT
        assert LOFI_NAVIGATION_BLOCK in LUCY_SYSTEM_PROMPT, (
            "LOFI_NAVIGATION_BLOCK must be present in LUCY_SYSTEM_PROMPT so Lucy "
            "can link users to the relevant Lofi app pages."
        )

    def test_support_system_prompt_includes_navigation_block(self):
        from lucy.agents.support_agent import SUPPORT_SYSTEM_PROMPT
        assert LOFI_NAVIGATION_BLOCK in SUPPORT_SYSTEM_PROMPT, (
            "LOFI_NAVIGATION_BLOCK must be present in SUPPORT_SYSTEM_PROMPT so the "
            "Support agent can guide users to the correct settings page."
        )


# ---------------------------------------------------------------------------
# Specialist agents must NOT include navigation prompting
# ---------------------------------------------------------------------------

class TestNavigationExcludedFromSpecialistAgents:
    def test_campaign_planner_system_prompt_excludes_navigation_block(self):
        from lucy.agents.campaign_planner_agent import CAMPAIGN_PLANNER_SYSTEM_PROMPT
        assert LOFI_NAVIGATION_BLOCK not in CAMPAIGN_PLANNER_SYSTEM_PROMPT, (
            "LOFI_NAVIGATION_BLOCK must NOT appear in CAMPAIGN_PLANNER_SYSTEM_PROMPT. "
            "Navigation hints can corrupt the strict JSON output format this agent produces."
        )

    def test_creative_director_orchestrator_prompt_excludes_navigation_block(self):
        from lucy.agents.creative_director_agent import CREATIVE_DIRECTOR_ORCHESTRATOR_PROMPT
        assert LOFI_NAVIGATION_BLOCK not in CREATIVE_DIRECTOR_ORCHESTRATOR_PROMPT, (
            "LOFI_NAVIGATION_BLOCK must NOT appear in CREATIVE_DIRECTOR_ORCHESTRATOR_PROMPT. "
            "The creative director produces structured deliverables, not product guidance."
        )

    def test_performance_analyst_system_prompt_excludes_navigation_block(self):
        from lucy.agents.performance_analyst_agent import PERFORMANCE_ANALYST_SYSTEM_PROMPT
        assert LOFI_NAVIGATION_BLOCK not in PERFORMANCE_ANALYST_SYSTEM_PROMPT, (
            "LOFI_NAVIGATION_BLOCK must NOT appear in PERFORMANCE_ANALYST_SYSTEM_PROMPT. "
            "Analyst output is focused on campaign data; navigation hints are distracting."
        )


# ---------------------------------------------------------------------------
# Navigation block content sanity checks
# ---------------------------------------------------------------------------

class TestNavigationBlockContent:
    def test_navigation_block_contains_core_routes(self):
        """Key Lofi routes must be present so agents can produce valid links."""
        required_routes = [
            "/campaigns",
            "/settings/brand",
            "/settings/ad-accounts",
            "/",
        ]
        for route in required_routes:
            assert route in LOFI_NAVIGATION_BLOCK, (
                f"Route '{route}' is missing from LOFI_NAVIGATION_BLOCK"
            )

    def test_navigation_block_contains_do_not_invent_instruction(self):
        """Agents must be told not to invent routes."""
        assert "do not invent" in LOFI_NAVIGATION_BLOCK.lower(), (
            "Navigation block must instruct agents not to invent routes"
        )

    def test_navigation_block_instructs_conditional_link_inclusion(self):
        """Agents should link only when relevant, not on every response."""
        block_lower = LOFI_NAVIGATION_BLOCK.lower()
        assert "only include a link" in block_lower or "when it is genuinely" in block_lower, (
            "Navigation block must instruct agents to include links conditionally, not always"
        )
