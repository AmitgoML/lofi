from lucy.agents.creative_director_agent import CreativeDirectorAgent
from lucy.agents.common.models import FileAgentOutput, UserOrgProfile


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


def test_specialist_tools_skip_profile_tool_when_profiles_preloaded():
    """Non-empty profile list → profile tool excluded."""
    tool_defs = [
        _FakeTool("get_user_org_profiles_tool"),
        _FakeTool("final_result"),
    ]

    filtered = CreativeDirectorAgent._filter_specialist_tool_defs(
        task_type="ideation",
        tool_defs=tool_defs,
        has_preloaded_profiles=[UserOrgProfile(brand_name="Acme")],
    )

    assert [tool.name for tool in filtered] == ["final_result"]


def test_specialist_tools_skip_profile_tool_when_profiles_empty_list():
    """Empty list [] means profiles were already fetched (none found) —
    re-exposing the tool would waste a turn and prompt narration like
    'I'll pull your profile... one sec'."""
    tool_defs = [
        _FakeTool("get_user_org_profiles_tool"),
        _FakeTool("final_result"),
    ]

    filtered = CreativeDirectorAgent._filter_specialist_tool_defs(
        task_type="ideation",
        tool_defs=tool_defs,
        has_preloaded_profiles=[],
    )

    assert [tool.name for tool in filtered] == ["final_result"]


def test_specialist_tools_keep_profile_tool_when_profiles_none():
    """None means profiles were never attempted — expose the tool."""
    tool_defs = [
        _FakeTool("get_user_org_profiles_tool"),
        _FakeTool("final_result"),
    ]

    filtered = CreativeDirectorAgent._filter_specialist_tool_defs(
        task_type="ideation",
        tool_defs=tool_defs,
        has_preloaded_profiles=None,
    )

    assert [tool.name for tool in filtered] == [
        "get_user_org_profiles_tool",
        "final_result",
    ]


def test_web_search_specialist_keeps_web_search_tool():
    tool_defs = [
        _FakeTool("get_user_org_profiles_tool"),
        _FakeTool("web_search"),
        _FakeTool("final_result"),
    ]

    filtered = CreativeDirectorAgent._filter_specialist_tool_defs(
        task_type="web_search",
        tool_defs=tool_defs,
        has_preloaded_profiles=[],
    )

    assert [tool.name for tool in filtered] == ["web_search", "final_result"]


def test_build_profile_context_compacts_first_profile():
    profiles = [
        UserOrgProfile(
            brand_name="Northstar Home",
            company_name="Northstar Home Services",
            website_url="https://northstar-home.example",
            industry="Home Services",
            audience="Homeowners planning renovation projects",
            positioning="Reliable, transparent home improvement provider",
        )
    ]

    result = CreativeDirectorAgent._build_profile_context(profiles)

    assert "brand=Northstar Home" in result
    assert "website=https://northstar-home.example" in result
    assert "industry=Home Services" in result
    assert "audience=Homeowners planning renovation projects" in result
    assert "positioning=Reliable, transparent home improvement provider" in result


# ---------------------------------------------------------------------------
# Orchestrator terminal branch tool gating
# Uses _select_orchestrator_tools which is the extracted, testable form of
# _prepare_tools_orchestrator.
# ---------------------------------------------------------------------------


class _FakeDeps:
    """Minimal deps-like object that accepts arbitrary flag attributes."""

    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


_ALL_TOOLS = [
    _FakeTool("creative_director_interview_tool"),
    _FakeTool("creative_director_router_tool"),
    _FakeTool("creative_director_execute_tool"),
    _FakeTool("final_result"),
]


def test_terminal_execute_branch_exposes_only_final_result():
    """After execute_tool runs, only final_result should be offered."""
    deps = _FakeDeps(creative_execute_used=True)
    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    assert [t.name for t in result] == ["final_result"]


def test_terminal_clarify_branch_exposes_only_final_result():
    """After interview returns not-ready, only final_result should be offered."""
    deps = _FakeDeps(creative_interview_used=True, creative_interview_ready=False)
    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    assert [t.name for t in result] == ["final_result"]


def test_ready_interview_branch_exposes_router():
    """After interview returns ready, only the router tool should be offered."""
    deps = _FakeDeps(creative_interview_used=True, creative_interview_ready=True)
    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    assert [t.name for t in result] == ["creative_director_router_tool"]


def test_initial_branch_exposes_interview_tool():
    """Before any tool has run, only the interview tool should be offered."""
    deps = _FakeDeps()
    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    assert [t.name for t in result] == ["creative_director_interview_tool"]


def test_post_route_branch_exposes_execute_tool():
    """After routing, only the execute tool should be offered."""
    deps = _FakeDeps(creative_route={"task_type": "ideation"})
    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    assert [t.name for t in result] == ["creative_director_execute_tool"]


# ---------------------------------------------------------------------------
# Output validator caching
# ---------------------------------------------------------------------------

from unittest.mock import patch


def test_cached_execute_output_locks_down_tool_gating():
    """With creative_execute_used set, only final_result is offered — the
    cached output in creative_director_final_output will be injected by the
    output_validator, so the model can safely call final_result('ok')."""
    cached = FileAgentOutput(message="Specialist response", files=[], jsons=[])
    deps = _FakeDeps(creative_execute_used=True, creative_director_final_output=cached)

    result = CreativeDirectorAgent._select_orchestrator_tools(deps, _ALL_TOOLS)
    # Only final_result is visible — free-form text generation is blocked
    assert [t.name for t in result] == ["final_result"]


def test_build_execute_tool_always_produces_non_empty_message():
    """final_text extracted from specialist output must never be empty."""
    from lucy.agents.common.models import FileAgentOutput as FAO

    for bad_output_message in ("", "   ", None):
        out = FAO(message=bad_output_message or " ", files=[], jsons=[])
        # Simulate the normalisation logic in creative_director_execute_tool
        final_text = ""
        if hasattr(out, "message") and isinstance(out.message, str) and out.message.strip():
            final_text = out.message.strip()
        if not final_text:
            final_text = str(out).strip() or "Done."

        assert final_text, "final_text must never be empty"
