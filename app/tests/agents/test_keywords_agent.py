from types import SimpleNamespace

from lucy.agents.keywords_agent import KeywordsAgent


class TestKeywordsAgent:
    def test_select_tools_requires_keyword_tool_first(self):
        tool_defs = [
            SimpleNamespace(name="final_result"),
            SimpleNamespace(name="keyword_trends_exec_tool"),
        ]
        deps = SimpleNamespace()

        selected = KeywordsAgent._select_tools(deps, tool_defs)

        assert [tool.name for tool in selected] == ["keyword_trends_exec_tool"]

    def test_select_tools_switches_to_final_result_after_keyword_tool(self):
        tool_defs = [
            SimpleNamespace(name="final_result"),
            SimpleNamespace(name="keyword_trends_exec_tool"),
        ]
        deps = SimpleNamespace(keywords_tool_used=True)

        selected = KeywordsAgent._select_tools(deps, tool_defs)

        assert [tool.name for tool in selected] == ["final_result"]
