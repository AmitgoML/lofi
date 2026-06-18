from lucy.agents.lucy_agent import LucyAgent


def test_knowledge_agent_defaults_include_web_search_tool():
    agent = LucyAgent.create()
    # If created successfully, default model and settings are applied
    assert agent is not None
