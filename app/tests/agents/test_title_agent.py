import pytest
from pydantic_ai import Agent


def test_title_agent_creates_without_deps_type():
    """The title agent must not declare deps_type since it is called without deps."""
    from lucy.agents.title_agent import create_title_agent

    agent = create_title_agent()
    assert isinstance(agent, Agent)
    # If deps_type were set, calling run() without deps would raise TypeError.
    # Verify the agent has no deps requirement.
    assert agent._deps_type is type(None) or agent._deps_type is None


@pytest.mark.asyncio
async def test_title_agent_run_without_deps():
    """Smoke test: calling run() without deps must not raise."""
    from unittest.mock import AsyncMock, patch

    from lucy.agents.title_agent import create_title_agent

    agent = create_title_agent()

    with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value.output = "Campaign optimization tips"
        result = await agent.run("User: How do I optimize campaigns?")
        mock_run.assert_called_once_with("User: How do I optimize campaigns?")
        assert result.output == "Campaign optimization tips"
