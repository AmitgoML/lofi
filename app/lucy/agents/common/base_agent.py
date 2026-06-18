from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from lucy.agents.common.models import ChatDeps


@dataclass
class PreRunResult:
    """Returned by pre_run_check to short-circuit the agent run."""
    message: str


class LofiAgent(ABC):
    REMINDER_HEADER: str = ""

    def create(self, model_name):
        """
        Create the agent.
        """
        pass

    @classmethod
    def reminder_header(cls) -> str:
        """Return the per-turn reminder header injected into every agent turn."""
        return cls.REMINDER_HEADER

    @classmethod
    async def pre_run_check(cls, deps: "ChatDeps") -> Optional[PreRunResult]:
        """Run a lightweight check before the agent executes.

        Return a PreRunResult to bypass the agent entirely (e.g. to ask a
        clarifying question), or None to proceed with the normal agent run.
        """
        return None

    @classmethod
    async def get_streaming_agent(cls, deps: "ChatDeps") -> "Optional[Tuple[Agent, str]]":
        """Return a (agent, prompt) tuple to stream directly, bypassing the default agent.

        Called after pre_run_check returns None (i.e. the agent is ready to run).
        When a tuple is returned, chat.py will stream that agent with the given prompt
        instead of the default agent returned by create().
        Return None to use the default agent (the normal path for all agents).
        """
        return None
