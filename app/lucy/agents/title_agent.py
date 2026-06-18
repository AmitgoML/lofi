from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.model_config import Models


def create_title_agent(model_name: str = Models.TITLE) -> Agent:
    """
    Create a deterministic agent specialized in producing short, human-friendly titles.
    """
    logger.info(f"Creating title agent with model '{model_name}'")

    system_prompt = (
        "You are Lucy's title generator for conversations about advertising and marketing in the Lofi platform. "
        "Generate concise, human-friendly chat titles that reflect marketing/ads context (goals, channels, campaigns, optimization). "
        "Rules: max 6 words; sentence case; no quotes/backticks; no punctuation at the end; "
        "avoid emojis and marketing fluff; return title only with no extra text."
    )

    return Agent(
        model=model_name,
        system_prompt=system_prompt,
    )
