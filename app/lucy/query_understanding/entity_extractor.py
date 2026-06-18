from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.model_config import Models, to_responses_model
from lucy.query_understanding.models import ExtractedDetails

ENTITY_EXTRACTOR_SYSTEM_PROMPT = """
You are a helpful assistant that extracts structured advertising context from user messages.

Extract only what is explicitly stated or clearly implied. Leave fields null when unknown.

Fields:
- brand: company or product brand name
- platform: ad platform or channel (e.g. Meta, Google Ads, TikTok, Instagram)
- campaign: campaign name or descriptive label the user refers to
- budget: numeric budget amount in the user's currency (no currency symbols)
- date_range: natural-language or explicit date window (e.g. "last 30 days", "Q1 2025")
- additional_details: any other relevant key-value facts (metric names, goals, locations, etc.)

Do not invent data. Use additional_details for extra entities that do not fit the primary fields.
""".strip()


class EntityExtractionResult(BaseModel):
    brand: Optional[str] = None
    platform: Optional[str] = None
    campaign: Optional[str] = None
    budget: Optional[float] = None
    date_range: Optional[str] = None
    additional_details: Dict[str, Any] = Field(default_factory=dict)


_entity_extractor: Optional[Agent] = None


def _get_entity_extractor() -> Agent:
    """Return the entity-extraction agent, creating it on first call."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = Agent(
            model=to_responses_model(Models.AGENT_NANO),
            model_settings=ModelSettings(temperature=0.2, max_tokens=400),
            system_prompt=ENTITY_EXTRACTOR_SYSTEM_PROMPT,
            output_type=EntityExtractionResult,
        )
    return _entity_extractor


def _to_extracted_details(result: EntityExtractionResult) -> ExtractedDetails:
    return ExtractedDetails(
        brand=result.brand,
        platform=result.platform,
        campaign=result.campaign,
        budget=result.budget,
        date_range=result.date_range,
        additional_details=dict(result.additional_details),
    )


async def extract_entities(query: str) -> ExtractedDetails:
    """Extract advertising entities from a user query."""
    try:
        result = await _get_entity_extractor().run(query)
        extraction: EntityExtractionResult = result.output
        details = _to_extracted_details(extraction)
        logger.info("Extracted query entities: {}", details.model_dump())
        return details
    except Exception as exc:
        logger.exception("Entity extraction failed: {}", exc)
        return ExtractedDetails()
