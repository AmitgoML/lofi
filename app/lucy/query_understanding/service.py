from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from lucy.query_understanding.context_retriever import retrieve_context
from lucy.query_understanding.entity_extractor import extract_entities
from lucy.query_understanding.intent_classifier import classify_intent
from lucy.query_understanding.models import DEFAULT_INTENT, ExtractedDetails, QueryContext


async def build_query_context(
    query: str,
    user_id: Optional[str] = None,
) -> QueryContext:
    """Classify intent, extract entities, and optionally retrieve context."""
    intent = DEFAULT_INTENT
    extracted_details = ExtractedDetails()
    retrieved_context: Dict[str, Any] = {}

    try:
        intent = await classify_intent(query)
        logger.info("Query understanding intent: {}", intent)
    except Exception as exc:
        logger.exception("Query understanding intent classification error: {}", exc)

    try:
        extracted_details = await extract_entities(query)
        logger.info("Query understanding extracted details: {}", extracted_details.model_dump())
    except Exception as exc:
        logger.exception("Query understanding entity extraction error: {}", exc)

    if user_id:
        try:
            retrieved_context = await retrieve_context(user_id, intent, extracted_details)
        except Exception as exc:
            logger.exception("Query understanding context retrieval error: {}", exc)
    else:
        logger.info("Skipping context retrieval: no user_id provided")

    return QueryContext(
        intent=intent,
        extracted_details=extracted_details,
        retrieved_context=retrieved_context,
    )
