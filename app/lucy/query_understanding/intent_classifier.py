from __future__ import annotations

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.model_config import Models
from lucy.query_understanding.models import DEFAULT_INTENT, QueryIntent

INTENT_CLASSIFIER_SYSTEM_PROMPT = """
You are Lucy's query intent classifier for advertising and marketing questions.

Classify the user's message into exactly ONE intent:

- campaign_performance — Analysis of a single campaign's metrics, trends, anomalies,
  or optimization (ROAS, CPA, CTR, spend efficiency, "how is my campaign doing").
- campaign_comparison — Comparing, ranking, or benchmarking multiple campaigns
  ("which campaign is best", "compare my campaigns", "top performers").
- brand_analysis — Brand positioning, competitors, audience, tone, or brand-level
  strategy (not tied to a specific campaign's performance data).
- budget_analysis — Budget allocation, pacing, spend limits, reallocation, or
  ROI at the budget level ("$500 budget", "how should I split budget").
- creative_analysis — Creative assets, ad copy, visuals, briefs, or creative
  performance (not general image/video generation requests).
- general_query — Everything else: general marketing advice, platform help,
  keyword research, or ambiguous requests.

Return structured JSON with:
- intent: one of the supported intent labels above
- reasoning: brief explanation of why this intent was chosen

When uncertain, prefer general_query.
""".strip()


class IntentClassificationResult(BaseModel):
    intent: QueryIntent
    reasoning: str = Field(default="", description="Brief rationale for the chosen intent")


_intent_classifier: Optional[Agent] = None


def _get_intent_classifier() -> Agent:
    """Return the intent-classification agent, creating it on first call."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = Agent(
            model=Models.AGENT_FAST,
            model_settings=ModelSettings(temperature=0.0, max_tokens=300),
            system_prompt=INTENT_CLASSIFIER_SYSTEM_PROMPT,
            output_type=IntentClassificationResult,
            retries=1,
            output_retries=1,
        )
    return _intent_classifier


async def classify_intent(query: str) -> str:
    """Classify a user query into a supported intent label."""
    try:
        result = await _get_intent_classifier().run(query)
        classification: IntentClassificationResult = result.output
        logger.info(
            "Classified query intent: {} ({})",
            classification.intent,
            classification.reasoning,
        )
        return classification.intent
    except Exception as exc:
        logger.exception("Intent classification failed: {}", exc)
        return DEFAULT_INTENT
