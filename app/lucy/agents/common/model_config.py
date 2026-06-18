from __future__ import annotations

import os
from typing import Optional

from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings


def get_provider(model_name: str) -> str:
    """Extract the provider prefix from a model string.

    Examples:
        "openai:gpt-5.2"           -> "openai"
        "anthropic:claude-sonnet-4" -> "anthropic"
        "google-gla:gemini-2.5-pro" -> "google-gla"
        "gpt-5.2"                   -> "openai"  (legacy bare name)
    """
    if ":" in model_name:
        return model_name.split(":")[0]
    return "openai"


def to_responses_model(model_name: str) -> str:
    """Convert an OpenAI Chat model string to use the Responses API.

    Required for builtin tools like WebSearchTool that only work with
    ``OpenAIResponsesModel``.  Non-OpenAI providers are returned unchanged.

    Examples:
        "openai:gpt-5.2"      -> "openai-responses:gpt-5.2"
        "openai-chat:gpt-5.2" -> "openai-responses:gpt-5.2"
        "anthropic:claude-…"  -> "anthropic:claude-…"  (unchanged)
    """
    provider = get_provider(model_name)
    if provider in ("openai", "openai-chat"):
        _, _, bare_name = model_name.partition(":")
        return f"openai-responses:{bare_name or model_name}"
    return model_name


def get_model_settings(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
) -> ModelSettings:
    """Build a ModelSettings dict populated only with non-None values."""
    settings: ModelSettings = {}
    if temperature is not None:
        settings["temperature"] = temperature
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    if top_p is not None:
        settings["top_p"] = top_p
    return settings


class _ModelsConfig:
    """
    Lazy runtime resolver for all LLM model identifiers.

    Attribute access reads the current ``os.environ`` at call time, so values
    always reflect env vars loaded by :func:`lucy.core.bootstrap.bootstrap`
    (or any earlier ``os.environ`` mutation).  The class is safe to import
    before bootstrap has run — you get hardcoded defaults, no crash, and no
    stale snapshot captured at class-definition time.

    Model strings use the pydantic-ai provider-prefix format:
        "<provider>:<model-name>"

    Supported providers:
        openai:      OpenAI (default)
        anthropic:   Anthropic Claude
        google-gla:  Google Gemini (via Google AI / Generative Language API)

    Two-level hierarchy:
      1. Role-based defaults (AGENT_PRIMARY, AGENT_FAST, etc.) — change one
         env var to upgrade an entire tier of agents at once.
      2. Per-agent overrides — set a specific env var to pin one agent to a
         different model without affecting others.

    Environment variables:
      Role-based:
        LUCY_MODEL_AGENT_PRIMARY    — default for all full-capability agents
        LUCY_MODEL_AGENT_FAST       — default for routing/classification/extraction
        LUCY_MODEL_AGENT_NANO       — default for cheapest/fastest single-label classification
        LUCY_MODEL_IMAGE_GEN        — image generation model (e.g. gpt-image-1.5, dall-e-3)
        LUCY_MODEL_VIDEO_GEN        — video generation model (e.g. sora-2, kling-v1)
        LUCY_MODEL_FILE_PDF         — model used to parse PDF attachments
        LUCY_MODEL_FILE_DEFAULT     — model used to parse non-PDF file attachments
        LUCY_MODEL_CREATIVE_VISION  — vision model for creative asset analysis (default gpt-4o-mini)

      Per-agent (each falls back to the role-based tier above):
        LUCY_MODEL_ROUTER
        LUCY_MODEL_TITLE
        LUCY_MODEL_LUCY
        LUCY_MODEL_SUPPORT
        LUCY_MODEL_KEYWORDS
        LUCY_MODEL_IMAGE
        LUCY_MODEL_VIDEO
        LUCY_MODEL_PERFORMANCE
        LUCY_MODEL_CAMPAIGN_PLANNER
        LUCY_MODEL_CREATIVE_DIRECTOR
        LUCY_MODEL_CD_ROUTER         — internal task-type router inside the Creative Director agent
        LUCY_MODEL_PERF_EXTRACTION   — campaign-name/metric extraction inside Performance Analyst

    Provider notes:
        - AGENT_PRIMARY / FAST / NANO default to Anthropic Claude.
        - WebSearchTool works with both OpenAI and Anthropic providers.
        - LUCY_MODEL_FILE_PDF must stay on OpenAI (uses the OpenAI Files API).
        - IMAGE_GENERATION and VIDEO_GENERATION are always OpenAI (gpt-image / Sora).

    Example: revert entire primary tier to OpenAI
        LUCY_MODEL_AGENT_PRIMARY=openai:gpt-5.2
        LUCY_MODEL_AGENT_FAST=openai:gpt-5-mini
        LUCY_MODEL_AGENT_NANO=openai:gpt-5-nano

    Current Anthropic defaults:
        AGENT_PRIMARY  -> anthropic:claude-sonnet-4-6        (latest Sonnet)
        AGENT_FAST     -> anthropic:claude-haiku-4-5-20251001 (latest Haiku)
        AGENT_NANO     -> anthropic:claude-haiku-4-5-20251001 (latest Haiku)
    """

    # Role-based tier: attribute_name -> (env_var, hard_default)
    _TIER: dict[str, tuple[str, str]] = {
        "AGENT_PRIMARY":         ("LUCY_MODEL_AGENT_PRIMARY",  "anthropic:claude-sonnet-4-6"),
        "AGENT_FAST":            ("LUCY_MODEL_AGENT_FAST",     "anthropic:claude-haiku-4-5-20251001"),
        "AGENT_NANO":            ("LUCY_MODEL_AGENT_NANO",     "anthropic:claude-haiku-4-5-20251001"),
        "IMAGE_GENERATION":      ("LUCY_MODEL_IMAGE_GEN",      "gpt-image-1.5"),
        "VIDEO_GENERATION":      ("LUCY_MODEL_VIDEO_GEN",      "sora-2"),
        # PDF analysis uses the OpenAI Files API directly -- must stay on OpenAI.
        "FILE_ANALYSIS_PDF":     ("LUCY_MODEL_FILE_PDF",       "openai:gpt-5"),
        "FILE_ANALYSIS_DEFAULT": ("LUCY_MODEL_FILE_DEFAULT",   "openai:gpt-5-mini"),
        # Vision analysis for creative assets (images/video frames). Defaults to
        # gpt-4o-mini which supports image inputs at low cost (~$0.01-0.03/image).
        "CREATIVE_VISION":       ("LUCY_MODEL_CREATIVE_VISION", "openai:gpt-4o-mini"),
    }

    # Per-agent overrides: attribute_name -> (env_var, fallback_tier_key)
    _PER_AGENT: dict[str, tuple[str, str]] = {
        "ROUTER":                   ("LUCY_MODEL_ROUTER",           "AGENT_FAST"),
        "TITLE":                    ("LUCY_MODEL_TITLE",            "AGENT_NANO"),
        "LUCY":                     ("LUCY_MODEL_LUCY",             "AGENT_PRIMARY"),
        "SUPPORT":                  ("LUCY_MODEL_SUPPORT",          "AGENT_PRIMARY"),
        "KEYWORDS":                 ("LUCY_MODEL_KEYWORDS",         "AGENT_PRIMARY"),
        "IMAGE":                    ("LUCY_MODEL_IMAGE",            "AGENT_PRIMARY"),
        "VIDEO":                    ("LUCY_MODEL_VIDEO",            "AGENT_PRIMARY"),
        "PERFORMANCE":              ("LUCY_MODEL_PERFORMANCE",      "AGENT_PRIMARY"),
        "CAMPAIGN_PLANNER":         ("LUCY_MODEL_CAMPAIGN_PLANNER", "AGENT_PRIMARY"),
        "CREATIVE_DIRECTOR":        ("LUCY_MODEL_CREATIVE_DIRECTOR","AGENT_PRIMARY"),
        "CREATIVE_DIRECTOR_ROUTER": ("LUCY_MODEL_CD_ROUTER",        "AGENT_FAST"),
        "PERF_EXTRACTION":          ("LUCY_MODEL_PERF_EXTRACTION",  "AGENT_NANO"),
    }

    def __getattr__(self, name: str) -> str:
        if name in self._TIER:
            env_var, default = self._TIER[name]
            return os.getenv(env_var, default)
        if name in self._PER_AGENT:
            env_var, tier_key = self._PER_AGENT[name]
            return os.getenv(env_var, getattr(self, tier_key))
        raise AttributeError(f"Models has no attribute {name!r}")

    def __repr__(self) -> str:
        tier_vals = {k: getattr(self, k) for k in self._TIER}
        agent_vals = {k: getattr(self, k) for k in self._PER_AGENT}
        return f"<Models tier={tier_vals} per_agent={agent_vals}>"


# Module-level singleton — attribute access always reflects current os.environ.
Models = _ModelsConfig()


def get_thinking_model_settings() -> AnthropicModelSettings:
    """Runtime model settings override that enables Anthropic extended thinking.

    Intended to be passed as model_settings= to agent.run_stream() on a
    per-request basis, not baked into agent creation. This keeps agents
    unchanged and allows the feature to be gated per-request (e.g. by
    subscription tier).

    Token budget is controlled by LUCY_THINKING_BUDGET (default 10000).
    Temperature is forced to 1 as required by the Anthropic thinking API.

    max_tokens must be greater than budget_tokens per Anthropic API requirements.
    We reserve 2000 tokens for the visible response on top of the thinking budget.
    """
    budget = int(os.getenv("LUCY_THINKING_BUDGET", "10000"))
    max_tokens = budget + 2000
    return AnthropicModelSettings(
        temperature=1,
        max_tokens=max_tokens,
        anthropic_thinking={"type": "enabled", "budget_tokens": budget},
    )
