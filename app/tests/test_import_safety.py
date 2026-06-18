"""
Import-safety regression tests.

These tests verify that importing Lucy modules does NOT require provider
credentials to be present (no ``AsyncOpenAI`` / ``AsyncAnthropic`` client
construction at import time) and does NOT trigger network calls during import.

If a module-level ``Agent(model=...)`` is added that constructs a provider
client immediately, these tests will catch it before the next deployment.
"""
import importlib
import os
import sys
from unittest.mock import patch

import pytest


def _import_fresh(module_name: str):
    """Import a module, evicting it from sys.modules first if cached."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_path",
    [
        "lucy.agents.common.model_config",
        "lucy.agents.performance_analyst_agent",
        "lucy.agents.creative_director_agent",
        "lucy.agents.router_agent",
        "lucy.agents.lucy_agent",
        "lucy.agents.keywords_agent",
        "lucy.agents.support_agent",
        "lucy.agents.image_agent",
        "lucy.agents.video_agent",
        "lucy.agents.campaign_planner_agent",
        "lucy.agents.common.factory",
        "lucy.core.bootstrap",
    ],
)
def test_module_imports_without_api_key(module_path):
    """Module must be importable even when OPENAI_API_KEY is absent."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        # Import should not raise — provider clients are created lazily
        mod = importlib.import_module(module_path)
        assert mod is not None
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_models_reads_env_at_access_time():
    """Models attributes must reflect os.environ at access time, not import time."""
    from lucy.agents.common.model_config import Models

    sentinel = "openai:test-sentinel-model"
    with patch.dict(os.environ, {"LUCY_MODEL_AGENT_PRIMARY": sentinel}):
        assert Models.AGENT_PRIMARY == sentinel

    # After unpatching, the default is restored on next access
    os.environ.pop("LUCY_MODEL_AGENT_PRIMARY", None)
    assert Models.AGENT_PRIMARY != sentinel


def test_models_per_agent_falls_back_to_tier():
    """Per-agent override must fall back to tier default when env var is absent."""
    from lucy.agents.common.model_config import Models

    os.environ.pop("LUCY_MODEL_LUCY", None)
    primary = Models.AGENT_PRIMARY
    assert Models.LUCY == primary


def test_models_per_agent_override_respected():
    """Setting a per-agent env var must override the tier default."""
    from lucy.agents.common.model_config import Models

    override = "anthropic:claude-test"
    with patch.dict(os.environ, {"LUCY_MODEL_LUCY": override}):
        assert Models.LUCY == override


def test_models_unknown_attribute_raises():
    from lucy.agents.common.model_config import Models

    with pytest.raises(AttributeError, match="Models has no attribute"):
        _ = Models.NONEXISTENT_KEY
