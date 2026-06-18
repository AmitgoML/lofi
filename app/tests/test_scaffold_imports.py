"""Verify new engagement modules are importable without side effects."""

import importlib
import pytest

SCAFFOLD_MODULES = [
    "lucy.orchestration",
    "lucy.orchestration.engine",
    "lucy.orchestration.state",
    "lucy.orchestration.hitl",
    "lucy.orchestration.streaming",
    "lucy.orchestration.workflows",
    "lucy.orchestration.workflows.campaign_planner",
    "lucy.orchestration.workflows.campaign_planner.tools",
    "lucy.pipeline",
    "lucy.pipeline.connectors",
    "lucy.pipeline.connectors.google_ads",
    "lucy.pipeline.connectors.meta",
    "lucy.pipeline.connectors.tiktok",
    "lucy.pipeline.connectors.spotify",
    "lucy.pipeline.layers.l1_raw",
    "lucy.pipeline.layers.l2_source_aligned",
    "lucy.pipeline.layers.l3_canonical",
    "lucy.pipeline.layers.l4_serving",
    "lucy.pipeline.mapping",
    "lucy.pipeline.normalization",
    "lucy.pipeline.scheduling",
    "lucy.domain",
    "lucy.domain.workflows",
    "lucy.domain.pipeline",
    "lucy.domain.competitive",
    "lucy.competitive",
    "lucy.competitive.connectors",
    "lucy.competitive.storage",
    "lucy.competitive.queries",
    "lucy.eval",
    "lucy.eval.datasets",
    "lucy.eval.judges",
    "lucy.eval.runner",
    "lucy.api.workflows",
    "lucy.api.pipeline",
    "lucy.api.common.pipeline_config",
]


@pytest.mark.parametrize("module_name", SCAFFOLD_MODULES)
def test_scaffold_module_imports(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None
