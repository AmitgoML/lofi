"""
Bootstrap regression tests.

Verifies that :func:`lucy.core.bootstrap.bootstrap` is idempotent,
calls ``load_envs`` exactly once on first invocation, and that
:func:`lucy.core.bootstrap.is_bootstrapped` reflects the state correctly.
"""
import importlib
import sys
from unittest.mock import call, patch

import pytest


def _reload_bootstrap():
    """Return a freshly imported bootstrap module with reset module state."""
    mod_name = "lucy.core.bootstrap"
    sys.modules.pop(mod_name, None)
    return importlib.import_module(mod_name)


def test_bootstrap_calls_load_envs_once():
    """bootstrap() must call load_envs exactly once on first call."""
    bootstrap_mod = _reload_bootstrap()

    with patch("lucy.utils.secrets.load_envs") as mock_load:
        bootstrap_mod.bootstrap()
        bootstrap_mod.bootstrap()  # second call — should be a no-op
        bootstrap_mod.bootstrap()  # third call — should be a no-op

    mock_load.assert_called_once()


def test_bootstrap_is_idempotent():
    """bootstrap() is safe to call multiple times without error."""
    bootstrap_mod = _reload_bootstrap()

    with patch("lucy.utils.secrets.load_envs"):
        for _ in range(5):
            bootstrap_mod.bootstrap()  # must not raise


def test_is_bootstrapped_reflects_state():
    """is_bootstrapped() must return False before and True after bootstrap()."""
    bootstrap_mod = _reload_bootstrap()

    assert bootstrap_mod.is_bootstrapped() is False

    with patch("lucy.utils.secrets.load_envs"):
        bootstrap_mod.bootstrap()

    assert bootstrap_mod.is_bootstrapped() is True


def test_bootstrap_module_importable_without_credentials():
    """lucy.core.bootstrap must be importable without any env vars set."""
    sys.modules.pop("lucy.core.bootstrap", None)
    mod = importlib.import_module("lucy.core.bootstrap")
    assert callable(mod.bootstrap)
    assert callable(mod.is_bootstrapped)
