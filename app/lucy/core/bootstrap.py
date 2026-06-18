"""
Single runtime bootstrap entrypoint for the Lucy application.

Call ``bootstrap()`` exactly once, before importing any Lucy module that
initialises provider clients or reads environment-derived configuration.
The function is idempotent — subsequent calls are no-ops.
"""
from __future__ import annotations

from loguru import logger

_bootstrapped: bool = False


def bootstrap() -> None:
    """Load environment configuration and prepare the application runtime.

    Performs the following steps in order:
    1. Loads env vars from ``.env`` (local) or AWS Secrets Manager (deployed),
       via :func:`lucy.utils.secrets.load_envs`.
    2. Marks the bootstrap as complete so repeated calls are skipped.

    This function must be called before any Lucy module that constructs
    LLM provider clients (pydantic-ai ``Agent``, OpenAI ``AsyncOpenAI``, …)
    or reads model/configuration env vars at module level.
    """
    global _bootstrapped
    if _bootstrapped:
        return

    # Import here to avoid circular imports and to ensure this module itself
    # can be imported safely without triggering side effects.
    from lucy.utils.secrets import load_envs

    load_envs()
    _bootstrapped = True
    logger.debug("Lucy bootstrap complete")


def is_bootstrapped() -> bool:
    """Return True if :func:`bootstrap` has already been called."""
    return _bootstrapped
