# secrets.py
import os
import json
import sys
from pathlib import Path
from typing import Optional

import boto3
from dotenv import load_dotenv
from loguru import logger


APP_ENV = os.getenv("APP_ENV", "local")
AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME") or f"{APP_ENV}/lucy/app-config"


def _is_testing() -> bool:
    """
    Check if we're running in a test environment.

    Returns True if pytest is running or SKIP_ENV_LOAD is set.
    """
    # Check if pytest is in the process name or modules
    if "pytest" in sys.modules:
        return True

    # Check for explicit test environment variable
    if os.getenv("SKIP_ENV_LOAD") == "true":
        return True

    # Check if APP_ENV is explicitly set to test
    if os.getenv("APP_ENV") == "test":
        return True

    return False


def _extract_value_for_env(env_key: str, raw_secret: str) -> str:
    """
    Return the value for ``env_key`` from a raw secret string.

    - If JSON: prefer key ``env_key``; else if exactly one string field, use it; else error.
    - If not JSON: treat as plain string.

    :param env_key: Environment variable name to extract.
    :param raw_secret: Raw secret value as returned by Secrets Manager.
    :return: Extracted value to set for the environment variable.
    :rtype: str
    """
    try:
        data = json.loads(raw_secret)
        if isinstance(data, dict):
            if env_key in data and isinstance(data[env_key], str):
                return data[env_key]
            str_items = [(k, v) for k, v in data.items() if isinstance(v, str)]
            if len(str_items) == 1:
                return str_items[0][1]
            raise ValueError(
                f"Secret JSON ambiguous for '{env_key}': provide a '{env_key}' key "
                f"or ensure the secret has exactly one top-level string field."
            )
    except json.JSONDecodeError:
        pass
    return raw_secret  # plain string


def load_envs(
    region: Optional[str] = None,
) -> None:
    """
    Load env vars from ``.env`` if present; else from AWS Secrets Manager.

    Skips loading if running in test environment (pytest or SKIP_ENV_LOAD=true).

    :param region: AWS region; if None, uses ``AWS_REGION`` env or default session.
    :return: None
    :rtype: None
    """
    # Skip loading environment variables during tests
    if _is_testing():
        logger.info(
            "Running in test environment - skipping .env and AWS Secrets Manager load"
        )
        return

    # Prefer .env for local dev
    logger.info(f"Current environment: {APP_ENV} environment")

    if APP_ENV == "local":
        env_path = Path(__file__).parent.parent.parent / ".env"
        logger.info(f"Loading secrets to env variables from {env_path}")
        load_dotenv(env_path)
        logger.info(os.environ)
        return

    client = boto3.client(
        "secretsmanager", region_name=region or os.getenv("AWS_REGION")
    )

    resp = client.get_secret_value(SecretId=AWS_SECRET_NAME)
    raw = resp.get("SecretString")
    if raw is None:
        raw = resp["SecretBinary"].decode("utf-8")

    secrets_json = json.loads(raw)
    logger.info("Loading secrets to env variables from AWS Secrets Manager")
    for env_key, value in secrets_json.items():
        logger.info(f"Loaded {env_key}")

        # Respect pre-set env vars (allow overrides from deployment config)
        if os.getenv(env_key):
            continue

        os.environ[env_key] = value
