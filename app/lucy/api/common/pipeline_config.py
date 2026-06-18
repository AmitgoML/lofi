"""Pipeline configuration — env-backed settings for connectors and scheduling."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline runtime configuration. Values from environment."""

    raw_s3_bucket: str = os.getenv("PIPELINE_RAW_S3_BUCKET", "")
    raw_s3_prefix: str = os.getenv("PIPELINE_RAW_S3_PREFIX", "l1-raw")
    restatement_days: int = int(os.getenv("PIPELINE_RESTATEMENT_DAYS", "7"))
    reporting_timezone: str = os.getenv("PIPELINE_REPORTING_TIMEZONE", "America/New_York")
    reporting_currency: str = os.getenv("PIPELINE_REPORTING_CURRENCY", "USD")

    google_ads_developer_token: str = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
    meta_app_id: str = os.getenv("META_APP_ID", "")
    meta_app_secret: str = os.getenv("META_APP_SECRET", "")
    tiktok_app_id: str = os.getenv("TIKTOK_APP_ID", "")
    tiktok_app_secret: str = os.getenv("TIKTOK_APP_SECRET", "")
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")


@dataclass(frozen=True)
class OrchestrationConfig:
    """Workflow orchestration configuration."""

    max_parallel_tasks: int = int(os.getenv("ORCHESTRATION_MAX_PARALLEL_TASKS", "4"))
    default_checkpoint_timeout_s: int = int(
        os.getenv("ORCHESTRATION_CHECKPOINT_TIMEOUT_S", "86400")
    )


@dataclass(frozen=True)
class CompetitiveConfig:
    """Competitive intelligence configuration."""

    searchapi_api_key: str = os.getenv("SEARCHAPI_API_KEY", "")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation framework configuration."""

    regression_threshold: float = float(os.getenv("EVAL_REGRESSION_THRESHOLD", "0.05"))
    ci_mode: bool = os.getenv("EVAL_CI_MODE", "false").lower() == "true"
