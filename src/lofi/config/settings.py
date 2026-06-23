"""Application configuration loaded from environment variables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    aws_region: str
    bedrock_model_id: str
    supabase_url: str
    supabase_key: str
    s3_bucket: str

    @classmethod
    def from_env(cls) -> "Settings":
        raise NotImplementedError
