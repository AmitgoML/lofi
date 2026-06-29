"""Application configuration loaded from environment variables (and .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str
    bedrock_model_id: str
    supabase_url: str
    supabase_key: str
    s3_bucket: str
