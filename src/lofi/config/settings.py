"""Application configuration loaded from environment variables (and .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str
    aws_profile: str | None
    bedrock_model_id: str
    supabase_url: str
    supabase_key: str
    s3_bucket: str
    image_model_id: str

    @classmethod
    def from_env(cls) -> "Settings":
        required = ("AWS_REGION", "BEDROCK_MODEL_ID", "SUPABASE_URL", "SUPABASE_KEY", "S3_BUCKET", "IMAGE_MODEL_ID")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            aws_region=os.environ["AWS_REGION"],
            aws_profile=os.environ.get("AWS_PROFILE") or None,
            bedrock_model_id=os.environ["BEDROCK_MODEL_ID"],
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=os.environ["SUPABASE_KEY"],
            s3_bucket=os.environ["S3_BUCKET"],
            image_model_id=os.environ["IMAGE_MODEL_ID"],
        )

_settings: "Settings | None" = None


def get_settings() -> "Settings":
    """Return the singleton Settings, loading from env vars on first call."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
