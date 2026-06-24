"""Application configuration loaded from environment variables."""

import os
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
        required = ("AWS_REGION", "BEDROCK_MODEL_ID", "SUPABASE_URL", "SUPABASE_KEY", "S3_BUCKET")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            aws_region=os.environ["AWS_REGION"],
            bedrock_model_id=os.environ["BEDROCK_MODEL_ID"],
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=os.environ["SUPABASE_KEY"],
            s3_bucket=os.environ["S3_BUCKET"],
        )
