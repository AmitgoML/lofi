"""Amazon S3 persistence for creative assets."""

import boto3

from lofi.config.settings import Settings


class S3CreativeStorage:
    """Stores and retrieves generated image/video creative assets."""

    def __init__(self, settings: Settings, client=None) -> None:
        self._bucket = settings.s3_bucket
        self._client = client or boto3.client("s3", region_name=settings.aws_region)

    def upload_asset(self, file_bytes: bytes, key: str, content_type: str) -> str:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
        )
        return f"s3://{self._bucket}/{key}"

    def search_existing_assets(self, brand: str, creative_format: str) -> list[dict]:
        prefix = f"creatives/{brand}/{creative_format}/"
        response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return [
            {
                "key": obj["Key"],
                "s3_url": f"s3://{self._bucket}/{obj['Key']}",
                "size": obj["Size"],
            }
            for obj in response.get("Contents", [])
        ]
