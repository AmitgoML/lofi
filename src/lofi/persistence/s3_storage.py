"""Amazon S3 persistence for creative assets."""


class S3CreativeStorage:
    """Stores and retrieves generated image/video creative assets."""

    def upload_asset(self, file_bytes: bytes, key: str, content_type: str) -> str:
        raise NotImplementedError

    def search_existing_assets(self, brand: str, creative_format: str) -> list[dict]:
        raise NotImplementedError
