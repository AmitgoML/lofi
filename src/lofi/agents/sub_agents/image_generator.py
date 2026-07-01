"""Image Generator Agent: produces image creatives via Amazon Nova Canvas and stores them in S3."""

import base64
import json
import uuid

from lofi.config.settings import get_settings
from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.s3_storage import S3CreativeStorage
from lofi.prompts.image_generator import DEFAULT_NEGATIVE_PROMPT_V1
from lofi.schemas.common import CreativeFormat, Platform
from lofi.schemas.creative_director import AssetRef

# Nova Canvas / Titan-compatible dimensions (multiples of 16, 320–4096) per platform ad spec
_PLATFORM_DIMENSIONS: dict[Platform, tuple[int, int]] = {
    Platform.META: (1024, 1024),    # 1:1 square
    Platform.TIKTOK: (576, 1024),   # ~9:16 vertical
    Platform.GOOGLE: (1024, 576),   # ~16:9 horizontal display
    Platform.SPOTIFY: (640, 640),   # 1:1 square
}


class ImageGeneratorAgent:
    """Generates display ads and static creatives via Amazon Nova Canvas on Bedrock."""

    def __init__(
        self,
        bedrock_client: BedrockClient,
        s3_storage: S3CreativeStorage,
    ) -> None:
        self._bedrock_client = bedrock_client
        self._s3 = s3_storage
        self._model_id = get_settings().image_model_id

    def generate(
        self,
        prompt: str,
        platforms: list[Platform],
        brand_name: str,
        negative_prompt: str | None = None,
    ) -> list[AssetRef]:
        """Generate one image per platform and upload to S3. Returns AssetRef list."""
        neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT_V1
        assets: list[AssetRef] = []
        for platform in platforms:
            width, height = _PLATFORM_DIMENSIONS.get(platform, (1024, 1024))
            image_bytes = self._invoke_image_model(prompt, neg, width, height)
            key = f"creatives/{brand_name}/{platform.value}/{uuid.uuid4()}.png"
            s3_url = self._s3.upload_asset(image_bytes, key, "image/png")
            assets.append(
                AssetRef(
                    asset_url=s3_url,
                    creative_format=CreativeFormat.IMAGE,
                    platform=platform,
                )
            )
        return assets

    def _invoke_image_model(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
    ) -> bytes:
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt,
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "width": width,
                "height": height,
                "cfgScale": 8.0,
            },
        }
        payload = self._bedrock_client.invoke_model(self._model_id, json.dumps(body))
        return base64.b64decode(payload["images"][0])
