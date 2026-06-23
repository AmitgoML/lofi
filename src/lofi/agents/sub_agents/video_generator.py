"""Video Generator Agent: produces video creatives and stores them in S3."""

from lofi.schemas.creative_director import AssetRef, BrandGuidelines, CreativeBrief


class VideoGeneratorAgent:
    """Generates reels, stories, TikTok videos, and video ads."""

    def generate(
        self, creative_brief: CreativeBrief, brand_guidelines: BrandGuidelines, platform_requirements: dict
    ) -> list[AssetRef]:
        raise NotImplementedError
