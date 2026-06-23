"""Image Generator Agent: produces image creatives and stores them in S3."""

from lofi.schemas.creative_director import AssetRef, BrandGuidelines, CreativeBrief


class ImageGeneratorAgent:
    """Generates display ads, banners, static and Meta creatives."""

    def generate(
        self, creative_brief: CreativeBrief, brand_guidelines: BrandGuidelines, platform_requirements: dict
    ) -> list[AssetRef]:
        raise NotImplementedError
