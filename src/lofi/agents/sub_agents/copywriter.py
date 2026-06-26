"""Copywriter Agent: generates brand-voice-aware ad copy via Claude on Bedrock."""

from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.models import BrandRow
from lofi.prompts.copywriter import copy_generation_v1
from lofi.schemas.common import AudienceSpec, CampaignGoal, Platform
from lofi.schemas.creative_director import CreativeBrief, TextAsset


class CopywriterAgent:
    """Generates headlines, descriptions, CTAs, hooks, keywords, and long headlines."""

    def __init__(self, bedrock_client: BedrockClient) -> None:
        self._bedrock_client = bedrock_client

    def generate(
        self,
        audience: AudienceSpec,
        goal: CampaignGoal,
        platform: Platform,
        offer: str | None,
        brand_data: BrandRow,
    ) -> TextAsset:
        prompt = copy_generation_v1(
            goal=goal,
            platform=platform,
            offer=offer,
            audience=audience,
            brand_name=brand_data.brand_name,
            brand_description=brand_data.brand_description,
            brand_tone_of_voice=brand_data.brand_tone_of_voice,
            brand_copywriting_tone=brand_data.brand_copywriting_tone,
            brand_core_values=brand_data.brand_core_values,
            brand_messaging_pillars=brand_data.brand_messaging_pillars,
            brand_tagline=brand_data.brand_tagline,
            brand_positioning=brand_data.brand_positioning,
            brand_dos_and_donts=brand_data.brand_dos_and_donts,
            brand_competitors=brand_data.brand_competitors,
            brand_keyword_blacklist=brand_data.brand_keyword_blacklist,
            brand_audiences=brand_data.brand_audiences,
            product_descriptions=brand_data.product_descriptions,
            brand_goal_config=brand_data.brand_goal_config,
        )
        return self._bedrock_client.extract_structured(prompt, TextAsset)
