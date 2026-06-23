"""Copywriter Agent: generates ad copy assets."""

from lofi.schemas.common import AudienceSpec, CampaignGoal, Platform
from lofi.schemas.creative_director import CreativeBrief, TextAsset


class CopywriterAgent:
    """Generates headlines, descriptions, CTAs, hooks, keywords, long headlines."""

    def generate(
        self,
        audience: AudienceSpec,
        goal: CampaignGoal,
        platform: Platform,
        offer: str | None,
        creative_brief: CreativeBrief,
    ) -> TextAsset:
        raise NotImplementedError
