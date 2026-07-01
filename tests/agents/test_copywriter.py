"""Tests for CopywriterAgent and copy_generation_v1 prompt.

Unit tests run always (mocked deps).
Integration tests hit real AWS Bedrock — run with:
    pytest tests/agents/test_copywriter.py -m integration -s
Requires env vars: AWS_REGION, BEDROCK_MODEL_ID, IMAGE_MODEL_ID,
SUPABASE_URL, SUPABASE_KEY, S3_BUCKET.
"""

import os
from unittest.mock import MagicMock

import pytest

from lofi.agents.sub_agents.copywriter import CopywriterAgent
from lofi.persistence.models import BrandRow
from lofi.prompts.copywriter import copy_generation_v1
from lofi.schemas.common import AudienceSpec, CampaignGoal, Platform
from lofi.schemas.creative_director import TextAsset

BRAND_ID = "8cfb9676-8fd5-42d8-aa3c-c6f678d5cf5e"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_brand_row(**overrides) -> BrandRow:
    defaults = dict(
        brand_id=BRAND_ID,
        brand_name="Acme Corp",
        brand_description="Workflow automation for modern ops teams",
        brand_tone_of_voice="Direct, confident, no fluff",
        brand_copywriting_tone="Conversational but credible",
        brand_core_values="Clarity, speed, reliability",
        brand_messaging_pillars="Automation, visibility, trust",
        brand_tagline="Work less, ship more",
        brand_positioning="The ops-first automation platform",
        brand_dos_and_donts="Do: use plain language. Don't: use jargon.",
        brand_competitors=["Zapier", "Make"],
        brand_keyword_blacklist=["synergy", "leverage"],
        brand_audiences=[{"segment": "ops managers", "age_range": "25-45"}],
        product_descriptions=[{"name": "AutoFlow", "desc": "No-code workflow builder"}],
        brand_goal_config={"primary": "lead_gen", "kpi": "CPL"},
    )
    defaults.update(overrides)
    return BrandRow(**defaults)


def make_audience() -> AudienceSpec:
    return AudienceSpec(age_min=25, age_max=45, genders=["all"])


def make_text_asset() -> TextAsset:
    return TextAsset(
        headlines=["Work less", "Ship more", "No-code ops", "Automate now", "Built for ops"],
        descriptions=["Cut manual work in half.", "Trusted by 500+ ops teams.", "Set up in minutes."],
        cta="Start Free",
        hooks=["Tired of manual work?", "Ops teams love this.", "Automate in 5 min"],
        keywords=["workflow automation", "no-code", "ops tooling", "productivity", "automation platform"],
        long_headlines=["The automation platform built for ops teams", "Less busywork, more shipping"],
    )


# ---------------------------------------------------------------------------
# Unit: CopywriterAgent.generate()
# ---------------------------------------------------------------------------

class TestCopywriterAgentGenerate:
    @pytest.fixture
    def bedrock_client(self):
        mock = MagicMock()
        mock.extract_structured.return_value = make_text_asset()
        return mock

    @pytest.fixture
    def agent(self, bedrock_client):
        return CopywriterAgent(bedrock_client=bedrock_client)

    def test_returns_text_asset(self, agent):
        result = agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer="30-day free trial",
            brand_data=make_brand_row(),
        )
        assert isinstance(result, TextAsset)

    def test_calls_extract_structured_with_text_asset_schema(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer=None,
            brand_data=make_brand_row(),
        )
        bedrock_client.extract_structured.assert_called_once()
        _, schema_cls = bedrock_client.extract_structured.call_args[0]
        assert schema_cls is TextAsset

    def test_prompt_contains_brand_name(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.AWARENESS,
            platform=Platform.GOOGLE,
            offer=None,
            brand_data=make_brand_row(brand_name="LoFi"),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "LoFi" in prompt

    def test_prompt_contains_platform(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.TRAFFIC,
            platform=Platform.TIKTOK,
            offer=None,
            brand_data=make_brand_row(),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "tiktok" in prompt.lower()

    def test_prompt_contains_blacklisted_words(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer=None,
            brand_data=make_brand_row(brand_keyword_blacklist=["synergy", "leverage"]),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "synergy" in prompt
        assert "leverage" in prompt

    def test_prompt_contains_competitors(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer=None,
            brand_data=make_brand_row(brand_competitors=["Zapier", "Make"]),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "Zapier" in prompt

    def test_prompt_includes_offer_when_provided(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.LEAD_GEN,
            platform=Platform.META,
            offer="50% off first month",
            brand_data=make_brand_row(),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "50% off first month" in prompt

    def test_prompt_handles_none_offer(self, agent, bedrock_client):
        agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.AWARENESS,
            platform=Platform.SPOTIFY,
            offer=None,
            brand_data=make_brand_row(),
        )
        prompt, _ = bedrock_client.extract_structured.call_args[0]
        assert "No specific offer" in prompt

    def test_handles_brand_with_all_optional_fields_none(self, agent):
        sparse_brand = BrandRow(brand_id=BRAND_ID, brand_name="MinimalBrand")
        result = agent.generate(
            audience=make_audience(),
            goal=CampaignGoal.TRAFFIC,
            platform=Platform.GOOGLE,
            offer=None,
            brand_data=sparse_brand,
        )
        assert isinstance(result, TextAsset)


# ---------------------------------------------------------------------------
# Unit: copy_generation_v1 prompt function
# ---------------------------------------------------------------------------

class TestCopyGenerationV1:
    def _call(self, **overrides):
        defaults = dict(
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer="Free trial",
            audience=make_audience(),
            brand_name="Acme",
            brand_description="Automation tools",
            brand_tone_of_voice="Direct",
            brand_copywriting_tone="Conversational",
            brand_core_values="Speed, trust",
            brand_messaging_pillars="Automation, reliability",
            brand_tagline="Work less",
            brand_positioning="Ops-first platform",
            brand_dos_and_donts="Do: be clear",
            brand_competitors=["Zapier"],
            brand_keyword_blacklist=["synergy"],
            brand_audiences=[{"segment": "ops managers"}],
            product_descriptions=[{"name": "AutoFlow"}],
            brand_goal_config={"kpi": "CPL"},
        )
        defaults.update(overrides)
        return copy_generation_v1(**defaults)

    def test_returns_string(self):
        assert isinstance(self._call(), str)

    def test_includes_brand_name(self):
        prompt = self._call(brand_name="BrandXYZ")
        assert "BrandXYZ" in prompt

    def test_includes_goal(self):
        prompt = self._call(goal=CampaignGoal.AWARENESS)
        assert "awareness" in prompt.lower()

    def test_includes_platform(self):
        prompt = self._call(platform=Platform.TIKTOK)
        assert "tiktok" in prompt.lower()

    def test_includes_blacklist(self):
        prompt = self._call(brand_keyword_blacklist=["toxic_word", "another_bad"])
        assert "toxic_word" in prompt
        assert "another_bad" in prompt

    def test_includes_competitors(self):
        prompt = self._call(brand_competitors=["CompetitorAlpha", "CompetitorBeta"])
        assert "CompetitorAlpha" in prompt

    def test_includes_tagline(self):
        prompt = self._call(brand_tagline="Just ship it")
        assert "Just ship it" in prompt

    def test_includes_dos_and_donts(self):
        prompt = self._call(brand_dos_and_donts="Do: be bold. Don't: be vague.")
        assert "Do: be bold" in prompt

    def test_handles_empty_competitors_list(self):
        prompt = self._call(brand_competitors=[])
        assert "N/A" in prompt

    def test_handles_empty_blacklist(self):
        prompt = self._call(brand_keyword_blacklist=[])
        assert "none" in prompt.lower()

    def test_handles_empty_audiences(self):
        prompt = self._call(brand_audiences=[])
        assert "no audience profiles defined" in prompt.lower()

    def test_handles_empty_product_descriptions(self):
        prompt = self._call(product_descriptions=[])
        assert "no product descriptions defined" in prompt.lower()

    def test_handles_all_none_optional_fields(self):
        prompt = copy_generation_v1(
            goal=CampaignGoal.TRAFFIC,
            platform=Platform.GOOGLE,
            offer=None,
            audience=make_audience(),
            brand_name="Bare",
            brand_description=None,
            brand_tone_of_voice=None,
            brand_copywriting_tone=None,
            brand_core_values=None,
            brand_messaging_pillars=None,
            brand_tagline=None,
            brand_positioning=None,
            brand_dos_and_donts=None,
            brand_competitors=[],
            brand_keyword_blacklist=[],
            brand_audiences=[],
            product_descriptions=[],
            brand_goal_config=None,
        )
        assert isinstance(prompt, str)
        assert "Bare" in prompt


# ---------------------------------------------------------------------------
# Integration: real Bedrock call
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCopywriterIntegration:
    @pytest.fixture(autouse=True)
    def _skip_if_no_env(self):
        required = ("AWS_REGION", "BEDROCK_MODEL_ID", "SUPABASE_URL", "SUPABASE_KEY", "S3_BUCKET", "IMAGE_MODEL_ID")
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            pytest.skip(f"Missing env vars: {', '.join(missing)}")

    @pytest.fixture
    def agent(self):
        from lofi.config.settings import get_settings
        from lofi.llm.bedrock_client import BedrockClient
        return CopywriterAgent(bedrock_client=BedrockClient(get_settings()))

    def test_generate_returns_valid_text_asset(self, agent):
        brand = BrandRow(
            brand_id=BRAND_ID,
            brand_name="Acme Corp",
            brand_description="Workflow automation for modern ops teams",
            brand_tone_of_voice="Direct, confident, no fluff",
            brand_copywriting_tone="Conversational but credible",
            brand_tagline="Work less, ship more",
            brand_competitors=["Zapier"],
            brand_keyword_blacklist=["synergy"],
        )
        result = agent.generate(
            audience=AudienceSpec(age_min=25, age_max=45, genders=["all"]),
            goal=CampaignGoal.CONVERSIONS,
            platform=Platform.META,
            offer="30-day free trial",
            brand_data=brand,
        )
        assert isinstance(result, TextAsset)
        assert len(result.headlines) > 0
        assert len(result.descriptions) > 0
        assert result.cta
        for h in result.headlines:
            assert len(h) <= 30, f"Headline too long: {h!r}"
        for d in result.descriptions:
            assert len(d) <= 90, f"Description too long: {d!r}"
        assert len(result.cta) <= 15, f"CTA too long: {result.cta!r}"

        print("\n── Copywriter Output ──────────────────────────────")
        print(f"Headlines:      {result.headlines}")
        print(f"Descriptions:   {result.descriptions}")
        print(f"CTA:            {result.cta}")
        print(f"Hooks:          {result.hooks}")
        print(f"Keywords:       {result.keywords}")
        print(f"Long headlines: {result.long_headlines}")
        print("───────────────────────────────────────────────────")
