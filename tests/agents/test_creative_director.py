"""Tests for CreativeDirectorAgent.

Unit tests run always (mocked deps).
Integration tests hit real AWS Bedrock + Supabase + S3 — run with:
    pytest tests/agents/test_creative_director.py -m integration
Requires all env vars set: AWS_REGION, BEDROCK_MODEL_ID, SUPABASE_URL,
SUPABASE_KEY, S3_BUCKET, IMAGE_MODEL_ID.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lofi.agents.creative_director import CreativeDirectorAgent
from lofi.agents.sub_agents.copywriter import CopywriterAgent
from lofi.agents.sub_agents.image_generator import ImageGeneratorAgent
from lofi.persistence.models import BrandRow
from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, CreativeFormat, Platform
from lofi.schemas.creative_director import (
    ABVariant,
    AssetDecision,
    AssetRef,
    CreativeBrief,
    CreativeDirectorInput,
    CreativeStrategy,
    TextAsset,
)
from lofi.schemas.performance_analyst import (
    AudienceRecommendation,
    CreativeRecommendation,
    LocationRecommendation,
    PerformanceAnalystOutput,
    PlatformRecommendation,
)

# ── Real DB identifiers (used by integration tests) ──────────────────────────
BRAND_ID = "8cfb9676-8fd5-42d8-aa3c-c6f678d5cf5e"
ASSET_ID = "527c0bdc-66c7-4150-9154-ff001bf40ef0"


# ── Shared test data factories ────────────────────────────────────────────────

def make_brand_row(**overrides) -> BrandRow:
    defaults = dict(
        brand_id=BRAND_ID,
        brand_name="Acme Corp",
        brand_description="We make great things.",
        brand_primary_color="#0057FF",
        brand_secondary_color="#FFFFFF",
        brand_imagery_style="Clean, modern, minimalist",
        brand_tone_of_voice="Confident and approachable",
        brand_copywriting_tone="Direct, benefit-led",
        brand_tagline="Build better, together.",
        brand_positioning="Premium B2B SaaS for forward-thinking teams",
        brand_core_values="Innovation, Reliability, Clarity",
        brand_messaging_pillars="Speed, simplicity, results",
        brand_dos_and_donts="Do: use action verbs. Don't: use jargon.",
        brand_keyword_blacklist=["cheap", "free trial"],
        brand_competitors=["Competitor A", "Competitor B"],
        brand_logo_usage_rules="Logo must appear on white or brand-blue backgrounds only.",
        product_descriptions=[{"name": "Acme Platform", "description": "End-to-end workflow automation"}],
        brand_audiences=[{"segment": "Tech leads at 50-500 person companies", "age_range": "28-45"}],
    )
    defaults.update(overrides)
    return BrandRow(**defaults)


def make_performance_insights(brand_id: str = BRAND_ID) -> PerformanceAnalystOutput:
    return PerformanceAnalystOutput(
        brand_id=brand_id,
        platform_recommendations=[
            PlatformRecommendation(platform=Platform.META, historical_roas=4.2, historical_ctr=0.035)
        ],
        audience_recommendations=[
            AudienceRecommendation(audience_segment="age_range:28-44", historical_performance_score=3.8)
        ],
        location_recommendations=[
            LocationRecommendation(location={"city": "San Francisco", "country": "USA"})
        ],
        creative_recommendations=[
            CreativeRecommendation(
                creative_format=CreativeFormat.IMAGE,
                historical_engagement_rate=0.04,
                asset_id=ASSET_ID,
            )
        ],
    )


def make_creative_brief() -> CreativeBrief:
    return CreativeBrief(
        goal=CampaignGoal.CONVERSIONS,
        audience=AudienceSpec(age_min=28, age_max=45, genders=["all"]),
        platforms=[Platform.META, Platform.GOOGLE],
        offer="30-day free pilot",
    )


def make_strategy() -> CreativeStrategy:
    return CreativeStrategy(
        best_creative_format=CreativeFormat.IMAGE,
        best_messaging_angle="Automate the work your team hates, so they can do the work they love.",
        asset_decisions=[AssetDecision(creative_format=CreativeFormat.IMAGE, action="generate")],
        variant_a=ABVariant(
            variant_label="A",
            image_prompt=(
                "Bold product screenshot of Acme Platform dashboard, deep blue #0057FF background, "
                "white UI elements, company logo top-right corner, clean studio lighting, "
                "high contrast, professional SaaS aesthetic, centered composition"
            ),
            negative_prompt="blurry, watermark, text overlay, low quality, distorted",
            rationale=(
                "Brand-hero: product as the focal point with dominant brand colors. "
                "Tests direct conversion intent — users who already know the category."
            ),
        ),
        variant_b=ABVariant(
            variant_label="B",
            image_prompt=(
                "Diverse team of professionals collaborating around a laptop in a bright modern office, "
                "natural window light, warm neutral tones, Acme Platform visible on screen in background, "
                "candid and authentic, brand blue accent in clothing, logo subtle bottom-left"
            ),
            negative_prompt="blurry, watermark, text overlay, low quality, staged looking",
            rationale=(
                "Lifestyle/context: aspirational scene showing the human outcome. "
                "Tests upper-funnel awareness — speaks to teams imagining a better workflow."
            ),
        ),
        rationale="Split brand-direct vs aspiration to learn whether this audience responds to product or outcome messaging.",
    )


def make_text_asset() -> TextAsset:
    return TextAsset(
        headlines=["Build faster", "Less busywork", "Automate today", "Ship smarter", "Your team, upgraded"],
        descriptions=[
            "Acme automates the repetitive work so your team ships faster.",
            "Connect your tools, cut the noise, and focus on what matters.",
            "Enterprise-grade workflow automation, built for fast-moving teams.",
        ],
        cta="Start Pilot",
        hooks=["Tired of manual handoffs?", "Your competitors already automate.", "What if Mondays didn't suck?"],
        keywords=["workflow automation", "SaaS productivity", "team efficiency", "process automation", "B2B software"],
        long_headlines=[
            "Build better products faster with Acme workflow automation",
            "Stop losing hours to manual work — Acme automates it for you",
        ],
    )


def make_asset_ref(platform: Platform = Platform.META) -> AssetRef:
    return AssetRef(
        asset_url=f"s3://test-bucket/creatives/Acme Corp/{platform.value}/test-uuid.png",
        creative_format=CreativeFormat.IMAGE,
        platform=platform,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bedrock_client() -> MagicMock:
    client = MagicMock()

    def _extract_structured(prompt: str, schema_cls):
        if schema_cls is CreativeStrategy:
            return make_strategy()
        if schema_cls is TextAsset:
            return make_text_asset()
        raise ValueError(f"Unexpected schema_cls in extract_structured: {schema_cls}")

    client.extract_structured.side_effect = _extract_structured
    return client


@pytest.fixture
def supabase_client() -> MagicMock:
    client = MagicMock()
    client.get_brand.return_value = make_brand_row()
    return client


@pytest.fixture
def s3_storage() -> MagicMock:
    storage = MagicMock()
    storage.upload_asset.return_value = "s3://test-bucket/creatives/Acme Corp/meta/test-uuid.png"
    return storage


@pytest.fixture
def mock_image_generator() -> MagicMock:
    gen = MagicMock(spec=ImageGeneratorAgent)
    gen.generate.side_effect = lambda prompt, platforms, brand_name, negative_prompt=None: [
        make_asset_ref(p) for p in platforms
    ]
    return gen


@pytest.fixture
def mock_copywriter() -> MagicMock:
    cw = MagicMock(spec=CopywriterAgent)
    cw.generate.return_value = make_text_asset()
    return cw


@pytest.fixture
def agent(bedrock_client, supabase_client, s3_storage, mock_image_generator, mock_copywriter) -> CreativeDirectorAgent:
    return CreativeDirectorAgent(
        bedrock_client=bedrock_client,
        supabase_client=supabase_client,
        s3_storage=s3_storage,
        image_generator=mock_image_generator,
        copywriter=mock_copywriter,
    )


@pytest.fixture
def director_input(supabase_client) -> CreativeDirectorInput:
    return CreativeDirectorInput(
        creative_brief=make_creative_brief(),
        brand_data=make_brand_row(),
        performance_analysis=make_performance_insights(),
    )


# ── Unit: _determine_creative_strategy ───────────────────────────────────────

class TestDetermineCreativeStrategy:
    def test_calls_bedrock_extract_structured_with_correct_schema(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput, bedrock_client: MagicMock
    ) -> None:
        strategy = agent._determine_creative_strategy(director_input)

        call_args = bedrock_client.extract_structured.call_args
        assert call_args[0][1] is CreativeStrategy

    def test_returns_creative_strategy_with_two_ab_variants(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput
    ) -> None:
        strategy = agent._determine_creative_strategy(director_input)

        assert isinstance(strategy, CreativeStrategy)
        assert strategy.variant_a.variant_label == "A"
        assert strategy.variant_b.variant_label == "B"
        assert strategy.variant_a.image_prompt != strategy.variant_b.image_prompt

    def test_strategy_includes_asset_decisions(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput
    ) -> None:
        strategy = agent._determine_creative_strategy(director_input)

        assert len(strategy.asset_decisions) >= 1
        assert strategy.asset_decisions[0].creative_format == CreativeFormat.IMAGE


# ── Unit: produce_assets ─────────────────────────────────────────────────────

class TestProduceAssets:
    def test_output_contains_all_three_slots(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput
    ) -> None:
        output = agent.produce_assets(director_input)

        # Slot 1: historical recommendation from perf insights
        assert len(output.recommended_assets) == 1
        assert output.recommended_assets[0].asset_id == ASSET_ID
        assert output.recommended_assets[0].creative_format == CreativeFormat.IMAGE

        # Slot 2 & 3: generated variants
        assert len(output.variant_a) > 0
        assert len(output.variant_b) > 0

    def test_variant_a_and_b_are_generated_for_each_platform(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput, mock_image_generator: MagicMock
    ) -> None:
        output = agent.produce_assets(director_input)

        assert mock_image_generator.generate.call_count == 2  # once per variant

        # Both variants cover both platforms from the brief
        assert {a.platform for a in output.variant_a} == {Platform.META, Platform.GOOGLE}
        assert {a.platform for a in output.variant_b} == {Platform.META, Platform.GOOGLE}

    def test_variant_a_uses_brand_hero_prompt(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput, mock_image_generator: MagicMock
    ) -> None:
        agent.produce_assets(director_input)

        first_call = mock_image_generator.generate.call_args_list[0]
        assert "blue" in first_call.kwargs["prompt"].lower() or "product" in first_call.kwargs["prompt"].lower()

    def test_variant_b_uses_lifestyle_prompt(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput, mock_image_generator: MagicMock
    ) -> None:
        agent.produce_assets(director_input)

        second_call = mock_image_generator.generate.call_args_list[1]
        prompt = second_call.kwargs["prompt"].lower()
        assert any(word in prompt for word in ["team", "office", "collaborat", "natural", "lifestyle"])

    def test_recommended_assets_skips_recs_without_asset_id(
        self, agent: CreativeDirectorAgent
    ) -> None:
        perf_no_asset_id = make_performance_insights()
        perf_no_asset_id.creative_recommendations[0].asset_id = None

        director_input = CreativeDirectorInput(
            creative_brief=make_creative_brief(),
            brand_data=make_brand_row(),
            performance_analysis=perf_no_asset_id,
        )

        output = agent.produce_assets(director_input)

        assert output.recommended_assets == []

    def test_copywriter_called_with_brand_data(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput, mock_copywriter: MagicMock
    ) -> None:
        agent.produce_assets(director_input)

        mock_copywriter.generate.assert_called_once()
        call_kwargs = mock_copywriter.generate.call_args.kwargs
        assert call_kwargs["brand_data"].brand_name == "Acme Corp"

    def test_output_includes_texts(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput
    ) -> None:
        output = agent.produce_assets(director_input)

        assert len(output.texts.headlines) == 5
        assert output.texts.cta == "Start Pilot"

    def test_output_carries_rationale_for_each_variant(
        self, agent: CreativeDirectorAgent, director_input: CreativeDirectorInput
    ) -> None:
        output = agent.produce_assets(director_input)

        assert "brand" in output.variant_a_rationale.lower() or "product" in output.variant_a_rationale.lower()
        assert "lifestyle" in output.variant_b_rationale.lower() or "context" in output.variant_b_rationale.lower()


# ── Unit: run (workflow state) ────────────────────────────────────────────────

class TestRun:
    def test_writes_creative_director_output_to_state(
        self, agent: CreativeDirectorAgent, supabase_client: MagicMock
    ) -> None:
        from lofi.schemas.campaign_planner import CampaignPlannerInput

        brief = CampaignPlannerInput(
            user_request="Launch a conversion campaign",
            brand="Acme Corp",
            organization_id="org-test",
            goal=CampaignGoal.CONVERSIONS,
            budget=BudgetSpec(total_budget=5000.0),
            campaign_timing=CampaignTiming(start_date="2026-08-01"),
        )
        state = {
            "user_request": "Launch a conversion campaign",
            "campaign_brief": brief,
            "performance_insights": make_performance_insights(brand_id=BRAND_ID),
        }

        result = agent.run(state)

        assert "creative_director_output" in result
        output = result["creative_director_output"]
        assert output.variant_a
        assert output.variant_b
        assert output.texts is not None

    def test_run_fetches_brand_from_supabase_using_brand_id(
        self, agent: CreativeDirectorAgent, supabase_client: MagicMock
    ) -> None:
        from lofi.schemas.campaign_planner import CampaignPlannerInput

        brief = CampaignPlannerInput(
            user_request="Run campaign",
            brand="Acme Corp",
            organization_id="org-test",
            goal=CampaignGoal.AWARENESS,
            budget=BudgetSpec(total_budget=1000.0),
            campaign_timing=CampaignTiming(start_date="2026-08-01"),
        )
        state = {
            "user_request": "Run campaign",
            "campaign_brief": brief,
            "performance_insights": make_performance_insights(brand_id=BRAND_ID),
        }

        agent.run(state)

        supabase_client.get_brand.assert_called_once_with(BRAND_ID)

    def test_run_raises_when_brand_id_missing(
        self, agent: CreativeDirectorAgent
    ) -> None:
        from lofi.schemas.campaign_planner import CampaignPlannerInput

        brief = CampaignPlannerInput(
            user_request="Run campaign",
            brand="Acme Corp",
            organization_id="org-test",
            goal=CampaignGoal.AWARENESS,
            budget=BudgetSpec(total_budget=1000.0),
            campaign_timing=CampaignTiming(start_date="2026-08-01"),
        )
        state = {
            "user_request": "Run campaign",
            "campaign_brief": brief,
            "performance_insights": make_performance_insights(brand_id=None),
        }

        with pytest.raises(ValueError, match="No brand_id found"):
            agent.run(state)


# ── Integration tests (real Bedrock + Supabase + S3) ─────────────────────────

@pytest.mark.integration
class TestCreativeDirectorIntegration:
    """
    Runs the full A/B creative flow against real services.

    Generated images are stored in S3 at:
        s3://<S3_BUCKET>/creatives/<brand_name>/<platform>/uuid.png

    Run with:
        pytest tests/agents/test_creative_director.py -m integration -s
    """

    @pytest.fixture(autouse=True)
    def _require_env(self):
        required = ("AWS_REGION", "BEDROCK_MODEL_ID", "SUPABASE_URL", "SUPABASE_KEY", "S3_BUCKET", "IMAGE_MODEL_ID")
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            pytest.skip(f"Missing env vars for integration test: {', '.join(missing)}")

    @pytest.fixture
    def real_services(self):
        from lofi.config.settings import Settings
        from lofi.llm.bedrock_client import BedrockClient
        from lofi.persistence.s3_storage import S3CreativeStorage
        from lofi.persistence.supabase_client import SupabaseClient

        settings = Settings.from_env()
        return {
            "bedrock": BedrockClient(settings),
            "supabase": SupabaseClient(settings),
            "s3": S3CreativeStorage(settings),
            "s3_bucket": settings.s3_bucket,
        }

    def test_produce_assets_generates_ab_variants_and_uploads_to_s3(self, real_services) -> None:
        """Full end-to-end: fetch brand → Claude strategy → Titan images × 2 → S3 upload."""
        bedrock = real_services["bedrock"]
        supabase = real_services["supabase"]
        s3 = real_services["s3"]
        s3_bucket = real_services["s3_bucket"]

        # Fetch the real brand from Supabase
        brand_row = supabase.get_brand(BRAND_ID)
        assert brand_row.brand_id == BRAND_ID
        print(f"\nBrand: {brand_row.brand_name}")

        agent = CreativeDirectorAgent(
            bedrock_client=bedrock,
            supabase_client=supabase,
            s3_storage=s3,
        )

        director_input = CreativeDirectorInput(
            creative_brief=CreativeBrief(
                goal=CampaignGoal.CONVERSIONS,
                audience=AudienceSpec(age_min=25, age_max=44, genders=["all"]),
                platforms=[Platform.META],   # one platform to keep test fast
                offer="Limited time offer",
            ),
            brand_data=brand_row,
            performance_analysis=make_performance_insights(brand_id=BRAND_ID),
        )

        output = agent.produce_assets(director_input)

        # Slot 1: historical recommendation
        print(f"Recommended assets: {[a.asset_id for a in output.recommended_assets]}")

        # Slot 2: Variant A
        assert len(output.variant_a) == 1
        assert output.variant_a[0].asset_url.startswith(f"s3://{s3_bucket}/creatives/")
        assert output.variant_a[0].platform == Platform.META
        print(f"Variant A (brand-hero): {output.variant_a[0].asset_url}")
        print(f"Variant A rationale: {output.variant_a_rationale}")

        # Slot 3: Variant B
        assert len(output.variant_b) == 1
        assert output.variant_b[0].asset_url.startswith(f"s3://{s3_bucket}/creatives/")
        print(f"Variant B (lifestyle): {output.variant_b[0].asset_url}")
        print(f"Variant B rationale: {output.variant_b_rationale}")

        # Copy
        assert len(output.texts.headlines) >= 3
        assert output.texts.cta
        print(f"Headlines: {output.texts.headlines}")
        print(f"CTA: {output.texts.cta}")

    def test_brand_fetched_from_supabase_has_required_fields(self, real_services) -> None:
        """Sanity check the brand row shape before the full flow."""
        supabase = real_services["supabase"]
        brand_row = supabase.get_brand(BRAND_ID)

        assert brand_row.brand_id == BRAND_ID
        assert brand_row.brand_name
        print(f"\nBrand name: {brand_row.brand_name}")
        print(f"Primary color: {brand_row.brand_primary_color}")
        print(f"Tone of voice: {brand_row.brand_tone_of_voice}")
        print(f"Tagline: {brand_row.brand_tagline}")
        print(f"Positioning: {brand_row.brand_positioning}")
        print(f"Copywriting tone: {brand_row.brand_copywriting_tone}")
