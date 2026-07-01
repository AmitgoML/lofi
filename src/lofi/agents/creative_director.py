"""Creative Director Agent: owns creative strategy and coordinates sub-agents."""

from pathlib import Path

from lofi.agents.sub_agents.copywriter import CopywriterAgent
from lofi.agents.sub_agents.image_generator import ImageGeneratorAgent
from lofi.agents.sub_agents.video_generator import VideoGeneratorAgent
from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.s3_storage import S3CreativeStorage
from lofi.persistence.supabase_client import SupabaseClient
from lofi.prompts.creative_director import creative_strategy_v1
from lofi.schemas.common import AudienceSpec, CreativeFormat, Platform
from lofi.schemas.creative_director import (
    AssetDecision,
    CreativeBrief,
    CreativeDirectorInput,
    CreativeDirectorOutput,
    CreativeStrategy,
    RecommendedAsset,
    TextAsset,
)
from lofi.state.workflow_state import WorkflowState

# Folder of placeholder creative assets used by produce_static_sample() while
# the real generation sub-agents (CopywriterAgent, ImageGeneratorAgent,
# VideoGeneratorAgent) are still unimplemented stubs.
STATIC_CREATIVES_DIR = Path(__file__).parent / "static_creatives"
VALVOLINE_SAMPLE_IMAGE = STATIC_CREATIVES_DIR / "valvoline_sample.jpg"


class CreativeDirectorAgent:
    """Fetches brand data, determines A/B creative strategy, and delegates to sub-agents."""

    def __init__(
        self,
        bedrock_client: BedrockClient,
        supabase_client: SupabaseClient,
        s3_storage: S3CreativeStorage,
        copywriter: CopywriterAgent | None = None,
        image_generator: ImageGeneratorAgent | None = None,
        video_generator: VideoGeneratorAgent | None = None,
    ) -> None:
        self._bedrock_client = bedrock_client
        self._supabase_client = supabase_client
        self._s3_storage = s3_storage
        self.copywriter = copywriter or CopywriterAgent(bedrock_client)
        self.image_generator = image_generator or ImageGeneratorAgent(bedrock_client, s3_storage)
        self.video_generator = video_generator or VideoGeneratorAgent()

    def run(self, state: WorkflowState) -> WorkflowState:
        performance_insights = state["performance_insights"]
        campaign_brief = state["campaign_brief"]

        brand_id = performance_insights.brand_id
        if not brand_id:
            raise ValueError(
                f"No brand_id found in performance_insights for brand '{campaign_brief.brand}'. "
                "Ensure the brand exists in the brands table."
            )

        brand_data = self._supabase_client.get_brand(brand_id)

        creative_brief = state.get("creative_brief") or CreativeBrief(
            goal=campaign_brief.goal,
            audience=campaign_brief.target_audience or AudienceSpec(age_min=18, age_max=65),
            platforms=campaign_brief.platforms or [Platform.META],
            offer=None,
        )

        director_input = CreativeDirectorInput(
            creative_brief=creative_brief,
            existing_campaign_assets=[],
            brand_data=brand_data,
            performance_analysis=performance_insights,
        )

        state["creative_director_output"] = self.produce_assets(director_input)
        return state

    def produce_assets(self, director_input: CreativeDirectorInput) -> CreativeDirectorOutput:
        strategy = self._determine_creative_strategy(director_input)
        brand = director_input.brand_data
        platforms = director_input.creative_brief.platforms

        # Slot 1: historical best-performing assets (no generation — just references)
        recommended_assets = [
            RecommendedAsset(
                asset_id=rec.asset_id,
                creative_format=rec.creative_format,
                historical_engagement_rate=rec.historical_engagement_rate,
            )
            for rec in director_input.performance_analysis.creative_recommendations
            if rec.asset_id
        ]

        # Slot 2: Variant A — brand-hero (product front and center, dominant brand colors)
        variant_a_assets = self.image_generator.generate(
            prompt=strategy.variant_a.image_prompt,
            platforms=platforms,
            brand_name=brand.brand_name,
            negative_prompt=strategy.variant_a.negative_prompt,
        )

        # Slot 3: Variant B — lifestyle/context (aspirational scene, subtle branding)
        variant_b_assets = self.image_generator.generate(
            prompt=strategy.variant_b.image_prompt,
            platforms=platforms,
            brand_name=brand.brand_name,
            negative_prompt=strategy.variant_b.negative_prompt,
        )

        texts = self.copywriter.generate(
            audience=director_input.creative_brief.audience,
            goal=director_input.creative_brief.goal,
            platform=(
                platforms[0] if platforms else Platform.META
            ),
            offer=director_input.creative_brief.offer,
            brand_data=brand,
        )

        return CreativeDirectorOutput(
            recommended_assets=recommended_assets,
            variant_a=variant_a_assets,
            variant_b=variant_b_assets,
            best_creative_format=strategy.best_creative_format,
            best_messaging_angle=strategy.best_messaging_angle,
            variant_a_rationale=strategy.variant_a.rationale,
            variant_b_rationale=strategy.variant_b.rationale,
            asset_decisions=strategy.asset_decisions,
            texts=texts,
        )

    def _determine_creative_strategy(self, director_input: CreativeDirectorInput) -> CreativeStrategy:
        existing_assets = [
            {
                "format": a.creative_format.value,
                "tags": a.tags,
                "score": a.performance_score,
                "asset_id": a.asset_id,
            }
            for a in director_input.existing_campaign_assets
        ]
        prompt = creative_strategy_v1(
            brief=director_input.creative_brief,
            brand=director_input.brand_data,
            perf=director_input.performance_analysis,
            existing_assets=existing_assets,
        )
        return self._bedrock_client.extract_structured(prompt, CreativeStrategy)
