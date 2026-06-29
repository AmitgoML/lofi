"""Creative Director Agent: owns creative strategy and delegates to sub-agents."""

from pathlib import Path

from lofi.agents.sub_agents.copywriter import CopywriterAgent
from lofi.agents.sub_agents.image_generator import ImageGeneratorAgent
from lofi.agents.sub_agents.video_generator import VideoGeneratorAgent
from lofi.schemas.common import CreativeFormat, Platform
from lofi.schemas.creative_director import (
    AssetDecision,
    AssetRef,
    CreativeDirectorInput,
    CreativeDirectorOutput,
    TextAsset,
)
from lofi.state.workflow_state import WorkflowState

# Folder of placeholder creative assets used by produce_static_sample() while
# the real generation sub-agents (CopywriterAgent, ImageGeneratorAgent,
# VideoGeneratorAgent) are still unimplemented stubs.
STATIC_CREATIVES_DIR = Path(__file__).parent / "static_creatives"
VALVOLINE_SAMPLE_IMAGE = STATIC_CREATIVES_DIR / "valvoline_sample.jpg"


class CreativeDirectorAgent:
    """Decides asset reuse vs. generation and coordinates copy/image/video sub-agents."""

    def __init__(
        self,
        copywriter: CopywriterAgent | None = None,
        image_generator: ImageGeneratorAgent | None = None,
        video_generator: VideoGeneratorAgent | None = None,
    ) -> None:
        self.copywriter = copywriter or CopywriterAgent()
        self.image_generator = image_generator or ImageGeneratorAgent()
        self.video_generator = video_generator or VideoGeneratorAgent()

    def run(self, state: WorkflowState) -> WorkflowState:
        # produce_assets() is still an unimplemented stub (no live
        # copywriter/image-generation backend yet), so this node serves the
        # static sample output instead of crashing the graph.
        state["creative_director_output"] = self.produce_static_sample()
        return state

    def produce_assets(self, director_input: CreativeDirectorInput) -> CreativeDirectorOutput:
        raise NotImplementedError

    def _check_asset_reuse(self, director_input: CreativeDirectorInput) -> bool:
        raise NotImplementedError

    def _determine_creative_strategy(self, director_input: CreativeDirectorInput) -> dict:
        raise NotImplementedError

    @staticmethod
    def produce_static_sample() -> CreativeDirectorOutput:
        """Sample CreativeDirectorOutput backed by a static asset on disk.

        Stands in for produce_assets() (still a NotImplementedError stub
        above) wherever a caller needs a real CreativeDirectorOutput-shaped
        value without a live LLM/image-generation backend - e.g. local UI
        development or manual testing.
        """
        return CreativeDirectorOutput(
            asset_decisions=[
                AssetDecision(creative_format=CreativeFormat.IMAGE, action="reuse", reused_asset_id="valvoline_sample")
            ],
            best_creative_format=CreativeFormat.IMAGE,
            best_messaging_angle="Trusted protection for every mile",
            assets=[
                AssetRef(
                    asset_url=str(VALVOLINE_SAMPLE_IMAGE),
                    creative_format=CreativeFormat.IMAGE,
                    platform=Platform.META,
                )
            ],
            texts=TextAsset(
                headlines=[
                    "Valvoline: Protection That Goes the Distance",
                    "Change Your Oil. Change Your Drive.",
                ],
                descriptions=[
                    "Valvoline's advanced full synthetic formula protects your engine "
                    "for up to 12,000 miles, so you can drive with confidence between "
                    "every change.",
                ],
                cta="Find a Location Near You",
                hooks=["Your engine works hard. Give it the protection it deserves."],
                keywords=["synthetic oil", "oil change", "engine protection", "valvoline"],
                long_headlines=[
                    "Valvoline Full Synthetic: Engineered to Protect Your Engine for the Long Haul",
                ],
            ),
        )
