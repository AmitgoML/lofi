"""Creative Director Agent: owns creative strategy and delegates to sub-agents."""

from lofi.agents.sub_agents.copywriter import CopywriterAgent
from lofi.agents.sub_agents.image_generator import ImageGeneratorAgent
from lofi.agents.sub_agents.video_generator import VideoGeneratorAgent
from lofi.schemas.creative_director import CreativeDirectorInput, CreativeDirectorOutput
from lofi.state.workflow_state import WorkflowState


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
        director_input = CreativeDirectorInput(
            creative_brief=state["creative_brief"],
            performance_analysis=state["performance_insights"],
        )
        state["creative_director_output"] = self.produce_assets(director_input)
        return state

    def produce_assets(self, director_input: CreativeDirectorInput) -> CreativeDirectorOutput:
        raise NotImplementedError

    def _check_asset_reuse(self, director_input: CreativeDirectorInput) -> bool:
        raise NotImplementedError

    def _determine_creative_strategy(self, director_input: CreativeDirectorInput) -> dict:
        raise NotImplementedError
