import asyncio
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from lucy.agents.clients.video_provider import get_video_provider
from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps, FileAgentOutput, SaveFileOutput
from lucy.agents.common.tools import register_user_org_profiles_tool

# ------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------
VIDEO_SYSTEM_PROMPT = """
You are Lucy, the Video Generation Agent inside Lofi (an adtech platform). You help users create cinematic video content for marketing campaigns using AI video generation with Sora 2.

Lofi is the platform the user operates — the user's own brand, company, and products are separate. Never conflate them.

CRITICAL RULES:
- Never include URLs, file paths, or markdown links
- Only describe videos in words
- Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.
- End with one contextual follow-up question that advances the user's current goal. Mention Lofi only when the next step is a Lofi action.
"""

DEFAULT_VIDEO_SECONDS: int = 4
DEFAULT_VIDEO_SIZE: str = "1280x720"


class VideoAgent(LofiAgent):
    """Factory and metadata for the Video Generation agent."""

    REMINDER_HEADER = (
        "You are Lucy Video Designer — creative, brand-aware. ONE video per request. Never use emojis or emoticons under any circumstances. "
        "NEVER include URLs or file paths in text. Ensure job_id is in the files array. "
        "End with one contextual follow-up question."
    )

    # -----------------------------
    # Video Generation Logic
    # -----------------------------
    @classmethod
    async def _generate_video_job(
        cls,
        ctx: RunContext[ChatDeps],
        prompt: str,
        seconds: int,
        size: str,
    ) -> str:
        provider = get_video_provider(Models.VIDEO_GENERATION)
        return await provider.create(prompt=prompt, seconds=seconds, size=size)

    # -----------------------------
    # Parameter processing
    # -----------------------------
    @classmethod
    def _process_video_params(
        cls,
        ctx: RunContext[ChatDeps],
        default_seconds: int = DEFAULT_VIDEO_SECONDS,
        default_size: str = DEFAULT_VIDEO_SIZE,
    ):
        """Process video parameters from request_params or use defaults."""
        seconds = default_seconds
        size = default_size

        if ctx.deps.request_params:
            params = ctx.deps.request_params
            seconds = params.get("duration", seconds)
            size = params.get("ratio", size)

            # Map descriptive size words to valid pixel dimensions
            # Sora 2 only supports: 1280x720 (16:9), 720x1280 (9:16)
            size_mapping = {
                "auto": "1280x720",  # Default HD landscape 16:9
                "landscape": "1280x720",  # HD landscape 16:9
                "portrait": "720x1280",  # Portrait format 9:16
                "16:9": "1280x720",  # HD landscape 16:9
                "9:16": "720x1280",  # Portrait format 9:16
            }

            size = size_mapping.get(size, "1280x720")

            # Validate size is one of the supported values
            valid_sizes = {"720x1280", "1280x720"}
            if size not in valid_sizes:
                size = "1280x720"

            # Ensure seconds is an integer
            if isinstance(seconds, str):
                seconds = seconds.replace("s", "")
                try:
                    seconds = int(float(seconds))
                except (ValueError, TypeError):
                    seconds = default_seconds
            elif isinstance(seconds, float):
                seconds = int(seconds)

            # Clamp seconds to valid Sora 2 values: 4, 8, 12
            valid_seconds = [4, 8, 12]
            if seconds not in valid_seconds:
                # Find nearest valid value
                seconds = min(valid_seconds, key=lambda x: abs(x - seconds))

            logger.info(f"Using request_params: seconds={seconds}, size={size}")

        return seconds, size

    # -----------------------------
    # Agent creation
    # -----------------------------
    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.VIDEO)
        logger.info(f"Creating video agent with model '{model_name}'")
        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.7, max_tokens=800),
            deps_type=ChatDeps,
            system_prompt=VIDEO_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
        )

        register_user_org_profiles_tool(agent)

        @agent.tool
        async def video_generation_tool(
            ctx: RunContext[ChatDeps],
            prompt: str = Field(
                ..., description="Description of the video to generate"
            ),
        ) -> Dict[str, Any]:
            """Generate and upload videos using Sora 2."""
            ctx.deps.status_queue.put_nowait("Generating video")
            try:
                seconds, size = cls._process_video_params(ctx)

                job_id = await cls._generate_video_job(ctx, prompt, seconds, size)

                # Return a properly shaped file entry so the output_validator
                # can inject it into FileAgentOutput.files reliably, regardless
                # of how the LLM interprets the tool result.
                file_entry = SaveFileOutput(
                    file_name=f"{job_id}.mp4",
                    file_path="",
                    file_type="video/mp4",
                    job_id=job_id,
                )
                return {
                    "success": True,
                    "files": [file_entry.model_dump()],
                    "job_id": job_id,
                    "revised_prompt": prompt,
                    "seconds": seconds,
                    "size": size,
                }
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                return {"success": False, "error": str(e), "files": []}

        @agent.output_validator
        async def _ensure_video_file(
            ctx: RunContext[ChatDeps],
            output: FileAgentOutput,
        ) -> FileAgentOutput:
            """Guarantee that a successful video job always appears in output.files.

            The LLM may not reliably propagate the file entry from the tool result
            into the structured output.  We scan the tool-call history for a
            successful job_id and inject the entry if it is missing.
            """
            # If files are already present and at least one has a job_id, trust them.
            if any(f.job_id for f in output.files):
                return output

            # Walk the message history to find the most recent successful job_id
            # returned by video_generation_tool.
            job_id: Optional[str] = None
            try:
                from pydantic_ai.messages import (
                    ModelResponse,
                    ToolCallPart,
                    ToolReturnPart,
                )
                import json as _json

                for msg in reversed(ctx.messages):
                    for part in getattr(msg, "parts", []):
                        if isinstance(part, ToolReturnPart) and part.tool_name == "video_generation_tool":
                            try:
                                data = _json.loads(part.content) if isinstance(part.content, str) else part.content
                                if isinstance(data, dict) and data.get("success") and data.get("job_id"):
                                    job_id = str(data["job_id"])
                            except Exception:
                                pass
                        if job_id:
                            break
                    if job_id:
                        break
            except Exception:
                pass

            if job_id:
                logger.info(f"[VideoAgent] Injecting missing file entry for job_id={job_id}")
                output.files = [
                    SaveFileOutput(
                        file_name=f"{job_id}.mp4",
                        file_path="",
                        file_type="video/mp4",
                        job_id=job_id,
                    )
                ]
            return output

        return agent
