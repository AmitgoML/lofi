"""Provider-agnostic video generation interface.

To add a new provider (e.g. fal.ai Kling, Google Veo, Runway):
  1. Implement the VideoProvider Protocol below.
  2. Add a branch to get_video_provider() that matches the model string.

Model string conventions:
    "sora-*"        -> OpenAIVideoProvider
    "kling-*"       -> FalVideoProvider (not yet implemented)
    "veo-*"         -> GoogleVideoProvider (not yet implemented)
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class VideoProvider(Protocol):
    """Protocol for video generation providers."""

    async def create(self, *, prompt: str, seconds: int, size: str) -> str:
        """Start a video generation job. Returns an opaque job_id."""
        ...

    async def retrieve_status(self, job_id: str) -> dict:
        """Poll job status. Returns a dict with at least 'status' (str) and
        optionally 'progress' (float 0-100)."""
        ...

    async def download_content(self, job_id: str) -> bytes:
        """Download completed video bytes for the given job_id."""
        ...


class OpenAIVideoProvider:
    """Video provider backed by the OpenAI Videos API (Sora)."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def create(self, *, prompt: str, seconds: int, size: str) -> str:
        from lucy.agents.clients.openai_client import get_openai_client

        client = get_openai_client()
        video = await asyncio.to_thread(
            client.videos.create,
            model=self.model,
            prompt=prompt,
            seconds=str(seconds),
            size=size,
        )
        logger.info(f"Video generation started: {video.id} ({video.status})")
        if getattr(video, "status", "") == "failed":
            raise RuntimeError("Video generation failed immediately")
        return video.id

    async def retrieve_status(self, job_id: str) -> dict:
        from lucy.agents.clients.openai_client import get_openai_client

        client = get_openai_client()
        video = await asyncio.to_thread(client.videos.retrieve, job_id)
        status_str = str(getattr(video, "status", "") or "")
        progress = getattr(video, "progress", None)
        try:
            progress_val = float(progress) if progress is not None else None
        except Exception:
            progress_val = None

        return {
            "status": status_str,
            "progress": progress_val,
        }

    async def download_content(self, job_id: str) -> bytes:
        import requests as req
        from lucy.agents.clients.openai_client import get_openai_client

        client = get_openai_client()
        url = f"https://api.openai.com/v1/videos/{job_id}/content"
        headers = {"Authorization": f"Bearer {client.api_key}"}
        resp = await asyncio.to_thread(req.get, url, headers=headers)
        resp.raise_for_status()
        return resp.content


def get_video_provider(model: str) -> VideoProvider:
    """Return the appropriate VideoProvider for the given model string.

    Routing rules (first match wins):
        "sora-*"    -> OpenAIVideoProvider
        default     -> OpenAIVideoProvider (safe fallback)

    To add fal.ai Kling, Google Veo, or Runway support, add branches here
    and implement the corresponding provider class above.
    """
    model_lower = model.lower()
    if model_lower.startswith("sora"):
        return OpenAIVideoProvider(model)

    logger.warning(
        f"No specific VideoProvider for model '{model}', falling back to OpenAI."
    )
    return OpenAIVideoProvider(model)
