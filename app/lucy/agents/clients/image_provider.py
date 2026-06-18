"""Provider-agnostic image generation and editing interface.

To add a new provider (e.g. fal.ai, Google Imagen):
  1. Implement the ImageProvider Protocol below.
  2. Add a branch to get_image_provider() that matches the model string.

Model string conventions:
    "gpt-image-*"     -> OpenAIImageProvider
    "fal/*"           -> FalImageProvider (not yet implemented)
    "imagen-*"        -> GoogleImageProvider (not yet implemented)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import List, Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class ImageProvider(Protocol):
    """Protocol for image generation and editing providers."""

    async def generate(
        self,
        *,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        n: int = 1,
    ) -> List[bytes]:
        """Generate new images from a text prompt. Returns raw PNG bytes per image."""
        ...

    async def edit(
        self,
        *,
        images: List[bytes],
        prompt: str,
        size: str,
        quality: str,
        background: str,
    ) -> List[bytes]:
        """Edit/modify existing images. Returns raw PNG bytes per result."""
        ...


class OpenAIImageProvider:
    """Image provider backed by the OpenAI Images API (gpt-image-*)."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def generate(
        self,
        *,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        n: int = 1,
    ) -> List[bytes]:
        import tempfile
        import os
        from lucy.agents.clients.openai_client import get_openai_client

        client = get_openai_client()
        response = await asyncio.to_thread(
            client.images.generate,
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            n=n,
        )
        if not response.data:
            raise ValueError("Empty response from image model")

        return [base64.b64decode(img.b64_json) for img in response.data]

    async def edit(
        self,
        *,
        images: List[bytes],
        prompt: str,
        size: str,
        quality: str,
        background: str,
    ) -> List[bytes]:
        import tempfile
        import os
        from lucy.agents.clients.openai_client import get_openai_client

        client = get_openai_client()

        tmp_paths: List[str] = []
        for img_bytes in images:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            tmp_paths.append(tmp.name)

        try:
            with contextlib.ExitStack() as stack:
                opened = [stack.enter_context(open(p, "rb")) for p in tmp_paths]
                resp = await asyncio.to_thread(
                    client.images.edit,
                    model=self.model,
                    image=opened if len(opened) > 1 else opened[0],
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    background=background,
                )
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        if not resp.data:
            raise ValueError("Empty response from image model on edit")

        return [base64.b64decode(img.b64_json) for img in resp.data]


def get_image_provider(model: str) -> ImageProvider:
    """Return the appropriate ImageProvider for the given model string.

    Routing rules (first match wins):
        "gpt-image-*"   -> OpenAIImageProvider
        "dall-e-*"      -> OpenAIImageProvider
        default         -> OpenAIImageProvider (safe fallback for unknown models)

    To add fal.ai or Google Imagen support, add branches here and implement
    the corresponding provider class above.
    """
    model_lower = model.lower()
    if model_lower.startswith("gpt-image") or model_lower.startswith("dall-e"):
        return OpenAIImageProvider(model)

    logger.warning(
        f"No specific ImageProvider for model '{model}', falling back to OpenAI."
    )
    return OpenAIImageProvider(model)
