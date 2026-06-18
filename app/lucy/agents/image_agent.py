import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.messages import ModelResponse, TextPart

from lucy.agents.clients.image_provider import get_image_provider
from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps, FileAgentOutput, SaveFileOutput
from lucy.agents.common.tools import register_user_org_profiles_tool
from lucy.database.supabase_client import (
    upload_file_to_storage,
    generate_short_uuid,
    get_client,
    resign_storage_url,
)
from lucy.database.creative_assets_client import (
    insert_generated_creative_asset_metadata,
    insert_generated_creative_asset_version,
)

# ------------------------------------------------------------
# System Prompt
# ------------------------------------------------------------
IMAGE_SYSTEM_PROMPT = """
You are Lucy, the Image Generation Agent inside Lofi (an adtech platform). You help users create and modify visual content for marketing campaigns using AI image generation.

Lofi is the platform the user operates — the user's own brand, company, and products are separate. Never conflate them.

CRITICAL RULES:
- Generate ONLY ONE image per request
- Never include URLs, file paths, or markdown links
- Only describe images in words
- Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.
- End with one contextual follow-up question that advances the user's current goal. Mention Lofi only when the next step is a Lofi action.
"""

DEFAULT_IMAGE_SIZE: str = "1024x1024"
DEFAULT_QUALITY: str = "medium"
DEFAULT_BACKGROUND: str = "transparent"
IMAGE_MODEL_NAME: str = Models.IMAGE_GENERATION


class ImageAgent(LofiAgent):
    """Factory and metadata for the Image Generation agent."""

    REMINDER_HEADER = (
        "You are Lucy Image Designer — creative, brand-aware. ONE image per request. Never use emojis or emoticons under any circumstances. "
        "Attachments: use image_modification_tool. No attachments: use image_generation_tool. "
        "NEVER include URLs or file paths in text. End with one contextual follow-up."
    )

    @staticmethod
    def _save_temp_image(image_bytes: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.close()
        return tmp.name

    @staticmethod
    def _cleanup_temp(path: str):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")

    @staticmethod
    def _upload_to_supabase(
        file_path: str, user_id: str, file_name: Optional[str] = None
    ) -> SaveFileOutput:
        """Upload file to Supabase and return metadata."""
        file_name = file_name or f"{generate_short_uuid()}.png"
        result = upload_file_to_storage(
            file_path=file_path,
            user_id=user_id,
            bucket_name="lucy-files",
            file_name=file_name,
            content_type="image/png",
        )
        return SaveFileOutput(
            file_name=file_name,
            file_path=result.get("file_path", ""),
            file_type="image/png",
        )

    @staticmethod
    def _get_previous_image_urls(ctx: RunContext[ChatDeps]) -> List[str]:
        """Extract image URLs from attachments or conversation history."""
        return [u for (u, _asset_id) in ImageAgent._get_previous_image_refs(ctx)]

    @staticmethod
    def _get_previous_image_refs(
        ctx: RunContext[ChatDeps],
    ) -> List[tuple[str, Optional[str]]]:
        """
        Return a list of (signed_url, asset_id) tuples from attachments or message history.

        Parent selection for versioning uses the first entry.
        """
        # Priority 1: attachments from current request
        if ctx.deps.attachments:
            attachments = ctx.deps.attachments
            all_refs: List[tuple[str, Optional[str]]] = []
            for attachment in attachments:
                url = None
                asset_id = None
                if isinstance(attachment, dict):
                    url = attachment.get("url")
                    asset_id = attachment.get("asset_id")
                else:
                    url = getattr(attachment, "url", None)
                    asset_id = getattr(attachment, "asset_id", None)

                if url:
                    signed_url = resign_storage_url(url) or url
                    all_refs.append((signed_url, str(asset_id) if asset_id else None))

            if all_refs:
                logger.info(f"Found {len(all_refs)} image URL(s) from attachments")
                return all_refs

        # Priority 2: fall back to conversation history
        if not ctx.deps.message_history:
            return []

        supabase = get_client()
        for msg in reversed(ctx.deps.message_history):
            if not isinstance(msg, ModelResponse):
                continue
            for part in msg.parts:
                if not isinstance(part, TextPart):
                    continue
                try:
                    data = json.loads(part.content)
                except Exception:
                    continue

                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    continue

                for item in data:
                    if isinstance(item, dict) and item.get("file_type", "").startswith(
                        "image/"
                    ):
                        file_path = item.get("file_path")
                        if not file_path:
                            continue
                        signed = supabase.storage.from_("lucy-files").create_signed_url(
                            file_path, expires_in=3600
                        )
                        url = (
                            signed.get("signedURL")
                            if isinstance(signed, dict)
                            else str(signed)
                        )
                        if url:
                            asset_id = item.get("asset_id")
                            return [(url, str(asset_id) if asset_id else None)]
        return []

    @staticmethod
    async def _save_image_metadata_best_effort(
        ctx: RunContext[ChatDeps],
        uploaded: SaveFileOutput,
        tmp_path: str,
        *,
        parent_asset_id: Optional[str] = None,
    ) -> None:
        """
        DRY wrapper around creative_assets writes.

        - Derives user_id/org_id from ctx
        - Calls DB helper (versioned when parent_asset_id provided)
        - Sets uploaded.asset_id when available
        - Never raises (best-effort)
        """
        try:
            user_id = ctx.deps.user_id or ""
            profiles = ctx.deps.user_profiles or []
            org_id = getattr(profiles[0], "org_id", None) if profiles else None
            if not user_id or not org_id:
                logger.warning(
                    "Skipping creative_assets metadata insert: missing user_id or org_id"
                )
                return

            file_size_bytes = os.path.getsize(tmp_path)
            if parent_asset_id:
                row = await asyncio.to_thread(
                    insert_generated_creative_asset_version,
                    user_id=str(user_id),
                    org_id=str(org_id),
                    parent_asset_id=str(parent_asset_id),
                    storage_path=uploaded.file_path,
                    file_name=uploaded.file_name,
                    file_size_bytes=file_size_bytes,
                    mime_type="image/png",
                    asset_type="image",
                    favorite=False,
                )
            else:
                row = await asyncio.to_thread(
                    insert_generated_creative_asset_metadata,
                    user_id=str(user_id),
                    org_id=str(org_id),
                    storage_path=uploaded.file_path,
                    file_name=uploaded.file_name,
                    file_size_bytes=file_size_bytes,
                    mime_type="image/png",
                    asset_type="image",
                    favorite=False,
                )

            asset_id = row.get("asset_id") if isinstance(row, dict) else None
            if asset_id:
                uploaded.asset_id = str(asset_id)
        except Exception as e:
            logger.warning(f"creative_assets metadata insert failed: {e}")

    # -----------------------------
    # Image Generation Logic
    # -----------------------------
    @classmethod
    async def _generate_and_upload_images(
        cls,
        ctx: RunContext[ChatDeps],
        prompt: str,
        size: str,
        quality: str,
        background: str,
    ) -> List[SaveFileOutput]:
        provider = get_image_provider(IMAGE_MODEL_NAME)
        images_bytes = await provider.generate(
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            n=1,
        )

        files: List[SaveFileOutput] = []
        for img_bytes in images_bytes:
            tmp_path = cls._save_temp_image(img_bytes)
            try:
                uploaded = cls._upload_to_supabase(tmp_path, ctx.deps.user_id)
                files.append(uploaded)
                await cls._save_image_metadata_best_effort(ctx, uploaded, tmp_path)
            finally:
                cls._cleanup_temp(tmp_path)

        return files

    @classmethod
    async def _modify_and_upload_image(
        cls,
        ctx: RunContext[ChatDeps],
        prompt: str,
        size: str,
        quality: str,
        background: str,
    ) -> List[SaveFileOutput]:
        refs = cls._get_previous_image_refs(ctx)
        if not refs:
            raise ValueError("No previous images found in conversation history")

        image_urls = [u for (u, _asset_id) in refs]
        parent_asset_id = refs[0][1]
        logger.info(f"Found {len(image_urls)} previous image(s)")

        # Download all source images
        source_images: List[bytes] = []
        for idx, image_url in enumerate(image_urls):
            logger.info(
                f"Downloading image {idx + 1}/{len(image_urls)}: {image_url[:80]}..."
            )
            response = await asyncio.to_thread(requests.get, image_url)
            if response.status_code != 200:
                logger.warning(
                    f"Failed to download image {idx + 1} (HTTP {response.status_code})"
                )
                continue
            source_images.append(response.content)

        if not source_images:
            raise ValueError("Failed to download any images")

        provider = get_image_provider(IMAGE_MODEL_NAME)
        edited_bytes = await provider.edit(
            images=source_images,
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
        )

        files: List[SaveFileOutput] = []
        for img_bytes in edited_bytes:
            tmp_out = cls._save_temp_image(img_bytes)
            try:
                uploaded = cls._upload_to_supabase(tmp_out, ctx.deps.user_id)
                files.append(uploaded)
                await cls._save_image_metadata_best_effort(
                    ctx,
                    uploaded,
                    tmp_out,
                    parent_asset_id=parent_asset_id,
                )
            finally:
                cls._cleanup_temp(tmp_out)

        return files

    # -----------------------------
    # Parameter processing
    # -----------------------------
    @classmethod
    def _process_image_params(
        cls,
        ctx: RunContext[ChatDeps],
        ratio: Optional[str] = None,
        quality_override: Optional[str] = None,
        bg_override: Optional[str] = None,
        default_size: str = DEFAULT_IMAGE_SIZE,
        default_quality: str = DEFAULT_QUALITY,
        default_background: str = DEFAULT_BACKGROUND,
    ) -> Tuple[str, str, str]:
        """
        Combine context params and tool args into final size, quality, background.

        Precedence:
        1. Tool arguments (ratio, quality_override, bg_override) if provided
        2. ctx.deps.request_params (ratio, quality, bg)
        3. Hard defaults
        """
        size: Optional[str] = default_size
        quality: Optional[str] = default_quality
        background: Optional[str] = default_background

        params = ctx.deps.request_params or {}

        # 1. request_params if present
        if "ratio" in params:
            size = params.get("ratio") or size
        if "quality" in params:
            quality = params.get("quality") or quality
        if "bg" in params:
            background = params.get("bg") or background

        # 2. explicit tool args override request_params
        if ratio is not None:
            size = ratio
        if quality_override is not None:
            quality = quality_override
        if bg_override is not None:
            background = bg_override

        # 3. normalize size based on simple keywords, but do not clobber explicit WxH
        normalized_ratio = (size or "").lower()
        if any(ch.isalpha() for ch in normalized_ratio):
            # Only apply keyword mapping if it is not a pure size like "1536x1024"
            if "landscape" in normalized_ratio:
                size = "1536x1024"
            elif "portrait" in normalized_ratio:
                size = "1024x1536"
            elif "square" in normalized_ratio:
                size = "1024x1024"
            else:
                size = DEFAULT_IMAGE_SIZE
        else:
            # If size looks like "1536x1024" leave as is, otherwise default
            size = size or DEFAULT_IMAGE_SIZE

        # 4. normalize quality
        if quality in (None, "auto"):
            quality = DEFAULT_QUALITY

        # 5. normalize background
        if background in (None, "auto"):
            background = DEFAULT_BACKGROUND

        logger.info(
            f"Using image params: size={size}, quality={quality}, background={background}"
        )
        return size, quality, background

    # -----------------------------
    # Agent creation
    # -----------------------------
    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.IMAGE)
        logger.info(f"Creating image agent with model '{model_name}'")
        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.7, max_tokens=800),
            deps_type=ChatDeps,
            system_prompt=IMAGE_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
        )

        register_user_org_profiles_tool(agent)

        @agent.tool
        async def image_generation_tool(
            ctx: RunContext[ChatDeps],
            prompt: str = Field(
                ...,
                description="Description of the image to generate (generate ONLY ONE image)",
            ),
            ratio: Optional[str] = Field(
                None,
                description=(
                    "Image orientation or ratio. Examples: "
                    "square, landscape, landscape orientation, portrait, 1536x1024"
                ),
            ),
            quality: Optional[str] = Field(
                None,
                description="Image quality: low, medium, high, or auto",
            ),
            background: Optional[str] = Field(
                None,
                description="Background setting: transparent, opaque, auto, etc.",
            ),
        ) -> Dict[str, Any]:
            """Generate and upload ONE image. Do not use if attachments are provided."""
            ctx.deps.status_queue.put_nowait("Generating image")
            try:
                if ctx.deps.attachments:
                    raise ValueError(
                        "Attachments provided - use image_modification_tool instead"
                    )

                size, quality_val, bg_val = cls._process_image_params(
                    ctx,
                    ratio=ratio,
                    quality_override=quality,
                    bg_override=background,
                )

                files = await cls._generate_and_upload_images(
                    ctx, prompt, size, quality_val, bg_val
                )
                return {
                    "success": True,
                    "files": [f.model_dump() for f in files],
                    "revised_prompt": prompt,
                }
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                return {"success": False, "error": str(e), "files": []}

        @agent.tool
        async def image_modification_tool(
            ctx: RunContext[ChatDeps],
            prompt: str = Field(..., description="Description of modifications"),
            ratio: Optional[str] = Field(
                None,
                description=(
                    "Image orientation or ratio for the modified image. Examples: "
                    "square, landscape, landscape orientation, portrait, 1536x1024"
                ),
            ),
            quality: Optional[str] = Field(
                None,
                description="Image quality for the modified image: low, medium, high, or auto",
            ),
            background: Optional[str] = Field(
                None,
                description="Background setting for the modified image.",
            ),
        ) -> Dict[str, Any]:
            """Modify an existing image based on previous attachments or history."""
            ctx.deps.status_queue.put_nowait("Modifying image")
            try:
                size, quality_val, bg_val = cls._process_image_params(
                    ctx,
                    ratio=ratio,
                    quality_override=quality,
                    bg_override=background,
                    default_background="transparent",
                )

                files = await cls._modify_and_upload_image(
                    ctx, prompt, size, quality_val, bg_val
                )
                return {
                    "success": True,
                    "files": [f.model_dump() for f in files],
                    "revised_prompt": prompt,
                }
            except Exception as e:
                logger.error(f"Image modification failed: {e}")
                return {"success": False, "error": str(e), "files": []}

        return agent
