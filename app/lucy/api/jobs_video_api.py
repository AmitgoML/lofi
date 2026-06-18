from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict

from loguru import logger
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from lucy.agents.clients.video_provider import get_video_provider
from lucy.agents.common.model_config import Models
from lucy.utils.auth import verify_jwt, extract_user_id
from lucy.database.supabase_client import (
    get_client,
    upload_file_to_storage,
    storage_file_exists,  # <- uses the helper you added
    get_user_org_profiles,
)
from lucy.database.creative_assets_client import (
    insert_generated_creative_asset_metadata,
)
from lucy.agents.video_agent import VideoAgent
from lucy.agents.common.models import SaveFileOutput


router = APIRouter()


class VideoJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    ready: bool
    file: SaveFileOutput | None = None
    signed_url: str | None = None
    error: str | None = None


def _save_temp_video(video_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes)
    tmp.close()
    return tmp.name


def _cleanup_temp(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


def _create_signed_url(
    bucket_name: str, file_path: str, expires_in_seconds: int
) -> str:
    sb = get_client()
    res = sb.storage.from_(bucket_name).create_signed_url(
        file_path,
        expires_in_seconds,
    )
    url = ""
    try:
        url = (res or {}).get("signedURL") or (res or {}).get("signedUrl") or ""
    except Exception:
        url = ""
    return str(url or "")


async def _save_generated_video_metadata(
    *,
    user_id: str,
    storage_path: str,
    file_name: str,
    tmp_path: str,
) -> None:
    """Best-effort insert into public.creative_assets for a generated video."""
    try:
        profiles = await asyncio.to_thread(get_user_org_profiles, user_id)
        org_id = None
        if isinstance(profiles, list) and profiles:
            first = profiles[0]
            if isinstance(first, dict):
                org_id = first.get("org_id")

        if not user_id or not org_id:
            logger.warning(
                "Skipping creative_assets metadata insert for video: missing user_id or org_id"
            )
            return

        file_size_bytes = os.path.getsize(tmp_path)
        await asyncio.to_thread(
            insert_generated_creative_asset_metadata,
            user_id=str(user_id),
            org_id=str(org_id),
            storage_path=storage_path,
            file_name=file_name,
            file_size_bytes=file_size_bytes,
            mime_type="video/mp4",
            asset_type="video",
            favorite=False,
        )
    except Exception as e:
        logger.warning(f"creative_assets metadata insert for video failed: {e}")


@router.get(
    "/videos/{job_id}/status",
    response_model=VideoJobStatusResponse,
)
async def get_video_job_status(
    job_id: str,
    payload: Dict[str, Any] = Depends(verify_jwt),
    save_if_ready: bool = Query(
        True,
        description="If true, download, upload to Supabase, and return a signed URL when completed",
    ),
    expires_in_seconds: int = Query(
        3600,
        ge=60,
        le=7 * 24 * 3600,
        description="Signed URL expiry in seconds",
    ),
) -> VideoJobStatusResponse:
    user_id = extract_user_id(payload)

    try:

        file_name = f"{job_id}.mp4"
        expected_path = f"users/{user_id}/{file_name}"

        # If the file already exists, skip download and upload
        exists = await asyncio.to_thread(
            storage_file_exists,
            file_path=expected_path,
        )
        if exists:
            signed = await asyncio.to_thread(
                _create_signed_url,
                "lucy-files",
                expected_path,
                int(expires_in_seconds),
            )

            file_out = SaveFileOutput(
                file_name=file_name,
                file_path=expected_path,
                file_type="video/mp4",
                job_id=job_id,
            )

            return VideoJobStatusResponse(
                job_id=job_id,
                status="completed",
                progress=100.0,
                ready=True,
                file=file_out,
                signed_url=signed or None,
            )

        provider = get_video_provider(Models.VIDEO_GENERATION)

        status_data = await provider.retrieve_status(job_id)
        status_str = status_data.get("status", "")
        progress_val = status_data.get("progress")

        if status_str in ("failed",):
            return VideoJobStatusResponse(
                job_id=job_id,
                status=status_str,
                progress=progress_val,
                ready=False,
                error="Video generation failed",
            )

        if status_str not in ("completed", "succeeded", "ready"):
            return VideoJobStatusResponse(
                job_id=job_id,
                status=status_str,
                progress=progress_val,
                ready=False,
            )

        logger.info(f"Downloading video content for job {job_id}")
        video_bytes = await provider.download_content(job_id)

        tmp_path = _save_temp_video(video_bytes)
        try:
            uploaded = await asyncio.to_thread(
                upload_file_to_storage,
                file_path=tmp_path,
                user_id=user_id,
                bucket_name="lucy-files",
                file_name=file_name,
                content_type="video/mp4",
            )

            file_out = SaveFileOutput(
                file_name=file_name,
                file_path=str(uploaded.get("file_path", expected_path)),
                file_type="video/mp4",
                job_id=job_id,
            )

            await _save_generated_video_metadata(
                user_id=user_id,
                storage_path=file_out.file_path,
                file_name=file_out.file_name,
                tmp_path=tmp_path,
            )

            signed = await asyncio.to_thread(
                _create_signed_url,
                "lucy-files",
                file_out.file_path,
                int(expires_in_seconds),
            )

            return VideoJobStatusResponse(
                job_id=job_id,
                status=status_str,
                progress=progress_val,
                ready=True,
                file=file_out,
                signed_url=signed or None,
            )
        finally:
            _cleanup_temp(tmp_path)

    except Exception as e:
        return VideoJobStatusResponse(
            job_id=job_id,
            status="failed",
            progress=100.0,
            ready=True,
            file=None,
            signed_url=None,
        )
