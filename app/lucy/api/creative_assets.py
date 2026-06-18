from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Literal, Set, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger

from lucy.agents.clients.image_provider import get_image_provider
from lucy.agents.common.model_config import Models
from lucy.utils.auth import verify_jwt, extract_user_id
from lucy.database.supabase_client import get_user_org_profiles, get_client, update_storage_object_owner
from lucy.database.creative_assets_client import (
    insert_generated_creative_asset_metadata,
    insert_generated_creative_asset_variant,
    insert_generated_creative_asset_version,
)


router = APIRouter()


async def modify_image(
    image_url: str,
    prompt: str,
    size: str,
    quality: str = "medium",
    background: str = "opaque",
) -> bytes:
    """Modify an image using the configured image generation provider.

    Args:
        image_url: Signed URL of the image to modify.
        prompt: Text prompt describing the desired modification.
        size: Target size in format "widthxheight" (e.g., "1024x1024").
        quality: Image quality ("low", "medium", "high", or "auto").
        background: Background style ("opaque", "transparent", or "auto").

    Returns:
        Modified image as PNG bytes.
    """
    import requests

    response = await asyncio.to_thread(requests.get, image_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image: HTTP {response.status_code}")

    provider = get_image_provider(Models.IMAGE_GENERATION)
    results = await provider.edit(
        images=[response.content],
        prompt=prompt,
        size=size,
        quality=quality,
        background=background,
    )

    if not results:
        raise ValueError("Empty response from image provider")

    return results[0]


AssetType = Literal["image", "video", "audio"]


class CreativeAssetCreateRequest(BaseModel):
    org_id: str = Field(
        ..., description="Organization ID (must be accessible by caller)"
    )
    parent_asset_id: Optional[str] = Field(
        None, description="Optional parent asset_id to create a new version"
    )
    storage_path: str = Field(..., description="Path in Supabase Storage")
    file_name: str = Field(..., description="Original filename")
    file_size_bytes: Optional[int] = Field(None, ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="MIME type, e.g. image/png")
    asset_type: AssetType = Field(..., description="image | video | audio")
    version_notes: Optional[str] = Field(
        None, description="Optional notes about this version"
    )
    favorite: bool = Field(False, description="Mark as favorite")


class CreativeAssetResponse(BaseModel):
    """Response model for created creative assets (allows extra DB fields)."""

    model_config = ConfigDict(extra="allow")

    asset_id: str
    org_id: str
    user_id: str
    storage_path: str
    file_name: str
    mime_type: str
    asset_type: str


def _authorized_org_ids(user_id: str) -> Set[str]:
    """Get set of org_ids the user has access to."""
    rows = get_user_org_profiles(user_id) or []
    return {r["org_id"] for r in rows if isinstance(r, dict) and r.get("org_id")}


async def _get_verified_user(payload: Dict[str, Any]) -> str:
    """Extract and validate user_id from JWT payload."""
    user_id = extract_user_id(payload)
    if not user_id or user_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user identity",
        )
    return user_id


def _verify_org_access(user_id: str, org_id: str) -> None:
    """Raise 403 if user doesn't have access to org_id."""
    if org_id not in _authorized_org_ids(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for org_id",
        )


# Aspect ratio dimensions mapping
ASPECT_RATIOS = {
    "square": (1024, 1024),  # 1:1
    "portrait": (1024, 1536),  # 9:16
    "landscape": (1536, 1024),  # 16:9
}


class CreateVariantsRequest(BaseModel):
    org_id: str = Field(..., description="Organization ID")
    parent_asset_id: str = Field(
        ..., description="Asset ID from creative_assets table"
    )
    storage_path: str = Field(..., description="Source image storage path")
    storage_bucket: str = Field("creative", description="Storage bucket name")
    formats: List[Literal["square", "portrait", "landscape"]] = Field(
        ..., description="List of aspect ratios to generate"
    )
    version_notes: Optional[str] = Field(
        None, description="Optional notes about these variants"
    )


class VariantResponse(BaseModel):
    asset_id: str
    storage_path: str
    storage_bucket: str
    file_name: str
    mime_type: str
    format: str
    width: int
    height: int


class CreateVariantsResponse(BaseModel):
    success: bool
    variants: List[VariantResponse]
    message: str


def _get_signed_url_for_storage_path(
    storage_path: str, bucket: str = "creative", expires_in: int = 3600
) -> str:
    """Get a signed URL for a storage path."""
    sb = get_client()
    try:
        result = sb.storage.from_(bucket).create_signed_url(storage_path, expires_in)
        if isinstance(result, dict):
            return result.get("signedURL", "")
        return str(result)
    except Exception as e:
        logger.error(f"Error creating signed URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create signed URL: {str(e)}",
        )


def _check_upload_error(response: Any) -> None:
    """Raise ValueError if upload response indicates an error."""
    error = response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
    if error:
        raise ValueError(f"Storage upload error: {error}")


async def _create_single_variant(
    user_id: str,
    org_id: str,
    parent_asset_id: str,
    signed_url: str,
    format_name: str,
    target_width: int,
    target_height: int,
    version_notes: Optional[str],
) -> Optional[VariantResponse]:
    """Create a single aspect ratio variant. Returns None on failure."""
    from lucy.database.supabase_client import generate_short_uuid
    
    # Generate variant image using OpenAI
    prompt = f"Resize this image to {format_name} format ({target_width}x{target_height}) without changing the content. Center crop if needed."
    img_bytes = await modify_image(
        image_url=signed_url,
        prompt=prompt,
        size=f"{target_width}x{target_height}",
        quality="medium",
        background="opaque",
    )
    
    # Upload to storage (user_id prefix required for RLS policies)
    short_uuid = generate_short_uuid()
    file_name = f"{user_id}_{format_name}_{target_width}x{target_height}_{short_uuid}.png"
    bucket_name = "creative"
    storage_path = f"images/{file_name}"
    
    sb = get_client()
    upload_response = sb.storage.from_(bucket_name).upload(
        path=storage_path,
        file=img_bytes,
        file_options={"content-type": "image/png"},
    )
    _check_upload_error(upload_response)
    
    # Update owner_id so the file shows up in the user's asset library
    # (service role uploads don't automatically set owner_id to the user)
    update_storage_object_owner(storage_path, user_id, bucket_name)
    
    # Create metadata record
    notes = version_notes or f"Generated {format_name} variant ({target_width}x{target_height})"
    row = insert_generated_creative_asset_variant(
        user_id=user_id,
        org_id=org_id,
        parent_asset_id=parent_asset_id,
        storage_path=storage_path,
        file_name=file_name,
        file_size_bytes=len(img_bytes),
        mime_type="image/png",
        asset_type="image",
        version_notes=notes,
        favorite=False,
    )
    
    if not row:
        return None
    
    return VariantResponse(
        asset_id=row.get("asset_id", ""),
        storage_path=storage_path,
        storage_bucket=bucket_name,
        file_name=file_name,
        mime_type="image/png",
        format=format_name,
        width=target_width,
        height=target_height,
    )


@router.post(
    "/creative-assets/variants",
    response_model=CreateVariantsResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_asset_variants(
    req: CreateVariantsRequest,
    payload: Dict[str, Any] = Depends(verify_jwt),
) -> CreateVariantsResponse:
    """
    Generate aspect ratio variants of an existing creative asset using Lucy image agent.
    
    Creates a signed URL for the original image, calls the image agent's modification
    tool for each format, and creates metadata records with versioning.
    """
    user_id = await _get_verified_user(payload)
    _verify_org_access(user_id, req.org_id)
    
    if not req.formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one format must be specified",
        )
    
    # Get signed URL for the source image
    signed_url = _get_signed_url_for_storage_path(req.storage_path, req.storage_bucket)
    if not signed_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access original image",
        )
    
    variants: List[VariantResponse] = []
    
    for format_name in req.formats:
        if format_name not in ASPECT_RATIOS:
            logger.warning(f"Unknown format: {format_name}, skipping")
            continue
        
        target_width, target_height = ASPECT_RATIOS[format_name]
        try:
            variant = await _create_single_variant(
                user_id=user_id,
                org_id=req.org_id,
                parent_asset_id=req.parent_asset_id,
                signed_url=signed_url,
                format_name=format_name,
                target_width=target_width,
                target_height=target_height,
                version_notes=req.version_notes,
            )
            if variant:
                variants.append(variant)
        except Exception as e:
            logger.error(f"Error creating {format_name} variant: {e}")
            continue
    
    if not variants:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create any variants",
        )
    
    return CreateVariantsResponse(
        success=True,
        variants=variants,
        message=f"Successfully created {len(variants)} variant(s)",
    )


class CreateVersionRequest(BaseModel):
    prompt: str = Field(..., description="Description of modifications for new version")
    size: Optional[str] = Field(None, description="Output size (e.g., '1024x1024')")
    quality: Optional[str] = Field("medium", description="low, medium, high")
    background: Optional[str] = Field("opaque", description="opaque, transparent")


@router.post(
    "/creative-assets/{asset_id}/versions",
    response_model=CreativeAssetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_asset_version(
    asset_id: str,
    req: CreateVersionRequest,
    payload: Dict[str, Any] = Depends(verify_jwt),
) -> CreativeAssetResponse:
    """
    Create a new version of an asset using AI image modification.
    
    This creates a new asset record with:
    - Incremented version_number
    - parent_asset_id pointing to the root asset
    - is_latest = true (previous versions marked false)
    """
    user_id = await _get_verified_user(payload)
    
    sb = get_client()
    
    # Fetch the source asset
    try:
        result = (
            sb.table("creative_assets")
            .select("*")
            .eq("asset_id", asset_id)
            .eq("is_deleted", False)
            .limit(1)
            .execute()
        )
        rows = getattr(result, "data", None) or []
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source asset not found",
            )
        source_asset = rows[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source asset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch source asset",
        )
    
    # Verify user has access to the org
    org_id = source_asset.get("org_id")
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source asset missing org_id",
        )
    _verify_org_access(user_id, org_id)
    
    # Get storage path and bucket
    storage_path = source_asset.get("storage_path")
    storage_bucket = source_asset.get("storage_bucket", "creative")
    
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source asset missing storage_path",
        )
    
    # Get signed URL for the source image
    signed_url = _get_signed_url_for_storage_path(storage_path, storage_bucket)
    if not signed_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access source image",
        )
    
    # Determine output size (default to source asset dimensions if available)
    output_size = req.size
    if not output_size:
        # Try to infer from source asset or use default
        output_size = "1024x1024"
    
    # Call OpenAI to modify the image
    try:
        img_bytes = await modify_image(
            image_url=signed_url,
            prompt=req.prompt,
            size=output_size,
            quality=req.quality,
            background=req.background,
        )
    except Exception as e:
        logger.error(f"Error modifying image with OpenAI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate new version: {str(e)}",
        )
    
    # Upload to storage
    from lucy.database.supabase_client import generate_short_uuid
    
    short_uuid = generate_short_uuid()
    file_name = f"{user_id}_v{short_uuid}.png"
    new_storage_path = f"images/{file_name}"
    
    try:
        upload_response = sb.storage.from_(storage_bucket).upload(
            path=new_storage_path,
            file=img_bytes,
            file_options={"content-type": "image/png"},
        )
        _check_upload_error(upload_response)
        
        # Update owner_id so the file shows up in the user's asset library
        # (service role uploads don't automatically set owner_id to the user)
        update_storage_object_owner(new_storage_path, user_id, storage_bucket)
    except Exception as e:
        logger.error(f"Error uploading new version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload new version: {str(e)}",
        )
    
    # Create version record
    version_notes = f"AI-modified: {req.prompt[:100]}"
    try:
        row = insert_generated_creative_asset_version(
            user_id=user_id,
            org_id=org_id,
            parent_asset_id=asset_id,
            storage_path=new_storage_path,
            file_name=file_name,
            file_size_bytes=len(img_bytes),
            mime_type="image/png",
            asset_type="image",
            version_notes=version_notes,
            favorite=False,
        )
    except Exception as e:
        logger.error(f"Error creating version metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create version metadata: {str(e)}",
        )
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create version record",
        )
    
    return CreativeAssetResponse(**row)


class GenerateStandaloneRequest(BaseModel):
    org_id: str = Field(..., description="Organization ID")
    source_asset_id: str = Field(..., description="Asset ID to use as base image")
    prompt: str = Field(..., description="Description of desired variant")
    name: Optional[str] = Field(None, description="Display name for the variant")
    size: Optional[str] = Field(None, description="Output size (e.g., '1024x1024')")
    quality: Optional[str] = Field("medium", description="low, medium, high")
    background: Optional[str] = Field("opaque", description="opaque, transparent")


@router.post(
    "/creative-assets/generate-standalone",
    response_model=CreativeAssetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_standalone_variant(
    req: GenerateStandaloneRequest,
    payload: Dict[str, Any] = Depends(verify_jwt),
) -> CreativeAssetResponse:
    """
    Generate a standalone variant of an asset using AI image modification.
    
    Unlike create_asset_version, this creates a completely new root asset
    (not a child of the current asset family).
    
    This creates a new asset record with:
    - No parent_asset_id (independent asset)
    - version_number = 1 (it's a new root)
    - is_latest = true
    """
    user_id = await _get_verified_user(payload)
    _verify_org_access(user_id, req.org_id)
    
    sb = get_client()
    
    # Fetch the source asset
    try:
        result = (
            sb.table("creative_assets")
            .select("*")
            .eq("asset_id", req.source_asset_id)
            .eq("is_deleted", False)
            .limit(1)
            .execute()
        )
        rows = getattr(result, "data", None) or []
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source asset not found",
            )
        source_asset = rows[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source asset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch source asset",
        )
    
    # Verify user has access to the org
    source_org_id = source_asset.get("org_id")
    if not source_org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source asset missing org_id",
        )
    _verify_org_access(user_id, source_org_id)
    
    # Get storage path and bucket
    storage_path = source_asset.get("storage_path")
    storage_bucket = source_asset.get("storage_bucket", "creative")
    
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source asset missing storage_path",
        )
    
    # Get signed URL for the source image
    signed_url = _get_signed_url_for_storage_path(storage_path, storage_bucket)
    if not signed_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to access source image",
        )
    
    # Determine output size
    output_size = req.size
    if not output_size:
        output_size = "1024x1024"
    
    # Call OpenAI to modify the image
    try:
        img_bytes = await modify_image(
            image_url=signed_url,
            prompt=req.prompt,
            size=output_size,
            quality=req.quality,
            background=req.background,
        )
    except Exception as e:
        logger.error(f"Error modifying image with OpenAI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate variant: {str(e)}",
        )
    
    # Upload to storage
    from lucy.database.supabase_client import generate_short_uuid
    
    short_uuid = generate_short_uuid()
    # Use provided name or generate one
    if req.name:
        # Sanitize the name for use in storage path
        sanitized_name = "".join(c if c.isalnum() or c in " -_" else "" for c in req.name)
        sanitized_name = sanitized_name.strip().replace(" ", "_")[:50]  # Limit length
        file_name = f"{sanitized_name}_{short_uuid}.png"
    else:
        file_name = f"{user_id}_variant_{short_uuid}.png"
    new_storage_path = f"images/{file_name}"
    
    # Determine the display name for the variant
    display_name = req.name if req.name else f"Variant of {source_asset.get('file_name', 'asset')}"
    
    try:
        upload_response = sb.storage.from_(storage_bucket).upload(
            path=new_storage_path,
            file=img_bytes,
            file_options={
                "content-type": "image/png",
                "x-upsert": "true",
            },
        )
        _check_upload_error(upload_response)
        
        # Update owner_id and user_metadata so the file shows up in the user's asset library
        # (service role uploads don't automatically set owner_id to the user)
        update_storage_object_owner(
            new_storage_path,
            user_id,
            storage_bucket,
            user_metadata={
                "name": display_name,
                "campaign_usage": [],
                "favorite": False,
            },
        )
    except Exception as e:
        logger.error(f"Error uploading variant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload variant: {str(e)}",
        )
    
    # Create standalone asset record (no parent_asset_id)
    version_notes = f"AI-generated variant: {req.prompt[:100]}"
    try:
        row = insert_generated_creative_asset_metadata(
            user_id=user_id,
            org_id=req.org_id,
            storage_path=new_storage_path,
            file_name=file_name,
            file_size_bytes=len(img_bytes),
            mime_type="image/png",
            asset_type="image",
            version_notes=version_notes,
            favorite=False,
        )
    except Exception as e:
        logger.error(f"Error creating standalone variant metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create variant metadata: {str(e)}",
        )
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create variant record",
        )
    
    return CreativeAssetResponse(**row)


