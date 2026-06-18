from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional, Literal

from loguru import logger

from lucy.database.supabase_client import get_client

AssetType = Literal["image", "video", "audio"]


def list_creative_assets(
    *,
    org_id: str,
    campaign_id: Optional[str] = None,
    asset_type: Optional[str] = None,
    source: Optional[str] = None,
    latest_only: bool = True,
    limit: int = 20,
) -> List[dict[str, Any]]:
    """Query creative assets for an organization with optional filters.

    Returns all metadata columns with signed URLs for each asset.
    """
    sb = get_client()
    query = (
        sb.table("creative_assets")
        .select("*")
        .eq("org_id", org_id)
        .eq("is_deleted", False)
    )
    if asset_type:
        query = query.eq("asset_type", asset_type)
    if source:
        query = query.eq("source", source)
    if latest_only:
        query = query.eq("is_latest", True)
    query = query.order("created_at", desc=True).limit(limit)

    try:
        result = query.execute()
        rows = getattr(result, "data", None) or []
    except Exception as e:
        logger.warning(f"list_creative_assets failed: {e}")
        return []

    sb = get_client()
    enriched: List[dict[str, Any]] = []
    for row in rows:
        storage_path = row.get("storage_path", "")
        signed_url = ""
        if storage_path:
            try:
                result = sb.storage.from_("lucy-files").create_signed_url(
                    storage_path, expires_in=3600
                )
                if isinstance(result, dict):
                    signed_url = (
                        result.get("signedURL")
                        or result.get("signed_url")
                        or ""
                    )
                elif isinstance(result, str):
                    signed_url = result
            except Exception:
                pass
        row["signed_url"] = signed_url
        enriched.append(row)

    return enriched


def insert_generated_creative_asset_metadata(
    *,
    user_id: str,
    org_id: str,
    storage_path: str,
    file_name: str,
    mime_type: str,
    asset_type: AssetType,
    file_size_bytes: Optional[int] = None,
    version_notes: Optional[str] = None,
    favorite: bool = False,
) -> dict[str, Any]:
    """
    Insert a creative asset metadata row for generated content.

    Notes:
    - Insert-only for this first increment (no idempotency, no version chain updates).
    - Versioning/lifecycle defaults are handled by DB defaults per ADR.
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("user_id is required")
    if not org_id or not str(org_id).strip():
        raise ValueError("org_id is required")
    if not storage_path or not str(storage_path).strip():
        raise ValueError("storage_path is required")
    if not file_name or not str(file_name).strip():
        raise ValueError("file_name is required")
    if not mime_type or not str(mime_type).strip():
        raise ValueError("mime_type is required")
    if asset_type not in ("image", "video", "audio"):
        raise ValueError("asset_type must be one of: image, video, audio")

    payload: dict[str, Any] = {
        "user_id": user_id,
        "org_id": org_id,
        "storage_path": storage_path,
        "file_name": file_name,
        "mime_type": mime_type,
        "asset_type": asset_type,
        "source": "ai_generated",
        "favorite": bool(favorite),
        "is_deleted": False,
    }
    if file_size_bytes is not None:
        payload["file_size_bytes"] = int(file_size_bytes)
    if version_notes is not None:
        payload["version_notes"] = version_notes

    sb = get_client()
    res = sb.table("creative_assets").insert(payload).execute()
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else {}


def insert_generated_creative_asset_version(
    *,
    user_id: str,
    org_id: str,
    parent_asset_id: str,
    storage_path: str,
    file_name: str,
    mime_type: str,
    asset_type: AssetType,
    file_size_bytes: Optional[int] = None,
    version_notes: Optional[str] = None,
    favorite: bool = False,
) -> dict[str, Any]:
    """
    Insert a new creative asset version row, maintaining the version chain.

    Behavior:
    - Resolves the chain root as: root = parent.parent_asset_id or parent.asset_id
    - Sets version_number = max(version_number across chain) + 1
    - Sets previous versions in the chain to is_latest=false (best-effort)
    - Inserts the new row with parent_asset_id=root and is_latest=true
    """
    if not parent_asset_id or not str(parent_asset_id).strip():
        raise ValueError("parent_asset_id is required")

    # Reuse validation from base insert
    if not user_id or not str(user_id).strip():
        raise ValueError("user_id is required")
    if not org_id or not str(org_id).strip():
        raise ValueError("org_id is required")
    if not storage_path or not str(storage_path).strip():
        raise ValueError("storage_path is required")
    if not file_name or not str(file_name).strip():
        raise ValueError("file_name is required")
    if not mime_type or not str(mime_type).strip():
        raise ValueError("mime_type is required")
    if asset_type not in ("image", "video", "audio"):
        raise ValueError("asset_type must be one of: image, video, audio")

    sb = get_client()

    # Fetch parent row to determine chain root and version_number
    parent_res = (
        sb.table("creative_assets")
        .select("asset_id,parent_asset_id,version_number")
        .eq("asset_id", parent_asset_id)
        .limit(1)
        .execute()
    )
    parent_rows = getattr(parent_res, "data", None) or []
    if not parent_rows:
        raise ValueError(f"Parent asset {parent_asset_id} not found")

    parent = parent_rows[0] if isinstance(parent_rows[0], dict) else {}
    root = parent.get("parent_asset_id") or parent.get("asset_id") or parent_asset_id
    
    # Query max version in the chain (root + all versions)
    max_res = (
        sb.table("creative_assets")
        .select("version_number")
        .eq("parent_asset_id", root)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    max_rows = getattr(max_res, "data", None) or []
    if max_rows:
        max_version = int(max_rows[0].get("version_number") or 1)
    else:
        # No versions exist yet, use root's version (always 1 for original)
        try:
            max_version = int(parent.get("version_number") or 1)
        except Exception:
            max_version = 1
    next_version = max_version + 1

    payload: dict[str, Any] = {
        "user_id": user_id,
        "org_id": org_id,
        "storage_path": storage_path,
        "file_name": file_name,
        "mime_type": mime_type,
        "asset_type": asset_type,
        "source": "ai_generated",
        "favorite": bool(favorite),
        "parent_asset_id": root,
        "version_number": next_version,
        "is_latest": False,  # Will be set to true via set_asset_as_latest
        "is_deleted": False,
    }
    if file_size_bytes is not None:
        payload["file_size_bytes"] = int(file_size_bytes)
    if version_notes is not None:
        payload["version_notes"] = version_notes

    res = sb.table("creative_assets").insert(payload).execute()
    rows = getattr(res, "data", None) or []
    new_row = rows[0] if rows else {}
    
    if not new_row:
        return {}
    
    # Use set_asset_as_latest to ensure only this new version is marked as latest
    # This properly clears is_latest across the entire family (root, versions, and variants)
    new_asset_id = new_row.get("asset_id")
    if new_asset_id:
        try:
            return set_asset_as_latest(asset_id=new_asset_id, org_id=org_id)
        except Exception as e:
            # If set_asset_as_latest fails, return the row as-is but log the error
            # The row was still created, just may not be marked as latest
            import sys
            print(f"Warning: Failed to set new version as latest: {e}", file=sys.stderr)
            return new_row
    
    return new_row


def insert_generated_creative_asset_variant(
    *,
    user_id: str,
    org_id: str,
    parent_asset_id: str,
    storage_path: str,
    file_name: str,
    mime_type: str,
    asset_type: AssetType,
    file_size_bytes: Optional[int] = None,
    version_notes: Optional[str] = None,
    favorite: bool = False,
) -> dict[str, Any]:
    """
    Insert a creative asset variant row without bumping version number.

    Behavior:
    - Keeps parent_asset_id as provided (the specific version the variant belongs to)
    - Sets version_number = parent.version_number (no increment)
    - Does not update other rows in the chain
    - Inserts the new row with is_latest=false
    
    Note: Unlike versions which always point to the chain root, variants point
    directly to their source version so they can be queried per-version.
    """
    if not parent_asset_id or not str(parent_asset_id).strip():
        raise ValueError("parent_asset_id is required")

    # Reuse validation from base insert
    if not user_id or not str(user_id).strip():
        raise ValueError("user_id is required")
    if not org_id or not str(org_id).strip():
        raise ValueError("org_id is required")
    if not storage_path or not str(storage_path).strip():
        raise ValueError("storage_path is required")
    if not file_name or not str(file_name).strip():
        raise ValueError("file_name is required")
    if not mime_type or not str(mime_type).strip():
        raise ValueError("mime_type is required")
    if asset_type not in ("image", "video", "audio"):
        raise ValueError("asset_type must be one of: image, video, audio")

    sb = get_client()

    # Fetch parent row to get version_number (but keep parent_asset_id as provided)
    parent_res = (
        sb.table("creative_assets")
        .select("asset_id,version_number")
        .eq("asset_id", parent_asset_id)
        .limit(1)
        .execute()
    )
    parent_rows = getattr(parent_res, "data", None) or []
    if not parent_rows:
        raise ValueError(f"Parent asset {parent_asset_id} not found")

    parent = parent_rows[0] if isinstance(parent_rows[0], dict) else {}
    try:
        parent_version = int(parent.get("version_number") or 1)
    except Exception:
        parent_version = 1

    payload: dict[str, Any] = {
        "user_id": user_id,
        "org_id": org_id,
        "storage_path": storage_path,
        "file_name": file_name,
        "mime_type": mime_type,
        "asset_type": asset_type,
        "source": "ai_generated",
        "favorite": bool(favorite),
        "parent_asset_id": parent_asset_id,  # Keep as the direct parent (the version)
        "version_number": parent_version,
        "is_latest": False,
        "is_deleted": False,
    }
    if file_size_bytes is not None:
        payload["file_size_bytes"] = int(file_size_bytes)
    if version_notes is not None:
        payload["version_notes"] = version_notes

    res = sb.table("creative_assets").insert(payload).execute()
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else {}


def set_asset_as_latest(
    *,
    asset_id: str,
    org_id: str,
) -> dict[str, Any]:
    """
    Set a specific asset version/variant as the latest in its family tree.

    Behavior:
    - Traverses up to find the true root (original asset with no parent)
    - Marks ALL assets in the family (root, versions, and variants) as is_latest=false
    - Marks the specified asset as is_latest=true
    
    Args:
        asset_id: The asset_id to mark as latest
        org_id: Organization ID for authorization check
    
    Returns:
        The updated asset row
    """
    if not asset_id or not str(asset_id).strip():
        raise ValueError("asset_id is required")
    if not org_id or not str(org_id).strip():
        raise ValueError("org_id is required")

    sb = get_client()

    # Fetch the asset to start traversing up
    asset_res = (
        sb.table("creative_assets")
        .select("asset_id,parent_asset_id,org_id")
        .eq("asset_id", asset_id)
        .eq("is_deleted", False)
        .limit(1)
        .execute()
    )
    asset_rows = getattr(asset_res, "data", None) or []
    if not asset_rows:
        raise ValueError(f"Asset {asset_id} not found")

    asset = asset_rows[0] if isinstance(asset_rows[0], dict) else {}
    
    # Verify org_id matches
    if asset.get("org_id") != org_id:
        raise ValueError("Unauthorized access to asset")
    
    # Traverse up to find the true root (asset with no parent_asset_id)
    root_id = asset.get("asset_id")
    current_parent = asset.get("parent_asset_id")
    
    while current_parent:
        parent_res = (
            sb.table("creative_assets")
            .select("asset_id,parent_asset_id")
            .eq("asset_id", current_parent)
            .eq("is_deleted", False)
            .limit(1)
            .execute()
        )
        parent_rows = getattr(parent_res, "data", None) or []
        if not parent_rows:
            break
        parent = parent_rows[0] if isinstance(parent_rows[0], dict) else {}
        root_id = parent.get("asset_id")
        current_parent = parent.get("parent_asset_id")

    # Collect all asset IDs in the family tree (root + all descendants)
    family_ids = {root_id}
    
    # Find all direct children of root (versions)
    children_res = (
        sb.table("creative_assets")
        .select("asset_id")
        .eq("parent_asset_id", root_id)
        .eq("is_deleted", False)
        .execute()
    )
    children_rows = getattr(children_res, "data", None) or []
    version_ids = [r.get("asset_id") for r in children_rows if r.get("asset_id")]
    family_ids.update(version_ids)
    
    # Find all variants (children of versions)
    for version_id in version_ids:
        variant_res = (
            sb.table("creative_assets")
            .select("asset_id")
            .eq("parent_asset_id", version_id)
            .eq("is_deleted", False)
            .execute()
        )
        variant_rows = getattr(variant_res, "data", None) or []
        variant_ids = [r.get("asset_id") for r in variant_rows if r.get("asset_id")]
        family_ids.update(variant_ids)

    # Mark ALL assets in the family as not latest (bulk update)
    family_ids_list = [fid for fid in family_ids if fid]
    if family_ids_list:
        try:
            sb.table("creative_assets").update({"is_latest": False}).in_(
                "asset_id", family_ids_list
            ).execute()
        except Exception:
            # Fallback to per-row updates if bulk update fails
            for fam_id in family_ids_list:
                try:
                    sb.table("creative_assets").update({"is_latest": False}).eq(
                        "asset_id", fam_id
                    ).execute()
                except Exception:
                    pass

    # Mark the target asset as latest
    res = (
        sb.table("creative_assets")
        .update({"is_latest": True})
        .eq("asset_id", asset_id)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else {}


def soft_delete_creative_asset(
    *,
    asset_id: str,
    org_id: str,
    deleted_by: str,
) -> List[str]:
    """
    Soft-delete a creative asset and its entire family tree.

    Behavior:
    - Traverses up to find the true root (original asset with no parent)
    - Collects all descendants from root (arbitrary depth)
    - Bulk updates all family members: is_deleted=True, deleted_at=now(), deleted_by=user_id
    - Storage files are retained; only database records are marked as deleted

    Args:
        asset_id: The asset_id to soft delete
        org_id: Organization ID for authorization check
        deleted_by: User ID of the user performing the delete

    Returns:
        List of soft-deleted asset_ids
    """
    if not asset_id or not str(asset_id).strip():
        raise ValueError("asset_id is required")
    if not org_id or not str(org_id).strip():
        raise ValueError("org_id is required")
    if not deleted_by or not str(deleted_by).strip():
        raise ValueError("deleted_by is required")

    sb = get_client()

    # Fetch the target asset.
    asset_res = (
        sb.table("creative_assets")
        .select("asset_id,parent_asset_id,org_id")
        .eq("asset_id", asset_id)
        .limit(1)
        .execute()
    )
    asset_rows = getattr(asset_res, "data", None) or []
    if not asset_rows:
        raise ValueError(f"Asset {asset_id} not found")

    asset = asset_rows[0] if isinstance(asset_rows[0], dict) else {}

    # Verify org_id matches
    if asset.get("org_id") != org_id:
        raise ValueError("Unauthorized access to asset")

    # Traverse up to find the true root
    root_id = asset.get("asset_id")
    current_parent = asset.get("parent_asset_id")
    visited_ancestors = {root_id}

    while current_parent:
        if current_parent in visited_ancestors:
            raise ValueError("Invalid asset hierarchy: cycle detected")
        visited_ancestors.add(current_parent)

        parent_res = (
            sb.table("creative_assets")
            .select("asset_id,parent_asset_id,org_id")
            .eq("asset_id", current_parent)
            .limit(1)
            .execute()
        )
        parent_rows = getattr(parent_res, "data", None) or []
        if not parent_rows:
            break
        parent = parent_rows[0] if isinstance(parent_rows[0], dict) else {}
        if parent.get("org_id") != org_id:
            raise ValueError("Invalid asset hierarchy: cross-org parent detected")
        root_id = parent.get("asset_id")
        current_parent = parent.get("parent_asset_id")

    # Collect all descendants from root with breadth-first traversal.
    # This handles arbitrary depth, not only root->version->variant.
    family_ids = {root_id}
    frontier = [root_id]
    while frontier:
        children_res = (
            sb.table("creative_assets")
            .select("asset_id")
            .in_("parent_asset_id", frontier)
            .eq("org_id", org_id)
            .execute()
        )
        children_rows = getattr(children_res, "data", None) or []

        next_frontier = []
        for row in children_rows:
            child_id = row.get("asset_id") if isinstance(row, dict) else None
            if child_id and child_id not in family_ids:
                family_ids.add(child_id)
                next_frontier.append(child_id)

        frontier = next_frontier

    family_ids_list = sorted([fid for fid in family_ids if fid])
    if not family_ids_list:
        return []

    deleted_at = datetime.now(timezone.utc).isoformat()
    update_payload = {
        "is_deleted": True,
        "deleted_at": deleted_at,
        "deleted_by": deleted_by,
    }

    sb.table("creative_assets").update(update_payload).in_(
        "asset_id", family_ids_list
    ).eq("org_id", org_id).execute()

    return family_ids_list
