from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Optional, List
import time
import threading
import uuid
import requests
from urllib.parse import urlparse

from loguru import logger
from supabase import create_client, Client, ClientOptions


@dataclass
class SupabaseConfig:
    url: str
    key: str


def _load_config() -> SupabaseConfig:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_*_KEY environment variables"
        )
    return SupabaseConfig(url=url, key=key)


_client: Optional[Client] = None

# Simple in-memory TTL cache for user org profiles
_PROFILE_CACHE: dict[str, tuple[float, List[dict[str, Any]]]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()
_PROFILE_CACHE_TTL_SEC = float(os.getenv("PROFILE_CACHE_TTL_SEC", "300"))

# In-memory TTL cache for full brand data (keyed by brand_id)
_BRAND_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_BRAND_CACHE_LOCK = threading.Lock()
_BRAND_CACHE_TTL_SEC = float(os.getenv("BRAND_CACHE_TTL_SEC", "300"))

# Conversation list caching removed to always return fresh results


def get_client() -> Client:
    global _client
    if _client is None:
        cfg = _load_config()
        # Cap PostgREST at 10 s so a stale/idle connection fails fast rather than
        # hanging for the OS TCP timeout (~120 s default).  We intentionally do NOT
        # pass a shared httpx_client — supabase-py v2 forwards it to every sub-client
        # (auth, postgrest, storage) via the same object reference, which causes them
        # to overwrite each other's base_url and breaks storage uploads.
        _client = create_client(
            cfg.url,
            cfg.key,
            options=ClientOptions(
                postgrest_client_timeout=10,
            ),
        )
        logger.info("Supabase client initialized (10 s PostgREST timeout)")
    return _client


def get_user_profile_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """
    Fetch a user row by user_id from your auth schema or a users table, if present.
    Adjust the table/select below to match your schema.
    """
    sb = get_client()
    # Example assumes a "users" table with id column; change as needed
    res = sb.table("profiles").select("*").eq("id", user_id).limit(1).execute()
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else None


def get_user_org_profiles(user_id: str) -> List[dict[str, Any]]:
    """
    Fetch richer user + organization context via a join.

    SQL logical equivalent:
    select first_name, last_name, website_url, company_name, industry, states
    from profiles pr
    join permissions pe on pe.profile_id=pr.id
    join organizations o on pe.org_id=o.id
    where pr.id = :user_id
    """
    # Cache hit check (TTL)
    now = time.time()
    try:
        with _PROFILE_CACHE_LOCK:
            cached = _PROFILE_CACHE.get(user_id)
            if cached is not None:
                ts, rows = cached
                if now - ts < _PROFILE_CACHE_TTL_SEC:
                    # Return a shallow copy to avoid accidental mutation of cache
                    return [dict(r) for r in rows]
                else:
                    # Expired cache; drop it
                    _PROFILE_CACHE.pop(user_id, None)
    except Exception:
        pass

    sb = get_client()
    # Join via permissions (FKs: permissions.profile_id -> profiles.id, permissions.org_id -> organizations.id)
    res = (
        sb.table("permissions")
        .select("org_id, profiles(*), organizations(*)")
        .eq("profile_id", user_id)
        .execute()
    )
    data = getattr(res, "data", None) or []

    # Fetch brands separately (many brands per org possible; pick the newest per org)
    org_ids: List[str] = []
    for r in data:
        oid = r.get("org_id") if isinstance(r, dict) else None
        if isinstance(oid, str) and oid:
            org_ids.append(oid)

    brand_by_org: dict[str, dict[str, Any]] = {}
    if org_ids:
        try:
            brands_res = (
                sb.table("brands")
                .select("*")
                .in_("associated_organization_id", org_ids)
                .order("created_at", desc=True)
                .execute()
            )
            brand_rows = getattr(brands_res, "data", None) or []
            for b in brand_rows:
                if not isinstance(b, dict):
                    continue
                assoc = b.get("associated_organization_id")
                if isinstance(assoc, str) and assoc and assoc not in brand_by_org:
                    brand_by_org[assoc] = b
        except Exception:
            # Best-effort: if brands query fails, still return org + profile basics
            brand_by_org = {}

    def _stringify(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        # Convert arrays/json to a stable string for downstream prompt usage
        if isinstance(v, list):
            # If list contains non-strings, fall back to json
            if all(isinstance(x, str) for x in v):
                return ", ".join(v)
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    out: List[dict[str, Any]] = []
    for row in data:
        prof = row.get("profiles")
        org = row.get("organizations")

        # Some PostgREST responses may return lists; normalize to dicts
        if isinstance(prof, list):
            prof = prof[0] if prof else {}
        if isinstance(org, list):
            org = org[0] if org else {}

        prof = prof or {}
        org = org or {}

        org_id = row.get("org_id")
        brand = brand_by_org.get(org_id, {}) if isinstance(org_id, str) else {}

        # Merge raw DB data first (brand → org → profile) so any new columns are
        # automatically visible via UserOrgProfile's extra="allow" config.
        # Normalized aliases are applied last so they are never clobbered by raw keys.
        merged: dict[str, Any] = {"org_id": org_id}
        for k, v in brand.items():
            merged[k] = _stringify(v)
        for k, v in org.items():
            merged[k] = _stringify(v)
        for k, v in prof.items():
            merged[k] = _stringify(v)

        # Stable normalized keys consumed by UserOrgProfile and downstream callers
        # such as _build_profile_summary in chat.py. Always written last so they
        # cannot be clobbered by raw DB column merging above.
        merged["org_id"] = org_id  # permissions-derived, never overwritten
        merged["first_name"] = prof.get("first_name")
        merged["last_name"] = prof.get("last_name")
        merged["company_name"] = org.get("company_name")
        merged["website_url"] = _stringify(
            brand.get("brand_website_url") or brand.get("website_url")
        )
        merged["industry"] = _stringify(
            brand.get("brand_industry") or brand.get("industry")
        )
        merged["states"] = _stringify(
            brand.get("brand_states_licensed") or brand.get("states")
        )
        # Brand narrative fields — all declared on UserOrgProfile
        merged["brand_name"] = _stringify(brand.get("brand_name"))
        merged["description"] = _stringify(brand.get("brand_description"))
        merged["tone_of_voice"] = _stringify(brand.get("brand_tone_of_voice"))
        merged["purpose"] = _stringify(brand.get("brand_purpose"))
        merged["mission_vision"] = _stringify(brand.get("brand_mission_vision"))
        merged["core_values"] = _stringify(brand.get("brand_core_values"))
        merged["audience"] = _stringify(brand.get("brand_audiences"))
        merged["positioning"] = _stringify(brand.get("brand_positioning"))
        merged["design_elements"] = _stringify(brand.get("brand_design_elements"))
        merged["messaging_pillars"] = _stringify(brand.get("brand_messaging_pillars"))
        merged["copywriting_tone"] = _stringify(brand.get("brand_copywriting_tone"))

        out.append(merged)
    # Store in cache
    try:
        with _PROFILE_CACHE_LOCK:
            _PROFILE_CACHE[user_id] = (time.time(), out)
    except Exception:
        pass

    return out


def get_full_brand(brand_id: str) -> Optional[dict[str, Any]]:
    """Fetch full brand record by brand_id with in-memory TTL cache."""
    if not brand_id:
        return None

    now = time.time()
    try:
        with _BRAND_CACHE_LOCK:
            cached = _BRAND_CACHE.get(brand_id)
            if cached is not None:
                ts, data = cached
                if now - ts < _BRAND_CACHE_TTL_SEC:
                    return dict(data)
                _BRAND_CACHE.pop(brand_id, None)
    except Exception:
        pass

    try:
        sb = get_client()
        result = (
            sb.table("brands")
            .select("*")
            .eq("brand_id", brand_id)
            .limit(1)
            .execute()
        )
        rows = getattr(result, "data", None) or []
        if not rows:
            return None

        brand = rows[0]

        try:
            with _BRAND_CACHE_LOCK:
                _BRAND_CACHE[brand_id] = (time.time(), brand)
        except Exception:
            pass

        return brand
    except Exception as e:
        logger.warning(f"Failed to fetch brand {brand_id}: {e}")
        return None


if __name__ == "__main__":
    from lucy.api.common.secrets import load_envs

    load_envs()
    from loguru import logger

    sb = get_client()

    test_user_id = os.getenv("TEST_USER_ID")
    user = get_user_profile_by_id(test_user_id)
    print(user)


# ---------------- Lucy chat history helpers ----------------


def fetch_lucy_messages(
    user_id: str, conversation_id: str, desc=False
) -> List[dict[str, Any]]:
    """
    Return rows for a conversation ordered by message_order.

    Expected columns: id, message, role, created_at, message_order, user_feedback, attachments
    """
    sb = get_client()
    res = (
        sb.table("lucy_messages")
        .select("id,message,role,created_at,message_order,user_feedback,attachments")
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .order("message_order", desc=desc)
        .execute()
    )
    return getattr(res, "data", None) or []


def update_lucy_message_attachments(message_id: int, attachments: list) -> None:
    """Persist attachment metadata (file_name, file_type, storage_path) on a message row."""
    sb = get_client()
    sb.table("lucy_messages").update({"attachments": attachments}).eq(
        "id", message_id
    ).execute()


def insert_lucy_message(row: dict[str, Any]) -> int:
    """
    Insert a single message row into lucy_messages.

    Row must include: user_id, conversation_id, message, role
    created_at can be omitted to use the DB default.

    Returns:
        Inserted message ID
    """
    sb = get_client()
    result = sb.table("lucy_messages").insert(row).execute()
    return result.data[0]["id"] if result.data else None


def ensure_lucy_conversation_exists(user_id: str, conversation_id: str) -> None:
    """
    Ensure lucy_conversations has a row with the given id. If missing, insert it.

    Only sets id and user_id, relying on DB defaults for other fields.
    """
    sb = get_client()
    # Check existence
    res = (
        sb.table("lucy_conversations")
        .select("id")
        .eq("id", conversation_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    if rows:
        return
    # Insert conversation
    try:
        sb.table("lucy_conversations").insert(
            {"id": conversation_id, "user_id": user_id}
        ).execute()
    except Exception as e:
        # Best-effort: ignore if it was created concurrently
        logger.debug(f"ensure_lucy_conversation_exists insert ignored: {e}")


def create_lucy_conversation(
    user_id: str, title: Optional[str] = None
) -> dict[str, Any]:
    """
    Create a new conversation row in lucy_conversations and return key fields.

    Relies on DB defaults for id and created_at. If title is provided, it will be set;
    otherwise the DB default empty string will be used.
    """
    sb = get_client()
    payload: dict[str, Any] = {"user_id": user_id}
    if title is not None:
        payload["title"] = title
    # Insert row; some client versions don't support chaining select() after insert
    sb.table("lucy_conversations").insert(payload).execute()
    # Fetch the most recent conversation for this user
    res = (
        sb.table("lucy_conversations")
        .select("id,created_at,title")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else {}


def delete_lucy_conversation(user_id: str, conversation_id: str) -> None:
    """
    Delete a conversation and its messages for the given user.

    Assumes FK from lucy_messages.conversation_id references lucy_conversations.id
    with ON DELETE CASCADE, or we manually delete messages first to be safe.
    """
    sb = get_client()
    # Best-effort: remove messages first
    try:
        sb.table("lucy_messages").delete().eq("user_id", user_id).eq(
            "conversation_id", conversation_id
        ).execute()
    except Exception:
        pass
    # Delete conversation row
    sb.table("lucy_conversations").delete().eq("id", conversation_id).eq(
        "user_id", user_id
    ).execute()


def list_lucy_conversations(
    user_id: str,
    limit: Optional[int] = None,
    ids_only: bool = False,
    q: Optional[str] = None,
) -> List[dict[str, Any]] | List[str]:
    """
    Return user conversations ordered by created_at desc.

    - When ids_only=True, returns a list[str] of conversation ids (up to limit)
    - Otherwise, returns a list of dicts with id, created_at, title
    - If q is provided, filter case-insensitively on title containing q
    """
    sb = get_client()
    select_fields = "id" if ids_only else "id,created_at,title"
    qy = sb.table("lucy_conversations").select(select_fields).eq("user_id", user_id)
    if q:
        try:
            qy = qy.ilike("title", f"%{q}%")
        except Exception:
            qy = qy.like("title", f"%{q}%")
    qy = qy.order("created_at", desc=True)
    if limit is not None:
        qy = qy.limit(int(limit))
    res = qy.execute()
    rows = getattr(res, "data", None) or []

    if ids_only:
        out: List[str] = []
        for r in rows:
            v = r.get("id")
            if isinstance(v, str):
                out.append(v)
        return out
    return rows


def list_lucy_conversations_ids(user_id: str, limit: int = 200) -> List[str]:
    """
    Deprecated: use list_lucy_conversations(user_id, limit=..., ids_only=True).
    """
    return list_lucy_conversations(user_id, limit=limit, ids_only=True)  # type: ignore[return-value]


def update_lucy_conversation_title(
    user_id: str, conversation_id: str, title: str
) -> dict[str, Any]:
    """
    Update the conversation title and return id, created_at, title.
    """
    sb = get_client()
    # Perform update
    sb.table("lucy_conversations").update({"title": title}).eq(
        "id", conversation_id
    ).eq("user_id", user_id).execute()
    # Fetch updated row
    res2 = (
        sb.table("lucy_conversations")
        .select("id,created_at,title")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res2, "data", None) or []
    return rows[0] if rows else {"id": conversation_id, "title": title}


def get_conversation_last_agent(user_id: str, conversation_id: str) -> Optional[str]:
    """Return the last_agent stored on a conversation, or None if unset."""
    sb = get_client()
    res = (
        sb.table("lucy_conversations")
        .select("last_agent")
        .eq("id", conversation_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []
    if rows and rows[0].get("last_agent"):
        return rows[0]["last_agent"]
    return None


def set_conversation_last_agent(
    user_id: str, conversation_id: str, agent: str
) -> None:
    """Persist the most recently routed agent on the conversation row."""
    sb = get_client()
    try:
        sb.table("lucy_conversations").update({"last_agent": agent}).eq(
            "id", conversation_id
        ).eq("user_id", user_id).execute()
    except Exception as e:
        logger.debug(f"set_conversation_last_agent failed: {e}")


def update_lucy_message_feedback(
    user_id: str, conversation_id: str, message_id: str, user_feedback: bool | None
) -> dict[str, Any]:
    """
    Update the user_feedback field for a specific message in lucy_messages.

    Args:
        user_id: User ID for authorization
        conversation_id: Conversation ID
        message_id: Message ID to update (UUID string)
        user_feedback: Boolean feedback value (True for positive, False for negative, None to remove vote)

    Returns:
        dict with updated message data or error info
    """
    sb = get_client()

    # First verify the message exists and belongs to the user
    res = (
        sb.table("lucy_messages")
        .select("id,message,role,created_at,message_order,user_feedback")
        .eq("id", message_id)
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .limit(1)
        .execute()
    )
    rows = getattr(res, "data", None) or []

    if not rows:
        raise ValueError(f"Message {message_id} not found or access denied")

    # Update the user_feedback field
    sb.table("lucy_messages").update({"user_feedback": user_feedback}).eq(
        "id", message_id
    ).eq("user_id", user_id).eq("conversation_id", conversation_id).execute()

    # Return updated message data
    updated_res = (
        sb.table("lucy_messages")
        .select("id,message,role,created_at,message_order,user_feedback")
        .eq("id", message_id)
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .limit(1)
        .execute()
    )
    updated_rows = getattr(updated_res, "data", None) or []
    return (
        updated_rows[0]
        if updated_rows
        else {"id": message_id, "user_feedback": user_feedback}
    )


# ---------------- Supabase Storage helpers ----------------


def generate_short_uuid() -> str:
    """
    Generate a short UUID (8 characters) for file naming.

    Returns:
        8-character UUID string
    """
    return str(uuid.uuid4()).replace("-", "")[:8]


def upload_file_to_storage(
    file_url: Optional[str] = None,
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    user_id: str = "",
    bucket_name: str = "lucy-files",
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Uploads a file (from URL, local path, or bytes) to Supabase Storage under a user-specific folder.

    Args:
        file_url: Optional URL to download before uploading.
        file_path: Optional local path to an existing file.
        file_bytes: Optional raw bytes (for in-memory uploads).
        user_id: User ID for organizing files in user-specific folder.
        bucket_name: Supabase Storage bucket name (default: "lucy-files").
        file_name: Optional custom file name (default: auto-generated).
        content_type: Optional MIME type override.

    Returns:
        dict: Success status, signed_url, and file_path metadata.
    """
    import time
    import mimetypes
    import requests
    from urllib.parse import urlparse

    try:
        sb = get_client()
        max_retries = 3

        # --- Step 1: Derive file name ---
        if not file_name:
            if file_url:
                parsed_url = urlparse(file_url)
                ext = os.path.splitext(parsed_url.path)[1] or ".bin"
            elif file_path:
                ext = os.path.splitext(file_path)[1] or ".bin"
            else:
                ext = ".bin"
            short_uuid = generate_short_uuid()
            file_name = f"{short_uuid}{ext}"

        user_path = f"users/{user_id}/{file_name}"

        # --- Step 2: Acquire file data ---
        file_data = None

        if file_bytes:
            # Direct in-memory data
            file_data = file_bytes
        elif file_path:
            with open(file_path, "rb") as f:
                file_data = f.read()
        elif file_url:
            # Download file from URL with retries
            logger.info(f"Downloading file from: {file_url}")
            response = None
            for attempt in range(max_retries):
                try:
                    response = requests.get(file_url, timeout=30)
                    response.raise_for_status()
                    break
                except (
                    requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed: {e}. Retrying..."
                        )
                        time.sleep(2**attempt)
                    else:
                        raise e
            file_data = response.content
            if not content_type:
                content_type = response.headers.get("content-type")

        if not file_data:
            raise ValueError(
                "No file data provided (file_url, file_path, or file_bytes required)."
            )

        # --- Step 3: Determine content type ---
        detected_type, _ = mimetypes.guess_type(file_name)
        final_content_type = content_type or detected_type or "application/octet-stream"

        # --- Step 4: Upload with retry logic ---
        logger.info(f"Uploading to Supabase bucket={bucket_name}, path={user_path}")
        for attempt in range(max_retries):
            try:
                sb.storage.from_(bucket_name).upload(
                    path=user_path,
                    file=file_data,
                    file_options={"content-type": final_content_type},
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Upload attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    time.sleep(2**attempt)
                else:
                    raise e

        # --- Step 5: Create signed URL for user access ---
        signed_url = sb.storage.from_(bucket_name).create_signed_url(
            user_path, expires_in=3600  # 1 hour
        )

        return {
            "success": True,
            "signed_url": signed_url,
            "file_path": user_path,
            "bucket": bucket_name,
            "user_id": user_id,
            "original_source": file_url or file_path or "bytes",
            "content_type": final_content_type,
        }

    except Exception as e:
        logger.error(f"Error uploading file to Supabase Storage: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_source": file_url or file_path or "bytes",
        }


def delete_file_from_storage(
    file_path: str, user_id: str, bucket_name: str = "lucy-files"
) -> dict[str, Any]:
    """
    Delete a file from Supabase Storage (user-specific).

    Args:
        file_path: Path to the file in storage (relative to user folder)
        user_id: User ID for access control
        bucket_name: Supabase Storage bucket name (default: "lucy-files")

    Returns:
        dict with success status
    """
    try:
        sb = get_client()

        # Create user-specific path
        user_path = f"users/{user_id}/{file_path}"

        logger.info(f"Deleting file from bucket: {bucket_name}, path: {user_path}")
        result = sb.storage.from_(bucket_name).remove([user_path])

        return {
            "success": True,
            "file_path": user_path,
            "bucket": bucket_name,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"Error deleting file from Supabase Storage: {e}")
        return {"success": False, "error": str(e), "file_path": file_path}


def list_user_files_in_storage(
    user_id: str, bucket_name: str = "lucy-files", limit: int = 100
) -> List[dict[str, Any]]:
    """
    List files in Supabase Storage bucket for a specific user.

    Args:
        user_id: User ID to list files for
        bucket_name: Supabase Storage bucket name (default: "lucy-files")
        limit: Maximum number of files to return

    Returns:
        List of file information dictionaries with signed URLs
    """
    try:
        sb = get_client()

        user_folder = f"users/{user_id}/"
        logger.info(f"Listing files in bucket: {bucket_name} for user: {user_id}")
        result = sb.storage.from_(bucket_name).list(path=user_folder, limit=limit)

        files = []
        for file_info in result:
            file_path = f"{user_folder}{file_info['name']}"
            # Create signed URL for private access
            signed_url = sb.storage.from_(bucket_name).create_signed_url(
                file_path, expires_in=3600  # 1 hour expiration
            )
            files.append(
                {
                    "name": file_info["name"],
                    "path": file_path,
                    "size": file_info.get("metadata", {}).get("size"),
                    "created_at": file_info.get("created_at"),
                    "signed_url": signed_url,
                    "user_id": user_id,
                }
            )

        return files

    except Exception as e:
        logger.error(f"Error listing files in Supabase Storage: {e}")
        return []


def create_signed_url_for_user_file(
    user_id: str,
    file_name: str,
    bucket_name: str = "lucy-files",
    expires_in: int = 3600,
) -> dict[str, Any]:
    """
    Create a signed URL for a user's file.

    Args:
        user_id: User ID who owns the file
        file_name: Name of the file
        bucket_name: Supabase Storage bucket name (default: "lucy-files")
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        dict with success status and signed URL
    """
    try:
        sb = get_client()

        user_path = f"users/{user_id}/{file_name}"

        logger.info(f"Creating signed URL for user: {user_id}, file: {file_name}")
        signed_url = sb.storage.from_(bucket_name).create_signed_url(
            user_path, expires_in=expires_in
        )

        return {
            "success": True,
            "signed_url": signed_url,
            "file_path": user_path,
            "user_id": user_id,
            "expires_in": expires_in,
        }

    except Exception as e:
        logger.error(f"Error creating signed URL: {e}")
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "file_name": file_name,
        }


def resign_storage_url(
    url_or_path: str,
    bucket_name: str | None = None,
    expires_in: int = 3600,
) -> Optional[str]:
    """
    Re-sign a Supabase storage URL or path to get a fresh signed URL.

    Supports inputs like:
      - 'users/abc123/image.png'
      - 'https://api-staging.app.meetlofi.com/storage/v1/object/sign/lucy-user-uploads/...'
    """
    try:
        sb = get_client()
        parsed = urlparse(url_or_path)

        # Case 1: Plain storage path (not URL)
        if not parsed.scheme:
            # example: users/abc123/image.png
            bucket = bucket_name or "lucy-files"
            logger.info(f"Re-signing plain path: bucket={bucket}, path={url_or_path}")
            result = sb.storage.from_(bucket).create_signed_url(
                url_or_path, expires_in=expires_in
            )
            return result.get("signedURL") if isinstance(result, dict) else str(result)

        # Case 2: Full Supabase signed URL
        # Example: https://api-staging.app.meetlofi.com/storage/v1/object/sign/<bucket>/<path>?token=...
        path_parts = parsed.path.split("/")

        if "object" in path_parts and "sign" in path_parts:
            try:
                sign_idx = path_parts.index("sign")
                bucket = path_parts[sign_idx + 1]
                object_path = "/".join(path_parts[sign_idx + 2 :])
            except (IndexError, ValueError):
                raise ValueError("Could not parse Supabase storage URL path")

            bucket = bucket_name or bucket
            logger.info(f"Re-signing Supabase URL: bucket={bucket}, path={object_path}")

            result = sb.storage.from_(bucket).create_signed_url(
                object_path, expires_in=expires_in
            )
            return result.get("signedURL") if isinstance(result, dict) else str(result)

        # If not a recognized Supabase storage pattern, return original
        logger.debug(f"URL not recognized as Supabase storage path: {url_or_path}")
        return url_or_path

    except Exception as e:
        logger.error(f"Error re-signing storage URL: {e}")
        return url_or_path


def cleanup_user_files(
    user_id: str, older_than_days: int = 30, bucket_name: str = "lucy-files"
) -> dict[str, Any]:
    """
    Clean up old files for a user (optional maintenance function).

    Args:
        user_id: User ID to clean up files for
        older_than_days: Delete files older than this many days
        bucket_name: Supabase Storage bucket name (default: "lucy-files")

    Returns:
        dict with cleanup results
    """
    try:
        from datetime import datetime, timedelta

        sb = get_client()
        user_folder = f"users/{user_id}/"

        # List all user files
        result = sb.storage.from_(bucket_name).list(path=user_folder)

        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_files = []

        for file_info in result:
            created_at = file_info.get("created_at")
            if created_at:
                file_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if file_date < cutoff_date:
                    file_path = f"{user_folder}{file_info['name']}"
                    sb.storage.from_(bucket_name).remove([file_path])
                    deleted_files.append(file_info["name"])

        return {
            "success": True,
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files,
            "user_id": user_id,
            "older_than_days": older_than_days,
        }

    except Exception as e:
        logger.error(f"Error cleaning up user files: {e}")
        return {"success": False, "error": str(e), "user_id": user_id}


def create_storage_bucket(
    bucket_name: str = "lucy-files", is_public: bool = True
) -> dict[str, Any]:
    """
    Create a new Supabase Storage bucket.
    If the bucket already exists, returns success without error.

    Args:
        bucket_name: Name of the bucket to create
        is_public: Whether the bucket should be public

    Returns:
        dict with success status
    """
    try:
        sb = get_client()

        # Check if bucket already exists
        try:
            buckets = sb.storage.list_buckets()
            existing_bucket = next(
                (b for b in buckets if b.get("name") == bucket_name), None
            )
            if existing_bucket:
                logger.info(f"Storage bucket '{bucket_name}' already exists")
                return {
                    "success": True,
                    "bucket_name": bucket_name,
                    "is_public": is_public,
                    "already_exists": True,
                }
        except Exception as check_error:
            # If we can't check, proceed with creation attempt
            logger.debug(f"Could not check for existing bucket: {check_error}")

        logger.info(f"Creating storage bucket: {bucket_name}")
        result = sb.storage.create_bucket(bucket_name, options={"public": is_public})

        return {"success": True, "bucket_name": bucket_name, "is_public": is_public}

    except Exception as e:
        error_str = str(e)
        # Check if error is due to bucket already existing
        if (
            "409" in error_str
            or "Duplicate" in error_str
            or "already exists" in error_str.lower()
        ):
            logger.info(
                f"Storage bucket '{bucket_name}' already exists (duplicate error)"
            )
            return {
                "success": True,
                "bucket_name": bucket_name,
                "is_public": is_public,
                "already_exists": True,
            }
        logger.error(f"Error creating storage bucket: {e}")
        return {"success": False, "error": str(e), "bucket_name": bucket_name}


def save_chart_config_to_storage(
    x_key: str,
    series: List[dict[str, Any]],
    data: List[dict[str, Any]],
    user_id: str,
    file_name: Optional[str] = None,
    bucket_name: str = "lucy-files",
    title: Optional[str] = None,
    name_key: Optional[str] = None,
    value_key: Optional[str] = None,
    inner_radius: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate and save a Recharts chart configuration file to Supabase Storage as JSON.

    Args:
        x_key: The key for the x-axis (e.g., "month", "date")
        series: List of series definitions, each with id, label, type, dataKey.
            Series may include a ``stack`` field for stacked bar charts.
            Example: [{"id": "revenue", "label": "Revenue", "type": "bar", "dataKey": "revenue"}]
        data: List of data points, each with the x_key value and series dataKey values
            Example: [{"month": "Jan", "revenue": 120000, "profit": 30000}]
        user_id: User ID for organizing files in user-specific folder
        file_name: Optional custom file name (default: auto-generated with .json extension)
        bucket_name: Supabase Storage bucket name (default: "lucy-files")
        title: Optional human-readable chart title
        name_key: For pie/radar charts — field name that holds the label for each slice/spoke
            (maps to Recharts ``nameKey``).
        value_key: For pie charts — field name that holds the numeric value for each slice
            (maps to Recharts ``valueKey``).
        inner_radius: For donut charts — inner radius in px (e.g. 60). Omit for standard pie.

    Returns:
        dict with success status, signed_url, and file_path metadata
    """
    try:
        import json

        # Generate file name if not provided
        if not file_name:
            short_uuid = generate_short_uuid()
            file_name = f"chart_config_{short_uuid}.json"

        # Build the JSON config object
        config: dict[str, Any] = {
            "xKey": x_key,
            "series": series,
            "data": data,
        }
        if title:
            config["title"] = title
        if name_key:
            config["nameKey"] = name_key
        if value_key:
            config["valueKey"] = value_key
        if inner_radius is not None:
            config["innerRadius"] = inner_radius

        # Format as JSON
        json_content = json.dumps(config, indent=2)

        # Convert to bytes
        file_bytes = json_content.encode("utf-8")

        # Upload to storage using existing function
        result = upload_file_to_storage(
            file_bytes=file_bytes,
            user_id=user_id,
            bucket_name=bucket_name,
            file_name=file_name,
            content_type="application/json",
        )

        if result.get("success"):
            logger.info(
                f"Chart config saved successfully: {result.get('file_path')} for user {user_id}"
            )
        else:
            logger.error(f"Failed to save chart config: {result.get('error')}")

        return result

    except Exception as e:
        logger.error(f"Error saving chart config to storage: {e}")
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "file_name": file_name,
        }


def update_storage_object_owner(
    storage_path: str,
    user_id: str,
    bucket_name: str = "creative",
    user_metadata: dict | None = None,
) -> bool:
    """
    Update the owner_id (and optionally user_metadata) of a storage object.
    
    This is necessary when files are uploaded via the service role key,
    as the owner_id is not automatically set to the user who initiated the upload.
    
    Args:
        storage_path: Path to the file in storage (e.g., "images/abc123.png")
        user_id: User ID to set as the owner
        bucket_name: Storage bucket name (default: "creative")
        user_metadata: Optional dict of user metadata to set (e.g., {"name": "...", "campaign_usage": [], "favorite": False})
    
    Returns:
        True if update succeeded, False otherwise
    """
    try:
        sb = get_client()
        
        # Build the update payload
        update_payload = {"owner_id": user_id}
        if user_metadata is not None:
            update_payload["user_metadata"] = user_metadata
        
        # Update the storage.objects record
        # The object is identified by bucket_id + name (storage_path)
        result = (
            sb.schema("storage")
            .from_("objects")
            .update(update_payload)
            .eq("bucket_id", bucket_name)
            .eq("name", storage_path)
            .execute()
        )
        
        rows = getattr(result, "data", None) or []
        if rows:
            logger.info(f"Updated storage object owner: {storage_path} -> {user_id}")
            return True
        else:
            logger.warning(f"No storage object found to update: {storage_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating storage object owner: {e}")
        return False


def storage_file_exists(
    *,
    file_path: str,
    bucket_name: str = "lucy-files",
) -> bool:
    try:
        sb = get_client()

        parts = [p for p in file_path.split("/") if p]
        if len(parts) < 2:
            return False

        parent = "/".join(parts[:-1])
        target_name = parts[-1]

        # list(path, options) with "search"
        items = (
            sb.storage.from_(bucket_name).list(
                parent,
                {"limit": 20, "offset": 0, "search": target_name},
            )
            or []
        )

        return any(
            isinstance(it, dict) and it.get("name") == target_name for it in items
        )

    except Exception as e:
        logger.warning(f"storage_file_exists failed: {e}")
        return False
