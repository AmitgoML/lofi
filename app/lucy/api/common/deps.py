from __future__ import annotations

from typing import Optional, Dict

from fastapi import Request, HTTPException, status

from pydantic_ai import Agent

from lucy.agents.lucy_agent import LucyAgent
from lucy.agents.support_agent import SupportAgent
from lucy.database.history import HistoryStore
from lucy.database.supabase_client import get_client, create_storage_bucket


def get_history_store(request: Request) -> HistoryStore:
    try:
        return request.app.state.history
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History store not available",
        )


def get_supabase_client(request: Request):
    """Get the Supabase client from app state or create a new one."""
    try:
        return request.app.state.supabase_client
    except AttributeError:
        # Create client if not in app state
        client = get_client()
        request.app.state.supabase_client = client
        return client


def ensure_storage_bucket(
    request: Optional[Request], bucket_name: str = "lucy-files", is_public: bool = False
):
    """Ensure the Supabase storage bucket exists."""
    try:
        # During app startup, request might be None
        if request is not None:
            # Check if bucket is already ensured in app state
            bucket_key = f"bucket_{bucket_name}"
            if hasattr(request.app.state, bucket_key):
                return request.app.state.__dict__[bucket_key]

        # Create bucket
        bucket_result = create_storage_bucket(bucket_name, is_public=is_public)

        # Store result in app state if request is available
        if request is not None:
            setattr(request.app.state, bucket_key, bucket_result)

        return bucket_result
    except Exception as e:
        return {"success": False, "error": str(e), "bucket_name": bucket_name}
