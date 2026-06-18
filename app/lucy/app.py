from __future__ import annotations

"""
Lucy Chat Application - FastAPI Backend

A streaming chat application with AI agents, user authentication, and persistent chat history.
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import fastapi
from fastapi import Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi_pagination import add_pagination

from lucy.core.bootstrap import bootstrap

bootstrap()

from lucy.database.history import HistoryStore
from lucy.database.history import SupabaseHistoryStore
from lucy.api.chat import router as chat_router
from lucy.api.creative_assets import router as creative_assets_router
from lucy.api.brand_documents import router as brand_doc_analysis_router
from lucy.api.jobs_video_api import router as jobs_video_router
from lucy.api.locations import router as locations_router
from lucy.api.common.deps import (
    get_history_store as get_history_store,
    ensure_storage_bucket,
)
from lucy.agents.common.model_config import Models
from lucy.utils.logging import setup_logging


THIS_DIR = Path(__file__).parent.resolve()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> Any:
    """
    Application lifespan manager for database connection.

    :param app: FastAPI application instance
    :yield: None
    """
    logger = logging.getLogger(__name__)

    # Startup
    try:
        logger.info(
            f"Model config — primary: {Models.AGENT_PRIMARY}, fast: {Models.AGENT_FAST}, "
            f"image_gen: {Models.IMAGE_GENERATION}, file_pdf: {Models.FILE_ANALYSIS_PDF}, "
            f"file_default: {Models.FILE_ANALYSIS_DEFAULT}"
        )
        logger.info("Initializing Supabase history backend...")
        app.state.history = SupabaseHistoryStore()
        logger.info("Using SupabaseHistoryStore for chat history")

        # Initialize storage bucket for image agent
        logger.info("Ensuring Supabase storage bucket exists...")
        bucket_result = ensure_storage_bucket(None, "lucy-files", is_public=False)
        if bucket_result["success"]:
            logger.info("Storage bucket 'lucy-files' is ready")
        elif "already exists" in bucket_result.get("error", "").lower():
            logger.info("Storage bucket 'lucy-files' already exists")
        else:
            logger.warning(
                f"Storage bucket initialization failed: {bucket_result.get('error')}"
            )

        yield
    except Exception as e:
        logger.error(f"Failed to initialize history backend: {e}")
        raise
    finally:
        logger.info("Lifespan cleanup completed")


# Initialize logging before creating the app
setup_logging()

app = fastapi.FastAPI(lifespan=lifespan)
# Enable pagination utilities (reads page/size from query params when used)
add_pagination(app)

# CORS for your IDE dev server + same-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Static files (serve the demo UI) ----------
@app.get("/")
async def index() -> FileResponse:
    # Provide your own minimal UI file here if you like
    return FileResponse(THIS_DIR / "ui/chat.html", media_type="text/html")


@app.get("/health", response_model=Dict[str, Any])
async def health() -> Dict[str, Any]:
    """
    Health check endpoint.

    :return: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "version": "1.0.0",
    }


# ---------- Dev server ----------
if __name__ == "__main__":
    """
    Development server entry point.
    """
    import uvicorn
    import os

    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")

    # Validate configuration
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"Invalid port number: {port}")

    print(f"Starting Lucy Chat Server on {host}:{port}")
    print(f"Reload: {reload_enabled}, Log Level: {log_level}")

    # Run server
    uvicorn.run(
        "lucy.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
        # Disable reload delay for better development experience
        reload_delay=0.1 if reload_enabled else None,
    )

# ---------- Routers ----------
# Mount routers at the end so dependency overrides in tests can see exported functions above
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(creative_assets_router, prefix="/api", tags=["creative-assets"])
app.include_router(brand_doc_analysis_router, prefix="/api", tags=["brand-doc-analysis"])
app.include_router(jobs_video_router, prefix="/jobs/video", tags=["video-jobs"])
app.include_router(locations_router, prefix="/locations", tags=["locations"])
