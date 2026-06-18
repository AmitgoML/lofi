from __future__ import annotations

from typing import Optional, Dict

import fastapi
from fastapi import Request, HTTPException, status

from pydantic_ai import Agent

from lucy.agents.lucy_agent import LucyAgent
from lucy.agents.support_agent import SupportAgent
from lucy.database import LocalDatabase
from lucy.database.history import HistoryStore


def get_db(request: Request) -> LocalDatabase:
    try:
        return request.app.state.db
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )


def get_history_store(request: Request) -> HistoryStore:
    try:
        return request.app.state.history
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="History store not available",
        )
