import os
from typing import List

import pytest
from fastapi.testclient import TestClient

from pydantic_ai.messages import ModelMessage


class DummyHistory:
    def __init__(self):
        self.saved: list[bytes] = []

    async def get_messages(
        self, user_id: str, session_id: str
    ) -> List[tuple[ModelMessage, int, int, None]]:
        return []

    async def add_messages(
        self, user_id: str, session_id: str, messages: List[ModelMessage]
    ) -> dict[str, List[int]]:
        self.saved.append(messages)
        return {"user": [], "model": [], "file": []}  # Return empty dict for testing

    async def get_recent_sanitized_messages(
        self, user_id: str, session_id: str, limit=None
    ):
        return []


@pytest.fixture()
def client_with_dummy_history(client: TestClient):
    from lucy.app import app as fastapi_app

    dummy = DummyHistory()
    get_history_store = __import__(
        "lucy.app", fromlist=["get_history_store"]
    ).get_history_store
    # Backup and set only the needed override
    prev = dict(fastapi_app.dependency_overrides)  # shallow copy
    fastapi_app.dependency_overrides[get_history_store] = lambda: dummy  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        fastapi_app.dependency_overrides.clear()
        fastapi_app.dependency_overrides.update(prev)


def test_stream_uses_history_override(client_with_dummy_history: TestClient):
    """Test that streaming uses the overridden history store"""
    from unittest.mock import patch

    # Mock the agent to avoid needing real API keys
    class MockAgent:
        def run_stream(self, *args, **kwargs):
            class MockRunContext:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

                async def stream(self, debounce_by=0.01):
                    yield {"type": "status", "value": "generating"}
                    yield {"type": "delta", "value": "Hello"}
                    yield {"type": "status", "value": "completed"}

                def timestamp(self):
                    from datetime import datetime, timezone

                    return datetime.now(tz=timezone.utc)

                def new_messages_json(self):
                    return b"[]"

            return MockRunContext()

    with patch("lucy.app.create_agent_for_query", return_value=MockAgent()):
        r = client_with_dummy_history.post(
            "/chat/stream",
            json={"message": "Hello", "session_id": "s-ovr"},
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 200
        # Ensure some deltas or statuses are returned despite empty history
        assert r.text.strip() != ""
