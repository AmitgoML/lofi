import os
import pytest
from unittest.mock import patch, MagicMock


def _read_ndjson_lines(resp_text: str):
    return [line for line in resp_text.splitlines() if line.strip()]


def test_json_chat_stream_emits_status_and_deltas(client):
    """Test that streaming chat emits status and delta events"""

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
                    yield {"type": "delta", "value": " world"}
                    yield {"type": "status", "value": "completed"}

                def timestamp(self):
                    from datetime import datetime, timezone

                    return datetime.now(tz=timezone.utc)

                def new_messages_json(self):
                    return b"[]"

            return MockRunContext()

    with patch("lucy.app.create_agent_for_query", return_value=MockAgent()):
        r = client.post(
            "/chat/stream",
            json={"message": "Hello", "session_id": "s1"},
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 200
        lines = _read_ndjson_lines(r.text)
        # Should include at least one status and one delta
        assert any('"type": "status"' in ln for ln in lines)
        assert any('"type": "delta"' in ln for ln in lines)


# Note: The streaming message ID tests are complex to mock due to FastAPI dependency injection
# The core voting functionality is already comprehensively tested in test_conversations.py
# and test_supabase_client.py. The streaming integration would require more sophisticated
# mocking of the agent creation and dependency injection system.
