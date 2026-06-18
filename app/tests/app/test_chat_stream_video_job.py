"""Tests for async video job stream contract.

Verifies that when a video agent produces a job_id the streaming endpoint:
1. Emits a ``type: "file"`` event with the correct fields (file_name, file_type,
   job_id, signed_url=null).
2. Emits ``file_message_id`` so the frontend can track the persisted message.
3. Does NOT emit a signed_url for the pending job.
4. The ``save_model_response`` helper skips signed-URL generation for job_id files.
"""
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from lucy.agents.common.models import FileAgentOutput, SaveFileOutput
from lucy.api.chat import save_model_response, create_file_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ndjson(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def _make_video_output(job_id: str) -> FileAgentOutput:
    return FileAgentOutput(
        message="I've started generating your video.",
        files=[
            SaveFileOutput(
                file_name=f"{job_id}.mp4",
                file_path="",
                file_type="video/mp4",
                job_id=job_id,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatStreamVideoFileEvent:
    """Verify the ``type: "file"`` stream event contract for async video jobs."""

    def test_video_file_event_includes_required_fields(self, client: TestClient):
        """file event must include file_name, file_type, job_id, and signed_url=None."""
        fake_job_id = "job-abc-123"
        fake_output = _make_video_output(fake_job_id)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            agent_class = _make_agent_class_with_output(fake_output)
            mock_agent = _make_mock_agent_with_output(fake_output)
            mock_route.return_value = ("video", agent_class, mock_agent)

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create a product video",
                    "request_type": "video",
                    "request_params": {"duration": 4, "ratio": "landscape"},
                },
            )

        assert response.status_code == 200
        events = _parse_ndjson(response.text)

        file_events = [e for e in events if e.get("type") == "file"]
        assert file_events, "Expected at least one 'file' event in the stream"

        evt = file_events[0]
        assert evt["file_type"] == "video/mp4", "file_type must be video/mp4"
        assert evt["file_name"] == f"{fake_job_id}.mp4", "file_name must match job_id.mp4"
        assert evt["job_id"] == fake_job_id, "job_id must be present"
        assert evt["signed_url"] is None, "signed_url must be null for a pending async job"

    def test_video_stream_emits_file_message_id(self, client: TestClient):
        """file_message_id event must be emitted so the frontend can track the card.

        We patch save_model_response to simulate a successful file-message
        persistence (file id=42) so the stream emits the file_message_id event.
        In production the real history store returns the persisted row ID.
        """
        fake_job_id = "job-xyz-456"
        fake_output = _make_video_output(fake_job_id)

        async def _fake_save_model_response(*args, **kwargs):
            return {"model": [99], "file": [42]}

        with (
            patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route,
            patch("lucy.api.chat.save_model_response", side_effect=_fake_save_model_response),
        ):
            agent_class = _make_agent_class_with_output(fake_output)
            mock_agent = _make_mock_agent_with_output(fake_output)
            mock_route.return_value = ("video", agent_class, mock_agent)

            response = client.post(
                "/chat/stream",
                json={"message": "Generate a short clip", "request_type": "video"},
            )

        assert response.status_code == 200
        events = _parse_ndjson(response.text)

        file_message_id_events = [
            e for e in events if e.get("type") == "file_message_id"
        ]
        assert file_message_id_events, (
            "Expected at least one 'file_message_id' event so the frontend can "
            "match the loading card to the persisted message"
        )
        assert file_message_id_events[0]["value"] == 42

    def test_video_stream_does_not_emit_signed_url(self, client: TestClient):
        """No signed URL should be included for a pending async job."""
        fake_job_id = "job-pending-789"
        fake_output = _make_video_output(fake_job_id)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            agent_class = _make_agent_class_with_output(fake_output)
            mock_agent = _make_mock_agent_with_output(fake_output)
            mock_route.return_value = ("video", agent_class, mock_agent)

            response = client.post(
                "/chat/stream",
                json={"message": "Make a video reel", "request_type": "video"},
            )

        assert response.status_code == 200
        events = _parse_ndjson(response.text)

        for evt in events:
            if evt.get("type") == "file" and evt.get("job_id") == fake_job_id:
                assert not evt.get("signed_url"), (
                    "signed_url must be absent or null for a pending async video job"
                )


class TestSaveModelResponseVideoJob:
    """Verify that save_model_response never generates a signed URL for async jobs."""

    @pytest.mark.asyncio
    async def test_no_signed_url_generated_for_job_id_file(self):
        """When a file has job_id set, signed URL generation is skipped."""
        file_with_job = SaveFileOutput(
            file_name="job-001.mp4",
            file_path="",
            file_type="video/mp4",
            job_id="job-001",
        )

        mock_supabase = MagicMock()
        mock_supabase.storage.from_().create_signed_url.return_value = {"signedURL": "https://bad.url"}

        mock_history_store = AsyncMock()
        mock_history_store.sanitize_history.return_value = []
        mock_history_store.add_messages.return_value = {"user": [], "model": [], "file": []}
        mock_history_store.strip_effective_prompt_header.side_effect = lambda s, header: s

        await save_model_response(
            user_id="u1",
            session_id="s1",
            response_text="Video is being generated.",
            streamed_files=[file_with_job],
            streamed_jsons=[],
            header="",
            history_store=mock_history_store,
            supabase_client=mock_supabase,
        )

        # The Supabase signed URL method must NOT have been called for the async job.
        mock_supabase.storage.from_().create_signed_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_signed_url_generated_for_synchronous_file(self):
        """For files without job_id, a signed URL should be generated normally."""
        sync_file = SaveFileOutput(
            file_name="image.png",
            file_path="users/u1/image.png",
            file_type="image/png",
            job_id=None,
        )

        mock_supabase = MagicMock()
        mock_supabase.storage.from_().create_signed_url.return_value = {
            "signedURL": "https://example.com/image.png"
        }

        mock_history_store = AsyncMock()
        mock_history_store.sanitize_history.return_value = []
        mock_history_store.add_messages.return_value = {"user": [], "model": [], "file": []}
        mock_history_store.strip_effective_prompt_header.side_effect = lambda s, header: s

        await save_model_response(
            user_id="u1",
            session_id="s1",
            response_text="Here is your image.",
            streamed_files=[sync_file],
            streamed_jsons=[],
            header="",
            history_store=mock_history_store,
            supabase_client=mock_supabase,
        )

        mock_supabase.storage.from_().create_signed_url.assert_called_once()


class TestCreateFileMessage:
    """Verify that create_file_message preserves job_id in the persisted payload."""

    def test_job_id_preserved_in_file_message(self):
        from lucy.api.chat import create_file_message

        f = SaveFileOutput(
            file_name="job-abc.mp4",
            file_path="",
            file_type="video/mp4",
            job_id="job-abc",
        )
        msg = create_file_message(f, signed_url=None, job_id="job-abc")

        # The TextPart content should be JSON with job_id intact.
        from pydantic_ai.messages import TextPart
        content = msg.parts[0].content
        payload = json.loads(content)
        assert payload["job_id"] == "job-abc"
        assert payload["signed_url"] is None
        assert payload["file_type"] == "video/mp4"
        assert payload["file_name"] == "job-abc.mp4"


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _make_agent_class_with_output(output: FileAgentOutput):
    """Return a minimal agent class that satisfies LofiAgent expectations."""

    class _AgentClass:
        @classmethod
        def reminder_header(cls):
            return ""

        @classmethod
        async def pre_run_check(cls, deps):
            return None

        @classmethod
        async def get_streaming_agent(cls, deps):
            return None

    return _AgentClass


def _make_mock_agent_with_output(output: FileAgentOutput):
    """Return a mock agent whose run_stream streams the given FileAgentOutput."""

    class _MockResult:
        def __init__(self, out: FileAgentOutput):
            self.output = out

        async def stream_output(self, *, debounce_by=None):
            yield self.output

        def all_messages(self):
            return []

    mock_result = _MockResult(output)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_result
    mock_ctx.__aexit__.return_value = False

    mock_agent = MagicMock()
    mock_agent.run_stream.return_value = mock_ctx
    return mock_agent
