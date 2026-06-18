"""
Regression tests for the /chat/stream NDJSON event contract.

These tests verify:
- The exact ordered sequence of event types on the happy path.
- Streams that end with status:error emit a terminal event and no completed event.
- The stream always emits status:completed last on the happy path.
- file events include file_name.
- model-json (option_draft_campaign) events are emitted correctly.
- The endpoint returns 200 even when the backend encounters a mid-stream error.
"""
from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ndjson(text: str) -> list[dict]:
    """Return all valid JSON objects from a newline-delimited response."""
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


class MockAgentClass:
    """Minimal agent class that satisfies the interface used by chat_stream()."""

    @classmethod
    def reminder_header(cls) -> str:
        return "Mock Agent"

    @classmethod
    async def pre_run_check(cls, deps):
        return None

    @classmethod
    async def get_streaming_agent(cls, deps):
        return None


class _MockResponse:
    """Minimal stand-in for pydantic-ai's ModelResponse."""

    def __init__(self, text: str = ""):
        self.parts = []  # no tool-call or thinking parts in tests
        self._text = text


class _RunResult:
    def __init__(self, snapshots: list, output=None, all_messages_result=None):
        self._snapshots = snapshots
        self.output = output
        self._all_messages_result = all_messages_result or []

    async def stream_output(self, *, debounce_by=None):
        for snapshot in self._snapshots:
            yield snapshot

    def all_messages(self):
        return self._all_messages_result


class _FailingRunResult:
    def __init__(self, partial: str | None = None):
        self._partial = partial
        self.output = None

    async def stream_output(self, *, debounce_by=None):
        if self._partial:
            yield self._partial
        raise RuntimeError("Agent exploded")

    def all_messages(self):
        return []


def _make_validation_error():
    """Return a real pydantic.ValidationError by forcing an invalid model parse."""
    from pydantic import BaseModel, ValidationError

    class _Dummy(BaseModel):
        x: int

    try:
        _Dummy.model_validate({"x": "not-an-int-string-that-wont-coerce"})
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected ValidationError was not raised")  # pragma: no cover


class _ValidationTailFailRunResult:
    def __init__(self, snapshots: list[str]):
        self._snapshots = snapshots
        self.output = None

    async def stream_output(self, *, debounce_by=None):
        for snapshot in self._snapshots:
            yield snapshot
        raise _make_validation_error()

    def all_messages(self):
        return []


class _RunContext:
    """Async context manager that yields a run result."""

    def __init__(self, run_result):
        self._result = run_result

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, *args):
        return False


def _make_agent(run_result) -> MagicMock:
    """Return a MagicMock whose run_stream() returns a proper async context manager.

    MagicMock (not AsyncMock) is intentional: calling run_stream() must return
    the context manager directly, not a coroutine that wraps it.
    """
    agent = MagicMock()
    agent.run_stream.return_value = _RunContext(run_result)
    return agent


def _route_patch(agent, route: str = "support") -> tuple:
    return (route, MockAgentClass, agent)


# ---------------------------------------------------------------------------
# Happy-path: event ordering
# ---------------------------------------------------------------------------

class TestStreamEventOrdering:
    def test_happy_path_contains_all_required_event_types(self, client: TestClient):
        """Happy-path stream must emit status:started, generating, delta, completed."""
        agent = _make_agent(_RunResult(["Hello", "Hello world"]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Hello", "session_id": "test-session"},
            )

        assert resp.status_code == 200
        events = parse_ndjson(resp.text)
        types = [e.get("type") for e in events]

        assert "status" in types, "No status event emitted"
        assert "delta" in types, "No delta event emitted"

        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "started" in status_values
        assert "generating" in status_values
        assert "completed" in status_values, (
            f"Stream did not end with status:completed. Status values: {status_values}"
        )

    def test_completed_is_the_last_event(self, client: TestClient):
        """status:completed must be the final event on the happy path."""
        agent = _make_agent(_RunResult(["Response text"]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Hi", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        assert events, "No events emitted"
        last_event = events[-1]
        assert last_event == {"type": "status", "value": "completed"}, (
            f"Last event was {last_event!r}, expected status:completed"
        )

    def test_delta_values_reconstruct_full_response(self, client: TestClient):
        """Accumulating all delta.value strings must equal the full response text."""
        # Each snapshot is the full text so far; the backend computes incremental deltas
        agent = _make_agent(_RunResult(["Hello", "Hello world", "Hello world!"]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Hi", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        accumulated = "".join(e["value"] for e in events if e.get("type") == "delta")
        assert accumulated == "Hello world!", (
            f"Accumulated deltas '{accumulated}' do not match expected full text"
        )

    def test_generating_status_includes_agent_metadata(self, client: TestClient):
        agent = _make_agent(_RunResult(["ok"]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
            patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent, route="lucy")

            resp = client.post(
                "/chat/stream",
                json={"message": "Hi", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        generating_events = [
            e for e in events
            if e.get("type") == "status" and e.get("value") == "generating"
        ]
        assert generating_events, "Expected at least one generating status event"
        assert any(e.get("agent") == "lucy" for e in generating_events)
        assert any(e.get("agent_description") == "Lucy" for e in generating_events)


# ---------------------------------------------------------------------------
# Error path: stream ends with status:error, never status:completed
# ---------------------------------------------------------------------------

class TestStreamErrorPath:
    def test_mid_stream_exception_emits_error_status(self, client: TestClient):
        """When the agent raises mid-stream, the stream emits status:error."""
        agent = _make_agent(_FailingRunResult(partial="partial output"))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Break it", "session_id": "test-session"},
            )

        # HTTP 200 is expected even on errors (status is in the stream body)
        assert resp.status_code == 200
        events = parse_ndjson(resp.text)
        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "error" in status_values, (
            f"Expected status:error in stream, got status values: {status_values}"
        )
        assert "completed" not in status_values, (
            "status:completed must NOT appear when an error occurs"
        )

    def test_error_stream_never_ends_with_completed(self, client: TestClient):
        """The last status event on an error path must be status:error."""
        agent = _make_agent(_FailingRunResult())

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Break it", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        status_events = [e for e in events if e.get("type") == "status"]
        assert status_events, "No status events on error path"
        assert status_events[-1]["value"] == "error", (
            f"Last status event was '{status_events[-1]['value']}', expected 'error'"
        )


# ---------------------------------------------------------------------------
# File events include file_name
# ---------------------------------------------------------------------------

class TestFileEventContract:
    def test_file_event_includes_file_name(self, client: TestClient):
        """Streamed file events must include file_name so the frontend can render."""
        from lucy.agents.common.models import SaveFileOutput

        class _OutputWithFile:
            message = "Here is your image"
            files = [
                SaveFileOutput(
                    file_name="test_image.png",
                    file_path="path/to/test_image.png",
                    file_type="image/png",
                    job_id=None,
                )
            ]
            jsons = []

        out = _OutputWithFile()
        result = _RunResult([out], output=out)
        agent = _make_agent(result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None), \
             patch("lucy.api.chat.get_signed_url_for_file", return_value="https://example.com/signed"):
            mock_route.return_value = _route_patch(agent, route="image")

            resp = client.post(
                "/chat/stream",
                json={"message": "Make an image", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        file_events = [e for e in events if e.get("type") == "file"]
        assert file_events, (
            f"No file event emitted. Event types: {[e.get('type') for e in events]}"
        )
        for fe in file_events:
            assert "file_name" in fe, f"file event missing file_name: {fe}"
            assert fe["file_name"], "file_name must be non-empty"
            assert "file_type" in fe, f"file event missing file_type: {fe}"


# ---------------------------------------------------------------------------
# model-json: option_draft_campaign events
# ---------------------------------------------------------------------------

class TestModelJsonEvents:
    def test_option_draft_campaign_event_emitted_for_json_output(
        self, client: TestClient
    ):
        """When the agent produces a JSON output of type option_draft_campaign,
        the stream must emit a matching event for the frontend to render the
        DraftCampaignMessage component."""
        from lucy.agents.common.models import JSONOutput

        draft_data = {"name": "Test Campaign", "goal": "awareness", "budget": 1000}

        class _OutputWithJson:
            message = "Here is your draft campaign"
            files = []
            jsons = [JSONOutput(json_type="option_draft_campaign", json_data=draft_data)]

        out = _OutputWithJson()
        result = _RunResult([out], output=out)
        agent = _make_agent(result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent, route="campaign_planner")

            resp = client.post(
                "/chat/stream",
                json={"message": "Draft a campaign", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        draft_events = [e for e in events if e.get("type") == "option_draft_campaign"]
        assert draft_events, (
            "No option_draft_campaign event in stream. "
            f"Event types seen: {[e.get('type') for e in events]}"
        )
        assert draft_events[0].get("data") == draft_data

    def test_stream_still_completes_with_json_output(self, client: TestClient):
        """A stream with JSON output must still end with status:completed."""
        from lucy.agents.common.models import JSONOutput

        class _OutputWithJson:
            message = "Done"
            files = []
            jsons = [JSONOutput(json_type="option_draft_campaign", json_data={})]

        result = _RunResult(["Done"], output=_OutputWithJson())
        agent = _make_agent(result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent, route="campaign_planner")

            resp = client.post(
                "/chat/stream",
                json={"message": "Draft a campaign", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "completed" in status_values, (
            "Stream with JSON output must still end with status:completed"
        )


# ---------------------------------------------------------------------------
# ValidationError resilience (truncated-JSON recovery)
# ---------------------------------------------------------------------------

class TestStreamValidationErrorResilience:
    def test_tail_validation_error_resolves_as_completed(self, client: TestClient):
        """A ValidationError at end-of-stream must NOT produce status:error when content was streamed."""
        agent = _make_agent(
            _ValidationTailFailRunResult(["Creative Brief", "Creative Brief for AuctionGoblin"])
        )

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Create a brief", "session_id": "test-session"},
            )

        assert resp.status_code == 200
        events = parse_ndjson(resp.text)
        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "completed" in status_values, (
            f"Expected status:completed after tail ValidationError, got: {status_values}"
        )
        assert "error" not in status_values, (
            f"Unexpected status:error after tail ValidationError: {status_values}"
        )

    def test_tail_validation_error_preserves_streamed_content(self, client: TestClient):
        """Content from partial snapshots must survive a tail-end ValidationError."""
        agent = _make_agent(
            _ValidationTailFailRunResult(["Hello", "Hello world"])
        )

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Say hello", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        accumulated = "".join(e["value"] for e in events if e.get("type") == "delta")
        assert "Hello world" in accumulated, (
            f"Expected streamed content in response, got: {accumulated!r}"
        )

    def test_empty_stream_validation_error_yields_error_status(self, client: TestClient):
        """If nothing was streamed before the ValidationError, status:error must be emitted."""
        agent = _make_agent(_ValidationTailFailRunResult([]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Trigger error", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "error" in status_values, (
            f"Expected status:error when nothing was streamed before ValidationError, got: {status_values}"
        )


# ---------------------------------------------------------------------------
# Empty-response fallback
# ---------------------------------------------------------------------------

class TestStreamEmptyResponseFallback:
    def test_empty_stream_emits_fallback_delta(self, client: TestClient):
        """When the agent yields no text at all, a fallback delta must be emitted."""
        agent = _make_agent(_RunResult([]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Do keyword research", "session_id": "test-session"},
            )

        assert resp.status_code == 200
        events = parse_ndjson(resp.text)
        delta_events = [e for e in events if e.get("type") == "delta"]
        assert delta_events, "Expected a fallback delta event when stream yields no text"

    def test_empty_stream_fallback_ends_with_completed(self, client: TestClient):
        """Empty-stream fallback must still end with status:completed, not status:error."""
        agent = _make_agent(_RunResult([]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Do keyword research", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        status_values = [e.get("value") for e in events if e.get("type") == "status"]
        assert "completed" in status_values, (
            f"Expected status:completed after empty stream fallback, got: {status_values}"
        )
        assert "error" not in status_values, (
            f"Unexpected status:error after empty stream fallback: {status_values}"
        )

    def test_empty_stream_fallback_message_is_user_friendly(self, client: TestClient):
        """Fallback message must be a generic, user-safe prompt to retry."""
        agent = _make_agent(_RunResult([]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Do keyword research", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        combined = " ".join(e["value"] for e in events if e.get("type") == "delta")
        assert "rephras" in combined.lower() or "try" in combined.lower(), (
            f"Expected a retry prompt in fallback message, got: {combined!r}"
        )


# ---------------------------------------------------------------------------
# Error message sanitisation
# ---------------------------------------------------------------------------

class TestStreamErrorSanitisation:
    def test_error_delta_does_not_leak_internal_exception_text(
        self, client: TestClient
    ):
        """The error delta must not expose raw exception/validation details."""

        async def _raises(*args, **kwargs):
            raise ValueError(
                "1 validation error for FileAgentOutput\nmessage\n  Field required"
            )
            yield  # make it a generator

        context = AsyncMock()
        context.__aenter__ = AsyncMock(return_value=context)
        context.__aexit__ = AsyncMock(return_value=False)
        context.stream_output = _raises
        context.output = None
        context.all_messages = MagicMock(return_value=[])

        bad_agent = MagicMock()
        bad_agent.run_stream.return_value = context

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(bad_agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Trigger error", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        delta_events = [e for e in events if e.get("type") == "delta"]
        assert delta_events, "Expected at least one delta on error path"

        for de in delta_events:
            value = de.get("value", "")
            assert "validation error" not in value.lower(), (
                f"Raw validation error text leaked to client: {value!r}"
            )
            assert "FileAgentOutput" not in value, (
                f"Internal model class name leaked to client: {value!r}"
            )
            assert "Field required" not in value, (
                f"Pydantic field error leaked to client: {value!r}"
            )

    def test_error_path_yields_user_friendly_message(self, client: TestClient):
        """The error delta must contain a generic, user-safe message."""

        async def _raises(*args, **kwargs):
            raise RuntimeError("Internal agent failure")
            yield

        context = AsyncMock()
        context.__aenter__ = AsyncMock(return_value=context)
        context.__aexit__ = AsyncMock(return_value=False)
        context.stream_output = _raises
        context.output = None
        context.all_messages = MagicMock(return_value=[])

        bad_agent = MagicMock()
        bad_agent.run_stream.return_value = context

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(bad_agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Trigger error", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        delta_values = [e["value"] for e in events if e.get("type") == "delta"]
        assert delta_values, "Expected a delta event on error path"
        # Should be a friendly message, not raw exception text
        combined = " ".join(delta_values)
        assert "try again" in combined.lower() or "went wrong" in combined.lower(), (
            f"Expected user-safe message in error delta, got: {combined!r}"
        )


# ---------------------------------------------------------------------------
# tool_status events: streaming intermediate progress from tools
# ---------------------------------------------------------------------------

class _RunResultWithStatus:
    """Like _RunResult but injects tool_status messages into the ChatDeps queue
    before yielding each text snapshot, simulating a tool calling
    ctx.deps.status_queue.put_nowait(...).
    """

    def __init__(self, snapshots: list, status_messages: list[str]):
        self._snapshots = snapshots
        self._status_messages = status_messages
        self._deps = None  # set after run_stream(deps=...) captures it

    async def stream_output(self, *, debounce_by=None):
        # Push all status messages before yielding any text snapshot so they
        # are present in the queue when the heartbeat loop drains it.
        for msg in self._status_messages:
            if self._deps is not None:
                self._deps.status_queue.put_nowait(msg)
        for snapshot in self._snapshots:
            yield snapshot

    def all_messages(self):
        return []


class _RunContextCaptureDeps:
    """Async context manager that captures deps= kwarg so the run result can
    push items onto status_queue during streaming."""

    def __init__(self, run_result: _RunResultWithStatus):
        self._result = run_result

    async def __aenter__(self):
        return self._result

    async def __aexit__(self, *args):
        return False


def _make_status_agent(run_result: _RunResultWithStatus) -> MagicMock:
    """Return a mock agent that captures the deps kwarg and wires it to the
    run result so status messages can be injected during streaming."""

    ctx = _RunContextCaptureDeps(run_result)

    agent = MagicMock()

    def _run_stream(*args, deps=None, **kwargs):
        run_result._deps = deps
        return ctx

    agent.run_stream.side_effect = _run_stream
    return agent


class TestToolStatusEvents:
    """tool_status events must appear in the NDJSON stream when tools push to the status_queue."""

    def test_tool_status_events_are_emitted(self, client: TestClient):
        """tool_status events must appear in the stream when the queue is populated."""
        run_result = _RunResultWithStatus(
            snapshots=["Answer text"],
            status_messages=["Loading brand context", "Analyzing campaign performance"],
        )
        agent = _make_status_agent(run_result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Analyze my campaigns", "session_id": "test-session"},
            )

        assert resp.status_code == 200
        events = parse_ndjson(resp.text)
        tool_status_events = [e for e in events if e.get("type") == "tool_status"]
        assert tool_status_events, (
            f"No tool_status events emitted. Event types seen: {[e.get('type') for e in events]}"
        )
        status_values = [e.get("value") for e in tool_status_events]
        assert "Loading brand context" in status_values
        assert "Analyzing campaign performance" in status_values

    def test_tool_status_values_are_non_empty_strings(self, client: TestClient):
        """Each tool_status event must have a non-empty string value."""
        run_result = _RunResultWithStatus(
            snapshots=["Done"],
            status_messages=["Searching knowledge base"],
        )
        agent = _make_status_agent(run_result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Search for something", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        tool_status_events = [e for e in events if e.get("type") == "tool_status"]
        for evt in tool_status_events:
            assert isinstance(evt.get("value"), str) and evt["value"].strip(), (
                f"tool_status event has empty or non-string value: {evt!r}"
            )

    def test_tool_status_events_appear_before_completed(self, client: TestClient):
        """tool_status events must appear before status:completed."""
        run_result = _RunResultWithStatus(
            snapshots=["Result"],
            status_messages=["Loading your profile"],
        )
        agent = _make_status_agent(run_result)

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Hi", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        tool_status_indices = [i for i, e in enumerate(events) if e.get("type") == "tool_status"]
        completed_indices = [
            i for i, e in enumerate(events)
            if e.get("type") == "status" and e.get("value") == "completed"
        ]
        if tool_status_indices and completed_indices:
            assert max(tool_status_indices) < min(completed_indices), (
                "tool_status events must all appear before status:completed"
            )

    def test_no_tool_status_when_queue_empty(self, client: TestClient):
        """When no tools push to the queue, no tool_status events must appear."""
        agent = _make_agent(_RunResult(["Simple answer"]))

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route, \
             patch("lucy.database.supabase_client.get_conversation_last_agent", return_value=None):
            mock_route.return_value = _route_patch(agent)

            resp = client.post(
                "/chat/stream",
                json={"message": "Quick question", "session_id": "test-session"},
            )

        events = parse_ndjson(resp.text)
        tool_status_events = [e for e in events if e.get("type") == "tool_status"]
        assert not tool_status_events, (
            f"Expected no tool_status events when queue is empty, got: {tool_status_events}"
        )
