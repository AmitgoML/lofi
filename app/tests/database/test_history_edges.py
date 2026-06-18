from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from lucy.database.history import HistoryStore


def test_tail_history_handles_empty():
    assert HistoryStore.tail_history([], limit=5) == []


def test_sanitize_history_drops_empty_response_parts():
    req = ModelRequest(parts=[UserPromptPart("hi")])
    now = datetime.now(tz=timezone.utc)
    resp_empty = ModelResponse(parts=[], timestamp=now)
    resp_text = ModelResponse(parts=[TextPart("x")], timestamp=now)
    out = HistoryStore.sanitize_history([req, resp_empty, resp_text])
    assert req in out
    assert resp_text in out
    assert resp_empty not in out
