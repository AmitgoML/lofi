from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from lucy.database.history import HistoryStore


def _ts():
    return datetime.now(tz=timezone.utc)


def test_tail_history_limits():
    msgs = [ModelRequest(parts=[UserPromptPart("hi")]) for _ in range(5)]
    out = HistoryStore.tail_history(msgs, limit=3)
    assert len(out) == 3
    assert out == msgs[-3:]


def test_sanitize_history_keeps_requests_and_text_responses_only():
    req = ModelRequest(parts=[UserPromptPart("hello")])
    resp_text = ModelResponse(parts=[TextPart("ok")], timestamp=_ts())
    # Simulate a response without text parts (e.g., tool-only)
    resp_no_text = ModelResponse(parts=[], timestamp=_ts())

    cleaned = HistoryStore.sanitize_history([req, resp_text, resp_no_text])
    assert req in cleaned
    assert any(isinstance(m, ModelResponse) for m in cleaned)
    assert all(
        not (isinstance(m, ModelResponse) and len(m.parts) == 0) for m in cleaned
    )


def test_trim_history_by_chars_takes_tail_to_fit_budget():
    msgs = [
        ModelRequest(parts=[UserPromptPart("a" * 5)]),
        ModelResponse(parts=[TextPart("b" * 5)], timestamp=_ts()),
        ModelRequest(parts=[UserPromptPart("c" * 5)]),
    ]
    out = HistoryStore.trim_history_by_chars(msgs, max_chars=7)
    # Should drop from the head until <= 7 visible chars remain; tail two messages are 10 chars,
    # so first drop yields remaining 10 > 7, drop second yields 5 <= 7
    assert len(out) == 1
    assert isinstance(out[0], ModelRequest)
    assert out[0].parts[0].content == "c" * 5


def test_trim_each_message_by_chars_truncates_individually():
    msgs = [
        ModelRequest(parts=[UserPromptPart("hello world")]),
        ModelResponse(parts=[TextPart("1234567890")], timestamp=_ts()),
    ]
    out = HistoryStore.trim_each_message_by_chars(msgs, per_message_max_chars=5)
    assert isinstance(out[0], ModelRequest)
    assert out[0].parts[0].content == "hello"
    assert isinstance(out[1], ModelResponse)
    assert out[1].parts[0].content == "12345"
