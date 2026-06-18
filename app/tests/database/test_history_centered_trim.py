from datetime import datetime, timezone

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from lucy.database.history import HistoryStore


def _req(text: str):
    return ModelRequest(parts=[UserPromptPart(text)])


def _resp(text: str):
    return ModelResponse(
        parts=[TextPart(text)], timestamp=datetime.now(tz=timezone.utc)
    )


def test_centered_trim_keeps_pivot_and_balanced_neighbors():
    # Build alternating user/model messages, lengths = 100 chars each
    msgs = [
        _req("a" * 100),
        _resp("x" * 100),
        _req("b" * 100),
        _resp("y" * 100),
        _req("c" * 100),  # pivot (last user request)
        _resp("z" * 100),
    ]

    # Budget fits pivot + one left + one right = 300
    out = HistoryStore.trim_history_centered_by_chars(msgs, 300)
    # Expect [resp(y), req(c), resp(z)] centered around the pivot
    assert out == [msgs[3], msgs[4], msgs[5]]


def test_centered_trim_no_user_centers_on_last_message():
    # Only model responses, each 80 chars
    msgs = [_resp("m1" + "_" * 77), _resp("m2" + "_" * 77), _resp("m3" + "_" * 77)]
    # Budget 160 should include last two
    out = HistoryStore.trim_history_centered_by_chars(msgs, 160)
    assert out == [msgs[1], msgs[2]]


def test_centered_trim_returns_all_when_within_budget():
    msgs = [_req("hello"), _resp("world"), _req("again"), _resp("!" * 10)]
    total_chars = sum(
        HistoryStore._message_char_count(m) for m in msgs  # type: ignore[attr-defined]
    )
    out = HistoryStore.trim_history_centered_by_chars(msgs, total_chars)
    assert out == msgs
