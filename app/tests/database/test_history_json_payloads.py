import json
from datetime import datetime, timezone

from pydantic_ai.messages import ModelResponse, TextPart

from lucy.database.history import to_chat_message


def _ts():
    return datetime.now(tz=timezone.utc)


def test_to_chat_message_parses_json_payload_as_model_json():
    payload = {"json_type": "example_schema_v1", "json_data": {"a": 1, "b": "x"}}
    msg = ModelResponse(parts=[TextPart(json.dumps(payload))], timestamp=_ts())

    out = to_chat_message(
        msg,
        supabase_client=None,
        message_order=3,
        message_id=42,
        user_feedback=None,
    )

    assert out["role"] == "model-json"
    assert out["content"] == ""
    assert out["file"] is None
    assert out["json"] == {"type": "example_schema_v1", "data": {"a": 1, "b": "x"}}
    assert out["message_order"] == 3
    assert out["message_id"] == 42


def test_to_chat_message_keeps_file_json_as_model_files():
    file_blob = {
        "file_name": "x.png",
        "file_path": "lucy-files/some/path.png",
        "file_type": "image/png",
    }
    msg = ModelResponse(parts=[TextPart(json.dumps(file_blob))], timestamp=_ts())

    out = to_chat_message(
        msg,
        supabase_client=None,
        message_order=1,
        message_id=7,
        user_feedback=None,
    )

    assert out["role"] == "model-files"
    assert out["content"] == ""
    assert out["json"] is None
    assert isinstance(out["file"], dict)
    assert out["file"]["file_name"] == "x.png"
    assert out["file"]["file_type"] == "image/png"
