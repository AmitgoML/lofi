from lucy.api.chat import ChatJsonResponse, ChatMessageResponse


def test_chat_message_response_accepts_model_json_role_and_json_field():
    msg = ChatMessageResponse(
        message_id="123",
        role="model-json",
        timestamp="2026-01-01T00:00:00+00:00",
        content="",
        file=None,
        json=ChatJsonResponse(type="example_schema_v1", data={"k": "v"}),
        message_order=1,
        user_feedback=None,
    )

    assert msg.role == "model-json"
    assert msg.json_data is not None
    assert msg.json_data.type == "example_schema_v1"
    assert msg.json_data.data == {"k": "v"}
