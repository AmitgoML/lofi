import json

from lucy.api.chat import create_file_message
from lucy.agents.common.models import SaveFileOutput


def test_create_file_message_includes_asset_id_in_json():
    f = SaveFileOutput(
        file_name="x.png",
        file_path="users/u1/x.png",
        file_type="image/png",
        asset_id="asset-123",
        job_id=None,
    )

    msg = create_file_message(f, signed_url="https://signed", job_id=f.job_id)
    # ModelFileResponse delegates to underlying ModelResponse
    payload = json.loads(msg.parts[0].content)

    assert payload["file_name"] == "x.png"
    assert payload["file_path"] == "users/u1/x.png"
    assert payload["file_type"] == "image/png"
    assert payload["asset_id"] == "asset-123"
