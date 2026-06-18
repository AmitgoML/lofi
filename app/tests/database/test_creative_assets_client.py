from types import SimpleNamespace

from lucy.database import creative_assets_client as cac


class _DummyInsertTable:
    def __init__(self, inserted_row: dict, capture: dict):
        self._inserted_row = inserted_row
        self._capture = capture

    def insert(self, payload):
        self._capture["payload"] = payload
        return self

    def execute(self):
        return SimpleNamespace(data=[self._inserted_row])


class _DummyClient:
    def __init__(self, inserted_row: dict, capture: dict):
        self._inserted_row = inserted_row
        self._capture = capture

    def table(self, name: str):
        self._capture["table"] = name
        return _DummyInsertTable(self._inserted_row, self._capture)


def test_insert_generated_creative_asset_metadata_inserts_expected_payload(monkeypatch):
    capture: dict = {}
    inserted = {"asset_id": "uuid-1", "source": "ai_generated"}

    def fake_get_client():
        return _DummyClient(inserted, capture)

    monkeypatch.setattr(cac, "get_client", fake_get_client)

    out = cac.insert_generated_creative_asset_metadata(
        user_id="user-1",
        org_id="org-1",
        storage_path="users/user-1/asset.png",
        file_name="asset.png",
        file_size_bytes=123,
        mime_type="image/png",
        asset_type="image",
        version_notes="v1",
        favorite=True,
    )

    assert out == inserted
    assert capture["table"] == "creative_assets"
    payload = capture["payload"]
    assert payload["user_id"] == "user-1"
    assert payload["org_id"] == "org-1"
    assert payload["storage_path"] == "users/user-1/asset.png"
    assert payload["file_name"] == "asset.png"
    assert payload["file_size_bytes"] == 123
    assert payload["mime_type"] == "image/png"
    assert payload["asset_type"] == "image"
    assert payload["source"] == "ai_generated"
    assert payload["version_notes"] == "v1"
    assert payload["favorite"] is True
