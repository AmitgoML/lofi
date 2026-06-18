from types import SimpleNamespace

from lucy.database import creative_assets_client as cac


class _DummyCreativeAssetsTable:
    def __init__(self, capture: dict, parent_row: dict, inserted_row: dict):
        self._capture = capture
        self._parent_row = parent_row
        self._inserted_row = inserted_row
        self._mode = None  # "select" | "insert" | "update"
        self._select_cols = None
        self._update_payload = None
        self._filters = []
        self._in_filter = None
        self._order_by = None
        self._order_desc = False

    def select(self, cols: str):
        self._mode = "select"
        self._select_cols = cols
        return self

    def update(self, payload: dict):
        self._mode = "update"
        self._update_payload = payload
        self._filters = []
        self._in_filter = None
        return self

    def insert(self, payload: dict):
        self._mode = "insert"
        self._capture["insert_payload"] = payload
        return self

    def eq(self, key: str, val):
        self._filters.append((key, val))
        return self

    def in_(self, key: str, vals):
        self._in_filter = (key, vals)
        return self

    def order(self, col: str, desc: bool = False):
        self._order_by = col
        self._order_desc = desc
        return self

    def limit(self, _n: int):
        return self

    def execute(self):
        if self._mode == "select":
            self._capture["select_cols"] = self._select_cols
            self._capture["select_filters"] = list(self._filters)
            # Check which asset is being queried
            asset_id_filter = next(
                (v for k, v in self._filters if k == "asset_id"), None
            )
            parent_id_filter = next(
                (v for k, v in self._filters if k == "parent_asset_id"), None
            )
            if asset_id_filter == self._inserted_row.get("asset_id"):
                # Return the inserted row (for set_asset_as_latest lookups)
                return SimpleNamespace(data=[self._inserted_row])
            if parent_id_filter is not None:
                # Query for max version - return empty (no other versions exist)
                return SimpleNamespace(data=[])
            return SimpleNamespace(data=[self._parent_row])
        if self._mode == "update":
            # Record each update call (payload + filters)
            self._capture.setdefault("updates", []).append(
                {"payload": self._update_payload, "filters": list(self._filters), "in_filter": self._in_filter}
            )
            # Return updated row for is_latest=True update
            if self._update_payload.get("is_latest") is True:
                return SimpleNamespace(data=[{**self._inserted_row, "is_latest": True}])
            return SimpleNamespace(data=[])
        if self._mode == "insert":
            return SimpleNamespace(data=[self._inserted_row])
        return SimpleNamespace(data=[])


class _DummyClient:
    def __init__(self, capture: dict, parent_row: dict, inserted_row: dict):
        self._capture = capture
        self._parent_row = parent_row
        self._inserted_row = inserted_row

    def table(self, name: str):
        assert name == "creative_assets"
        return _DummyCreativeAssetsTable(
            self._capture, self._parent_row, self._inserted_row
        )


def test_insert_generated_creative_asset_version_builds_versioned_payload(monkeypatch):
    capture: dict = {}
    parent = {"asset_id": "a1", "parent_asset_id": None, "version_number": 1, "org_id": "o1"}
    inserted = {"asset_id": "a2", "parent_asset_id": "a1", "org_id": "o1", "is_latest": True}

    def fake_get_client():
        return _DummyClient(capture, parent, inserted)

    monkeypatch.setattr(cac, "get_client", fake_get_client)

    out = cac.insert_generated_creative_asset_version(
        user_id="u1",
        org_id="o1",
        parent_asset_id="a1",
        storage_path="users/u1/new.png",
        file_name="new.png",
        mime_type="image/png",
        asset_type="image",
        file_size_bytes=10,
    )

    assert out == inserted
    payload = capture["insert_payload"]
    assert payload["parent_asset_id"] == "a1"  # root
    assert payload["version_number"] == 2
    assert payload["is_latest"] is False  # Initially inserted as False
    assert payload["source"] == "ai_generated"

    # set_asset_as_latest should be called after insert:
    # 1. Update all family members to is_latest=False (bulk via in_)
    # 2. Update target asset to is_latest=True
    assert isinstance(capture.get("updates"), list)
    assert len(capture["updates"]) == 2
    # Verify the final update sets is_latest=True
    final_update = capture["updates"][-1]
    assert final_update["payload"].get("is_latest") is True


def test_insert_generated_creative_asset_version_from_old_version_uses_max(monkeypatch):
    """Test that creating a version from an older version uses max version number + 1."""
    capture: dict = {}
    
    # Simulate version chain: root (v1), child1 (v2), child2 (v3)
    # We're creating from child1 (v2), so new version should be 4, not 3
    root = {"asset_id": "a1", "parent_asset_id": None, "version_number": 1, "org_id": "o1"}
    parent_v2 = {"asset_id": "a2", "parent_asset_id": "a1", "version_number": 2, "org_id": "o1"}
    max_version = {"asset_id": "a3", "parent_asset_id": "a1", "version_number": 3, "org_id": "o1"}
    inserted = {"asset_id": "a4", "parent_asset_id": "a1", "org_id": "o1", "is_latest": True}
    
    class _EnhancedDummyCreativeAssetsTable:
        def __init__(self, capture: dict):
            self._capture = capture
            self._mode = None
            self._select_cols = None
            self._update_payload = None
            self._filters = []
            self._in_filter = None
            self._order_by = None
            self._order_desc = False
            
        def select(self, cols: str):
            self._mode = "select"
            self._select_cols = cols
            return self
            
        def update(self, payload: dict):
            self._mode = "update"
            self._update_payload = payload
            self._filters = []
            self._in_filter = None
            return self
            
        def insert(self, payload: dict):
            self._mode = "insert"
            self._capture["insert_payload"] = payload
            return self
            
        def eq(self, key: str, val):
            self._filters.append((key, val))
            return self
            
        def in_(self, key: str, vals):
            self._in_filter = (key, vals)
            return self
            
        def order(self, col: str, desc: bool = False):
            self._order_by = col
            self._order_desc = desc
            return self
            
        def limit(self, _n: int):
            return self
            
        def execute(self):
            if self._mode == "select":
                # Check which query this is based on filters and columns
                asset_id_filter = next(
                    (v for k, v in self._filters if k == "asset_id"), None
                )
                parent_id_filter = next(
                    (v for k, v in self._filters if k == "parent_asset_id"), None
                )
                
                # Query for parent asset (by asset_id)
                if asset_id_filter == "a2":
                    return SimpleNamespace(data=[parent_v2])
                
                # Query for max version (by parent_asset_id with order)
                if parent_id_filter == "a1" and self._order_by == "version_number":
                    return SimpleNamespace(data=[max_version])
                
                # Query for inserted row (for set_asset_as_latest)
                if asset_id_filter == "a4":
                    return SimpleNamespace(data=[inserted])
                
                # Query for children (return empty)
                if parent_id_filter is not None:
                    return SimpleNamespace(data=[])
                    
                return SimpleNamespace(data=[])
                
            if self._mode == "update":
                self._capture.setdefault("updates", []).append(
                    {"payload": self._update_payload, "filters": list(self._filters), "in_filter": self._in_filter}
                )
                if self._update_payload.get("is_latest") is True:
                    return SimpleNamespace(data=[{**inserted, "is_latest": True}])
                return SimpleNamespace(data=[])
                
            if self._mode == "insert":
                return SimpleNamespace(data=[inserted])
                
            return SimpleNamespace(data=[])
    
    class _EnhancedDummyClient:
        def __init__(self, capture: dict):
            self._capture = capture
            
        def table(self, name: str):
            assert name == "creative_assets"
            return _EnhancedDummyCreativeAssetsTable(self._capture)
    
    def fake_get_client():
        return _EnhancedDummyClient(capture)
    
    monkeypatch.setattr(cac, "get_client", fake_get_client)
    
    # Create a new version from a2 (version 2), when a3 (version 3) already exists
    out = cac.insert_generated_creative_asset_version(
        user_id="u1",
        org_id="o1",
        parent_asset_id="a2",  # Creating from version 2
        storage_path="users/u1/new.png",
        file_name="new.png",
        mime_type="image/png",
        asset_type="image",
        file_size_bytes=10,
    )
    
    assert out == {**inserted, "is_latest": True}
    payload = capture["insert_payload"]
    assert payload["parent_asset_id"] == "a1"  # root
    assert payload["version_number"] == 4  # Should be max(3) + 1, not parent(2) + 1
    assert payload["is_latest"] is False  # Initially inserted as False
    assert payload["source"] == "ai_generated"
