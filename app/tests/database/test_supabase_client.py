from types import SimpleNamespace

import pytest

from lucy.database import supabase_client as sc


class _DummyTable:
    def __init__(self, data):
        self._data = data

    def select(self, _):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class _DummyClient:
    def __init__(self, data_by_table):
        self._data_by_table = data_by_table

    def table(self, _name):
        return _DummyTable(self._data_by_table.get(_name, []))


def test_get_user_org_profiles_flattens_single_rows(monkeypatch):
    # Ensure no cross-test cache leakage
    try:
        sc._PROFILE_CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    permissions = [
        {
            "org_id": "org-1",
            "profiles": {"first_name": "Alex", "last_name": "Arbat"},
            "organizations": {"company_name": "Durdom"},
        }
    ]
    brands = [
        {
            "associated_organization_id": "org-1",
            "brand_website_url": "some.com",
            "brand_industry": ["Funeral/Burial Services"],
            "brand_states_licensed": ["NJ"],
            "created_at": "2025-01-01T00:00:00+00:00",
        }
    ]

    def fake_get_client():
        return _DummyClient({"permissions": permissions, "brands": brands})

    monkeypatch.setattr(sc, "get_client", fake_get_client)
    out = sc.get_user_org_profiles("u1")
    expected = {
        "org_id": "org-1",
        "first_name": "Alex",
        "last_name": "Arbat",
        "website_url": "some.com",
        "company_name": "Durdom",
        "industry": "Funeral/Burial Services",
        "states": "NJ",
    }
    assert isinstance(out, list) and len(out) == 1
    row = out[0]
    for k, v in expected.items():
        assert row.get(k) == v


def test_get_user_org_profiles_handles_list_wrapped_joins(monkeypatch):
    # Ensure no cross-test cache leakage
    try:
        sc._PROFILE_CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    permissions = [
        {
            "org_id": "org-2",
            "profiles": [{"first_name": "A", "last_name": "B"}],
            "organizations": [{"company_name": "X"}],
        }
    ]
    brands = [
        {
            "associated_organization_id": "org-2",
            "brand_website_url": "ex.com",
            "brand_industry": None,
            "brand_states_licensed": None,
            "created_at": "2025-01-01T00:00:00+00:00",
        }
    ]

    def fake_get_client():
        return _DummyClient({"permissions": permissions, "brands": brands})

    monkeypatch.setattr(sc, "get_client", fake_get_client)
    out = sc.get_user_org_profiles("u1")
    assert out[0]["org_id"] == "org-2"
    assert out[0]["first_name"] == "A"
    assert out[0]["last_name"] == "B"
    assert out[0]["website_url"] == "ex.com"


# Voting functionality tests
class _DummyTableWithUpdate:
    def __init__(self, select_data, update_data=None):
        self._select_data = select_data
        self._update_data = update_data
        self._update_called = False
        self._update_args = None

    def select(self, _):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, _):
        return self

    def order(self, field, desc=False):
        return self

    def update(self, data):
        self._update_called = True
        self._update_args = data
        return self

    def execute(self):
        if self._update_called:
            # Return updated data after update
            if self._select_data and len(self._select_data) > 0:
                updated_data = self._select_data[0].copy()  # Get first item from list
                if self._update_args:
                    updated_data.update(self._update_args)
                return SimpleNamespace(data=[updated_data])
            else:
                # No data to update, return empty
                return SimpleNamespace(data=[])
        else:
            # Return select data
            return SimpleNamespace(data=self._select_data)


class _DummyClientWithUpdate:
    def __init__(self, select_data, update_data=None):
        self._select_data = select_data
        self._update_data = update_data

    def table(self, _name):
        return _DummyTableWithUpdate(self._select_data, self._update_data)


def test_update_lucy_message_feedback_upvote(monkeypatch):
    """Test updating message feedback to upvote (True)"""
    message_data = {
        "id": "msg-123",
        "message": "Test message",
        "role": "model",
        "created_at": "2025-01-01T00:00:00+00:00",
        "message_order": 1,
        "user_feedback": True,
    }

    def fake_get_client():
        return _DummyClientWithUpdate([message_data])

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.update_lucy_message_feedback("user-123", "conv-456", "msg-123", True)

    assert result["id"] == "msg-123"
    assert result["user_feedback"] is True
    assert result["message"] == "Test message"


def test_update_lucy_message_feedback_downvote(monkeypatch):
    """Test updating message feedback to downvote (False)"""
    message_data = {
        "id": "msg-123",
        "message": "Test message",
        "role": "model",
        "created_at": "2025-01-01T00:00:00+00:00",
        "message_order": 1,
        "user_feedback": False,
    }

    def fake_get_client():
        return _DummyClientWithUpdate([message_data])

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.update_lucy_message_feedback("user-123", "conv-456", "msg-123", False)

    assert result["id"] == "msg-123"
    assert result["user_feedback"] is False
    assert result["message"] == "Test message"


def test_update_lucy_message_feedback_remove_vote(monkeypatch):
    """Test removing vote by setting feedback to None"""
    message_data = {
        "id": "msg-123",
        "message": "Test message",
        "role": "model",
        "created_at": "2025-01-01T00:00:00+00:00",
        "message_order": 1,
        "user_feedback": None,
    }

    def fake_get_client():
        return _DummyClientWithUpdate([message_data])

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.update_lucy_message_feedback("user-123", "conv-456", "msg-123", None)

    assert result["id"] == "msg-123"
    assert result["user_feedback"] is None
    assert result["message"] == "Test message"


def test_update_lucy_message_feedback_message_not_found(monkeypatch):
    """Test updating feedback for a message that doesn't exist"""

    def fake_get_client():
        return _DummyClientWithUpdate([])  # Empty result

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    with pytest.raises(
        ValueError, match="Message msg-nonexistent not found or access denied"
    ):
        sc.update_lucy_message_feedback("user-123", "conv-456", "msg-nonexistent", True)


def test_update_lucy_message_feedback_uuid_message_id(monkeypatch):
    """Test updating feedback with UUID string message ID"""
    message_id = "d2d95851-5115-4119-b4dc-15d19016d243"
    message_data = {
        "id": message_id,
        "message": "Test message",
        "role": "model",
        "created_at": "2025-01-01T00:00:00+00:00",
        "message_order": 1,
        "user_feedback": True,
    }

    def fake_get_client():
        return _DummyClientWithUpdate([message_data])

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.update_lucy_message_feedback("user-123", "conv-456", message_id, True)

    assert result["id"] == message_id
    assert result["user_feedback"] is True


def test_fetch_lucy_messages_with_user_feedback(monkeypatch):
    """Test fetching messages includes user_feedback field"""
    messages_data = [
        {
            "id": "msg-1",
            "message": "First message",
            "role": "user",
            "created_at": "2025-01-01T00:00:00+00:00",
            "message_order": 1,
            "user_feedback": None,
        },
        {
            "id": "msg-2",
            "message": "Second message",
            "role": "model",
            "created_at": "2025-01-01T00:01:00+00:00",
            "message_order": 2,
            "user_feedback": True,
        },
        {
            "id": "msg-3",
            "message": "Third message",
            "role": "model",
            "created_at": "2025-01-01T00:02:00+00:00",
            "message_order": 3,
            "user_feedback": False,
        },
    ]

    def fake_get_client():
        return _DummyClientWithUpdate(messages_data)

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.fetch_lucy_messages("user-123", "conv-456")

    assert len(result) == 3
    assert result[0]["user_feedback"] is None
    assert result[1]["user_feedback"] is True
    assert result[2]["user_feedback"] is False
    assert result[0]["id"] == "msg-1"
    assert result[1]["id"] == "msg-2"
    assert result[2]["id"] == "msg-3"


def test_fetch_lucy_messages_desc_order(monkeypatch):
    """Test fetching messages in descending order"""
    messages_data = [
        {
            "id": "msg-3",
            "message": "Third message",
            "role": "model",
            "created_at": "2025-01-01T00:02:00+00:00",
            "message_order": 3,
            "user_feedback": False,
        },
        {
            "id": "msg-2",
            "message": "Second message",
            "role": "model",
            "created_at": "2025-01-01T00:01:00+00:00",
            "message_order": 2,
            "user_feedback": True,
        },
        {
            "id": "msg-1",
            "message": "First message",
            "role": "user",
            "created_at": "2025-01-01T00:00:00+00:00",
            "message_order": 1,
            "user_feedback": None,
        },
    ]

    def fake_get_client():
        return _DummyClientWithUpdate(messages_data)

    monkeypatch.setattr(sc, "get_client", fake_get_client)

    result = sc.fetch_lucy_messages("user-123", "conv-456", desc=True)

    assert len(result) == 3
    assert result[0]["id"] == "msg-3"  # Most recent first
    assert result[1]["id"] == "msg-2"
    assert result[2]["id"] == "msg-1"


# ---------------------------------------------------------------------------
# Full UserOrgProfile normalized contract for get_user_org_profiles
# ---------------------------------------------------------------------------

def _make_full_brand_client():
    """Return a dummy client with a brand row containing all profile fields."""
    permissions = [
        {
            "org_id": "org-full",
            "profiles": {"first_name": "Jane", "last_name": "Doe"},
            "organizations": {"company_name": "FullCo"},
        }
    ]
    brands = [
        {
            "associated_organization_id": "org-full",
            "brand_name": "FullBrand",
            "brand_website_url": "fullbrand.com",
            "brand_industry": ["E-commerce"],
            "brand_states_licensed": ["CA", "NY"],
            "brand_description": "We sell stuff",
            "brand_tone_of_voice": "Bold",
            "brand_purpose": "Inspire",
            "brand_mission_vision": "World domination",
            "brand_core_values": "Honesty, Speed",
            "brand_audiences": ["Millennials", "Gen Z"],
            "brand_positioning": "Premium but accessible",
            "brand_design_elements": "Minimalist sans-serif",
            "brand_messaging_pillars": "Quality, Value, Trust",
            "brand_copywriting_tone": "Conversational",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
    ]
    return _DummyClient({"permissions": permissions, "brands": brands})


def test_get_user_org_profiles_maps_all_normalized_fields(monkeypatch):
    """Every field declared on UserOrgProfile must be populated from the brand row."""
    try:
        sc._PROFILE_CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    monkeypatch.setattr(sc, "get_client", _make_full_brand_client)
    out = sc.get_user_org_profiles("u-full")
    assert out, "Expected at least one profile row"
    row = out[0]

    expected = {
        "org_id": "org-full",
        "first_name": "Jane",
        "last_name": "Doe",
        "company_name": "FullCo",
        "website_url": "fullbrand.com",
        "industry": "E-commerce",
        "states": "CA, NY",
        "brand_name": "FullBrand",
        "description": "We sell stuff",
        "tone_of_voice": "Bold",
        "purpose": "Inspire",
        "mission_vision": "World domination",
        "core_values": "Honesty, Speed",
        "audience": "Millennials, Gen Z",
        "positioning": "Premium but accessible",
        "design_elements": "Minimalist sans-serif",
        "messaging_pillars": "Quality, Value, Trust",
        "copywriting_tone": "Conversational",
    }

    for field, expected_value in expected.items():
        assert row.get(field) == expected_value, (
            f"Normalized field '{field}' mismatch: got {row.get(field)!r}, expected {expected_value!r}"
        )


def test_get_user_org_profiles_normalized_fields_compatible_with_model(monkeypatch):
    """UserOrgProfile(**row) must populate all typed fields without errors."""
    try:
        sc._PROFILE_CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    from lucy.agents.common.models import UserOrgProfile

    monkeypatch.setattr(sc, "get_client", _make_full_brand_client)
    out = sc.get_user_org_profiles("u-full")
    profile = UserOrgProfile(**out[0])

    assert profile.first_name == "Jane"
    assert profile.last_name == "Doe"
    assert profile.company_name == "FullCo"
    assert profile.website_url == "fullbrand.com"
    assert profile.description == "We sell stuff"
    assert profile.tone_of_voice == "Bold"
    assert profile.audience == "Millennials, Gen Z"
    assert profile.core_values == "Honesty, Speed"
    assert profile.positioning == "Premium but accessible"
    assert profile.messaging_pillars == "Quality, Value, Trust"
    assert profile.copywriting_tone == "Conversational"


def test_get_user_org_profiles_normalized_fields_null_when_brand_absent(monkeypatch):
    """When no brand row exists for the org, all brand narrative fields must be None."""
    try:
        sc._PROFILE_CACHE.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    permissions = [
        {
            "org_id": "org-nobrand",
            "profiles": {"first_name": "Bob", "last_name": "Smith"},
            "organizations": {"company_name": "NoBrandCo"},
        }
    ]
    monkeypatch.setattr(
        sc, "get_client",
        lambda: _DummyClient({"permissions": permissions, "brands": []})
    )

    out = sc.get_user_org_profiles("u-nobrand")
    row = out[0]

    brand_narrative_fields = [
        "description", "tone_of_voice", "purpose", "mission_vision",
        "core_values", "audience", "positioning", "design_elements",
        "messaging_pillars", "copywriting_tone",
    ]
    for field in brand_narrative_fields:
        assert row.get(field) is None, (
            f"Expected {field!r} to be None when no brand row exists, got {row.get(field)!r}"
        )
