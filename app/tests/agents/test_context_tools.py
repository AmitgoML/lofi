"""Tests for context tools: get_brand_context, get_ad_accounts, get_login_history.

Key contracts verified:
- Large JSONB fields are summarised, not dumped verbatim.
- Summary strings do NOT reference non-existent tools.
- get_ad_accounts and get_login_history return graceful error payloads on failure.
- get_ad_accounts and get_login_history return empty lists when no rows exist.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lucy.agents.common.models import ChatDeps


def _make_deps(brand: Dict[str, Any] | None = None, user_id: str = "u1") -> ChatDeps:
    return ChatDeps(user_id=user_id, brand_context=brand)


def _run(coro):
    """Run a coroutine synchronously for test purposes."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# get_brand_context: large-field summarisation
# ---------------------------------------------------------------------------

class TestGetBrandContext:
    def _invoke(self, brand: Dict[str, Any]) -> Dict[str, Any]:
        from pydantic_ai import Agent
        from lucy.agents.common.context_tools import register_brand_context_tool

        results: list[Dict] = []

        agent = MagicMock(spec=Agent)

        captured_fn = None

        def _capture_tool(fn):
            nonlocal captured_fn
            captured_fn = fn
            return fn

        agent.tool = _capture_tool
        register_brand_context_tool(agent)
        assert captured_fn is not None

        ctx = MagicMock()
        ctx.deps = _make_deps(brand=brand)

        return _run(captured_fn(ctx))

    def test_large_locations_list_is_summarised(self):
        brand = {"brand_locations": [{"city": f"City {i}"} for i in range(100)]}
        result = self._invoke(brand)
        locations_val = result.get("brand_locations", "")
        assert isinstance(locations_val, str)
        assert "100" in locations_val, f"Expected count in summary, got: {locations_val!r}"

    def test_large_customers_list_is_summarised(self):
        brand = {"brand_customers_lists": ["cust" + str(i) for i in range(75)]}
        result = self._invoke(brand)
        val = result.get("brand_customers_lists", "")
        assert "75" in val, f"Expected count in summary, got: {val!r}"

    def test_summary_does_not_reference_nonexistent_tool(self):
        brand = {"brand_locations": [{"city": "NY"}] * 5}
        result = self._invoke(brand)
        val = result.get("brand_locations", "")
        assert "get_brand_locations" not in val, (
            f"Summary must not reference a nonexistent tool, got: {val!r}"
        )

    def test_small_locations_list_is_still_summarised(self):
        """brand_locations is always in LARGE_JSONB_KEYS so even small lists are summarised."""
        brand = {"brand_locations": ["New York", "Los Angeles"]}
        result = self._invoke(brand)
        val = result.get("brand_locations", "")
        assert isinstance(val, str)
        # Always summarised because brand_locations is a known large field
        assert "2" in val and "entries" in val

    def test_available_true_when_brand_exists(self):
        result = self._invoke({"brand_name": "TestBrand"})
        assert result.get("available") is True

    def test_available_false_when_no_brand(self):
        from pydantic_ai import Agent
        from lucy.agents.common.context_tools import register_brand_context_tool

        agent = MagicMock(spec=Agent)
        captured_fn = None

        def _capture_tool(fn):
            nonlocal captured_fn
            captured_fn = fn
            return fn

        agent.tool = _capture_tool
        register_brand_context_tool(agent)

        ctx = MagicMock()
        ctx.deps = _make_deps(brand=None)

        result = _run(captured_fn(ctx))
        assert result.get("available") is False


# ---------------------------------------------------------------------------
# get_ad_accounts: graceful error handling
# ---------------------------------------------------------------------------

class TestGetAdAccounts:
    def _invoke(self, supabase_data=None, raise_exc=None) -> Dict[str, Any]:
        from pydantic_ai import Agent
        from lucy.agents.common.context_tools import register_ad_accounts_tool

        agent = MagicMock(spec=Agent)
        captured_fn = None

        def _capture_tool(fn):
            nonlocal captured_fn
            captured_fn = fn
            return fn

        agent.tool = _capture_tool
        register_ad_accounts_tool(agent)
        assert captured_fn is not None

        ctx = MagicMock()
        ctx.deps = _make_deps()

        if raise_exc:
            fake_client = MagicMock()
            fake_client.table.side_effect = raise_exc
            with patch("lucy.agents.common.context_tools.get_client", return_value=fake_client):
                return _run(captured_fn(ctx))

        fake_table = MagicMock()
        fake_table.select.return_value = fake_table
        fake_table.eq.return_value = fake_table
        from types import SimpleNamespace
        fake_table.execute.return_value = SimpleNamespace(data=supabase_data or [])
        fake_client = MagicMock()
        fake_client.table.return_value = fake_table

        with patch("lucy.agents.common.context_tools.get_client", return_value=fake_client):
            return _run(captured_fn(ctx))

    def test_returns_rows_on_success(self):
        rows = [{"account_id": "ga-1", "platform": "google_ads", "status": "active"}]
        result = self._invoke(supabase_data=rows)
        assert result["accounts"] == rows
        assert result["total"] == 1

    def test_returns_empty_list_when_no_rows(self):
        result = self._invoke(supabase_data=[])
        assert result["accounts"] == []
        assert result["total"] == 0

    def test_graceful_error_on_supabase_failure(self):
        result = self._invoke(raise_exc=RuntimeError("PostgREST error"))
        assert result["accounts"] == []
        assert result["total"] == 0
        assert "error" in result, "Expected 'error' key in graceful error response"

    def test_error_key_is_user_friendly_string(self):
        result = self._invoke(raise_exc=Exception("DB gone"))
        assert isinstance(result.get("error"), str)
        assert result["error"]  # non-empty


# ---------------------------------------------------------------------------
# get_login_history: graceful error handling and limit clamping
# ---------------------------------------------------------------------------

class TestGetLoginHistory:
    def _invoke(self, limit: int = 10, supabase_data=None, raise_exc=None) -> Dict[str, Any]:
        from pydantic_ai import Agent
        from lucy.agents.common.context_tools import register_login_history_tool

        agent = MagicMock(spec=Agent)
        captured_fn = None

        def _capture_tool(fn):
            nonlocal captured_fn
            captured_fn = fn
            return fn

        agent.tool = _capture_tool
        register_login_history_tool(agent)
        assert captured_fn is not None

        ctx = MagicMock()
        ctx.deps = _make_deps()

        if raise_exc:
            fake_client = MagicMock()
            fake_client.table.side_effect = raise_exc
            with patch("lucy.agents.common.context_tools.get_client", return_value=fake_client):
                return _run(captured_fn(ctx, limit=limit))

        from types import SimpleNamespace
        fake_table = MagicMock()
        fake_table.select.return_value = fake_table
        fake_table.eq.return_value = fake_table
        fake_table.order.return_value = fake_table
        fake_table.limit.return_value = fake_table
        fake_table.execute.return_value = SimpleNamespace(data=supabase_data or [])
        fake_client = MagicMock()
        fake_client.table.return_value = fake_table

        with patch("lucy.agents.common.context_tools.get_client", return_value=fake_client):
            return _run(captured_fn(ctx, limit=limit))

    def test_returns_rows_on_success(self):
        rows = [{"login_id": "l1", "last_login_at": "2025-01-01T00:00:00Z"}]
        result = self._invoke(supabase_data=rows)
        assert result["logins"] == rows
        assert result["total"] == 1

    def test_returns_empty_list_when_no_rows(self):
        result = self._invoke(supabase_data=[])
        assert result["logins"] == []
        assert result["total"] == 0

    def test_graceful_error_on_supabase_failure(self):
        result = self._invoke(raise_exc=RuntimeError("table not found"))
        assert result["logins"] == []
        assert result["total"] == 0
        assert "error" in result

    def test_error_key_is_user_friendly_string(self):
        result = self._invoke(raise_exc=Exception("timeout"))
        assert isinstance(result.get("error"), str)
        assert result["error"]
