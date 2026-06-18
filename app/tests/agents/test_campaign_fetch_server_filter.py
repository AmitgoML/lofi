"""Tests for _fetch_user_campaigns server-side filtering.

Verifies that when campaign_id is provided, the query pushes the filter to
the database rather than fetching all campaigns and filtering in Python.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_supabase(rows: list) -> MagicMock:
    """Return a mock supabase client that tracks query builder calls."""
    table_mock = MagicMock()
    table_mock.select.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.execute.return_value = SimpleNamespace(data=rows)
    client_mock = MagicMock()
    client_mock.table.return_value = table_mock
    return client_mock, table_mock


class TestFetchUserCampaignsServerFilter:
    def test_campaign_id_filter_pushed_to_db(self):
        """When campaign_id is provided, an eq(campaign_id=...) must be in the query."""
        from lucy.agents.performance_analyst_agent import _fetch_user_campaigns

        client_mock, table_mock = _make_mock_supabase([])

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            _run(_fetch_user_campaigns("user-1", "*", campaign_id="camp-42"))

        # Inspect all eq() calls: one for user_id, one for campaign_id
        eq_calls = table_mock.eq.call_args_list
        eq_kwargs = [(c.args[0], c.args[1]) for c in eq_calls]
        assert ("campaign_id", "camp-42") in eq_kwargs, (
            f"Expected server-side campaign_id filter in DB query, got eq calls: {eq_kwargs}"
        )

    def test_no_campaign_id_filter_when_not_provided(self):
        """When campaign_id is None, only the user_id eq filter must appear."""
        from lucy.agents.performance_analyst_agent import _fetch_user_campaigns

        client_mock, table_mock = _make_mock_supabase([])

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            _run(_fetch_user_campaigns("user-1", "*"))

        eq_calls = table_mock.eq.call_args_list
        eq_kwargs = [c.args[0] for c in eq_calls]
        assert "campaign_id" not in eq_kwargs, (
            f"campaign_id filter must not appear when not requested, got: {eq_kwargs}"
        )
        assert "user_id" in eq_kwargs

    def test_returns_only_matching_campaign(self):
        """The function returns exactly the rows the DB provides for the given campaign_id."""
        from lucy.agents.performance_analyst_agent import _fetch_user_campaigns

        row = {"campaign_id": "camp-42", "campaign_name": "My Campaign"}
        client_mock, table_mock = _make_mock_supabase([row])

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            result = _run(_fetch_user_campaigns("user-1", "*", campaign_id="camp-42"))

        assert result == [row]

    def test_returns_all_campaigns_when_no_id(self):
        """Without campaign_id, all rows returned by DB are passed through."""
        from lucy.agents.performance_analyst_agent import (
            _fetch_user_campaigns,
            _CAMPAIGN_SUMMARY_COLUMNS,
        )

        rows = [
            {"campaign_id": "c1", "campaign_name": "Alpha"},
            {"campaign_id": "c2", "campaign_name": "Beta"},
        ]
        client_mock, table_mock = _make_mock_supabase(rows)

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            result = _run(_fetch_user_campaigns("user-1", _CAMPAIGN_SUMMARY_COLUMNS))

        assert result == rows

    def test_summary_columns_used_for_list_fetch(self):
        """List fetches must use _CAMPAIGN_SUMMARY_COLUMNS, not select('*')."""
        from lucy.agents.performance_analyst_agent import (
            _fetch_user_campaigns,
            _CAMPAIGN_SUMMARY_COLUMNS,
        )

        client_mock, table_mock = _make_mock_supabase([])

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            _run(_fetch_user_campaigns("user-1", _CAMPAIGN_SUMMARY_COLUMNS))

        select_calls = table_mock.select.call_args_list
        assert select_calls, "Expected select() to be called"
        passed_columns = select_calls[0].args[0]
        assert passed_columns == _CAMPAIGN_SUMMARY_COLUMNS, (
            f"Expected summary columns for list fetch, got: {passed_columns!r}"
        )

    def test_full_columns_used_for_single_campaign_fetch(self):
        """Single-campaign fetches must use select('*') for complete data."""
        from lucy.agents.performance_analyst_agent import _fetch_user_campaigns

        client_mock, table_mock = _make_mock_supabase([])

        with patch(
            "lucy.agents.performance_analyst_agent.get_client",
            return_value=client_mock,
        ):
            _run(_fetch_user_campaigns("user-1", "*", campaign_id="camp-1"))

        select_calls = table_mock.select.call_args_list
        passed_columns = select_calls[0].args[0]
        assert passed_columns == "*", (
            f"Expected '*' columns for single-campaign deep-dive, got: {passed_columns!r}"
        )
