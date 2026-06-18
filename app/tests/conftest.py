import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def pytest_configure(config):
    """Set required env vars before collection so module-level Agent() calls succeed."""
    os.environ.setdefault("OPENAI_API_KEY", "test")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")


@pytest.fixture(scope="session", autouse=True)
def _ensure_openai_env():
    os.environ.setdefault("OPENAI_API_KEY", "test")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")
    # Mock Supabase environment variables for testing
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")


@pytest.fixture(scope="session", autouse=True)
def _prevent_env_loading():
    """Prevent tests from loading .env file by mocking the load_envs function"""
    with patch("lucy.utils.secrets.load_envs") as mock_load_envs:
        # Make the mock function do nothing instead of loading .env
        mock_load_envs.return_value = None
        yield mock_load_envs


@pytest.fixture(autouse=True)
def _clear_agent_cache():
    """Ensure each test starts with a fresh agent factory cache."""
    from lucy.agents.common.factory import clear_cache
    clear_cache()
    yield
    clear_cache()


class _FakeRunResult:
    def __init__(self, chunks: Iterable[str]):
        self._chunks = list(chunks)

    async def stream(self, debounce_by: float = 0.01):
        for chunk in self._chunks:
            # simulate async streaming
            await asyncio.sleep(0)
            yield chunk

    def timestamp(self):
        return datetime.now(tz=timezone.utc)

    def new_messages_json(self) -> bytes:
        # Return valid JSON bytes representing an empty message list
        return b"[]"


class _FakeRunContext:
    def __init__(self, result: _FakeRunResult):
        self._result = result

    async def __aenter__(self) -> _FakeRunResult:
        return self._result

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAgent:
    def __init__(self, chunks: Iterable[str] | None = None):
        self._chunks = list(chunks or ("Hello", " world"))

    def run_stream(self, *args, **kwargs) -> _FakeRunContext:
        return _FakeRunContext(_FakeRunResult(self._chunks))


@pytest.fixture(scope="session")
def client() -> TestClient:
    # Ensure OpenAI clients can construct without real credentials
    os.environ.setdefault("OPENAI_API_KEY", "test")
    # Import here to ensure app module-level setup runs once per session
    from lucy.app import app as fastapi_app
    from lucy.utils.auth import verify_jwt
    from lucy.api.common.deps import get_history_store, get_supabase_client
    from lucy.database.history import HistoryStore
    import lucy.app as app_module

    # Override auth dependency to avoid real JWT verification
    fastapi_app.dependency_overrides[verify_jwt] = lambda: {"sub": "test-user-id"}

    # Mock Supabase client
    class MockSupabaseClient:
        def __init__(self):
            pass

        def table(self, table_name: str):
            return MockSupabaseTable()

    class MockSupabaseTable:
        def __init__(self):
            pass

        def select(self, *args):
            return self

        def insert(self, *args):
            return self

        def update(self, *args):
            return self

        def delete(self, *args):
            return self

        def eq(self, *args):
            return self

        def order(self, *args):
            return self

        def limit(self, *args):
            return self

        def execute(self):
            return MockSupabaseResponse()

    class MockSupabaseResponse:
        def __init__(self):
            self.data = []
            self.count = 0

    # Mock history store
    class MockHistoryStore:
        def __init__(self):
            pass

        async def get_recent_sanitized_messages(
            self, user_id: str, session_id: str, limit: int = 10
        ):
            return []

        async def store_message(
            self, user_id: str, session_id: str, role: str, content: str
        ):
            pass

        async def get_user_org_profiles(self, user_id: str):
            return []

    # Override dependencies
    fastapi_app.dependency_overrides[get_supabase_client] = lambda: MockSupabaseClient()
    fastapi_app.dependency_overrides[get_history_store] = lambda: MockHistoryStore()

    # Override agent selection to avoid network/model calls
    def _fake_get_agent(agent_name: str | None = None):
        return FakeAgent(["Test ", "response"])  # two chunks

    async def _fake_create_agent_for_query(query: str, **kwargs):
        return FakeAgent(["Test ", "stream"])  # two chunks

    app_module.get_agent = _fake_get_agent  # type: ignore[attr-defined]
    app_module.create_agent_for_query = _fake_create_agent_for_query  # type: ignore[attr-defined]

    with TestClient(fastapi_app) as c:
        yield c
