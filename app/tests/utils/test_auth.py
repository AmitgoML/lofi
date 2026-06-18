import os
from typing import Optional

import pytest
from fastapi import HTTPException

from lucy.utils import auth as auth_mod


def test_extract_user_id_variants():
    assert auth_mod.extract_user_id({"sub": "u1"}) == "u1"
    assert auth_mod.extract_user_id({"user_id": 123}) == "123"
    assert auth_mod.extract_user_id({"user": {"id": "abc"}}) == "abc"
    assert auth_mod.extract_user_id({}) == "anonymous"
    assert auth_mod.extract_user_id(None) == "anonymous"


def test_auth_required_default_true(monkeypatch):
    # ensure default environment doesn't disable auth
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    # reload module to recompute AUTH_REQUIRED
    import importlib

    importlib.reload(auth_mod)
    assert auth_mod.AUTH_REQUIRED is True


@pytest.mark.parametrize(
    "env_val,expected",
    [("false", False), ("0", False), ("no", False), ("off", False), ("true", True)],
)
def test_auth_required_env_values(monkeypatch, env_val, expected):
    monkeypatch.setenv("AUTH_REQUIRED", env_val)
    import importlib

    importlib.reload(auth_mod)
    assert auth_mod.AUTH_REQUIRED is expected


def test_verify_jwt_missing_header_raises(monkeypatch):
    # Force auth required
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    import importlib

    importlib.reload(auth_mod)
    with pytest.raises(HTTPException) as ei:
        auth_mod.verify_jwt(credentials=None)  # type: ignore[arg-type]
    assert ei.value.status_code == 401
    assert "Missing Authorization header" in ei.value.detail
