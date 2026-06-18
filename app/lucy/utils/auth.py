from __future__ import annotations

import os
from typing import Optional
from functools import lru_cache

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from loguru import logger

from lucy.core.bootstrap import bootstrap

bootstrap()

# === Defaults ===
JWT_ALGOS = os.getenv("JWT_ALGOS", "ES256")

JWT_ISSUER = os.getenv("JWT_ISSUER", "https://djixfuimtdnejaiscqvy.supabase.co/auth/v1")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "authenticated")  # optional
# Require auth by default; only disable if explicitly set to a falsey token
JWT_JWKS_URL = os.getenv(
    "JWT_JWKS_URL",
    "https://api-staging.app.meetlofi.com/auth/v1/.well-known/jwks.json",
)

# New: Support for symmetric key JWT signing (development)
JWT_SECRET = os.getenv("JWT_SECRET")  # For HS256 symmetric signing
JWT_USE_SECRET = os.getenv("JWT_USE_SECRET", "false").lower() in {
    "true",
    "1",
    "yes",
    "on",
}

_AUTH_ENV = os.getenv("AUTH_REQUIRED")
AUTH_REQUIRED = (
    True
    if _AUTH_ENV is None
    else _AUTH_ENV.strip().lower() not in {"false", "0", "no", "off"}
)

# Always allow dependency to run; enforce requirement in verify_jwt
security = HTTPBearer(auto_error=False)


@lru_cache
def get_jwks(url=JWT_JWKS_URL):
    """
    Fetch JWKS and cache the result.

    :return: JWKS document as JSON.
    :rtype: dict
    """
    resp = httpx.get(url, timeout=5.0)
    resp.raise_for_status()
    return resp.json()


def get_public_key(token: str):
    """
    Find the matching JWK in JWKS for the token's key id (kid).

    :param token: Encoded JWT string.
    :return: Matching JWK dict.
    :rtype: dict
    :raises fastapi.HTTPException: If no matching key is found.
    """
    unverified_header = jwt.get_unverified_header(token)

    jwks = get_jwks()
    for jwk in jwks["keys"]:
        if jwk["kid"] == unverified_header["kid"]:
            return jwk
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Public key not found for token",
    )


def verify_jwt_with_secret(token: str) -> dict:
    """
    Verify JWT using symmetric key (HS256).

    :param token: Encoded JWT string.
    :return: Decoded JWT payload.
    :rtype: dict
    :raises fastapi.HTTPException: For invalid/expired tokens.
    """
    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET not configured for symmetric key verification",
        )

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            issuer=JWT_ISSUER,
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation error: {str(e)}",
        )


def verify_jwt_with_jwks(token: str) -> dict:
    """
    Verify JWT using asymmetric key from JWKS.

    :param token: Encoded JWT string.
    :return: Decoded JWT payload.
    :rtype: dict
    :raises fastapi.HTTPException: For invalid/expired tokens.
    """
    jwk = get_public_key(token)

    try:
        payload = jwt.decode(
            token,
            jwk,
            algorithms=[JWT_ALGOS],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation error: {str(e)}",
        )


def verify_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Verify JWT from the Authorization header.

    :param credentials: Parsed bearer token credentials (dependency injected).
    :return: Decoded JWT payload if valid; None if auth is disabled and header missing.
    :rtype: Optional[dict]
    :raises fastapi.HTTPException: For missing/expired/invalid tokens.
    """
    # Enforce requirement
    if AUTH_REQUIRED and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # If not required and no credentials, allow through
    if not AUTH_REQUIRED and credentials is None:
        return None

    # Validate provided token
    assert credentials is not None  # For type checkers
    token = credentials.credentials

    # Choose verification method based on configuration
    if JWT_USE_SECRET:
        logger.debug("Using symmetric key JWT verification")
        return verify_jwt_with_secret(token)
    else:
        logger.debug("Using JWKS asymmetric key JWT verification")
        return verify_jwt_with_jwks(token)


def extract_user_id(payload: Optional[dict]) -> str:
    """
    Extract the user id from a JWT payload.

    :param payload: JWT payload as a dict.
    :return: User id or "anonymous" if not present.
    :rtype: str
    """
    if isinstance(payload, dict):
        uid = payload.get("sub") or payload.get("user_id")
        if not uid:
            user = payload.get("user")
            if isinstance(user, dict):
                uid = user.get("id")
        if uid:
            return str(uid)
    return "anonymous"
