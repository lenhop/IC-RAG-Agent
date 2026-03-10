"""
JWT token creation and verification.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt

# Algorithm for JWT signing
ALGORITHM = "HS256"


def _get_secret() -> str:
    """Get JWT secret from env; use dev default when not set."""
    secret = os.getenv("AUTH_JWT_SECRET", "").strip()
    if not secret:
        secret = os.getenv("JWT_SECRET", "dev-secret-change-in-production").strip()
    return secret or "dev-secret-change-in-production"


def _get_expire_seconds() -> int:
    """Get token expiry in seconds from env."""
    return int(os.getenv("AUTH_JWT_EXPIRE_SECONDS", "86400"))


def create_token(
    user_id: str,
    user_name: str,
    role: str,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token for the user.

    Args:
        user_id: User UUID.
        user_name: User display name.
        role: User role (general, supervisor, admin).
        extra_claims: Optional additional claims.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + timedelta(seconds=_get_expire_seconds())
    payload = {
        "sub": user_id,
        "user_name": user_name,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT string (with or without "Bearer " prefix).

    Returns:
        Decoded payload dict, or None if invalid/expired.
    """
    t = (token or "").strip()
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    if not t:
        return None
    try:
        payload = jwt.decode(t, _get_secret(), algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_user_id_from_token(token: str) -> Optional[str]:
    """Extract user_id (sub) from token."""
    payload = verify_token(token)
    if not payload:
        return None
    return payload.get("sub")
