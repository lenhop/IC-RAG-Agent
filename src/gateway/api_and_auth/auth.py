"""
Gateway auth and user resolution.

AuthGuard: JWT validation and user_id resolution for query/rewrite/history endpoints.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import HTTPException

from ..schemas import QueryRequest
from src.auth.jwt_util import verify_token


class AuthGuard:
    """
    Auth and user resolution for the gateway.

    All methods are class methods; dependencies (request, authorization) passed as arguments.
    """

    @classmethod
    def auth_required(cls) -> bool:
        """Return True when gateway requires auth for query/rewrite."""
        return os.getenv("GATEWAY_AUTH_REQUIRED", "").lower() in ("1", "true", "yes", "on")

    @classmethod
    async def get_optional_user(cls, authorization: Optional[str]) -> Optional[dict]:
        """
        When GATEWAY_AUTH_REQUIRED=true, validate JWT and return payload or raise 401.
        When false, return None (no auth check).
        """
        if not cls.auth_required():
            return None
        if not authorization or not str(authorization).strip():
            raise HTTPException(status_code=401, detail="Authorization required")
        payload = verify_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return payload

    @classmethod
    async def require_user_for_history(cls, authorization: Optional[str]) -> dict:
        """
        Require valid JWT for user history endpoint.
        Returns payload with sub (user_id); raises 401 if missing or invalid.
        """
        if not authorization or not str(authorization).strip():
            raise HTTPException(status_code=401, detail="Authorization required")
        payload = verify_token(authorization)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return payload

    @classmethod
    def resolve_user_id(cls, request: QueryRequest, user_payload: Optional[dict]) -> Optional[str]:
        """
        Resolve effective user_id for memory operations.

        When GATEWAY_AUTH_REQUIRED=true: derive from JWT (sub claim).
        When auth optional: use request.user_id from body.
        """
        if user_payload and user_payload.get("sub"):
            return str(user_payload["sub"]).strip() or None
        if request.user_id and str(request.user_id).strip():
            return str(request.user_id).strip()
        return None
