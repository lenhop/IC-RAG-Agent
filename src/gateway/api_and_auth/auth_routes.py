"""
Auth API routes: register, sign-in, sign-out, me.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from src.auth.schemas import RegisterRequest, RegisterResponse, SignInRequest, SignInResponse, UserInfo
from src.auth.service import AuthService
from src.auth.jwt_util import verify_token
from src.uds.uds_client import QueryError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Dependency: return singleton AuthService."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


async def get_current_user(
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict[str, Any]:
    """
    Extract and verify JWT from Authorization header.

    Returns:
        Decoded payload with sub, user_name, role.

    Raises:
        HTTPException 401: Missing or invalid token.
    """
    if not authorization or not authorization.strip():
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    payload = verify_token(authorization)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


@router.post("/register", response_model=RegisterResponse)
async def register(
    body: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RegisterResponse:
    """
    Register a new user.

    Returns:
        RegisterResponse with user_id, user_name, role.
    """
    try:
        return auth_service.register(
            user_name=body.user_name,
            password=body.password,
            email=body.email,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except QueryError as e:
        logger.warning("Auth register failed (ClickHouse): %s", e)
        detail = str(e)
        if "Unknown table" in detail or "UNKNOWN_TABLE" in detail:
            detail = (
                "User table not initialized. Run: clickhouse-client --database ic_agent -f db/auth/create_user_table.sql"
            )
        raise HTTPException(status_code=503, detail=detail)


@router.post("/signin", response_model=SignInResponse)
async def signin(
    request: Request,
    body: SignInRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> SignInResponse:
    """
    Sign in and return JWT + user info.
    """
    client_ip = request.client.host if request.client else None
    try:
        return auth_service.sign_in(
            user_name=body.user_name,
            password=body.password,
            client_ip=client_ip,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except QueryError as e:
        logger.warning("Auth signin failed (ClickHouse): %s", e)
        detail = str(e)
        if "Unknown table" in detail or "UNKNOWN_TABLE" in detail:
            detail = (
                "User table not initialized. Run: clickhouse-client --database ic_agent -f db/auth/create_user_table.sql"
            )
        raise HTTPException(status_code=503, detail=detail)


@router.post("/signout")
async def signout() -> dict[str, str]:
    """
    Sign out. Client discards token; no server-side blacklist.
    Returns 204-style success.
    """
    return {"message": "Signed out"}


@router.get("/me", response_model=UserInfo)
async def me(
    payload: dict[str, Any] = Depends(get_current_user),
) -> UserInfo:
    """
    Get current user from JWT.
    """
    user_id = payload.get("sub", "")
    user_name = payload.get("user_name", "")
    role = payload.get("role", "general")
    return UserInfo(
        user_id=str(user_id),
        user_name=user_name,
        email=None,
        role=role,
        status="active",
    )
