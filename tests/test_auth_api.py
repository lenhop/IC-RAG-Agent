"""
Integration tests for auth API endpoints.

Uses mock AuthService to avoid ClickHouse dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.gateway.api import app
from src.auth.schemas import RegisterResponse, SignInResponse, UserInfo
from src.gateway.auth_routes import get_auth_service
from src.auth.service import AuthService


client = TestClient(app)


@pytest.fixture
def mock_auth_service():
    """Mock AuthService for register/signin using FastAPI dependency_overrides."""
    mock_svc = MagicMock(spec=AuthService)
    app.dependency_overrides[get_auth_service] = lambda: mock_svc
    yield mock_svc
    app.dependency_overrides.clear()


def test_register_200(mock_auth_service):
    """POST /api/v1/auth/register returns 200 with user info."""
    mock_auth_service.register.return_value = RegisterResponse(
        user_id="uuid-123",
        user_name="testuser",
        role="general",
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"user_name": "testuser", "password": "Pass1234", "email": "test@example.com"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_name"] == "testuser"
    assert data["role"] == "general"
    assert data["user_id"] == "uuid-123"


def test_register_400_validation_error(mock_auth_service):
    """POST /api/v1/auth/register returns 400 when validation fails."""
    mock_auth_service.register.side_effect = ValueError("user_name already exists")
    resp = client.post(
        "/api/v1/auth/register",
        json={"user_name": "existing", "password": "Pass1234"},
    )
    assert resp.status_code == 400
    assert "user_name already exists" in resp.json().get("detail", "")


def test_signin_200(mock_auth_service):
    """POST /api/v1/auth/signin returns 200 with token and user."""
    mock_auth_service.sign_in.return_value = SignInResponse(
        access_token="jwt-token-123",
        token_type="bearer",
        user=UserInfo(
            user_id="uuid-123",
            user_name="testuser",
            email="test@example.com",
            role="general",
            status="active",
        ),
    )
    resp = client.post(
        "/api/v1/auth/signin",
        json={"user_name": "testuser", "password": "Pass1234"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"] == "jwt-token-123"
    assert data["token_type"] == "bearer"
    assert data["user"]["user_name"] == "testuser"
    assert data["user"]["role"] == "general"


def test_signin_401_invalid_credentials(mock_auth_service):
    """POST /api/v1/auth/signin returns 401 for invalid credentials."""
    mock_auth_service.sign_in.side_effect = ValueError("Invalid user_name or password")
    resp = client.post(
        "/api/v1/auth/signin",
        json={"user_name": "testuser", "password": "wrong"},
    )
    assert resp.status_code == 401
    assert "Invalid" in resp.json().get("detail", "")


def test_signout_200():
    """POST /api/v1/auth/signout returns 200."""
    resp = client.post("/api/v1/auth/signout")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data


def test_me_200_with_valid_token():
    """GET /api/v1/auth/me returns 200 with user info when token valid."""
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer " + _create_test_token()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "user_id" in data
    assert "user_name" in data
    assert "role" in data


def test_me_401_without_token():
    """GET /api/v1/auth/me returns 401 without Authorization header."""
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


def test_me_401_with_invalid_token():
    """GET /api/v1/auth/me returns 401 with invalid token."""
    resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert resp.status_code == 401


def _create_test_token() -> str:
    """Create a valid JWT for testing."""
    from src.auth.jwt_util import create_token
    return create_token("test-user-id", "testuser", "general")
