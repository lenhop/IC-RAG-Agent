"""
Unit tests for auth module: password, JWT, AuthService, UserRepository.

Uses mocks for ClickHouse; no real database required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.auth.password import hash_password, validate_password_strength, verify_password
from src.auth.jwt_util import create_token, verify_token, get_user_id_from_token
from src.auth.service import AuthService
from src.auth.schemas import RegisterResponse, SignInResponse, UserInfo


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


def test_hash_password_returns_non_empty_string():
    """hash_password returns a non-empty bcrypt hash."""
    h = hash_password("test123")
    assert isinstance(h, str)
    assert len(h) > 20
    assert h.startswith("$2")


def test_verify_password_correct():
    """verify_password returns True for correct password."""
    h = hash_password("secret99")
    assert verify_password("secret99", h) is True


def test_verify_password_incorrect():
    """verify_password returns False for wrong password."""
    h = hash_password("secret99")
    assert verify_password("wrong", h) is False


def test_validate_password_strength_valid():
    """validate_password_strength accepts valid password."""
    valid, err = validate_password_strength("Pass1234")
    assert valid is True
    assert err == ""


def test_validate_password_strength_too_short():
    """validate_password_strength rejects password shorter than min_length."""
    valid, err = validate_password_strength("Ab1")
    assert valid is False
    assert "at least" in err


def test_validate_password_strength_no_letter():
    """validate_password_strength rejects password without letter."""
    valid, err = validate_password_strength("12345678")
    assert valid is False
    assert "letter" in err


def test_validate_password_strength_no_digit():
    """validate_password_strength rejects password without digit."""
    valid, err = validate_password_strength("Password")
    assert valid is False
    assert "digit" in err


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def test_create_token_returns_string():
    """create_token returns non-empty JWT string."""
    token = create_token("user-123", "alice", "general")
    assert isinstance(token, str)
    assert len(token) > 20


def test_verify_token_valid():
    """verify_token decodes valid token."""
    token = create_token("user-123", "alice", "general")
    payload = verify_token(token)
    assert payload is not None
    assert payload.get("sub") == "user-123"
    assert payload.get("user_name") == "alice"
    assert payload.get("role") == "general"
    assert "exp" in payload


def test_verify_token_with_bearer_prefix():
    """verify_token accepts Bearer prefix."""
    token = create_token("user-123", "alice", "general")
    payload = verify_token(f"Bearer {token}")
    assert payload is not None
    assert payload.get("sub") == "user-123"


def test_verify_token_invalid_returns_none():
    """verify_token returns None for invalid token."""
    assert verify_token("invalid.jwt.token") is None
    assert verify_token("") is None


def test_get_user_id_from_token():
    """get_user_id_from_token extracts sub from valid token."""
    token = create_token("user-456", "bob", "supervisor")
    assert get_user_id_from_token(token) == "user-456"
    assert get_user_id_from_token("invalid") is None


# ---------------------------------------------------------------------------
# AuthService (mock repository)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_repo():
    """Mock UserRepository."""
    repo = MagicMock()
    repo.get_by_user_name.return_value = None
    repo.get_by_email.return_value = None
    return repo


def test_register_success(mock_repo):
    """AuthService.register creates user and returns RegisterResponse."""
    svc = AuthService(repository=mock_repo)
    result = svc.register("testuser", "Pass1234", "test@example.com")
    assert isinstance(result, RegisterResponse)
    assert result.user_name == "testuser"
    assert result.role == "general"
    assert result.user_id
    mock_repo.create.assert_called_once()
    call_kwargs = mock_repo.create.call_args[1]
    assert call_kwargs["user_name"] == "testuser"
    assert call_kwargs["email"] == "test@example.com"
    assert call_kwargs["role"] == "general"
    assert "password_hash" in call_kwargs


def test_register_user_name_exists_raises(mock_repo):
    """AuthService.register raises when user_name already exists."""
    mock_repo.get_by_user_name.return_value = {"user_id": "existing"}
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="user_name already exists"):
        svc.register("existing", "Pass1234")


def test_register_email_exists_raises(mock_repo):
    """AuthService.register raises when email already exists."""
    mock_repo.get_by_email.return_value = {"user_id": "existing"}
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="email already exists"):
        svc.register("newuser", "Pass1234", "existing@example.com")


def test_register_weak_password_raises(mock_repo):
    """AuthService.register raises for weak password."""
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="Password"):
        svc.register("newuser", "short", None)


def test_sign_in_success(mock_repo):
    """AuthService.sign_in returns SignInResponse with token."""
    mock_repo.get_by_user_name.return_value = {
        "user_id": str(uuid4()),
        "user_name": "alice",
        "email": "alice@example.com",
        "password_hash": hash_password("Pass1234"),
        "role": "general",
        "status": "active",
    }
    svc = AuthService(repository=mock_repo)
    result = svc.sign_in("alice", "Pass1234")
    assert isinstance(result, SignInResponse)
    assert result.access_token
    assert result.token_type == "bearer"
    assert result.user.user_name == "alice"
    assert result.user.role == "general"
    assert result.user.user_id


def test_sign_in_wrong_password_raises(mock_repo):
    """AuthService.sign_in raises for wrong password."""
    mock_repo.get_by_user_name.return_value = {
        "user_id": str(uuid4()),
        "user_name": "alice",
        "password_hash": hash_password("Pass1234"),
        "role": "general",
        "status": "active",
    }
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="Invalid"):
        svc.sign_in("alice", "WrongPass")


def test_sign_in_user_not_found_raises(mock_repo):
    """AuthService.sign_in raises when user not found."""
    mock_repo.get_by_user_name.return_value = None
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="Invalid"):
        svc.sign_in("nonexistent", "Pass1234")


def test_sign_in_inactive_user_raises(mock_repo):
    """AuthService.sign_in raises for inactive user."""
    mock_repo.get_by_user_name.return_value = {
        "user_id": str(uuid4()),
        "user_name": "alice",
        "password_hash": hash_password("Pass1234"),
        "role": "general",
        "status": "inactive",
    }
    svc = AuthService(repository=mock_repo)
    with pytest.raises(ValueError, match="not active"):
        svc.sign_in("alice", "Pass1234")
