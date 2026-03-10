"""
Auth service: register, sign-in, sign-out logic.
"""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from .jwt_util import create_token
from .password import hash_password, validate_password_strength, verify_password
from .repository import UserRepository
from .schemas import RegisterResponse, SignInResponse, UserInfo

logger = logging.getLogger(__name__)

VALID_ROLES = frozenset({"general", "supervisor", "admin"})


def _get_password_min_length() -> int:
    """Get min password length from env."""
    return int(os.getenv("AUTH_PASSWORD_MIN_LENGTH", "8"))


class AuthService:
    """
    Handles user registration, sign-in, and sign-out.

    Validates passwords, checks uniqueness, issues JWT tokens.
    """

    def __init__(self, repository: UserRepository | None = None):
        """
        Initialize auth service.

        Args:
            repository: Optional UserRepository. If None, creates default.
        """
        self._repo = repository or UserRepository()

    def register(
        self,
        user_name: str,
        password: str,
        email: str | None = None,
    ) -> RegisterResponse:
        """
        Register a new user.

        Args:
            user_name: Display name (unique).
            password: Plaintext password.
            email: Optional email (unique if provided).

        Returns:
            RegisterResponse with user_id, user_name, role.

        Raises:
            ValueError: If validation fails or user_name/email already exists.
        """
        user_name = (user_name or "").strip()
        if not user_name:
            raise ValueError("user_name is required")

        # Validate password strength
        min_len = _get_password_min_length()
        valid, err = validate_password_strength(password, min_length=min_len)
        if not valid:
            raise ValueError(err)

        # Check user_name uniqueness
        existing = self._repo.get_by_user_name(user_name)
        if existing:
            raise ValueError("user_name already exists")

        # Check email uniqueness if provided
        email_clean = (email or "").strip()
        if email_clean:
            existing_email = self._repo.get_by_email(email_clean)
            if existing_email:
                raise ValueError("email already exists")

        user_id = uuid4()
        password_hash = hash_password(password)
        role = "general"

        self._repo.create(
            user_id=user_id,
            user_name=user_name,
            password_hash=password_hash,
            email=email_clean,
            role=role,
        )

        logger.info("User registered: user_name=%s user_id=%s", user_name, user_id)
        return RegisterResponse(
            user_id=str(user_id),
            user_name=user_name,
            role=role,
        )

    def sign_in(
        self,
        user_name: str,
        password: str,
        client_ip: str | None = None,
    ) -> SignInResponse:
        """
        Sign in user and return JWT + user info.

        Args:
            user_name: Display name.
            password: Plaintext password.
            client_ip: Optional client IP for last_login_ip.

        Returns:
            SignInResponse with access_token and user info.

        Raises:
            ValueError: If credentials invalid or user inactive/suspended.
        """
        user_name = (user_name or "").strip()
        if not user_name:
            raise ValueError("user_name is required")
        if not password:
            raise ValueError("password is required")

        user = self._repo.get_by_user_name(user_name)
        if not user:
            raise ValueError("Invalid user_name or password")

        status = (user.get("status") or "active").lower()
        if status != "active":
            raise ValueError("Account is not active")

        if not verify_password(password, user.get("password_hash", "")):
            raise ValueError("Invalid user_name or password")

        user_id = user.get("user_id")
        role = user.get("role") or "general"
        email = user.get("email") or None
        if email == "":
            email = None

        # Update last login
        try:
            from uuid import UUID

            self._repo.update_last_login(UUID(user_id), ip=client_ip)
        except Exception as e:
            logger.warning("update_last_login failed (non-fatal): %s", e)

        token = create_token(
            user_id=str(user_id),
            user_name=user_name,
            role=role,
        )

        logger.info("User signed in: user_name=%s", user_name)
        return SignInResponse(
            access_token=token,
            token_type="bearer",
            user=UserInfo(
                user_id=str(user_id),
                user_name=user_name,
                email=email,
                role=role,
                status=status,
            ),
        )

    def sign_out(self) -> None:
        """
        Sign out. Client-side token discard; no server-side blacklist for now.
        """
        pass
