"""
Password hashing and verification using bcrypt.
"""

from __future__ import annotations

import re

import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        password: Plaintext password.

    Returns:
        Bcrypt hash string.
    """
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Args:
        plain_password: User-provided password.
        hashed_password: Stored hash from database.

    Returns:
        True if password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def validate_password_strength(
    password: str,
    min_length: int = 8,
) -> tuple[bool, str]:
    """
    Validate password meets strength requirements.

    Requirements: min length, at least one letter, at least one digit.

    Args:
        password: Password to validate.
        min_length: Minimum length (default 8).

    Returns:
        Tuple of (is_valid, error_message). error_message empty when valid.
    """
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters"
    if not re.search(r"[a-zA-Z]", password):
        return False, "Password must contain at least one letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    return True, ""
