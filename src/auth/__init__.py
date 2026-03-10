"""
Auth module for IC-RAG-Agent.

Provides user registration, sign-in, sign-out with ClickHouse storage and JWT tokens.
"""

from .schemas import (
    RegisterRequest,
    RegisterResponse,
    SignInRequest,
    SignInResponse,
    UserInfo,
)
from .service import AuthService
from .repository import UserRepository

__all__ = [
    "AuthService",
    "UserRepository",
    "RegisterRequest",
    "RegisterResponse",
    "SignInRequest",
    "SignInResponse",
    "UserInfo",
]
