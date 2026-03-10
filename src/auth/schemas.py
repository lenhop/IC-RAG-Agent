"""
Auth request/response schemas.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    user_name: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)
    email: Optional[str] = Field(None, max_length=256)


class RegisterResponse(BaseModel):
    """Response after successful registration."""

    user_id: str
    user_name: str
    role: str
    message: str = "Registration successful"


class SignInRequest(BaseModel):
    """Request body for sign-in."""

    user_name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserInfo(BaseModel):
    """User info returned in auth responses."""

    user_id: str
    user_name: str
    email: Optional[str] = None
    role: str
    status: str


class SignInResponse(BaseModel):
    """Response after successful sign-in."""

    access_token: str
    token_type: str = "bearer"
    user: UserInfo
