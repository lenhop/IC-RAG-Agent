"""
SP-API agent exceptions (no dependency on external ai_toolkit.errors).
"""

from __future__ import annotations


class SPAPIAuthError(Exception):
    """Raised when LWA OAuth2 token exchange fails or returns invalid payload."""

    def __init__(self, message: str, *, auth_type: str = "lwa_oauth2") -> None:
        super().__init__(message)
        self.auth_type = auth_type
