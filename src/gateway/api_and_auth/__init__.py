"""Gateway API and auth package."""

from .api import app
from .auth_routes import get_auth_service, router

__all__ = ["app", "router", "get_auth_service"]
