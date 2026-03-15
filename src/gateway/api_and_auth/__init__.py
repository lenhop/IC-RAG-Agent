"""Gateway API and auth package."""

# No top-level imports to avoid pulling in api/auth_routes when only message is needed (e.g. router).
# Use: from src.gateway.api_and_auth.api import app
# Use: from src.gateway.api_and_auth.auth_routes import router, get_auth_service
# Use: from src.gateway.api_and_auth.message import ...
__all__: list[str] = []
