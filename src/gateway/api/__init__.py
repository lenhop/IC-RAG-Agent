"""Gateway API and auth package."""

# No top-level imports to avoid pulling in api/auth when only message is needed (e.g. router).
# Use: from src.gateway.api_and_auth.api import app
# Use: from src.gateway.api_and_auth.auth import AuthGuard, router, get_auth_service
# Use: from src.gateway.api_and_auth.config import GatewayConfig, GatewayEventLogger
# Use: from src.gateway.api_and_auth.view_helpers import IntentDetailsBuilder, PlanHelper, DebugTraceBuilder
# Use: from src.gateway.message import ...
__all__: list[str] = []
