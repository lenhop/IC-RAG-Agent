"""Gateway API package — HTTP entry-point, auth, config, view helpers."""

# No top-level imports to avoid pulling in api/auth when only message is needed (e.g. router).
# Use: from src.gateway.api.api import app
# Use: from src.gateway.api.auth import AuthGuard, router, get_auth_service
# Use: from src.gateway.api.config import GatewayConfig, GatewayEventLogger
# Use: from src.gateway.api.view_helpers import IntentDetailsBuilder, PlanHelper, DebugTraceBuilder
# Use: from src.gateway.message import ...
__all__: list[str] = []
