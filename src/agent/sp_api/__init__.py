"""
Amazon Selling Partner API read-only agent (Orders + Listings tools, ReAct, FastAPI).

Environment:
    SP_API_REFRESH_TOKEN, SP_API_CLIENT_ID, SP_API_CLIENT_SECRET (required for live calls)
    SP_API_SELLER_ID (required for getListingsItem)
    SP_API_MARKETPLACE_ID, SP_API_ENDPOINT (optional)
    SP_API_LLM_PROVIDER / SP_API_LLM_MODEL (agent reasoning)
    SP_API_TEST_MODE=true to skip Amazon and echo query (smoke tests)
"""

from .app import app
from .exceptions import SPAPIAuthError
from .sp_api_client import SPAPIClient, SPAPICredentials

__all__ = [
    "app",
    "SPAPIAuthError",
    "SPAPIClient",
    "SPAPICredentials",
]
