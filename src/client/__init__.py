"""
Client module for unified chat gateway.

Exports GatewayClient and query_sync for calling the gateway API.
"""

from .api_client import GatewayClient, GatewayClientError, VALID_WORKFLOWS


def query_sync(
    query: str,
    workflow: str = "auto",
    rewrite_enable: bool = True,
    session_id: str | None = None,
    base_url: str | None = None,
    timeout: int = 120,
) -> dict:
    """
    Convenience function to run a synchronous query against the gateway.

    Uses GATEWAY_API_URL from env if base_url is not provided.

    Args:
        query: User query string.
        workflow: Workflow selector (auto|general|amazon_docs|ic_docs|sp_api|uds).
        rewrite_enable: Whether to enable query rewriting.
        session_id: Optional session ID for multi-turn.
        base_url: Optional override for gateway URL.
        timeout: Request timeout in seconds.

    Returns:
        Response dict from gateway (or mock).
    """
    client = GatewayClient(base_url=base_url, timeout=timeout)
    return client.query_sync(
        query=query,
        workflow=workflow,
        rewrite_enable=rewrite_enable,
        session_id=session_id,
    )


# Alias: query = query_sync for API consistency
query = query_sync

__all__ = ["GatewayClient", "GatewayClientError", "query_sync", "query", "VALID_WORKFLOWS"]
