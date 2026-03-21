"""
Backward-compatible re-exports for gateway outbound HTTP clients.

Implementations live in ``dispatcher.clients``; import from here or from
``clients`` directly.
"""

from __future__ import annotations

from .clients import (
    IC_DOCS_NOT_READY_MESSAGE,
    RAG_API_URL,
    SP_API_URL,
    UDS_API_URL,
    UDS_BACKEND_TIMEOUT,
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)
from .clients.http_client import BackendHttpClient

# Shared timeout default (some callers may expect module-level symbol)
BACKEND_TIMEOUT = BackendHttpClient.default_timeout_seconds()

__all__ = [
    "BACKEND_TIMEOUT",
    "IC_DOCS_NOT_READY_MESSAGE",
    "RAG_API_URL",
    "SP_API_URL",
    "UDS_API_URL",
    "UDS_BACKEND_TIMEOUT",
    "call_amazon_docs",
    "call_general",
    "call_ic_docs",
    "call_sp_api",
    "call_uds",
]
