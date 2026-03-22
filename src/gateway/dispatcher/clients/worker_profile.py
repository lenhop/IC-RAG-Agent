"""
Gateway worker deployment profile: optionally stub UDS and/or SP-API.

Profiles:
    - ``rag_only``: stub both UDS and SP-API (no HTTP to 8001/8003).
    - ``rag_sp_api``: stub UDS only; SP-API client calls ``SP_API_URL`` (used by
      ``--dispatcher-rag-only`` in ``project_stack.sh``).

When stubbing, handlers return a **successful** envelope ``{"answer", "sources"}`` so
``DispatcherExecutor`` marks tasks ``completed`` and ``rule_merge`` includes the text.
(Errors would yield ``failed`` and be dropped from merged answers.)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# User-visible copy for RAG-only phase (Chinese product message).
STUB_UDS_ANSWER = (
    "数据分析（UDS）能力尚未在本环境开放；当前仅提供 RAG 问答（通用知识 / Amazon 文档等）。"
    "如需指标分析，请稍后再试或联系管理员。"
)

STUB_SP_API_ANSWER = (
    "卖家运营（SP-API）能力尚未在本环境开放；当前仅提供 RAG 问答（通用知识 / Amazon 文档等）。"
    "如需订单、库存等操作，请稍后再试或联系管理员。"
)


def _env_truthy(name: str) -> bool:
    """Return True when env var is a common affirmative string."""
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_rag_only_worker_profile() -> bool:
    """
    Return True when the gateway is configured for RAG-only workers (UDS and SP-API stubbed).

    Env:
        GATEWAY_WORKER_PROFILE: set to ``rag_only``, ``rag-only``, or ``rag``.

    Returns:
        True if profile matches (case-insensitive).
    """
    profile = (os.getenv("GATEWAY_WORKER_PROFILE") or "").strip().lower()
    return profile in ("rag_only", "rag-only", "rag")


def is_rag_sp_api_worker_profile() -> bool:
    """
    Return True for the lightweight stack: real RAG + real SP-API; UDS stubbed.

    Used by ``--dispatcher-rag-only`` in ``project_stack.sh``: starts rag (8002),
    sp_api (8003), and gateway (8000) without running UDS (8001).

    Env:
        GATEWAY_WORKER_PROFILE: ``rag_sp_api``, ``rag-sp-api``, ``dispatcher-rag-sp-api``,
        or ``dispatcher_rag_sp_api`` (case-insensitive).

    Returns:
        True if profile matches.
    """
    profile = (os.getenv("GATEWAY_WORKER_PROFILE") or "").strip().lower()
    return profile in (
        "rag_sp_api",
        "rag-sp-api",
        "dispatcher-rag-sp-api",
        "dispatcher_rag_sp_api",
    )


def should_stub_uds() -> bool:
    """
    Return True if UDS HTTP calls should be stubbed.

    Stub UDS when:
        - ``GATEWAY_STUB_UDS_SP_API`` is truthy (stubs both workers), or
        - ``GATEWAY_WORKER_PROFILE`` is ``rag_only`` / ``rag_sp_api`` (or aliases).

    Returns:
        Whether to short-circuit the UDS client.
    """
    if _env_truthy("GATEWAY_STUB_UDS_SP_API"):
        return True
    if is_rag_only_worker_profile():
        return True
    if is_rag_sp_api_worker_profile():
        return True
    return False


def should_stub_sp_api() -> bool:
    """
    Return True if SP-API HTTP calls should be stubbed.

    Stub SP-API when:
        - ``GATEWAY_STUB_UDS_SP_API`` is truthy, or
        - ``GATEWAY_WORKER_PROFILE`` is ``rag_only`` (RAG-only; no live seller agent).

    Does not stub when profile is ``rag_sp_api`` (live SP-API worker on 8003).

    Returns:
        Whether to short-circuit the SP-API client.
    """
    if _env_truthy("GATEWAY_STUB_UDS_SP_API"):
        return True
    if is_rag_only_worker_profile():
        return True
    return False


def should_stub_uds_and_sp_api() -> bool:
    """
    Return True when **both** UDS and SP-API would use stubs (legacy combined check).

    Prefer ``should_stub_uds()`` and ``should_stub_sp_api()`` for new code.

    Returns:
        True if both backends are stubbed under current env.
    """
    return should_stub_uds() and should_stub_sp_api()


def stub_response_for_workflow(workflow: str) -> Dict[str, Any]:
    """
    Build a completed-task payload for the given workflow key.

    Args:
        workflow: Normalized workflow name (``uds`` or ``sp_api``).

    Returns:
        Dict with ``answer`` and empty ``sources``.

    Raises:
        ValueError: If workflow is not stubbed by this module.
    """
    key = (workflow or "").strip().lower()
    if key == "uds":
        text = STUB_UDS_ANSWER
    elif key == "sp_api":
        text = STUB_SP_API_ANSWER
    else:
        raise ValueError(f"no stub defined for workflow {workflow!r}")

    logger.info("Worker stub active for workflow=%s (RAG-only profile)", key)
    return {"answer": text, "sources": []}
