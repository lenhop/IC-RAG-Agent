"""
Gateway worker deployment profile: optionally stub UDS / SP-API for RAG-only rollout.

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
    Return True when the gateway is configured for RAG-only workers.

    Env:
        GATEWAY_WORKER_PROFILE: set to ``rag_only``, ``rag-only``, or ``rag``.

    Returns:
        True if profile matches (case-insensitive).
    """
    profile = (os.getenv("GATEWAY_WORKER_PROFILE") or "").strip().lower()
    return profile in ("rag_only", "rag-only", "rag")


def should_stub_uds_and_sp_api() -> bool:
    """
    Return True if UDS and SP-API HTTP calls should be skipped in favor of stubs.

    Enabled when either:
        - ``GATEWAY_WORKER_PROFILE`` is RAG-only (see ``is_rag_only_worker_profile``), or
        - ``GATEWAY_STUB_UDS_SP_API`` is truthy (1/true/yes/on).

    Returns:
        Whether to short-circuit UDS/SP-API clients.
    """
    if is_rag_only_worker_profile():
        return True
    return _env_truthy("GATEWAY_STUB_UDS_SP_API")


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
