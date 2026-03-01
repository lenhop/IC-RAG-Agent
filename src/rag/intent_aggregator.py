"""
Intent signal aggregation for parallel four-strategy workflow.

All No -> mode from RAG_AGGREGATE_NO_MODE (general/documents/hybrid).
>=1 Yes -> documents or hybrid (from RAG_AGGREGATE_YES_MODE).
"""

from __future__ import annotations

import os
from typing import Literal

AnswerMode = Literal["documents", "general", "hybrid"]


def aggregate_intent_signals(
    signals: list[bool],
    yes_to_mode: str | None = None,
    no_to_mode: str | None = None,
) -> AnswerMode:
    """
    Aggregate Yes/No signals from four parallel methods.

    - All No -> mode from RAG_AGGREGATE_NO_MODE (general, documents, or hybrid)
    - At least one Yes -> documents or hybrid (from RAG_AGGREGATE_YES_MODE)

    Args:
        signals: List of bool (True=Yes, False=No) from Documents, Keywords, FAQ, LLM.
        yes_to_mode: Override for >=1 Yes. "documents" or "hybrid". Default from RAG_AGGREGATE_YES_MODE.
        no_to_mode: Override for All No. "general", "documents", or "hybrid". Default from RAG_AGGREGATE_NO_MODE.

    Returns:
        AnswerMode: documents, general, or hybrid.
    """
    if not signals:
        fallback = no_to_mode or os.getenv("RAG_AGGREGATE_NO_MODE", "general")
        return _normalize_mode(fallback)
    if not any(signals):
        fallback = no_to_mode or os.getenv("RAG_AGGREGATE_NO_MODE", "general")
        return _normalize_mode(fallback)
    mode = yes_to_mode or os.getenv("RAG_AGGREGATE_YES_MODE", "hybrid")
    if mode == "documents":
        return "documents"
    return "hybrid"


def _normalize_mode(mode: str) -> AnswerMode:
    """Ensure mode is valid AnswerMode; default to general if invalid."""
    m = (mode or "").lower().strip()
    if m in ("documents", "general", "hybrid"):
        return m  # type: ignore
    return "general"
