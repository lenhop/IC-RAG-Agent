"""
Intent signal aggregation for parallel four-strategy workflow.

All No -> general; >=1 Yes -> documents or hybrid (from RAG_AGGREGATE_YES_MODE).
"""

from __future__ import annotations

import os
from typing import Literal

AnswerMode = Literal["documents", "general", "hybrid"]


def aggregate_intent_signals(
    signals: list[bool],
    yes_to_mode: str | None = None,
) -> AnswerMode:
    """
    Aggregate Yes/No signals from four parallel methods.

    - All No -> general
    - At least one Yes -> documents or hybrid (from config)

    Args:
        signals: List of bool (True=Yes, False=No) from Documents, Keywords, FAQ, LLM.
        yes_to_mode: Override. "documents" or "hybrid". Default from RAG_AGGREGATE_YES_MODE.

    Returns:
        AnswerMode: documents, general, or hybrid.
    """
    if not signals:
        return "general"
    if not any(signals):
        return "general"
    mode = yes_to_mode or os.getenv("RAG_AGGREGATE_YES_MODE", "hybrid")
    if mode == "documents":
        return "documents"
    return "hybrid"
