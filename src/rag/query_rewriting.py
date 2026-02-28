"""
Query rewriting for RAG pipeline.

Layer 2: Lightweight term expansion and standardization to improve retrieval recall.
Reads RAG_QUERY_REWRITE_ENABLED, RAG_QUERY_REWRITE_MIN_LENGTH, RAG_QUERY_REWRITE_MAX_LENGTH.
"""

from __future__ import annotations

import os
import re


def rewrite_query_lightweight(question: str) -> str:
    """
    Lightweight query rewrite - term expansion and standardization.

    Expands short/ambiguous queries (e.g. "FBA fee", "cost") to improve retrieval recall.
    Reads RAG_QUERY_REWRITE_ENABLED, RAG_QUERY_REWRITE_MIN_LENGTH, RAG_QUERY_REWRITE_MAX_LENGTH
    from environment. Skips rewrite for long queries (> max_length words).

    Args:
        question: Raw user question.

    Returns:
        Rewritten question string.
    """
    if not question or not str(question).strip():
        return question

    enabled = os.getenv("RAG_QUERY_REWRITE_ENABLED", "true").lower() in ("true", "1", "yes")
    if not enabled:
        return question

    min_len = int(os.getenv("RAG_QUERY_REWRITE_MIN_LENGTH", "5"))
    max_len = int(os.getenv("RAG_QUERY_REWRITE_MAX_LENGTH", "10"))
    words = question.split()
    if len(words) > max_len:
        return question

    # Term expansion map (always applied when rewriting)
    # Longer phrases first to avoid double expansion (e.g. "FBA fee" before "fee")
    term_map = {
        "FBA fee": "Amazon FBA fee",
        "FBA": "Amazon FBA",
        "FBM": "Amazon FBM",
        "fee": "Amazon FBA fee",
        "cost": "fulfillment fee",
        "charge": "fulfillment fee",
    }
    # Standardization map (only for very short queries)
    std_map = {
        "cost": "fulfillment fee",
        "charge": "fulfillment fee",
        "price": "fulfillment fee",
    } if len(words) < min_len else {}

    combined = {**term_map, **std_map}
    rewritten = question
    # Use placeholders to avoid recursive expansion; process longer terms first
    placeholders: dict[str, str] = {}
    for i, (term, expansion) in enumerate(sorted(combined.items(), key=lambda x: -len(x[0]))):
        ph = f"__RAG_PH_{i}__"
        placeholders[ph] = expansion
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        rewritten = pattern.sub(ph, rewritten)
    for ph, expansion in placeholders.items():
        rewritten = rewritten.replace(ph, expansion)
    return rewritten


def rewrite_query(question: str) -> str:
    """
    Alias for rewrite_query_lightweight. Cleaner API for workflow integration.

    Args:
        question: Raw user question.

    Returns:
        Rewritten question string.
    """
    return rewrite_query_lightweight(question)
