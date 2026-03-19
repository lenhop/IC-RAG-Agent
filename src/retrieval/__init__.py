"""
Intent retrieval package: Keyword retrieval and Vector retrieval.

Public API:
  - KeywordRetrieval: keyword/regex-based intent matching (Layer 1).
  - VectorRetrieval: Chroma-based similarity search over intent_registry (Layer 2).
"""

from __future__ import annotations

from src.retrieval.keyword_retrieval import KeywordMatchResult, KeywordRetrieval
from src.retrieval.vector_retrieval import VectorCandidate, VectorRetrieval

__all__ = [
    "KeywordMatchResult",
    "KeywordRetrieval",
    "VectorCandidate",
    "VectorRetrieval",
]
