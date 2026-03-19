"""
Basic tests for src.retrieval.keyword_retrieval (KeywordRetrieval).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.keyword_retrieval import KeywordMatchResult, KeywordRetrieval


def test_keyword_retrieval_empty_returns_none() -> None:
    """Empty query returns None."""
    matcher = KeywordRetrieval()
    assert matcher.match("") is None
    assert matcher.match("   ") is None


def test_keyword_retrieval_match_returns_result() -> None:
    """Query containing pattern returns KeywordMatchResult."""
    matcher = KeywordRetrieval()
    # Default rules: order ID pattern or "latest" -> sp_api; "order"/"inventory" -> uds
    r = matcher.match("Get the latest status of order 112-1234567-8901234")
    assert r is not None
    assert isinstance(r, KeywordMatchResult)
    assert r.workflow in ("sp_api", "uds")
    assert r.source == "keyword"


def test_keyword_retrieval_uds_style() -> None:
    """UDS-style query matches uds workflow."""
    matcher = KeywordRetrieval()
    r = matcher.match("Show my inventory")
    assert r is not None
    assert r.workflow == "uds"
    assert r.intent_name == "inventory"
