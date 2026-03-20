"""
Basic tests for src.retrieval.query_process (QueryProcessor).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_process import QueryProcessor, normalize_query


def test_normalize_trim_and_collapse() -> None:
    """normalize trims and collapses whitespace."""
    assert QueryProcessor.normalize("  hello   world  ") == "hello world"
    assert QueryProcessor.normalize("a\n\tb") == "a b"
    assert QueryProcessor.normalize(None) == ""
    assert QueryProcessor.normalize("") == ""


def test_normalize_query_alias() -> None:
    """normalize_query is alias for QueryProcessor.normalize."""
    assert normalize_query("  x  y  ") == "x y"


def test_collapse_to_single_line() -> None:
    """collapse_to_single_line merges newlines to space."""
    assert QueryProcessor.collapse_to_single_line("a\nb\nc") == "a b c"
    assert QueryProcessor.collapse_to_single_line("  \n  x  \n  ") == "x"
    assert QueryProcessor.collapse_to_single_line(None) == ""
    assert QueryProcessor.collapse_to_single_line("") == ""


def test_apply_typo_and_filler_fixes() -> None:
    """Typo and filler fixes applied correctly."""
    assert "what" in QueryProcessor.apply_typo_and_filler_fixes("wat is this")
    assert "inventory" in QueryProcessor.apply_typo_and_filler_fixes("check invetory")
    s = QueryProcessor.apply_typo_and_filler_fixes("hey, show me inventory thx")
    assert s.startswith("show") and "inventory" in s
    assert QueryProcessor.apply_typo_and_filler_fixes(None) == ""
    # Whitespace-only: returns text or "" (original behavior)
    assert QueryProcessor.apply_typo_and_filler_fixes("   ") == "   "


def test_strip_echoed_context() -> None:
    """strip_echoed_context returns fallback when LLM echo detected."""
    fallback = "original query"
    assert QueryProcessor.strip_echoed_context("user: hello", fallback) == fallback
    assert QueryProcessor.strip_echoed_context("normalize: completed", fallback) == fallback
    assert QueryProcessor.strip_echoed_context("clean output", fallback) == "clean output"
    assert QueryProcessor.strip_echoed_context(None, fallback) == fallback
    assert QueryProcessor.strip_echoed_context("", fallback) == fallback


def test_clean_for_retrieval() -> None:
    """clean_for_retrieval runs full pipeline."""
    s = QueryProcessor.clean_for_retrieval("  hey  wat  is  invetory  thx  ")
    assert "what" in s and "inventory" in s
    assert s == s.lower()
    assert QueryProcessor.clean_for_retrieval(None) == ""
