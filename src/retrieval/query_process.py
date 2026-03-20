"""
Query preprocessing for retrieval pipeline.

Responsibilities:
  - Normalize whitespace (strip, collapse)
  - Collapse multi-line text to single line
  - Apply typo and filler-word fixes
  - Strip LLM-echoed context from rewrite output

Used by: keyword retrieval, vector retrieval, gateway rewrite flow.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueryProcessor — Unified query cleaning (grouped by business flow)
# ---------------------------------------------------------------------------


class QueryProcessor:
    """Unified query cleaning for retrieval and rewrite pipelines.

    Methods are grouped by responsibility:
      - Basic: normalize, collapse_to_single_line
      - Typo/filler: apply_typo_and_filler_fixes
      - Rewrite post-processing: strip_echoed_context
    """

    # Common typo corrections (pattern, replacement)
    _TYPO_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bwat\b", "what"),
        (r"\binvetory\b", "inventory"),
        (r"\binvnetory\b", "inventory"),
        (r"\btehm\b", "them"),
        (r"\bpls\b", "please"),
        (r"\bthx\b", "thanks"),
        (r"\bu\b", "you"),
        (r"\bur\b", "your"),
    ]

    # Patterns indicating LLM echoed context (rewrite output contamination)
    _ECHO_PATTERNS: tuple[str, ...] = (
        "normalize: completed",
        "integrate short-term memory",
        "rewrite backend",
        "rewrite time",
        "intent classification",
        "rewrite-only test mode",
        "user:",
        "assistant:",
    )

    @staticmethod
    def normalize(text: Optional[str]) -> str:
        """Trim and collapse whitespace. Always applied before any routing."""
        return re.sub(r"\s+", " ", (text or "").strip())

    @staticmethod
    def collapse_to_single_line(text: Optional[str]) -> str:
        """Collapse multi-line text to a single line (newlines -> space)."""
        if not text or not text.strip():
            return text or ""
        line = re.sub(r"\s*\n+\s*", " ", text.strip())
        line = re.sub(r"\s+", " ", line).strip()
        return line if line else text.strip()

    @classmethod
    def apply_typo_and_filler_fixes(cls, text: Optional[str]) -> str:
        """Apply typo corrections, punctuation normalization, filler removal, lowercase."""
        if not text or not text.strip():
            return text or ""
        s = text.strip()
        # Collapse repeated punctuation
        s = re.sub(r"\?+", "?", s)
        s = re.sub(r"!+", "!", s)
        # Typo corrections
        for pattern, repl in cls._TYPO_REPLACEMENTS:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        # Remove leading "hey" and trailing "thx/thanks"
        s = re.sub(r"^\s*hey\s*,?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*(thx|thanks)\s*[.!?]*\s*$", "", s, flags=re.IGNORECASE).strip()
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else text.strip()

    @classmethod
    def strip_echoed_context(cls, response: Optional[str], fallback_query: str) -> str:
        """Detect LLM-echoed context in rewrite output; return fallback if contaminated."""
        if not response or not response.strip():
            return fallback_query
        r_lower = response.strip().lower()
        if any(p in r_lower for p in cls._ECHO_PATTERNS):
            logger.debug("Rewrite output contains echoed context; using fallback query")
            return fallback_query
        return response.strip()

    @classmethod
    def clean_for_retrieval(cls, text: Optional[str]) -> str:
        """Full pipeline for query before keyword/vector retrieval: normalize + collapse + typo fixes."""
        s = cls.normalize(text)
        s = cls.collapse_to_single_line(s)
        s = cls.apply_typo_and_filler_fixes(s)
        return s


# ---------------------------------------------------------------------------
# Convenience functions (backward compatibility)
# ---------------------------------------------------------------------------


def normalize_query(text: Optional[str]) -> str:
    """Trim and collapse whitespace. Alias for QueryProcessor.normalize."""
    return QueryProcessor.normalize(text)
