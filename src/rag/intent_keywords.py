"""
Domain keyword and phrase loading for intent classification.

Loads RAG_DOMAIN_KEYWORDS from env and optionally phrases from RAG_TITLE_PHRASES_CSV.
Provides match_domain_signals() for use by the answer-mode classifier.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_domain_keywords() -> List[str]:
    """Load RAG_DOMAIN_KEYWORDS from env as a lowercase list."""
    raw = os.getenv("RAG_DOMAIN_KEYWORDS", "FBA,FBM,Amazon,eBay,库存,政策")
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


def _load_title_phrases(project_root: Path | None = None) -> List[str]:
    """
    Load phrases from RAG_TITLE_PHRASES_CSV if file exists.

    Returns empty list if path not set, file missing, or parse error.
    """
    root = project_root or PROJECT_ROOT
    path_val = os.getenv("RAG_TITLE_PHRASES_CSV", "")
    if not path_val:
        return []

    path = Path(path_val)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        return []

    phrases: List[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phrase = (row.get("phrase") or "").strip()
                if phrase:
                    phrases.append(phrase.lower())
    except (csv.Error, OSError):
        pass
    return phrases


def get_domain_signals(project_root: Path | None = None) -> List[str]:
    """
    Return combined list of domain keywords and title phrases (lowercase).

    Cached at module level for efficiency; call _invalidate_domain_signals_cache()
    to force reload (e.g. in tests).
    """
    if get_domain_signals._cache is not None:
        return get_domain_signals._cache
    keywords = _load_domain_keywords()
    phrases = _load_title_phrases(project_root)
    # Deduplicate while preserving order (phrases may overlap keywords)
    seen: set[str] = set()
    combined: List[str] = []
    for item in keywords + phrases:
        if item and item not in seen:
            seen.add(item)
            combined.append(item)
    # Sort by length descending so longer phrases match first
    combined.sort(key=len, reverse=True)
    get_domain_signals._cache = combined
    return combined


get_domain_signals._cache: List[str] | None = None


def _invalidate_domain_signals_cache() -> None:
    """Clear cache (for tests or config reload)."""
    get_domain_signals._cache = None


def get_general_prefixes() -> List[str]:
    """Load RAG_GENERAL_PREFIXES from env as a lowercase list."""
    raw = os.getenv("RAG_GENERAL_PREFIXES", "what is,define,什么是")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def match_domain_signals(question: str, project_root: Path | None = None) -> List[str]:
    """
    Return list of domain keywords/phrases that appear in the question (case-insensitive).

    Args:
        question: User question string.
        project_root: Optional project root for resolving CSV path.

    Returns:
        List of matched signals (for interpretability and debugging).
    """
    if not question or not str(question).strip():
        return []

    normalized = question.lower()
    signals = get_domain_signals(project_root)
    matches: List[str] = []
    for s in signals:
        if s and s in normalized:
            matches.append(s)
    return matches
