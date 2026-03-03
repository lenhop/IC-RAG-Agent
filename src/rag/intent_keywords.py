"""
Domain keyword and phrase loading for intent classification.

Loads keywords from THREE sources (when enabled):
  1. RAG_DOMAIN_KEYWORDS (env)
  2. RAG_TITLE_PHRASES_CSV (phrases_from_titles.csv)
  3. RAG_FAQ_CSV (amazon_fqa.csv) - optional, controlled by RAG_FAQ_KEYWORDS_ENABLED

Provides match_domain_signals() for use by the answer-mode classifier.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Set

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Minimal English stopwords for FAQ keyword extraction (no NLTK dependency)
_FAQ_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "if", "then", "else", "when", "where", "how",
    "what", "why", "who", "which", "your", "my", "our", "their",
}


def _load_domain_keywords() -> List[str]:
    """Load RAG_DOMAIN_KEYWORDS from env as a lowercase list."""
    raw = os.getenv("RAG_DOMAIN_KEYWORDS", "FBA,FBM,Amazon,eBay,inventory,policy")
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


def _tokenize_faq_text(text: str, stopwords: Set[str]) -> List[str]:
    """
    Tokenize text into words (supports English and CJK).

    Args:
        text: Raw text string (e.g. FAQ question).
        stopwords: Set of stopwords to filter.

    Returns:
        List of valid tokens (lowercased, non-stopword, length >= 2).
    """
    tokens = re.findall(r"[a-zA-Z0-9]{2,}|[\u4e00-\u9fff]+", text)
    return [t.lower() if t.isascii() else t for t in tokens if t.lower() not in stopwords]


def _extract_phrases_from_faq_questions(
    questions: List[str],
    top_n: int = 50,
    ngram_range: tuple = (1, 4),
) -> List[str]:
    """
    Extract keywords/phrases from FAQ questions using n-gram frequency.

    Includes single words (1-gram) and multi-word phrases (2-4 grams) for
    domain term coverage (e.g. "FBA", "amazon") and phrases (e.g. "fba fee").

    Args:
        questions: List of FAQ question strings.
        top_n: Max number of phrases to return.
        ngram_range: (min_n, max_n) for phrase length in words.

    Returns:
        List of phrase strings, sorted by frequency (desc).
    """
    counter: dict[str, int] = {}
    min_n, max_n = ngram_range

    for q in questions:
        if not q or not str(q).strip():
            continue
        tokens = _tokenize_faq_text(q, _FAQ_STOPWORDS)
        if len(tokens) < min_n:
            continue
        for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                counter[phrase] = counter.get(phrase, 0) + 1

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [p for p, _ in sorted_items[:top_n]]


def _load_faq_keywords(project_root: Path | None = None) -> List[str]:
    """
    Extract domain keywords from FAQ questions when RAG_FAQ_KEYWORDS_ENABLED.

    Uses question column from RAG_FAQ_CSV. Returns empty list if disabled,
    file missing, or parse error.

    Returns:
        List of lowercase keywords/phrases extracted from FAQ questions.
    """
    if os.getenv("RAG_FAQ_KEYWORDS_ENABLED", "false").lower() not in ("true", "1", "yes"):
        return []

    from src.rag.chroma_loaders import load_faq_questions

    root = project_root or PROJECT_ROOT
    questions = load_faq_questions(root)
    if not questions:
        return []

    top_n = int(os.getenv("RAG_FAQ_KEYWORDS_TOP_N", "50"))
    return _extract_phrases_from_faq_questions(questions, top_n=top_n)


def get_domain_signals(project_root: Path | None = None) -> List[str]:
    """
    Return combined list of domain keywords from THREE sources (lowercase).

    Sources:
      1. RAG_DOMAIN_KEYWORDS (env)
      2. RAG_TITLE_PHRASES_CSV (phrases_from_titles.csv)
      3. RAG_FAQ_CSV (amazon_fqa.csv) - when RAG_FAQ_KEYWORDS_ENABLED=true

    Cached at module level for efficiency; call _invalidate_domain_signals_cache()
    to force reload (e.g. in tests).
    """
    if get_domain_signals._cache is not None:
        return get_domain_signals._cache
    keywords = _load_domain_keywords()
    phrases = _load_title_phrases(project_root)
    faq_keywords = _load_faq_keywords(project_root)
    # Deduplicate while preserving order (sources may overlap)
    seen: set[str] = set()
    combined: List[str] = []
    for item in keywords + phrases + faq_keywords:
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
    raw = os.getenv("RAG_GENERAL_PREFIXES", "what is,define")
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
