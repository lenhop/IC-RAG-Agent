"""
Prompt loader for externalized LLM prompts.

Reads .txt files from src/gateway/route_llm/ subdirectories (clarification, rewriting, classification).
Usage:
    from src.gateway.prompt_loader import load_prompt
    prompt = load_prompt("clarification/clarification_detect_ambiguity")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Base: src/gateway/route_llm/ (prompts live next to their consumers)
_PROMPTS_BASE = Path(__file__).parent / "route_llm"
_cache: Dict[str, str] = {}


def load_prompt(name: str) -> str:
    """
    Load a prompt by name (filename without .txt extension).

    Reads from src/gateway/route_llm/{name}.txt, caches after first load.
    Supports subdirectory paths, e.g.:
        "clarification/clarification_detect_ambiguity"
        "rewriting/rewrite_query_clean"
        "classification/intent_split_query"

    Raises FileNotFoundError if the prompt file does not exist.

    Args:
        name: Prompt path, e.g. "clarification/clarification_detect_ambiguity".

    Returns:
        Prompt text content.
    """
    if name in _cache:
        return _cache[name]

    filepath = _PROMPTS_BASE / f"{name}.txt"
    if not filepath.is_file():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {filepath}")

    _cache[name] = text
    logger.debug("Loaded prompt '%s' from %s (%d chars)", name, filepath, len(text))
    return text


def reload_prompt(name: str) -> str:
    """Force reload a prompt from disk (bypasses cache)."""
    _cache.pop(name, None)
    return load_prompt(name)


def clear_cache() -> None:
    """Clear all cached prompts."""
    _cache.clear()
