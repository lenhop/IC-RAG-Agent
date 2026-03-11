"""
Prompt loader for externalized LLM prompts.

Reads .txt files from src/prompts/ directory and caches them in memory.
Usage:
    from src.gateway.prompt_loader import load_prompt
    prompt = load_prompt("clarification_detect_ambiguity")  # loads src/prompts/clarification_detect_ambiguity.txt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Points to src/prompts/ (one level up from src/gateway/, then into prompts/)
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_cache: Dict[str, str] = {}


def load_prompt(name: str) -> str:
    """
    Load a prompt by name (filename without .txt extension).

    Reads from src/prompts/{name}.txt, caches after first load.
    Raises FileNotFoundError if the prompt file does not exist.

    Args:
        name: Prompt name, e.g. "clarification", "rewrite", "route_classification".

    Returns:
        Prompt text content.
    """
    if name in _cache:
        return _cache[name]

    filepath = _PROMPTS_DIR / f"{name}.txt"
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
