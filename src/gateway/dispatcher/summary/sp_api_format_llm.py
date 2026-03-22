"""
Strict LLM pass to format SP-API worker text for chat (anti-hallucination).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from src.llm.text_generation_backend import complete_chat, resolve_text_generation_backend

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TEMPERATURE = 0.1


def _load_system_prompt() -> str:
    path = Path(__file__).resolve().parent / "sp_api_format_prompt.md"
    if not path.is_file():
        raise FileNotFoundError(f"sp_api_format_prompt.md missing at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("sp_api_format_prompt.md is empty")
    return text


def format_sp_api_worker_answer(
    worker_answer: str,
    *,
    user_sub_query: str = "",
    backend: Optional[str] = None,
) -> str:
    """
    Format SP-API tool/agent output using the configured text-generation backend.

    Args:
        worker_answer: Raw string from the sp_api worker (e.g. YAML in a code fence).
        user_sub_query: Optional task query for language/context hints.
        backend: ``deepseek`` or ``ollama``; default from env resolution.

    Returns:
        Formatted user-facing text.

    Raises:
        FileNotFoundError: If prompt file is missing.
        RuntimeError: If the LLM returns empty text.
    """
    raw = (worker_answer or "").strip()
    if not raw:
        raise ValueError("worker_answer must be non-empty")

    system_prompt = _load_system_prompt()
    sub = (user_sub_query or "").strip()
    user_content = (
        f"User sub-query (context only):\n{sub or '(none)'}\n\n"
        f"=== SP-API worker payload (authoritative) ===\n{raw}\n"
    )

    be = (backend or resolve_text_generation_backend()).strip().lower()
    if be not in ("deepseek", "ollama"):
        be = resolve_text_generation_backend()

    max_tokens = _DEFAULT_MAX_TOKENS
    raw_mt = (os.getenv("GATEWAY_SP_API_FORMAT_MAX_TOKENS") or "").strip()
    if raw_mt:
        try:
            max_tokens = max(256, int(raw_mt, 10))
        except ValueError:
            logger.warning("Invalid GATEWAY_SP_API_FORMAT_MAX_TOKENS=%r", raw_mt)

    started = time.perf_counter()
    out = complete_chat(
        be,  # type: ignore[arg-type]
        system_prompt,
        user_content,
        max_tokens=max_tokens,
        temperature=_DEFAULT_TEMPERATURE,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("SP-API format LLM completed in %sms backend=%s", elapsed_ms, be)
    text = (out or "").strip()
    if not text:
        raise RuntimeError("SP-API format LLM returned empty text")
    return text
