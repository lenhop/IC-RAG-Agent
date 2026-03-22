"""
Shared text-generation backend for gateway merge, RAG amazon_business merge, and SP-API formatting.

Reads ``GATEWAY_TEXT_GENERATION_BACKEND`` (``deepseek`` | ``ollama``). The RAG service (port 8002)
must set the same env var as the gateway (8000) if you want consistent behavior across processes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

BackendName = Literal["deepseek", "ollama"]


def resolve_text_generation_backend() -> BackendName:
    """
    Resolve which LLM backs post-retrieval / merge / formatting steps.

    Env:
        GATEWAY_TEXT_GENERATION_BACKEND: ``deepseek`` or ``ollama`` (case-insensitive).

    Default:
        ``deepseek`` when ``DEEPSEEK_API_KEY`` is non-empty; otherwise ``ollama``.

    Returns:
        ``deepseek`` or ``ollama``.
    """
    raw = (os.getenv("GATEWAY_TEXT_GENERATION_BACKEND") or "").strip().lower()
    if raw in ("deepseek", "ds"):
        return "deepseek"
    if raw in ("ollama", "local"):
        return "ollama"
    if (os.getenv("DEEPSEEK_API_KEY") or "").strip():
        return "deepseek"
    return "ollama"


def complete_chat(
    backend: BackendName,
    system_prompt: str,
    user_content: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Run one system+user completion on the selected backend.

    Args:
        backend: ``deepseek`` or ``ollama``.
        system_prompt: System instructions.
        user_content: User payload.
        max_tokens: Completion cap (DeepSeek: API max_tokens; Ollama: num_predict).
        temperature: Sampling temperature.

    Returns:
        Stripped assistant text.

    Raises:
        RuntimeError: If the backend call fails after any DeepSeek-to-Ollama fallback.
    """
    sys_t = (system_prompt or "").strip()
    usr_t = (user_content or "").strip()
    if backend == "deepseek":
        try:
            from src.llm.call_deepseek import DeepSeekChat

            out = DeepSeekChat().complete(
                sys_t,
                usr_t,
                max_tokens=max(256, int(max_tokens)),
                temperature=float(temperature),
            )
            text = (out or "").strip()
            if text:
                return text
            raise RuntimeError("DeepSeek returned empty text")
        except ValueError as exc:
            logger.warning(
                "GATEWAY_TEXT_GENERATION_BACKEND=deepseek but DeepSeek unavailable (%s); "
                "falling back to Ollama.",
                exc,
            )
            backend = "ollama"

    if backend == "ollama":
        from src.llm.call_ollama import OllamaClient, get_ollama_config

        cfg = get_ollama_config()
        prompt = (
            f"### System\n{sys_t}\n\n### User\n{usr_t}\n\n"
            "### Assistant\n"
        )
        client = OllamaClient()
        url = cfg.generate_url
        payload: dict = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max(256, int(max_tokens)),
                "temperature": float(temperature),
            },
        }
        try:
            import requests

            resp = requests.post(url, json=payload, timeout=cfg.timeout)
        except Exception as exc:
            logger.error("Ollama chat-style generate failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {detail}")
        try:
            data = resp.json()
            out = (data.get("response") or "").strip()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama invalid JSON: {exc}") from exc
        if not out:
            raise RuntimeError("Ollama returned empty text")
        return OllamaClient.strip_markdown_fences(out)

    raise ValueError(f"unknown backend: {backend!r}")
