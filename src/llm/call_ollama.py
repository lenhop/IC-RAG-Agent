"""
Unified Ollama HTTP client for Route LLM (generate + embed).

Configuration uses exactly four environment variables (no alternates):
  OLLAMA_BASE_URL       - API base, e.g. http://localhost:11434
  OLLAMA_GENERATE_MODEL - Model for /api/generate
  OLLAMA_REQUEST_TIMEOUT - Seconds for HTTP calls
  OLLAMA_EMBED_MODEL    - Model for /api/embed
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Single source of truth: these four keys only.
_ENV_BASE_URL = "OLLAMA_BASE_URL"
_ENV_GENERATE_MODEL = "OLLAMA_GENERATE_MODEL"
_ENV_TIMEOUT = "OLLAMA_REQUEST_TIMEOUT"
_ENV_EMBED_MODEL = "OLLAMA_EMBED_MODEL"


def _req(name: str) -> str:
    """Return stripped env value or raise ValueError with setup hint."""
    v = (os.getenv(name) or "").strip()
    if not v:
        raise ValueError(
            f"{name} is not set. Add to .env, e.g. {_ENV_BASE_URL}=http://localhost:11434 "
            f"and {_ENV_GENERATE_MODEL}=qwen3:1.7b, {_ENV_TIMEOUT}=120, {_ENV_EMBED_MODEL}=all-minilm:latest"
        )
    return v


@dataclass(frozen=True)
class OllamaRuntimeConfig:
    """Resolved Ollama settings from the four env vars."""

    generate_url: str
    base_url: str
    model: str
    timeout: int
    embed_model: str


def _ensure_generate_url(base_url: str) -> str:
    u = (base_url or "").strip().rstrip("/")
    if not u:
        raise ValueError(f"{_ENV_BASE_URL} is empty")
    return f"{u}/api/generate"


def get_ollama_config() -> OllamaRuntimeConfig:
    """
    Load Ollama settings from OLLAMA_BASE_URL, OLLAMA_GENERATE_MODEL,
    OLLAMA_REQUEST_TIMEOUT, OLLAMA_EMBED_MODEL.

    Raises:
        ValueError: If any variable is missing or timeout is not a positive int.
    """
    base_url = (os.getenv("OLLAMA_BASE_URL")).strip().rstrip("/")
    model = os.getenv("OLLAMA_GENERATE_MODEL")    
    embed_model = (os.getenv("OLLAMA_EMBED_MODEL")).strip()    

    if not model:
        raise ValueError("OLLAMA_GENERATE_MODEL is not set")
    if not base_url:
        raise ValueError("OLLAMA_BASE_URL is not set")
    if not embed_model:
        raise ValueError("OLLAMA_EMBED_MODEL is not set")

    generate_url = _ensure_generate_url(base_url)
    timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
    return OllamaRuntimeConfig(generate_url=generate_url, base_url=base_url, model=model, timeout=timeout, embed_model=embed_model)


class OllamaClient:
    """Shared Ollama /api/generate and /api/embed using the four env vars."""

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        raw = (text or "").strip()
        if not raw.startswith("```"):
            return raw
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return raw

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        generate_url: Optional[str] = None,
        empty_fallback: str = "",
    ) -> str:
        """
        POST /api/generate. Uses env model/url/timeout unless overrides passed.
        """
        cfg = get_ollama_config()
        url = generate_url or cfg.generate_url
        mdl = model or cfg.model
        timeout = timeout or cfg.timeout
        payload = {"model": mdl, "prompt": prompt or "", "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
            logger.error("Ollama generate failed (%s): %s", url, exc, exc_info=True)
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            logger.error("Ollama HTTP %s (%s): %s", resp.status_code, url, detail)
            raise RuntimeError(f"Ollama generate HTTP {resp.status_code}: {detail}")
        try:
            data = resp.json()
            out = (data.get("response") or "").strip()
            return out if out else empty_fallback
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.error("Ollama invalid JSON (%s): %s", url, exc, exc_info=True)
            raise RuntimeError(f"Ollama generate invalid response: {exc}") from exc

    def embed(
        self,
        inputs: List[str],
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> List[List[float]]:
        """
        POST /api/embed. Default model is OLLAMA_EMBED_MODEL.
        """
        cfg = get_ollama_config()
        base = (base_url or cfg.base_url).rstrip("/")
        url = f"{base}/api/embed"
        mdl = (model or "").strip() or cfg.embed_model
        to = timeout if timeout is not None else cfg.timeout
        try:
            resp = requests.post(
                url, json={"model": mdl, "input": inputs}, timeout=to
            )
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
            logger.error("Ollama embed failed (%s): %s", url, exc, exc_info=True)
            raise RuntimeError(f"Ollama embed failed: {exc}") from exc
        if resp.status_code != 200:
            resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(inputs):
            raise ValueError(
                f"Ollama embed: expected {len(inputs)} vectors, got {len(embeddings or [])}"
            )
        return embeddings
