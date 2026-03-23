"""
Unified resolution of chat LLM backend labels (deepseek vs ollama) per gateway stage.

Embedding and vector backends stay on OLLAMA_* / INTENT_REGISTRY_EMBED_BACKEND; this module
only covers chat completion stages (rewrite, clarification, intent_detect, route, text_generation).

Precedence for each stage:
  1. ``rewrite`` only: non-empty request ``rewrite_backend`` after normalization (ollama|deepseek).
  2. Stage-specific env (e.g. GATEWAY_REWRITE_BACKEND) when set and valid.
  3. ``GATEWAY_CHAT_LLM_BACKEND`` when set and valid.
  4. Default ``deepseek``.

Invalid env values are logged and ignored so callers fall through to the next tier.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

# Global default chain (new): used when stage-specific env is unset or invalid.
ENV_GATEWAY_CHAT_LLM_BACKEND = "GATEWAY_CHAT_LLM_BACKEND"

# Per-stage env keys (existing names, backward compatible).
_STAGE_ENV_KEYS: Dict[str, str] = {
    "rewrite": "GATEWAY_REWRITE_BACKEND",
    "clarification": "GATEWAY_CLARIFICATION_BACKEND",
    "intent_detect": "GATEWAY_INTENT_DETECT_BACKEND",
    "route": "GATEWAY_ROUTE_BACKEND",
    "text_generation": "GATEWAY_TEXT_GENERATION_BACKEND",
}

ChatBackendStage = Literal[
    "rewrite",
    "clarification",
    "intent_detect",
    "route",
    "text_generation",
]

ChatBackendLabel = Literal["deepseek", "ollama"]


def _normalize_chat_backend_label(raw: Optional[str]) -> Optional[ChatBackendLabel]:
    """
    Map env or request strings to a canonical backend label.

    Args:
        raw: Raw string from env or client (may be empty or unknown).

    Returns:
        ``deepseek``, ``ollama``, or None when empty/whitespace or unrecognized.
    """
    try:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if not s:
            return None
        if s in ("deepseek", "ds"):
            return "deepseek"
        if s in ("ollama", "local"):
            return "ollama"
        logger.warning("Invalid chat backend label %r; ignoring", raw)
        return None
    except (TypeError, ValueError) as exc:
        logger.warning("Chat backend label normalize failed for %r: %s", raw, exc)
        return None


def resolve_chat_backend(
    stage: ChatBackendStage,
    *,
    request_override: Optional[str] = None,
) -> ChatBackendLabel:
    """
    Resolve which chat backend to use for a logical gateway stage.

    Args:
        stage: Logical stage name.
        request_override: For ``rewrite`` only: optional client override (same allowed values
            as ``QueryRequest.rewrite_backend`` after validation; extra aliases ds/local accepted).

    Returns:
        ``deepseek`` or ``ollama`` (never None).
    """
    try:
        if stage not in _STAGE_ENV_KEYS:
            logger.error("Unknown chat backend stage %r; defaulting to deepseek", stage)
            return "deepseek"

        # 1) Request override applies only to rewrite.
        if stage == "rewrite":
            ro = _normalize_chat_backend_label(
                (request_override or "").strip() or None
            )
            if ro is not None:
                return ro

        # 2) Stage-specific env.
        stage_key = _STAGE_ENV_KEYS[stage]
        stage_raw = os.getenv(stage_key)
        stage_norm = _normalize_chat_backend_label(stage_raw)
        if stage_norm is not None:
            return stage_norm

        # 3) Global chat default env.
        global_norm = _normalize_chat_backend_label(os.getenv(ENV_GATEWAY_CHAT_LLM_BACKEND))
        if global_norm is not None:
            return global_norm

        # 4) Final fallback.
        return "deepseek"
    except Exception as exc:
        logger.error("resolve_chat_backend failed for stage=%r: %s", stage, exc, exc_info=True)
        return "deepseek"


def effective_backends_snapshot() -> Dict[str, Any]:
    """
    Build a JSON-serializable snapshot of effective chat backends (no secrets).

    Used by gateway hints and operator dashboards so clients stay aligned with server policy
    without duplicating env precedence logic.

    Returns:
        Dict with raw global env (or null), effective global default label, per-stage labels,
        and whether DEEPSEEK_API_KEY is non-empty.
    """
    try:
        stages: tuple[ChatBackendStage, ...] = (
            "rewrite",
            "clarification",
            "intent_detect",
            "route",
            "text_generation",
        )
        per_stage: Dict[str, str] = {s: resolve_chat_backend(s) for s in stages}
        raw_global = (os.getenv(ENV_GATEWAY_CHAT_LLM_BACKEND) or "").strip()
        norm_global = _normalize_chat_backend_label(raw_global if raw_global else None)
        effective_global = norm_global or "deepseek"
        key_set = bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())
        return {
            "gateway_chat_llm_backend_env": raw_global or None,
            "gateway_chat_llm_backend_effective": effective_global,
            "per_stage": per_stage,
            "deepseek_api_key_set": key_set,
        }
    except Exception as exc:
        logger.error("effective_backends_snapshot failed: %s", exc, exc_info=True)
        return {
            "gateway_chat_llm_backend_env": None,
            "gateway_chat_llm_backend_effective": "deepseek",
            "per_stage": {},
            "deepseek_api_key_set": False,
            "error": str(exc),
        }


__all__ = [
    "ChatBackendLabel",
    "ChatBackendStage",
    "ENV_GATEWAY_CHAT_LLM_BACKEND",
    "effective_backends_snapshot",
    "resolve_chat_backend",
]
