"""
Unified DeepSeek client for Route LLM.

Provides a single OpenAI-compatible chat path for clarification, rewriting,
and intent splitting. Configuration is read once via get_deepseek_config()
from the four standard DeepSeek environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSeekConfig:
    """
    Frozen snapshot of DeepSeek API settings loaded from the environment.

    Attributes:
        api_key: Bearer key for DeepSeek (OpenAI-compatible API).
        llm_model: Model id sent to chat.completions (e.g. deepseek-chat).
        base_url: API base URL without trailing slash.
        request_timeout: Client timeout in seconds for each HTTP request.
    """

    api_key: str
    llm_model: str
    base_url: str
    request_timeout: int


def get_deepseek_config() -> DeepSeekConfig:
    """
    Load DeepSeek settings from the process environment.

    Reads exactly these variables:

        DEEPSEEK_API_KEY (required)
            Secret key; raises ValueError if missing or blank.

        DEEPSEEK_LLM_MODEL (optional)
            Chat model name. Default: deepseek-chat

        DEEPSEEK_BASE_URL (optional)
            OpenAI-compatible base URL. Default: https://api.deepseek.com

        DEEPSEEK_REQUEST_TIMEOUT (optional)
            Per-request timeout in seconds. Must be positive integer.
            Default: 60

    Returns:
        DeepSeekConfig: Immutable config used by DeepSeekChat.complete().

    Raises:
        ValueError: If DEEPSEEK_API_KEY is unset, or timeout is invalid.
    """
    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY must be set for DeepSeek calls")

    llm_model = (os.getenv("DEEPSEEK_LLM_MODEL") or "deepseek-chat").strip()
    base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")

    raw_timeout = (os.getenv("DEEPSEEK_REQUEST_TIMEOUT") or "60").strip()
    try:
        request_timeout = int(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "DEEPSEEK_REQUEST_TIMEOUT must be a positive integer"
        ) from exc
    if request_timeout <= 0:
        raise ValueError("DEEPSEEK_REQUEST_TIMEOUT must be positive")

    return DeepSeekConfig(
        api_key=api_key,
        llm_model=llm_model,
        base_url=base_url,
        request_timeout=request_timeout,
    )


class DeepSeekChat:
    """
    Production client for DeepSeek chat completions (system + user messages).

    Thread-safe for typical FastAPI use: each complete() builds a short-lived
    OpenAI client from get_deepseek_config(). No shared mutable state.
    """

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Execute one chat completion and return the assistant message text.

        Sends messages: system (system_prompt), user (user_content). Uses
        DEEPSEEK_LLM_MODEL unless model_override is non-empty.

        Args:
            system_prompt: System role content (instructions).
            user_content: User role content (task input).
            max_tokens: Max completion tokens (passed to API).
            temperature: Sampling temperature.
            model_override: If set, replaces config llm_model for this call only.

        Returns:
            Stripped assistant content string.

        Raises:
            RuntimeError: If openai is not installed, the API errors, or the
                response has no non-empty assistant content.
        """
        cfg = get_deepseek_config()
        model = (model_override or "").strip() or cfg.llm_model

        if OpenAI is None:
            raise RuntimeError("openai client not installed; cannot call DeepSeek")

        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.request_timeout,
        )
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_content or ""},
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = response.choices[0] if response.choices else None
            if choice and choice.message and choice.message.content:
                return (choice.message.content or "").strip()
        except Exception as exc:
            logger.error("DeepSeek chat failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek failed: {exc}") from exc

        raise RuntimeError("DeepSeek returned empty assistant content")
