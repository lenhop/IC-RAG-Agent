"""
DeepSeek \"retrieval\" pass: model-generated evidence bullets (no vector store).

Uses src.llm.call_deepseek.DeepSeekChat only; no duplicate HTTP clients.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.call_deepseek import DeepSeekChat

logger = logging.getLogger(__name__)

# System prompt: ask for concise factual bullets the model can justify from training.
_EVIDENCE_SYSTEM = (
    "You are a research assistant for Amazon seller and e-commerce policy topics. "
    "Given a user question, output ONLY a bullet list of short factual claims or "
    "definitions that could help answer it. Do not cite documents you do not have. "
    "If unsure, say \"Uncertain:\" for that bullet. Use clear, separate lines "
    "starting with \"- \". Do not answer the question fully; only list supporting points."
)


class DeepSeekRetrieveFacade:
    """
    Classmethod facade for the second retrieval path (parametric / model knowledge).
    """

    @classmethod
    def evidence_for_query(
        cls,
        question: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        chat: Optional[DeepSeekChat] = None,
    ) -> str:
        """
        Produce structured bullet evidence from DeepSeek (not from Chroma).

        Args:
            question: User question text.
            max_tokens: Completion cap for the evidence pass.
            temperature: Sampling temperature (low for factual tone).
            chat: Optional DeepSeekChat for tests; default new instance.

        Returns:
            Model text (bullet list).

        Raises:
            ValueError: If question is empty.
            RuntimeError: Propagated from DeepSeekChat.complete on API failure.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty for DeepSeek evidence retrieval")

        client = chat or DeepSeekChat()
        user_content = f"Question:\n{q}\n\nProduce bullet points as instructed."
        try:
            out = client.complete(
                _EVIDENCE_SYSTEM,
                user_content,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("DeepSeek evidence retrieval failed: %s", exc, exc_info=True)
            raise

        logger.debug("DeepSeek evidence length=%d", len(out or ""))
        return out

    @classmethod
    def general_answer(
        cls,
        question: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        chat: Optional[DeepSeekChat] = None,
    ) -> str:
        """
        Single-path general Q&A (no Chroma).

        Args:
            question: User question.
            max_tokens: Completion cap.
            temperature: Sampling temperature.
            chat: Optional DeepSeekChat for tests.

        Returns:
            Assistant answer string.

        Raises:
            ValueError: If question is empty.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")

        system = (
            "You are a helpful assistant. Answer clearly and accurately. "
            "If the question is ambiguous, state reasonable assumptions briefly."
        )
        client = chat or DeepSeekChat()
        return client.complete(
            system,
            f"User question:\n{q}",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @classmethod
    def answer_from_chroma_context_only(
        cls,
        question: str,
        chroma_context: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        chat: Optional[DeepSeekChat] = None,
    ) -> str:
        """
        Grounded answer using only the provided Chroma context (documents mode).

        Args:
            question: User question.
            chroma_context: Concatenated excerpt text from Chroma.
            max_tokens: Completion cap.
            temperature: Sampling temperature.
            chat: Optional DeepSeekChat for tests.

        Returns:
            Assistant answer grounded on context.

        Raises:
            ValueError: If question is empty.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")

        system = (
            "You answer using ONLY the provided context excerpts. "
            "If the context does not contain enough information, say so explicitly "
            "and do not invent facts. Cite which excerpt supports key claims when possible."
        )
        user = f"Context:\n{chroma_context}\n\nQuestion:\n{q}"
        client = chat or DeepSeekChat()
        return client.complete(
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
