"""
Final merge step for Amazon Business: Chroma excerpts + DeepSeek evidence.

Conflict rule: when Chroma (Source A) and model evidence (Source B) disagree,
the final answer must follow Source A.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.text_generation_backend import complete_chat, resolve_text_generation_backend

logger = logging.getLogger(__name__)

_MERGE_SYSTEM = (
    "You synthesize two sources into one answer for the user.\n"
    "- Source A: excerpts from an internal document vector store (Chroma). "
    "These are authoritative for product/policy details that appear in them.\n"
    "- Source B: bullet points from a language model knowledge pass (not from the vector store).\n"
    "Rules:\n"
    "1) Integrate both sources when they agree.\n"
    "2) If Source A and Source B conflict on a factual point, you MUST prefer Source A (Chroma wins).\n"
    "3) Do not invent facts, SKUs, order IDs, or citations not present in Source A or B.\n"
    "4) If Source A has no relevant excerpt for a sub-question, you may use Source B "
    "but label it as general knowledge.\n"
    "5) Write a coherent answer in the same language as the user question when reasonable.\n"
    "6) Do not mention \"Source A/B\" labels in the final answer; write naturally.\n"
)


class MergeComposer:
    """Classmethod facade for the dual-path merge completion."""

    @classmethod
    def final_answer(
        cls,
        question: str,
        chroma_blocks: str,
        deepseek_evidence: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        text_generation_backend: Optional[str] = None,
    ) -> str:
        """
        Produce the user-facing answer from Chroma text and model evidence (Source B).

        Args:
            question: Original user question.
            chroma_blocks: Formatted string of Chroma excerpts (Source A).
            deepseek_evidence: Bullet list from DeepSeekRetrieveFacade (Source B).
            max_tokens: Completion cap.
            temperature: Sampling temperature.
            text_generation_backend: ``deepseek`` or ``ollama``; default from
                ``GATEWAY_TEXT_GENERATION_BACKEND`` / env resolution.

        Returns:
            Merged assistant answer.

        Raises:
            ValueError: If question is empty.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty for merge")

        a_block = (chroma_blocks or "").strip() or "(No document excerpts passed similarity threshold.)"
        b_block = (deepseek_evidence or "").strip() or "(No model evidence produced.)"

        user_content = (
            f"User question:\n{q}\n\n"
            f"=== Source A (Chroma document excerpts) ===\n{a_block}\n\n"
            f"=== Source B (Model knowledge bullets) ===\n{b_block}\n\n"
            "Write the final answer following the rules."
        )
        backend = (text_generation_backend or resolve_text_generation_backend()).strip().lower()
        if backend not in ("deepseek", "ollama"):
            backend = resolve_text_generation_backend()

        try:
            out = complete_chat(
                backend,  # type: ignore[arg-type]
                _MERGE_SYSTEM,
                user_content,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Merge LLM call failed (backend=%s): %s", backend, exc, exc_info=True)
            raise

        logger.debug("Merged answer length=%d", len(out or ""))
        return out
