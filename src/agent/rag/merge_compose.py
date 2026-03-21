"""
Final merge step for Amazon Business: Chroma excerpts + DeepSeek evidence.

Conflict rule: when Chroma (Source A) and model evidence (Source B) disagree,
the final answer must follow Source A.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.call_deepseek import DeepSeekChat

logger = logging.getLogger(__name__)

_MERGE_SYSTEM = (
    "You synthesize two sources into one answer for the user.\n"
    "- Source A: excerpts from an internal document vector store (Chroma). "
    "These are authoritative for product/policy details that appear in them.\n"
    "- Source B: bullet points from a language model knowledge pass (not from the vector store).\n"
    "Rules:\n"
    "1) Integrate both sources when they agree.\n"
    "2) If Source A and Source B conflict on a factual point, you MUST prefer Source A.\n"
    "3) If Source A has no relevant excerpt for a sub-question, you may use Source B "
    "but label it as general knowledge.\n"
    "4) Write a coherent answer in the same language as the user question when reasonable.\n"
    "5) Do not mention \"Source A/B\" labels in the final answer; write naturally.\n"
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
        chat: Optional[DeepSeekChat] = None,
    ) -> str:
        """
        Produce the user-facing answer from Chroma text and DeepSeek evidence.

        Args:
            question: Original user question.
            chroma_blocks: Formatted string of Chroma excerpts (Source A).
            deepseek_evidence: Bullet list from DeepSeekRetrieveFacade (Source B).
            max_tokens: Completion cap.
            temperature: Sampling temperature.
            chat: Optional DeepSeekChat for tests.

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
        client = chat or DeepSeekChat()
        try:
            out = client.complete(
                _MERGE_SYSTEM,
                user_content,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("Merge DeepSeek call failed: %s", exc, exc_info=True)
            raise

        logger.debug("Merged answer length=%d", len(out or ""))
        return out
