"""Reusable clarification service for gateway endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .clarification import check_ambiguity


@dataclass
class ClarificationCheckResult:
    """Normalized clarification check output used by API endpoints."""

    needs_clarification: bool
    clarification_question: Optional[str]
    backend: str
    conversation_context: Optional[str]


class ClarificationService:
    """Encapsulates clarification config, context loading, and ambiguity check."""

    def is_enabled(self) -> bool:
        """Return True when clarification check is enabled."""
        value = os.getenv("GATEWAY_CLARIFICATION_ENABLED", "true").strip().lower()
        return value not in ("0", "false", "no", "off")

    def resolve_backend(self) -> str:
        """
        Resolve clarification backend from GATEWAY_CLARIFICATION_BACKEND env.
        Raises ValueError if not set or empty.
        """
        value = (os.getenv("GATEWAY_CLARIFICATION_BACKEND") or "").strip().lower()
        if not value:
            raise ValueError(
                "GATEWAY_CLARIFICATION_BACKEND must be set (e.g. 'ollama' or 'deepseek')"
            )
        return value

    def resolve_memory_rounds(self) -> int:
        """Resolve rounds used for clarification context."""
        try:
            return max(1, int(os.getenv("GATEWAY_CLARIFICATION_MEMORY_ROUNDS", "5")))
        except (ValueError, TypeError):
            return 5

    def get_conversation_context(
        self,
        memory: Any,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[str]:
        """Build clarification context from short-term memory."""
        if not memory:
            return None

        clarification_rounds = self.resolve_memory_rounds()
        default_rounds = 3
        history: list[dict[str, Any]] = []

        # Always fetch the larger window; trim later if no clarification found.
        if user_id and str(user_id).strip():
            history = memory.get_history_by_user(str(user_id).strip(), last_n=clarification_rounds)
        elif session_id and str(session_id).strip():
            history = memory.get_history(str(session_id).strip(), last_n=clarification_rounds)

        if not history:
            return None

        has_clarification = any(
            (turn.get("workflow") or "").strip().lower() == "clarification"
            for turn in history
        )
        effective_rounds = clarification_rounds if has_clarification else default_rounds
        if len(history) > effective_rounds:
            history = history[-effective_rounds:]

        lines: list[str] = []
        for idx, turn in enumerate(history, start=1):
            query = (turn.get("query") or "").strip()
            answer = (turn.get("answer") or "").strip()
            if not query:
                continue
            lines.append(f'Turn {idx}: User asked "{query}" -> Answer: "{answer}"')
        return "\n".join(lines) if lines else None

    def check(
        self,
        query: str,
        memory: Any,
        user_id: Optional[str],
        session_id: Optional[str],
        ambiguity_checker: Callable[..., dict[str, Any]] = check_ambiguity,
    ) -> ClarificationCheckResult:
        """Run end-to-end clarification check with unified output."""
        backend = self.resolve_backend()
        context = self.get_conversation_context(memory, user_id, session_id)
        ambiguity_result = ambiguity_checker(query, conversation_context=context)

        question = ambiguity_result.get("clarification_question")
        if not isinstance(question, str):
            question = None

        return ClarificationCheckResult(
            needs_clarification=bool(ambiguity_result.get("needs_clarification")),
            clarification_question=question.strip() if question else None,
            backend=backend,
            conversation_context=context,
        )
