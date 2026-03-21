"""
Phase 5: required-field validation and clarification question generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...route_llm.classification import validate_intents

logger = logging.getLogger(__name__)


class FieldGate:
    """Intent required-field gate before execution (classmethod facade)."""

    @classmethod
    def run_validation(
        cls,
        intents_with_meta: List[Dict[str, Any]],
        conversation_context: Optional[str],
    ) -> Optional[str]:
        """
        Validate required fields per intent; return clarification question or None.

        Returns:
            Non-empty clarification string when validation fails; None on success
            or when validation raises (logged, treated as non-fatal).
        """
        if not intents_with_meta:
            return None
        try:
            return validate_intents(intents_with_meta, conversation_context)
        except Exception as exc:
            logger.warning("Intent field validation failed (non-fatal): %s", exc)
            return None
