"""
Intent and workflow details for rewrite endpoint.

IntentDetailsBuilder: build intents, intent_details, workflows from rewritten query.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentDetailsBuilder:
    """
    Build intent list, intent_details (per-intent workflow), and workflows list.

    Used by the rewrite endpoint to show UI preview. Uses LLM prompt-based
    classification (classification module) for all intent detection.
    """

    @classmethod
    def build_intent_details(
        cls, rewritten_query: str
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]], List[str]]:
        """
        Build (intents, intent_details, workflows) from rewritten query.

        Runs split_intents; classifies each intent via classify_intent().
        Falls back to "general" workflow when classification is unavailable.
        """
        intents: Optional[List[str]] = None
        intent_details: List[Dict[str, str]] = []
        workflows: List[str] = []
        if not (rewritten_query or "").strip():
            return intents, intent_details, workflows
        try:
            from ..route_llm.classification import (
                classify_intent,
                split_intents,
            )
            intents = split_intents(rewritten_query)
            if intents:
                for intent in intents:
                    q = (intent or "").strip()
                    if not q:
                        continue
                    result = classify_intent(q)
                    wf = result.workflow if result else "general"
                    intent_details.append(
                        {
                            "intent": q,
                            "workflow": wf,
                        }
                    )
                    if wf and wf not in workflows:
                        workflows.append(wf)
            if not workflows and (intents or []):
                q = (intents[0] if intents else rewritten_query or "").strip()
                if q:
                    result = classify_intent(q)
                    wf = result.workflow if result else "general"
                    if wf:
                        workflows = [wf]
        except Exception as exc:
            logger.warning(
                "Intent split or classification failed (rewrite response): %s", exc
            )
            # Fallback: single "general" intent for the whole query
            q = (rewritten_query or "").strip()
            if q:
                intent_details.append(
                    {
                        "intent": q,
                        "workflow": "general",
                    }
                )
                workflows = ["general"]
        return intents, intent_details, workflows
