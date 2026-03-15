"""
Intent and workflow details for rewrite endpoint.

IntentDetailsBuilder: build intents, intent_details, workflows from rewritten query.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ..route_llm.routing_heuristics import (
    apply_docs_preference,
    route_workflow_heuristic,
    split_multi_intent_clauses,
)

logger = logging.getLogger(__name__)


class IntentDetailsBuilder:
    """
    Build intent list, intent_details (per-intent workflow), and workflows list.

    Used by the rewrite endpoint to show UI preview. When intent classification
    is enabled, uses dual retrieval (keyword + vector); otherwise heuristic.
    """

    @classmethod
    def build_intent_details(
        cls, rewritten_query: str
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]], List[str]]:
        """
        Build (intents, intent_details, workflows) from rewritten query.

        Runs split_intents; when GATEWAY_INTENT_CLASSIFICATION_ENABLED uses
        dual retrieval per intent, else heuristic. Fallback uses split_multi_intent_clauses.
        """
        intents: Optional[List[str]] = None
        intent_details: List[Dict[str, str]] = []
        workflows: List[str] = []
        if not (rewritten_query or "").strip():
            return intents, intent_details, workflows
        try:
            from ..route_llm.classification import (
                get_keyword_vector_results,
                resolve_intent,
                split_intents,
            )
            intents = split_intents(rewritten_query)
            use_dual = (
                os.getenv("GATEWAY_INTENT_CLASSIFICATION_ENABLED", "").lower()
                in ("1", "true", "yes", "on")
            )
            if intents:
                for intent in intents:
                    q = (intent or "").strip()
                    if not q:
                        continue
                    if use_dual:
                        keyword_wf, vector_wf = get_keyword_vector_results(q)
                        final_wf = resolve_intent(keyword_wf, vector_wf)
                        intent_details.append(
                            {
                                "intent": q,
                                "keyword": keyword_wf,
                                "vector": vector_wf,
                                "workflow": final_wf or "general",
                            }
                        )
                        if final_wf and final_wf not in workflows:
                            workflows.append(final_wf)
                    else:
                        wf, _ = route_workflow_heuristic(q)
                        wf = apply_docs_preference(q, wf)
                        intent_details.append(
                            {
                                "intent": q,
                                "keyword": "—",
                                "vector": "—",
                                "workflow": wf or "general",
                            }
                        )
                        if wf and wf not in workflows:
                            workflows.append(wf)
            if not workflows and (intents or []):
                q = (intents[0] if intents else rewritten_query or "").strip()
                if q:
                    wf, _ = route_workflow_heuristic(q)
                    wf = apply_docs_preference(q, wf)
                    if wf:
                        workflows = [wf]
        except Exception as exc:
            logger.warning(
                "Intent split or dual retrieval failed (rewrite response): %s", exc
            )
            q = (rewritten_query or "").strip()
            if q:
                clauses = split_multi_intent_clauses(q)
                if len(clauses) >= 2:
                    seen: set[str] = set()
                    for clause in clauses:
                        wf, _ = route_workflow_heuristic(clause)
                        wf = apply_docs_preference(clause, wf)
                        intent_details.append(
                            {
                                "intent": clause,
                                "keyword": "—",
                                "vector": "—",
                                "workflow": wf or "general",
                            }
                        )
                        if wf and wf not in seen:
                            seen.add(wf)
                            workflows.append(wf)
                else:
                    wf, _ = route_workflow_heuristic(q)
                    wf = apply_docs_preference(q, wf)
                    intent_details.append(
                        {
                            "intent": q,
                            "keyword": "—",
                            "vector": "—",
                            "workflow": wf or "general",
                        }
                    )
                    if wf:
                        workflows = [wf]
        return intents, intent_details, workflows
