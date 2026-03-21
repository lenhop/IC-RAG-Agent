"""
Build RewritePlan from extracted intents and optional pre-classified metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...route_llm.classification import classify_intents_batch
from ...schemas import RewritePlan, TaskGroup, TaskItem

logger = logging.getLogger(__name__)


class PlanBuilder:
    """Planning helpers for intent-to-task graph (classmethod facade)."""

    @classmethod
    def build_from_extracted_intents(
        cls,
        intents: List[str],
        conversation_context: Optional[str] = None,
        classified_intents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[RewritePlan, List[Dict]]:
        """
        Build execution plan from extracted intents.

        Delegates batch classification to classify_intents_batch when
        classified_intents is not provided. Falls back to \"general\" workflow
        when classification fails.

        Returns:
            (RewritePlan, intents_with_meta) for Phase 5 field validation.
        """
        tasks: List[TaskItem] = []
        intents_with_meta: List[Dict] = []

        batch_results = classified_intents
        if batch_results is None:
            try:
                batch_results = classify_intents_batch(intents, conversation_context)
            except Exception as exc:
                logger.warning("Batch classification failed, falling back to general: %s", exc)
                batch_results = None

        if batch_results is not None:
            for idx, item in enumerate(batch_results, start=1):
                q = item["query"]
                wf = item["workflow"]
                intent_name = item["intent_name"]

                intents_with_meta.append({
                    "query": q,
                    "intent_name": intent_name,
                    "required_fields": item.get("required_fields") or [],
                    "clarification_template": item.get("clarification_template") or "",
                })
                tasks.append(
                    TaskItem(
                        task_id=f"t{idx}",
                        workflow=wf,
                        query=q,
                        depends_on=[],
                        reason=f"intent_classified:{intent_name}" if intent_name else "classified_fallback",
                    )
                )
        else:
            for idx, intent in enumerate(intents, start=1):
                q = (intent or "").strip()
                if not q:
                    continue
                intents_with_meta.append({
                    "query": q,
                    "intent_name": "",
                    "required_fields": [],
                    "clarification_template": "",
                })
                tasks.append(
                    TaskItem(
                        task_id=f"t{idx}",
                        workflow="general",
                        query=q,
                        depends_on=[],
                        reason="classification_unavailable_fallback",
                    )
                )

        if not tasks:
            fallback_q = (intents[0] if intents else "unable to extract intents").strip() or "unable to extract intents"
            plan = RewritePlan(
                plan_type="single_domain",
                merge_strategy="concat",
                task_groups=[
                    TaskGroup(
                        group_id="g1",
                        parallel=True,
                        tasks=[
                            TaskItem(
                                task_id="t1",
                                workflow="general",
                                query=fallback_q,
                                depends_on=[],
                                reason="empty_intents_fallback",
                            )
                        ],
                    )
                ],
            )
            return plan, []

        plan = RewritePlan(
            plan_type="hybrid" if len({t.workflow for t in tasks}) > 1 else "single_domain",
            merge_strategy="concat",
            task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
        )
        return plan, intents_with_meta
