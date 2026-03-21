"""
Assemble build_execution_plan from planning facades and route hints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...route_llm.rewriting.rewriters import _RewriteRouter
from ...schemas import QueryRequest, RewritePlan
from src.logger import get_logger_facade
from src.retrieval.query_process import QueryProcessor

from .builders import PlanBuilder
from .fallback import FallbackPlanBuilder
from .field_gate import FieldGate
from .postprocess import PlanPostProcessor

logger = logging.getLogger(__name__)

_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


class PlanPipeline:
    """High-level execution plan construction (classmethod facade)."""

    @classmethod
    def build_execution_plan(
        cls,
        request: QueryRequest,
        rewritten_query: str,
        intents: Optional[List[str]] = None,
        conversation_context: Optional[str] = None,
        classified_intents: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[RewritePlan, Optional[str]]:
        """
        Build a validated execution plan for query orchestration.

        Args:
            request: Parsed QueryRequest.
            rewritten_query: Optimized retrieval query from Route LLM.
            intents: Optional list of sub-queries from intent classification.
            conversation_context: Optional conversation history for classification.
            classified_intents: Optional pre-classified intent metadata from caller.

        Returns:
            (RewritePlan, clarification_question) where clarification_question is
            None when all required fields are present.
        """
        explicit = (request.workflow or "auto").strip().lower() or "auto"
        normalized_query = QueryProcessor.normalize(request.query or "")
        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="dispatcher_plan_start",
                    stage="dispatcher",
                    message="build_execution_plan started",
                    status="started",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    workflow=explicit,
                    query_raw=request.query or "",
                    query_rewritten=rewritten_query,
                    intent_list=intents or [],
                )
            except Exception:
                pass

        if explicit != "auto":
            task_query = (rewritten_query or normalized_query).strip() or normalized_query
            return FallbackPlanBuilder.build_single_task_plan(task_query, explicit), None

        if intents and len(intents) > 0:
            plan, intents_with_meta = PlanBuilder.build_from_extracted_intents(
                intents,
                conversation_context,
                classified_intents=classified_intents,
            )
            plan = PlanPostProcessor.correct_plan_workflows(plan)
            clarification_question = FieldGate.run_validation(
                intents_with_meta,
                conversation_context,
            )
            return plan, clarification_question

        multi_task_plan = FallbackPlanBuilder.build_multi_task_plan_from_query(
            rewritten_query or normalized_query
        )
        if multi_task_plan is not None:
            return multi_task_plan, None

        workflow, _, _, _, _ = _RewriteRouter.route_workflow(
            (rewritten_query or normalized_query).strip(), request
        )
        task_query = (rewritten_query or normalized_query).strip() or normalized_query
        final_plan = FallbackPlanBuilder.build_single_task_plan(task_query, workflow)
        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="dispatcher_plan_done",
                    stage="dispatcher",
                    message="build_execution_plan completed",
                    status="success",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    workflow=workflow,
                    query_raw=request.query or "",
                    query_rewritten=rewritten_query,
                    metadata={"plan_type": final_plan.plan_type},
                )
            except Exception:
                pass
        return final_plan, None
