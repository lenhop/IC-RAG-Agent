"""
Gateway task execution and workflow dispatch.

DispatcherExecutor: call workflow backends, execute tasks and plan.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List

from ..dispatcher.services import (
    call_amazon_docs,
    call_general,
    call_ic_docs,
    call_sp_api,
    call_uds,
)
from ..schemas import (
    QueryRequest,
    RewritePlan,
    TaskExecutionResult,
    TaskGroup,
    TaskItem,
)

logger = logging.getLogger(__name__)


class DispatcherExecutor:
    """
    Execute workflow backends and task plans.

    call_workflow_backend: invoke one worker (RAG/UDS/SP-API etc).
    execute_task / execute_task_group / execute_plan: run plan and return results.
    """

    @classmethod
    def call_workflow_backend(
        cls, workflow: str, query_text: str, session_id: str | None
    ) -> Dict[str, Any]:
        """Invoke one worker agent (RAG/UDS/SP-API) and return normalized dict."""
        workflow_lower = (workflow or "general").strip().lower()
        if workflow_lower == "general":
            return call_general(query_text, session_id)
        if workflow_lower == "amazon_docs":
            return call_amazon_docs(query_text, session_id)
        if workflow_lower == "ic_docs":
            return call_ic_docs(query_text, session_id)
        if workflow_lower == "sp_api":
            return call_sp_api(query_text, session_id)
        if workflow_lower == "uds":
            return call_uds(query_text, session_id)
        logger.warning("Unknown workflow '%s', falling back to general", workflow)
        return call_general(query_text, session_id)

    @classmethod
    def execute_task(cls, task: TaskItem, request: QueryRequest) -> TaskExecutionResult:
        """Execute a single task and return structured result."""
        started = time.perf_counter()
        task_query = (task.query or "").strip()
        if not task_query:
            return TaskExecutionResult(
                task_id=task.task_id,
                workflow=task.workflow,
                query=task_query,
                status="skipped",
                answer="",
                sources=[],
                error="Empty task query",
                duration_ms=0,
            )
        backend_result = cls.call_workflow_backend(
            task.workflow, task_query, request.session_id
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        error_msg = backend_result.get("error")
        if error_msg:
            return TaskExecutionResult(
                task_id=task.task_id,
                workflow=task.workflow,
                query=task_query,
                status="failed",
                answer="",
                sources=backend_result.get("sources", []),
                error=str(error_msg),
                duration_ms=duration_ms,
            )
        return TaskExecutionResult(
            task_id=task.task_id,
            workflow=task.workflow,
            query=task_query,
            status="completed",
            answer=str(backend_result.get("answer", "")),
            sources=backend_result.get("sources", []),
            error=None,
            duration_ms=duration_ms,
        )

    @classmethod
    def execute_task_group(
        cls, group: TaskGroup, request: QueryRequest
    ) -> List[TaskExecutionResult]:
        """Execute one task group; tasks in parallel when group.parallel is True."""
        if not group.tasks:
            return []
        if not group.parallel or len(group.tasks) == 1:
            return [cls.execute_task(task, request) for task in group.tasks]
        results: List[TaskExecutionResult] = []
        max_workers = min(4, len(group.tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(cls.execute_task, task, request): task.task_id
                for task in group.tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover
                    task_id = future_map[future]
                    logger.exception(
                        "Task execution failed for task_id=%s: %s", task_id, exc
                    )
                    results.append(
                        TaskExecutionResult(
                            task_id=task_id,
                            workflow="general",
                            query="",
                            status="failed",
                            answer="",
                            sources=[],
                            error=str(exc),
                            duration_ms=0,
                        )
                    )
        by_task_id = {r.task_id: r for r in results}
        return [by_task_id[t.task_id] for t in group.tasks if t.task_id in by_task_id]

    @classmethod
    def execute_plan(
        cls, plan: RewritePlan, request: QueryRequest
    ) -> List[TaskExecutionResult]:
        """Execute full plan; groups sequential, tasks within group in parallel."""
        all_results: List[TaskExecutionResult] = []
        for group in plan.task_groups:
            all_results.extend(cls.execute_task_group(group, request))
        return all_results
