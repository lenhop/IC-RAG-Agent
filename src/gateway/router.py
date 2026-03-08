"""
Route LLM (Planning): routing and query rewriting for the gateway.

Provides:
- rewrite_query: normalize and optionally rewrite via LLM (Ollama/DeepSeek).
- build_execution_plan: parse planner JSON or heuristic multi-task fallback.
- route_workflow: single-task workflow classification (Route LLM or heuristic).
- _correct_plan_workflows: heuristic override for LLM misclassifications.

Output: RewritePlan (task_groups with workflow + query per task) for the Dispatcher.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Tuple, Optional

from .rewriters import (
    parse_rewrite_plan_text,
    planner_rewrite_enabled,
    rewrite_intents_only,
)
from .schemas import QueryRequest, RewritePlan, TaskGroup, TaskItem

logger = logging.getLogger(__name__)

# Route LLM env: read at call time to avoid import-side side effects
def _route_llm_enabled() -> bool:
    v = os.getenv("GATEWAY_ROUTE_LLM_ENABLED", "false").strip().lower()
    return v in ("true", "1", "yes")


def _route_llm_backend() -> str:
    v = os.getenv("GATEWAY_ROUTE_LLM_BACKEND", "ollama").strip().lower()
    return v if v in ("ollama", "deepseek") else "ollama"


def _route_llm_conf_threshold() -> float:
    try:
        return float(os.getenv("GATEWAY_ROUTE_LLM_CONF_THRESHOLD", "0.7"))
    except (ValueError, TypeError):
        return 0.7


def _normalize(query: str) -> str:
    """Trim and collapse whitespace. Always applied before any rewrite."""
    text = query or ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_definition_query(query: str) -> bool:
    """Return True for definition/explanation style questions."""
    q = (query or "").strip().lower()
    return (
        q.startswith("what is ")
        or q.startswith("what's ")
        or q.startswith("define ")
        or q.startswith("explain ")
    )


def _is_fba_term_query(query: str) -> bool:
    """Return True when query focuses on FBA terminology."""
    q = (query or "").lower()
    return "fba" in q or "fbm" in q


def _apply_docs_preference(query: str, workflow: str) -> str:
    """
    Prefer IC docs for definition-style FBA/FBM questions.

    This prevents conceptual documentation queries (e.g. "what is FBA")
    from being routed to operational SP-API workflow.
    """
    if workflow == "sp_api" and _is_definition_query(query) and _is_fba_term_query(query):
        q = (query or "").lower()
        # Fee-policy style definitions should use Amazon docs rather than IC docs.
        if any(k in q for k in ("storage fee", "fba fee", "fees", "cost", "price")):
            return "amazon_docs"
        return "ic_docs"
    return workflow


def _split_multi_intent_clauses(query: str) -> List[str]:
    """Split a potentially mixed query into normalized clause candidates."""
    text = _normalize(query)
    if not text:
        return []
    # First try granular split on question-starter patterns (get order, which table, show me, etc.).
    # This handles "what is X get order Y which table Z show me W" -> 4 clauses.
    granular_raw = re.split(
        r" (?=how many |what is |what's |what does |which |what table |show me |get |get order |check )",
        text,
        flags=re.I,
    )
    granular = []
    for p in granular_raw:
        trimmed = p.removesuffix(" and") if p.endswith(" and") else p
        norm = _normalize(trimmed)
        if norm:
            granular.append(norm)
    if len(granular) >= 2:
        parts = granular
    else:
        # Fallback: split by comma, semicolon, or " and " before question words.
        # Do NOT split on comma when part of date (e.g. "September 1st, 2026", "January 1, 2025").
        # Use (?!\s*\d{4}\b) so "Month DD, YYYY" is not split (space before year).
        parts = re.split(
            r",\s*(?!\s*\d{4}\b)|;\s*|\s+and\s+(?=how|what|which|when|show|get|check)",
            text,
            flags=re.I,
        )
    clauses = []
    seen = set()
    for raw in parts:
        clause = _normalize(raw)
        if not clause:
            continue
        key = clause.lower()
        if key in seen:
            continue
        seen.add(key)
        clauses.append(clause)
    return clauses


def _build_plan_from_extracted_intents(intents: List[str]) -> RewritePlan:
    """
    Build execution plan from extracted_intents when LLM merges tasks.
    Routes each intent via heuristic and creates one task per intent.
    """
    tasks: List[TaskItem] = []
    for idx, intent in enumerate(intents, start=1):
        q = (intent or "").strip()
        if not q:
            continue
        wf, _ = _route_workflow_heuristic(q)
        wf = _apply_docs_preference(q, wf)
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=wf,
                query=q,
                depends_on=[],
                reason="extracted_intents_fallback",
            )
        )
    if not tasks:
        return RewritePlan(
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
                            query=(intents[0] if intents else "unable to extract intents").strip() or "unable to extract intents",
                            depends_on=[],
                            reason="empty_intents_fallback",
                        )
                    ],
                )
            ],
        )
    return RewritePlan(
        plan_type="hybrid" if len({t.workflow for t in tasks}) > 1 else "single_domain",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )


def _build_multi_task_plan_from_query(query: str) -> Optional[RewritePlan]:
    """
    Build a heuristic multi-task plan from comma/semicolon separated mixed query.

    This is used as a robust fallback when planner JSON is unavailable.
    """
    clauses = _split_multi_intent_clauses(query)
    if len(clauses) < 2:
        return None
    tasks: List[TaskItem] = []
    for idx, clause in enumerate(clauses, start=1):
        workflow, _ = _route_workflow_heuristic(clause)
        workflow = _apply_docs_preference(clause, workflow)
        tasks.append(
            TaskItem(
                task_id=f"t{idx}",
                workflow=workflow,
                query=clause,
                depends_on=[],
                reason="heuristic_multi_intent_fallback",
            )
        )
    return RewritePlan(
        plan_type="hybrid",
        merge_strategy="concat",
        task_groups=[TaskGroup(group_id="g1", parallel=True, tasks=tasks)],
    )


def rewrite_query(request: QueryRequest) -> str:
    """
    Normalize and optionally rewrite the incoming query before routing.

    Flow:
    1. Always normalize (trim, collapse whitespace).
    2. If rewrite_enable:
       - Resolve backend from request.rewrite_backend or GATEWAY_REWRITE_BACKEND (default "ollama").
       - If planner enabled: Phase 1 intent classification via rewrite_intents_only.
         On success: return JSON {"intents": [...]} for build_execution_plan.
         On failure: return normalized query; build_execution_plan uses heuristic fallback.
       - Else if backend in ("ollama", "deepseek"): call rewriter; return result or normalized.
       - Else: return normalized query.
    3. Else: return normalized query.

    Args:
        request: Parsed QueryRequest from the client.

    Returns:
        Rewritten query string used for routing and downstream services.
    """
    normalized = _normalize(request.query or "")

    if not request.rewrite_enable:
        return normalized

    backend = (
        (request.rewrite_backend or "").strip().lower()
        or os.getenv("GATEWAY_REWRITE_BACKEND", "ollama").strip().lower()
    )

    # Two-phase flow: planner uses intent classification only (Phase 1).
    if planner_rewrite_enabled():
        import json
        result = rewrite_intents_only(normalized, backend=backend)
        if result and result.get("intents"):
            return json.dumps(result)
        # Fallback: return normalized; build_execution_plan uses heuristic multi-intent split.
        return normalized

    if backend == "ollama":
        from .rewriters import rewrite_with_ollama
        return rewrite_with_ollama(normalized)
    if backend == "deepseek":
        from .rewriters import rewrite_with_deepseek
        return rewrite_with_deepseek(normalized)

    return normalized


def _expand_merged_tasks(plan: RewritePlan) -> RewritePlan:
    """
    Split any task whose query contains multiple distinct sub-queries into
    separate tasks. Ensures each sub-question is a separate plan item (Option B).
    """
    expanded_groups = []
    for group in plan.task_groups or []:
        expanded_tasks: List[TaskItem] = []
        next_id = 1
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            clauses = _split_multi_intent_clauses(q)
            if len(clauses) < 2:
                expanded_tasks.append(task)
                next_id += 1
                continue
            for clause in clauses:
                wf, _ = _route_workflow_heuristic(clause)
                wf = _apply_docs_preference(clause, wf)
                expanded_tasks.append(
                    TaskItem(
                        task_id=f"t{next_id}",
                        workflow=wf,
                        query=clause,
                        depends_on=task.depends_on or [],
                        reason=(task.reason or "") + "_expanded" if task.reason else "merged_split",
                    )
                )
                next_id += 1
        if expanded_tasks:
            expanded_groups.append(
                TaskGroup(
                    group_id=group.group_id,
                    parallel=group.parallel,
                    tasks=expanded_tasks,
                )
            )
    if not expanded_groups:
        return plan
    return RewritePlan(
        plan_type="hybrid" if len({t.workflow for g in expanded_groups for t in g.tasks}) > 1 else plan.plan_type,
        merge_strategy=plan.merge_strategy,
        task_groups=expanded_groups,
    )


def _correct_plan_workflows(plan: RewritePlan) -> RewritePlan:
    """
    Apply heuristic-based workflow correction to fix LLM misclassifications.

    When the heuristic returns a different workflow with confidence >= 0.9,
    override the task's workflow to correct obvious routing errors.
    """
    HEURISTIC_OVERRIDE_CONF = 0.9
    for group in plan.task_groups or []:
        for task in group.tasks or []:
            q = (task.query or "").strip()
            if not q:
                continue
            h_wf, h_conf = _route_workflow_heuristic(q)
            h_wf = _apply_docs_preference(q, h_wf)
            current = (task.workflow or "").strip().lower()
            if h_wf != current and h_conf >= HEURISTIC_OVERRIDE_CONF:
                logger.debug(
                    "Plan correction: task %s workflow %s -> %s (heuristic conf=%.2f)",
                    task.task_id,
                    current,
                    h_wf,
                    h_conf,
                )
                task.workflow = h_wf
    return plan


def _build_single_task_plan(query: str, workflow: str) -> RewritePlan:
    """Build a single-task execution plan for legacy/fallback flow."""
    task_query = (query or "").strip()
    task_workflow = (workflow or "general").strip().lower() or "general"
    return RewritePlan(
        plan_type="single_domain",
        merge_strategy="concat",
        task_groups=[
            TaskGroup(
                group_id="g1",
                parallel=True,
                tasks=[
                    TaskItem(
                        task_id="t1",
                        workflow=task_workflow,
                        query=task_query,
                        depends_on=[],
                        reason="single_task_fallback",
                    )
                ],
            )
        ],
    )


def build_execution_plan(request: QueryRequest, rewritten_query: str) -> RewritePlan:
    """
    Build a validated execution plan for query orchestration.

    Behavior:
    - Explicit workflow (non-auto): synthesize one task with that workflow.
    - Planner mode enabled (auto): parse structured planner output from rewritten text.
    - Fallback: route once and synthesize one task from routed workflow.
    """
    explicit = (request.workflow or "auto").strip().lower() or "auto"
    normalized_query = _normalize(request.query or "")

    if explicit != "auto":
        task_query = (rewritten_query or normalized_query).strip() or normalized_query
        return _build_single_task_plan(task_query, explicit)

    if planner_rewrite_enabled():
        parsed_plan = parse_rewrite_plan_text(
            text=rewritten_query or "",
            fallback_query=normalized_query,
        )
        if parsed_plan:
            task_count = sum(len(g.tasks) for g in (parsed_plan.task_groups or []))
            intents = parsed_plan.extracted_intents or []
            # When extracted_intents has more items than tasks (or no tasks), use intents.
            if intents and (len(intents) > task_count or task_count == 0):
                logger.info(
                    "Planner merged or empty tasks (%d tasks vs %d extracted_intents); "
                    "rebuilding from extracted_intents.",
                    task_count,
                    len(intents),
                )
                return _correct_plan_workflows(_build_plan_from_extracted_intents(intents))
            if parsed_plan.task_groups:
                # When LLM returns single task but query has multiple clauses, use heuristic.
                clauses = _split_multi_intent_clauses(rewritten_query or normalized_query)
                if task_count == 1 and len(clauses) >= 2:
                    logger.info(
                        "Planner returned single task for multi-clause query (%d clauses); "
                        "using heuristic multi-task fallback.",
                        len(clauses),
                    )
                    multi_task_plan = _build_multi_task_plan_from_query(
                        rewritten_query or normalized_query
                    )
                    if multi_task_plan:
                        return _correct_plan_workflows(multi_task_plan)
                # Expand any merged tasks (e.g. "how many X how many Y" -> two tasks).
                expanded = _expand_merged_tasks(parsed_plan)
                return _correct_plan_workflows(expanded)

    multi_task_plan = _build_multi_task_plan_from_query(rewritten_query or normalized_query)
    if multi_task_plan is not None:
        return multi_task_plan

    workflow, _, _, _, _ = route_workflow((rewritten_query or normalized_query).strip(), request)
    task_query = (rewritten_query or normalized_query).strip() or normalized_query
    return _build_single_task_plan(task_query, workflow)


def _route_workflow_heuristic(query: str) -> Tuple[str, float]:
    """
    Rule-based workflow selection from query keywords.

    Used when workflow is "auto" and either Route LLM is disabled or
    LLM confidence is below threshold. Returns (workflow, confidence).
    """
    q_lower = (query or "").lower()

    # SP-API workflow: real-time operations (order status, inventory, buy box, etc.).
    if any(
        k in q_lower
        for k in [
            "order status",
            "check order",
            "get order",
            "inventory placement",
            "request an inventory placement",
            "request inventory placement",
            "inventory health",
            "buy box status",
            "buy box",
            "check if asin",
            "has any active",
            "seller account verification",
            "verify my seller",
            "verify seller account",
            "request a detailed financial",
            "request a settlement report",
            "removal order",
            "create a removal order",
            "stranded inventory",
        ]
    ):
        return "sp_api", 0.92

    # UDS workflow: ASIN-level data retrieval (financial summary, fees, revenue from warehouse).
    if any(
        k in q_lower
        for k in [
            "financial summary for asin",
            "financial summary for",
            "get financial summary",
        ]
    ):
        return "uds", 0.92

    # UDS workflow: high-confidence analytical aggregations (top N, by X).
    if any(
        k in q_lower
        for k in [
            "top 5 ",
            "top 10 ",
            "top 20 ",
            "products by ",
            "by refund",
            "by product",
            "by category",
            "refund count",
            "refund rate",
        ]
    ):
        return "uds", 0.92

    # UDS workflow: strong table/schema cues (inclined to UDS agent).
    if any(
        k in q_lower
        for k in [
            "clickhouse table",
            "data table",
            "uds table",
            "which table",
            "what table",
            "table stores",
            "clickhouse",
            "uds",
        ]
    ):
        return "uds", 0.92

    # UDS workflow (analytical/table/schema cues).
    if any(
        k in q_lower
        for k in [
            "what columns",
            "column",
            "schema",
            "dataset",
            "last month",
            "last 30 days",
            "last quarter",
            "by month",
            "trend",
            "total ",
            "average ",
            "compare ",
            "breakdown",
            "historical",
            "financial report",
            "settlement report",            
        ]
    ):
        return "uds", 0.85

    # Amazon docs: explicit business rules/policies/requirements/fee definitions.
    if any(
        k in q_lower
        for k in [
            "policy",
            "policies",
            "requirements",
            "business rule",
            "business rules",
            "what does amazon",
            "fees",
            "fee structure",
            "fba removal",
            "oversize fee",
            "storage fee",
            "referral fee",
            "guidelines",
        ]
    ):
        return "amazon_docs", 0.9

    # Amazon docs: product / seller / API documentation questions.
    if any(k in q_lower for k in ["amazon docs", "seller central", "sp-api docs", "aws docs"]):
        return "amazon_docs", 0.9

    # IC docs: internal project / framework documentation.
    if any(k in q_lower for k in ["ic-rag-agent", "ic docs", "framework.md", "project.md"]):
        return "ic_docs", 0.9

    # SP-API workflow: order, shipment, catalog, finance focused on Amazon API.
    if any(
        k in q_lower
        for k in [
            "sp-api",
            "spapi",
            "amazon order",
            "fba",
            "shipment",
            "catalog",
            "seller api",
        ]
    ):
        return "sp_api", 0.85

    # UDS workflow: additional analytical fallback cues.
    if any(
        k in q_lower
        for k in [
            "sales",
            "revenue",
            "orders",
            "orders sold",
            "table",
            "dataset",
            "clickhouse",
            "uds",
        ]
    ):
        return "uds", 0.85

    # Fallback: general LLM workflow (no keyword match).
    return "general", 0.7


def route_workflow(
    query: str, request: QueryRequest
) -> Tuple[str, float, str, Optional[str], Optional[float]]:
    """
    Choose workflow and routing confidence based on query and request.

    Behavior:
    - If client sets workflow != "auto": return that workflow with confidence 1.0.
    - If workflow == "auto":
      - If GATEWAY_ROUTE_LLM_ENABLED is false: use heuristic keyword rules only.
      - If enabled: call Route LLM (backend from request.route_backend or env);
        if confidence >= GATEWAY_ROUTE_LLM_CONF_THRESHOLD use LLM result,
        else fall back to heuristic rules.

    Args:
        query: Rewritten query text from rewrite_query.
        request: Original QueryRequest (workflow, route_backend, etc.).

    Returns:
        (workflow, routing_confidence) tuple.
    """
    explicit = (request.workflow or "auto").strip().lower()
    if explicit != "auto":
        # explicit workflow is considered a manual override
        return explicit, 1.0, "manual", None, None

    # Auto routing: either Route LLM (when enabled and confident) or heuristic.
    backend = (request.route_backend or "").strip() or _route_llm_backend()
    threshold = _route_llm_conf_threshold()

    if not _route_llm_enabled():
        wf, conf = _route_workflow_heuristic(query or "")
        wf = _apply_docs_preference(query or "", wf)
        return wf, conf, "heuristic", None, None

    # Route LLM path: call LLM; fall back to heuristic if confidence too low.
    from .route_llm import route_with_llm

    workflow, confidence = route_with_llm(query or "", backend)
    workflow = _apply_docs_preference(query or "", workflow)
    if confidence >= threshold:
        logger.debug(
            "Route LLM selected workflow=%s confidence=%.2f (>= %.2f)",
            workflow,
            confidence,
            threshold,
        )
        return workflow, confidence, "llm", backend, confidence

    # Below threshold or LLM returned safe default: use heuristic.
    logger.debug(
        "Route LLM confidence %.2f < %.2f; using heuristic fallback",
        confidence,
        threshold,
    )
    wf2, conf2 = _route_workflow_heuristic(query or "")
    wf2 = _apply_docs_preference(query or "", wf2)
    return wf2, conf2, "heuristic", None, None


__all__ = ["rewrite_query", "route_workflow", "build_execution_plan"]

