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

from .rewriters import parse_rewrite_plan_text, planner_rewrite_enabled
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
    # Keep simple deterministic splitting for comma/semicolon mixed queries.
    parts = re.split(r"[,;]\s*", text)
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
       - If backend in ("ollama", "deepseek"): call rewriter; return result or normalized on failure.
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

    if backend == "ollama":
        from .rewriters import rewrite_with_ollama
        return rewrite_with_ollama(normalized)
    if backend == "deepseek":
        from .rewriters import rewrite_with_deepseek
        return rewrite_with_deepseek(normalized)

    return normalized


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
        if parsed_plan and parsed_plan.task_groups:
            return _correct_plan_workflows(parsed_plan)

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
        ]
    ):
        return "sp_api", 0.92

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

    # UDS workflow (analytical/table/schema cues).
    if any(
        k in q_lower
        for k in [
            "which table",
            "what table",
            "table stores",
            "what columns",
            "column",
            "schema",
            "dataset",
            "clickhouse",
            "uds",
            "last month",
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

