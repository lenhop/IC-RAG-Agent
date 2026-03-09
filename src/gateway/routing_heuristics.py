"""
Shared routing heuristics for workflow selection.

Extracted from router for use by both router (route_workflow) and dispatcher
(build_execution_plan). Avoids circular imports and centralizes keyword-based
routing logic.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def normalize_query(text: str) -> str:
    """Trim and collapse whitespace. Always applied before any routing."""
    text = text or ""
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


def apply_docs_preference(query: str, workflow: str) -> str:
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


def split_multi_intent_clauses(query: str) -> List[str]:
    """Split a potentially mixed query into normalized clause candidates."""
    text = normalize_query(query)
    if not text:
        return []
    # First try granular split on question-starter patterns.
    granular_raw = re.split(
        r" (?=how many |what is |what's |what does |which |what table |show me |get |get order |check )",
        text,
        flags=re.I,
    )
    granular = []
    for p in granular_raw:
        trimmed = p.removesuffix(" and") if p.endswith(" and") else p
        # Strip trailing comma/semicolon left by question-starter split.
        trimmed = re.sub(r"[,;]+\s*$", "", trimmed)
        norm = normalize_query(trimmed)
        if norm:
            granular.append(norm)
    if len(granular) >= 2:
        parts = granular
    else:
        # Fallback: split by comma, semicolon, or " and " before question words.
        # Do NOT split on comma when part of date (e.g. "September 1st, 2026").
        parts = re.split(
            r",\s*(?!\s*\d{4}\b)|;\s*|\s+and\s+(?=how|what|which|when|show|get|check)",
            text,
            flags=re.I,
        )
    clauses = []
    seen = set()
    for raw in parts:
        clause = normalize_query(raw)
        if not clause:
            continue
        key = clause.lower()
        if key in seen:
            continue
        seen.add(key)
        clauses.append(clause)
    return clauses


def route_workflow_heuristic(query: str) -> Tuple[str, float]:
    """
    Rule-based workflow selection from query keywords.

    Used when workflow is "auto" and either Route LLM is disabled or
    LLM confidence is below threshold. Returns (workflow, confidence).
    """
    q_lower = (query or "").lower()

    # SP-API workflow: real-time operations.
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

    # UDS workflow: ASIN-level data retrieval.
    if any(
        k in q_lower
        for k in [
            "financial summary for asin",
            "financial summary for",
            "get financial summary",
        ]
    ):
        return "uds", 0.92

    # UDS workflow: high-confidence analytical aggregations.
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

    # UDS workflow: strong table/schema cues.
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

    # SP-API workflow: order, shipment, catalog, finance.
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


__all__ = [
    "normalize_query",
    "apply_docs_preference",
    "split_multi_intent_clauses",
    "route_workflow_heuristic",
]
