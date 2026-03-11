"""
Phase 5: Per-intent required field validation for multi-intent clarification.

After intent classification resolves each sub-query to an intent, this module
checks whether the required fields for that intent are present in the query
(or conversation context). If any are missing, it returns a clarification
question aggregated across all sub-queries.

Required field patterns (deterministic, no LLM needed):
  - order_id:       \d{3}-\d{7}-\d{7}
  - asin_or_sku:    B0[0-9A-Z]{8} or SKU keyword
  - date_range:     year, month name, last month/quarter/year, Q1-Q4
  - report_type:    settlement, removal, financial (keyword match)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ── Field presence patterns ────────────────────────────────────────────────

_FIELD_PATTERNS: Dict[str, re.Pattern] = {
    "order_id": re.compile(
        r"\d{3}-\d{7}-\d{7}|order\s*id|order\s*number",
        re.IGNORECASE,
    ),
    "asin_or_sku": re.compile(
        r"B0[0-9A-Z]{8}|ASIN\s+\w+|\bSKU\b|\bASIN\b",
        re.IGNORECASE,
    ),
    "date_range": re.compile(
        r"\b(last\s+(month|quarter|year|week|30\s*days|6\s*months))"
        r"|\bQ[1-4]\b"
        r"|\b20\d{2}\b"
        r"|\b(january|february|march|april|may|june|july|august"
        r"|september|october|november|december)\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\bthis\s+(month|year|quarter)\b"
        r"|\byesterday\b|\btoday\b",
        re.IGNORECASE,
    ),
    "report_type": re.compile(
        r"\b(settlement|removal|financial|fee|sales|inventory)\s*(report|summary|data)?\b",
        re.IGNORECASE,
    ),
}


def _field_present(field: str, text: str) -> bool:
    """Return True if the required field is detectable in text."""
    pattern = _FIELD_PATTERNS.get(field)
    if pattern is None:
        # Unknown field — assume present to avoid false clarifications.
        return True
    return bool(pattern.search(text or ""))


def check_required_fields(
    query: str,
    required_fields: List[str],
    conversation_context: Optional[str] = None,
) -> List[str]:
    """
    Return list of required fields that are missing from query + context.

    Args:
        query: The sub-query text for this intent.
        required_fields: List of field names from intent registry.
        conversation_context: Optional prior conversation turns (may contain the field).

    Returns:
        List of missing field names. Empty list means all fields are present.
    """
    if not required_fields:
        return []

    # Search both the query and conversation context.
    search_text = query or ""
    if conversation_context and conversation_context.strip():
        search_text = f"{search_text}\n{conversation_context}"

    return [
        field for field in required_fields
        if not _field_present(field, search_text)
    ]


def build_clarification_for_missing_fields(
    missing_by_intent: List[Tuple[str, str, List[str]]],
) -> Optional[str]:
    """
    Build a combined clarification question for multiple intents with missing fields.

    Args:
        missing_by_intent: List of (intent_name, clarification_template, missing_fields).
            clarification_template: from intent registry (human-readable question).
            missing_fields: list of field names that are missing.

    Returns:
        Combined clarification string, or None if nothing is missing.
    """
    items = [
        (name, template, fields)
        for name, template, fields in missing_by_intent
        if fields
    ]
    if not items:
        return None

    if len(items) == 1:
        _, template, _ = items[0]
        return template or "Please provide more details."

    # Multiple intents with missing fields — combine into one message.
    lines = ["I need a few more details:"]
    for intent_name, template, _ in items:
        label = intent_name.replace("_", " ").title()
        question = template or "Please provide more details."
        lines.append(f"- For {label}: {question}")
    return "\n".join(lines)


def validate_intents(
    intents_with_meta: List[Dict],
    conversation_context: Optional[str] = None,
) -> Optional[str]:
    """
    Validate required fields for a list of classified intents.

    Args:
        intents_with_meta: List of dicts, each with:
            - query: str (the sub-query text)
            - intent_name: str
            - required_fields: List[str]
            - clarification_template: str
        conversation_context: Optional prior conversation turns.

    Returns:
        Clarification question string if any required fields are missing,
        or None if all intents are complete.
    """
    missing_by_intent: List[Tuple[str, str, List[str]]] = []

    for item in intents_with_meta:
        query = item.get("query", "")
        intent_name = item.get("intent_name", "")
        required_fields = item.get("required_fields") or []
        clarification_template = item.get("clarification_template", "")

        missing = check_required_fields(query, required_fields, conversation_context)
        if missing:
            missing_by_intent.append((intent_name, clarification_template, missing))

    return build_clarification_for_missing_fields(missing_by_intent)
