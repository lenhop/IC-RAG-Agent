"""
Keyword retrieval for intent classification (Layer 1).

Matches query against rules/keywords for strong-rule intents (e.g. SP-API, UDS).
Encapsulates config loading, rule compilation, and match execution in one class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Default rule source (can be overridden by env or constructor)
_SRC_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RULES_PATH = _SRC_ROOT / "gateway" / "route_llm" / "classification"


@dataclass
class KeywordMatchResult:
    """Result of a single keyword/regex match."""

    workflow: str
    intent_name: str
    confidence: str = "high"
    source: str = "keyword"


class KeywordRetrieval:
    """
    Keyword and regex-based intent matcher for strong-rule workflows (SP-API, UDS).

    Loads rules from config, compiles patterns, and returns the first matching
    workflow + intent. Used as Layer 1 before vector retrieval and LLM.
    """

    def __init__(
        self,
        rules_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            rules_path: Directory or file for rule config. If None, uses default.
        """
        self._rules_path = rules_path or _DEFAULT_RULES_PATH
        self._compiled: List[tuple] = []  # (pattern, workflow, intent_name)
        self._load_and_compile_rules()

    def _load_and_compile_rules(self) -> None:
        """Load rule config and compile regex/keyword lists. Ensures logic cohesion."""
        # Placeholder: no external file dependency by default; add CSV/YAML later.
        self._compiled = self._default_sp_api_uds_rules()

    def _default_sp_api_uds_rules(self) -> List[tuple]:
        """Default in-code rules for SP-API and UDS (order: SP-API first, then UDS)."""
        rules: List[tuple] = []
        # SP-API: order ID pattern + "latest/current" intent
        order_id_re = re.compile(r"\d{3}-\d{7}-\d{7}", re.IGNORECASE)
        latest_re = re.compile(r"\b(latest|current|real-time|最新|当前)\b", re.IGNORECASE)
        # UDS: order/product/inventory without "latest"
        uds_order_re = re.compile(r"\b(order|orders|订单)\b", re.IGNORECASE)
        uds_inv_re = re.compile(r"\b(inventory|stock|库存)\b", re.IGNORECASE)
        # Store as (pattern_or_func, workflow, intent_name); simplified as (re.Pattern, str, str)
        rules.append((order_id_re, "sp_api", "get_order_status"))
        rules.append((latest_re, "sp_api", "realtime_check"))
        rules.append((uds_order_re, "uds", "order_history"))
        rules.append((uds_inv_re, "uds", "inventory"))
        return rules

    def match(self, query: str) -> Optional[KeywordMatchResult]:
        """
        Run keyword/regex match on query. Returns first match or None.

        Args:
            query: User query text (e.g. rewritten query or intent clause).

        Returns:
            KeywordMatchResult if any rule matches, else None.
        """
        if not query or not query.strip():
            return None
        q = query.strip()
        for pattern, workflow, intent_name in self._compiled:
            if isinstance(pattern, re.Pattern) and pattern.search(q):
                return KeywordMatchResult(
                    workflow=workflow,
                    intent_name=intent_name,
                    confidence="high",
                    source="keyword",
                )
        return None


# Convenience: module-level instance for tests
def keyword_retrieve(query: str) -> Optional[KeywordMatchResult]:
    """One-shot keyword retrieval using default config."""
    return KeywordRetrieval().match(query)
