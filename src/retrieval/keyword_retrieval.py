"""
Keyword Retrieval — Public API Module (公共接口模块)

Architecture (强制):
  keyword_retrieval.py = 公共接口模块（Layer 1 唯一入口），负责关键字/正则意图匹配
  __init__.py         = 包入口，从本模块 re-export 公共 API

  下游模块（gateway、classification 等）应通过本模块或 src.retrieval 包导入，
  禁止直接依赖内部实现（_ 前缀方法与模块级 _* 常量）。

Workflow:
  Query → 加载/编译规则 → match(query) → 首个命中返回 KeywordMatchResult，否则 None

Internal (not exported):
  _SRC_ROOT, _DEFAULT_RULES_PATH     — 规则路径常量
  _load_and_compile_rules            — 加载并编译规则
  _default_sp_api_uds_rules          — 内置 SP-API/UDS 规则（正则）

Public API (exported via __init__.py):
  KeywordMatchResult                  — 匹配结果数据类（workflow, intent_name, confidence, source）
  KeywordRetrieval                    — 主类；match(query) → Optional[KeywordMatchResult]
  keyword_retrieve(query)             — 一次性便捷函数（测试/脚本用）
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Default rule source (can be overridden by env or constructor)
_SRC_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RULES_PATH = _SRC_ROOT / "gateway" / "route_llm" / "classification"
_AMAZON_INTENTS_CSV = _DEFAULT_RULES_PATH / "amazon_intents.csv"

# Workflow order for parallel detection: sp_api -> uds -> amazon_docs
_WORKFLOW_ORDER = ("sp_api", "uds", "amazon_docs")


# ---------------------------------------------------------------------------
# Public API — 公共接口（唯一对外入口，由 __init__.py re-export）
#
# 下游业务模块（gateway、classification、dispatcher 等）必须通过这些类/函数访问
# Layer 1 关键字检索能力。
# ---------------------------------------------------------------------------

@dataclass
class KeywordMatchResult:
    """Result of a single keyword/regex match (part of public API)."""

    workflow: str
    intent_name: str
    confidence: str = "high"
    source: str = "keyword"


class KeywordRetrieval:
    """
    Keyword and regex-based intent matcher for strong-rule workflows (SP-API, UDS).

    Primary public entry point: ``match(query)``.

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
        self._amazon_keywords_cache: Optional[List[str]] = None
        self._load_and_compile_rules()

    def _load_and_compile_rules(self) -> None:
        """Load rule config and compile regex/keyword lists. Ensures logic cohesion."""
        all_rules: List[tuple] = []
        for wf in _WORKFLOW_ORDER:
            all_rules.extend(self._get_rules_for_workflow(wf))
        self._compiled = all_rules

    def _get_rules_for_workflow(self, workflow: str) -> List[tuple]:
        """Return rules for a single workflow. Used by detect_parallel for per-workflow checks."""
        if workflow == "sp_api":
            order_id_re = re.compile(r"\d{3}-\d{7}-\d{7}", re.IGNORECASE)
            latest_re = re.compile(r"\b(latest|current|real-time|最新|当前)\b", re.IGNORECASE)
            return [
                (order_id_re, "sp_api", "get_order_status"),
                (latest_re, "sp_api", "realtime_check"),
            ]
        if workflow == "uds":
            uds_order_re = re.compile(r"\b(order|orders|订单)\b", re.IGNORECASE)
            uds_inv_re = re.compile(r"\b(inventory|stock|库存)\b", re.IGNORECASE)
            return [
                (uds_order_re, "uds", "order_history"),
                (uds_inv_re, "uds", "inventory"),
            ]
        if workflow == "amazon_docs":
            keywords = self._load_amazon_docs_keywords()
            return [(kw, "amazon_docs", "amazon_business") for kw in keywords]
        return []

    def _load_amazon_docs_keywords(self) -> List[str]:
        """Load Amazon Business keywords from amazon_intents.csv (skip header). Cached."""
        if self._amazon_keywords_cache is not None:
            return self._amazon_keywords_cache
        if not _AMAZON_INTENTS_CSV.is_file():
            self._amazon_keywords_cache = []
            return []
        try:
            lines = _AMAZON_INTENTS_CSV.read_text(encoding="utf-8").splitlines()
            # First line is header "keyword"
            self._amazon_keywords_cache = [line.strip() for line in lines[1:] if line.strip()]
            return self._amazon_keywords_cache
        except Exception:
            self._amazon_keywords_cache = []
            return []

    def _match_workflow(self, query: str, workflow: str) -> Optional[KeywordMatchResult]:
        """Check if query matches rules for a single workflow. Used by detect_parallel."""
        if not query or not query.strip():
            return None
        q = query.strip()
        rules = self._get_rules_for_workflow(workflow)
        for rule in rules:
            pattern_or_str, wf, intent_name = rule
            if isinstance(pattern_or_str, re.Pattern):
                if pattern_or_str.search(q):
                    return KeywordMatchResult(
                        workflow=wf,
                        intent_name=intent_name,
                        confidence="high",
                        source="keyword",
                    )
            elif isinstance(pattern_or_str, str):
                if pattern_or_str.lower() in q.lower():
                    return KeywordMatchResult(
                        workflow=wf,
                        intent_name=intent_name,
                        confidence="high",
                        source="keyword",
                    )
        return None

    def detect_parallel(self, query: str) -> Optional[KeywordMatchResult]:
        """
        Run parallel detection for sp_api, uds, amazon_docs. Return first non-None in order.

        Used by ClassificationImplementMethod.keyword_classification_method.
        """
        if not query or not query.strip():
            return None
        step_results: Dict[str, Optional[KeywordMatchResult]] = {}

        def _run_step(wf: str) -> None:
            step_results[wf] = self._match_workflow(query, wf)

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(_run_step, _WORKFLOW_ORDER))

        for wf in _WORKFLOW_ORDER:
            r = step_results.get(wf)
            if r is not None:
                return r
        return None

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
        for pattern_or_str, workflow, intent_name in self._compiled:
            matched = False
            if isinstance(pattern_or_str, re.Pattern):
                matched = bool(pattern_or_str.search(q))
            elif isinstance(pattern_or_str, str):
                matched = pattern_or_str.lower() in q.lower()
            if matched:
                return KeywordMatchResult(
                    workflow=workflow,
                    intent_name=intent_name,
                    confidence="high",
                    source="keyword",
                )
        return None


def keyword_retrieve(query: str) -> Optional[KeywordMatchResult]:
    """
    Optional one-shot helper (tests / scripts). Prefer ``KeywordRetrieval`` for DI and tests.
    """
    return KeywordRetrieval().match(query)
