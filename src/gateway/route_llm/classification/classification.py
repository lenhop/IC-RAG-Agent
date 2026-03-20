"""
Intent Classification — Public API Module (公共接口模块)

Architecture (强制):
  classification.py = 对外门面：公共函数 + _IntentValidator；委托实现到 implement_methods
  implement_methods.py = 实现层：IntentSplitMethod、ClassificationImplementMethod、IntentResult、ClassificationIntentVectorStore
  __init__.py       = 包入口，从本模块 re-export 公共 API

  下游（dispatcher, api 等）通过本包入口访问；禁止依赖 implement_methods 内部私有符号。

Workflow (per Intent-Classification-Workflow diagram):
  Rewritten Query → Split Intents → per-intent: keyword → vector → LLM → Merge → Dispatcher

Internal (in this module, not exported):
  _IntentValidator — 必填字段校验与追问

Implementation detail (implement_methods):
  IntentResult；ClassificationIntentVectorStore.llm_detect — LLM 并行三 workflow；amazon_intents.csv 示例注入

Public API (exported via __init__.py):
  split_intents, classify_intent, classify_intents_batch, validate_intents, IntentResult
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .implement_methods import (
    ClassificationImplementMethod,
    IntentResult,
    IntentSplitMethod,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部实现 — 门面层仅保留校验器
# ---------------------------------------------------------------------------


class _IntentValidator:
    """必填字段校验器：检查意图所需字段是否存在，缺失时生成追问文本。"""

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

    @classmethod
    def validate(
        cls,
        intents_with_meta: List[Dict[str, object]],
        conversation_context: Optional[str] = None,
    ) -> Optional[str]:
        """校验已分类意图的必填字段，缺失时生成追问文本。"""
        missing_by_intent: List[Tuple[str, str, List[str]]] = []
        for item in intents_with_meta:
            query = item.get("query", "")
            intent_name = item.get("intent_name", "")
            required_fields = item.get("required_fields") or []
            clarification_template = item.get("clarification_template", "")

            missing = cls._check_required_fields(query, required_fields, conversation_context)
            if missing:
                missing_by_intent.append((intent_name, clarification_template, missing))

        return cls._build_clarification(missing_by_intent)

    @classmethod
    def _check_required_fields(
        cls,
        query: str,
        required_fields: List[str],
        conversation_context: Optional[str] = None,
    ) -> List[str]:
        """检查查询中是否包含所需字段。"""
        if not required_fields:
            return []
        search_text = query or ""
        if conversation_context and conversation_context.strip():
            search_text = f"{search_text}\n{conversation_context}"
        return [
            field_name
            for field_name in required_fields
            if not cls._field_present(field_name, search_text)
        ]

    @classmethod
    def _build_clarification(
        cls,
        missing_by_intent: List[Tuple[str, str, List[str]]],
    ) -> Optional[str]:
        """根据缺失字段生成追问文本。"""
        items = [
            (intent_name, template, fields)
            for intent_name, template, fields in missing_by_intent
            if fields
        ]
        if not items:
            return None
        if len(items) == 1:
            _, template, _ = items[0]
            return template or "Please provide more details."

        lines = ["I need a few more details:"]
        for intent_name, template, _ in items:
            label = intent_name.replace("_", " ").title()
            question = template or "Please provide more details."
            lines.append(f"- For {label}: {question}")
        return "\n".join(lines)

    @classmethod
    def _field_present(cls, field_name: str, text: str) -> bool:
        pattern = cls._FIELD_PATTERNS.get(field_name)
        if pattern is None:
            return True
        return bool(pattern.search(text or ""))


# ---------------------------------------------------------------------------
# Public API — 委托 implement_methods
# ---------------------------------------------------------------------------


def split_intents(query: str, conversation_context: Optional[str] = None) -> List[str]:
    """将重写后的查询拆分为单意图子问题列表。"""
    return IntentSplitMethod.split(query, conversation_context=conversation_context)


def classify_intent(
    query: str,
    conversation_context: Optional[str] = None,
) -> IntentResult:
    """对单条查询执行 keyword → vector → LLM 串行短路分类。"""
    return ClassificationImplementMethod().detect(
        query, conversation_context=conversation_context
    )


_INTENT_CLASSIFY_PARALLEL_CHUNK = 3


def classify_intents_batch(
    intents: List[str],
    conversation_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对拆分后的意图列表分类；子句间按批并行（每批最多 3 条）。"""
    pairs: List[Tuple[int, str]] = [
        (i, (intent or "").strip())
        for i, intent in enumerate(intents)
        if (intent or "").strip()
    ]
    results: List[Dict[str, Any]] = []

    def _classify_one(pair: Tuple[int, str]) -> Tuple[int, Dict[str, Any]]:
        i, q = pair
        intent_start = time.perf_counter()
        classification = ClassificationImplementMethod().detect(
            q, conversation_context=conversation_context
        )
        intent_elapsed_ms = int((time.perf_counter() - intent_start) * 1000)
        q_preview = (q[:60] + "…") if len(q) > 60 else q
        logger.info(
            "[Perf] intent_classification intent[%d] %r -> %s: %d ms",
            i,
            q_preview,
            classification.workflow,
            intent_elapsed_ms,
        )
        item: Dict[str, Any] = {
            "query": q,
            "workflow": classification.workflow,
            "intent_name": classification.intent_name,
            "confidence": classification.confidence,
            "source": classification.source,
            "required_fields": classification.required_fields,
            "clarification_template": classification.clarification_template,
        }
        if classification.intent_elapsed_ms is not None:
            item["intent_elapsed_ms"] = classification.intent_elapsed_ms
        if classification.step_timings:
            item["step_timings"] = classification.step_timings
        return i, item

    for batch_start in range(0, len(pairs), _INTENT_CLASSIFY_PARALLEL_CHUNK):
        chunk = pairs[batch_start : batch_start + _INTENT_CLASSIFY_PARALLEL_CHUNK]
        chunk_out: List[Tuple[int, Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=_INTENT_CLASSIFY_PARALLEL_CHUNK) as executor:
            futures = [executor.submit(_classify_one, p) for p in chunk]
            for fut in futures:
                try:
                    chunk_out.append(fut.result())
                except Exception as exc:
                    logger.warning("Intent batch classify failed: %s", exc)
        chunk_out.sort(key=lambda x: x[0])
        for _, item in chunk_out:
            results.append(item)

    return results


def validate_intents(
    intents_with_meta: List[Dict[str, object]],
    conversation_context: Optional[str] = None,
) -> Optional[str]:
    """校验已分类意图的必填字段，缺失时生成追问文本。"""
    return _IntentValidator.validate(intents_with_meta, conversation_context)
