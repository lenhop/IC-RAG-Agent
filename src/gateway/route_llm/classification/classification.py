"""
Intent Classification — Public API Module (公共接口模块)

Architecture (强制):
  classification.py = 公共接口模块（唯一统一公共接口），负责全部意图分类逻辑
  __init__.py       = 包入口，从本模块 re-export 公共 API

  所有下游业务模块（dispatcher, intent_rewrite, api 等）禁止直接导入内部类，
  必须通过本模块 / 包入口完成所有意图分类相关操作。

Workflow (per Intent-Classification-Workflow diagram):
  Rewritten Query
    → Split Intents（拆分多意图）
    → Structure Intention List
    → Loop through intent list:
        1. SP-API Intent?   → Yes → sp_api
        2. UDS Intent?      → Yes → uds
        3. Amazon Business?  → Yes → amazon_docs
        4. All No           → general
    → Merge Intent Results
    → Delivery to Dispatcher

Approach:
  每个意图判定步骤使用独立的 prompt（sp_api_prompts.md, uds_prompts.md,
  amazon_prompts.md），将关键词和示例句子写在 prompt 中，喂给 LLM 做判断。
  按照流程图顺序依次检测：先 SP-API，再 UDS，再 Amazon Business，都不匹配
  则标记为 General。

Internal classes (not exported):
  _RuntimeConfig     — 运行时配置（LLM 后端选择等）
  _LLMIntentDetector — 基于 prompt + LLM 的并行意图检测器（三 workflow）
  _IntentValidator   — 必填字段校验 & 追问生成

Intent split 实现位于 implement_methods.IntentSplitMethod（由 split_intents 委托调用）。

Public API (exported via __init__.py):
  split_intents(query)                          → List[str]
  classify_intent(query, context)               → IntentResult
  classify_intents_batch(intents, context)       → List[Dict[str, Any]]
  validate_intents(intents_with_meta, context)   → Optional[str]
  IntentResult                                   — 分类结果数据类
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...prompt_loader import load_prompt
from src.logger import get_logger_facade
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient, get_ollama_config

logger = logging.getLogger(__name__)
_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


# ---------------------------------------------------------------------------
# Amazon intent examples loader（从 CSV 文件动态加载，缓存在模块级别）
# ---------------------------------------------------------------------------

_amazon_examples_cache: str | None = None


def _load_amazon_intent_examples() -> str:
    """从 amazon_intents.csv 加载关键词列表，返回 markdown bullet 列表字符串。

    首行为 header（跳过），每行一个关键词，缓存在模块级变量避免重复读取。
    """
    global _amazon_examples_cache
    if _amazon_examples_cache is not None:
        return _amazon_examples_cache

    csv_path = Path(__file__).parent / "amazon_intents.csv"
    try:
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        # 首行为 header，跳过；过滤空行
        keywords = [line.strip() for line in lines[1:] if line.strip()]
        _amazon_examples_cache = "\n".join(f"- {kw}" for kw in keywords)
    except Exception as exc:
        logger.warning("Failed to load amazon_intents.csv: %s; using empty examples", exc)
        _amazon_examples_cache = "(no examples available)"

    return _amazon_examples_cache


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """意图分类结果数据类。

    Attributes:
        intent_name: 意图名称（如 'get_order_status'），由 LLM 生成。
        workflow: 路由的目标工作流（'sp_api', 'uds', 'amazon_docs', 'general'）。
        confidence: 置信度等级（'high', 'medium', 'low'）。
        source: 匹配来源（'llm', 'fallback'）。
        required_fields: 该意图需要的必填字段列表。
        clarification_template: 缺少必填字段时的追问模板。
        intent_elapsed_ms: 该意图分类总耗时（毫秒），用于 UI/log 展示。
        step_timings: 每个 detection step 的耗时列表，每项 {step, workflow, ms}。
    """

    intent_name: str
    workflow: str
    confidence: str = "medium"
    source: str = "llm"
    required_fields: List[str] = field(default_factory=list)
    clarification_template: str = ""
    intent_elapsed_ms: Optional[int] = None
    step_timings: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# 内部实现类（不对外导出，下游模块禁止直接使用）
# ---------------------------------------------------------------------------


# ── 意图检测步骤定义（按流程图顺序：SP-API → UDS → Amazon Business） ──
# 每一步对应一个 prompt 文件和一个目标 workflow。

_DETECTION_STEPS: List[Dict[str, str]] = [
    {"prompt_name": "classification/sp_api_prompts", "workflow": "sp_api"},
    {"prompt_name": "classification/uds_prompts", "workflow": "uds"},
    {"prompt_name": "classification/amazon_prompts", "workflow": "amazon_docs"},
]


class _RuntimeConfig:
    """运行时配置：从环境变量读取 LLM 后端选择等。"""

    @staticmethod
    def get_intent_split_backend() -> str:
        """返回意图拆分使用的 LLM 后端（'ollama' 或 'deepseek'）。"""
        return (os.getenv("GATEWAY_INTENT_SPLIT_BACKEND") or "ollama").strip().lower()

    @staticmethod
    def get_intent_detect_backend() -> str:
        """返回意图检测使用的 LLM 后端（'ollama' 或 'deepseek'）。"""
        return (os.getenv("GATEWAY_INTENT_DETECT_BACKEND") or "ollama").strip().lower()


class _LLMIntentDetector:
    """基于 prompt + LLM 的并行意图检测器。

    按照流程图顺序并行执行三个检测步骤：
      1. SP-API Intent?   → sp_api_prompts.md
      2. UDS Intent?      → uds_prompts.md
      3. Amazon Business?  → amazon_prompts.md
      4. 全部不匹配      → general

    三个步骤并行执行，完成后按 sp_api -> uds -> amazon_docs 顺序取第一个 Yes。
    若全是 No，返回 general。
    """

    @classmethod
    def detect(cls, query: str, conversation_context: Optional[str] = None) -> IntentResult:
        """对单条查询并行检测意图类型，按 sp_api -> uds -> amazon_docs 取第一个 Yes。

        Args:
            query: 待分类的子查询。
            conversation_context: 可选的对话历史，注入到 prompt {history} 占位符。

        Returns:
            IntentResult，包含 workflow 和 intent_name。
        """
        if not query or not query.strip():
            return IntentResult(
                intent_name="general",
                workflow="general",
                confidence="low",
                source="fallback",
            )

        stripped = query.strip()
        step_timings: List[Dict[str, Any]] = []
        step_results: List[Tuple[Optional[IntentResult], int]] = []

        # ── 并行执行三个检测步骤 ──
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    cls._run_detection_step,
                    query=stripped,
                    prompt_name=step["prompt_name"],
                    target_workflow=step["workflow"],
                    conversation_context=conversation_context,
                )
                for step in _DETECTION_STEPS
            ]
            for step, future in zip(_DETECTION_STEPS, futures):
                try:
                    result, step_ms = future.result()
                    step_timings.append({
                        "step": step["prompt_name"],
                        "workflow": step["workflow"],
                        "ms": step_ms,
                    })
                    step_results.append((result, step_ms))
                except Exception as exc:
                    logger.warning(
                        "Intent detection step '%s' failed: %s; treating as None",
                        step["prompt_name"],
                        exc,
                    )
                    step_timings.append({
                        "step": step["prompt_name"],
                        "workflow": step["workflow"],
                        "ms": 0,
                    })
                    step_results.append((None, 0))

        # ── 按 sp_api -> uds -> amazon_docs 取第一个 Yes ──
        total_elapsed_ms = max((r[1] for r in step_results), default=0)
        for result, _ in step_results:
            if result is not None:
                result.intent_elapsed_ms = total_elapsed_ms
                result.step_timings = step_timings
                cls._log_detection_result(stripped, result)
                return result

        # ── 全部不匹配 → Mark as General ──
        general_result = IntentResult(
            intent_name="general",
            workflow="general",
            confidence="low",
            source="llm",
            intent_elapsed_ms=total_elapsed_ms,
            step_timings=step_timings if step_timings else None,
        )
        cls._log_detection_result(stripped, general_result)
        return general_result

    @classmethod
    def _run_detection_step(
        cls,
        query: str,
        prompt_name: str,
        target_workflow: str,
        conversation_context: Optional[str] = None,
    ) -> Tuple[Optional[IntentResult], int]:
        """执行单个检测步骤：加载 prompt → 替换 {history}/{query} → 调用 LLM → 解析结果。

        Args:
            query: 单条 intent clause（split_intents 切分后的子句）。
            prompt_name: prompt 文件路径（如 'classification/sp_api_prompts'）。
            target_workflow: 匹配成功时的 workflow 名称。
            conversation_context: 可选的对话历史文本，注入到 prompt {history} 占位符。

        Returns:
            (IntentResult if matched else None, step_elapsed_ms)
        """
        prompt_template = load_prompt(prompt_name)
        history_text = (conversation_context or "").strip() or "(no conversation history)"
        prompt_text = (
            prompt_template
            .replace("{history}", history_text)
            .replace("{query}", query)
        )
        # amazon_docs prompt 含 {examples} 占位符，从 CSV 动态注入
        if "{examples}" in prompt_text:
            prompt_text = prompt_text.replace("{examples}", _load_amazon_intent_examples())

        # ── 调用 LLM ──
        step_start = time.perf_counter()
        response_text = cls._call_llm(prompt_text)
        step_elapsed_ms = int((time.perf_counter() - step_start) * 1000)
        query_preview = (query[:50] + "…") if len(query) > 50 else query
        logger.info(
            "[Perf] intent_classification step %s (workflow=%s) query=%r: %d ms",
            prompt_name,
            target_workflow,
            query_preview,
            step_elapsed_ms,
        )
        if not response_text:
            return None, step_elapsed_ms

        # ── 解析 JSON 响应 ──
        parsed = cls._parse_llm_response(response_text)
        if parsed is None:
            return None, step_elapsed_ms

        match_value = parsed.get("match")
        # Support legacy {"result": "Yes"} format from prompts not yet updated
        if match_value is None:
            result_str = str(parsed.get("result", "")).strip().lower()
            match_value = result_str in ("yes", "true", "1")
        if not match_value:
            return None, step_elapsed_ms

        # ── 匹配成功：提取 confidence（intent_name 由 target_workflow 决定）──
        intent_name = target_workflow
        confidence = parsed.get("confidence", "medium") or "medium"

        return IntentResult(
            intent_name=intent_name,
            workflow=target_workflow,
            confidence=confidence,
            source="llm",
        ), step_elapsed_ms

    @classmethod
    def _call_llm(cls, prompt: str) -> str:
        """调用 LLM 后端（Ollama 或 DeepSeek）执行 prompt。"""
        backend = _RuntimeConfig.get_intent_detect_backend()

        if backend == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            try:
                return DeepSeekChat().complete(
                    system_prompt="You are an intent classifier. Output ONLY valid JSON.",
                    user_content=prompt,
                    max_tokens=256,
                )
            except Exception as exc:
                logger.warning("Intent detect DeepSeek failed: %s; trying Ollama", exc)

        # Ollama fallback / default
        try:
            return OllamaClient().generate(prompt, empty_fallback="")
        except Exception as exc:
            logger.warning("Intent detect Ollama failed: %s", exc)
            return ""

    @staticmethod
    def _parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON（容错处理 markdown fences）。"""
        raw = text.strip()
        # 去除 markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                raw = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(raw)
        except ValueError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except ValueError:
                    pass
            logger.debug("Intent detect JSON parse failed: %s", raw[:200])
            return None

    @staticmethod
    def _log_detection_result(query: str, result: IntentResult) -> None:
        """记录分类结果到网关日志系统。"""
        if not _gateway_logger:
            return
        try:
            _gateway_logger.log_runtime(
                event_name="intent_classification_resolved",
                stage="intent_classification",
                message="classify_intent resolved",
                status="success",
                workflow=result.workflow,
                query_raw=query,
                metadata={
                    "intent_name": result.intent_name,
                    "source": result.source,
                    "confidence": result.confidence,
                },
            )
        except Exception:
            pass


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
# Public API — 公共接口（唯一对外入口，由 __init__.py re-export）
#
# 下游业务模块（dispatcher, intent_rewrite, api 等）必须通过这些函数访问
# 所有意图分类、拆分、验证能力。
# ---------------------------------------------------------------------------

# Lazy import to avoid circular dependency (implement_methods imports from this module)
from .implement_methods import ClassificationImplementMethod, IntentSplitMethod


def split_intents(query: str, conversation_context: Optional[str] = None) -> List[str]:
    """将重写后的查询拆分为单意图子问题列表。

    Args:
        query: 经过 query rewrite 的用户查询文本。
        conversation_context: 可选的对话历史，注入到 prompt {history} 占位符。

    Returns:
        拆分后的子问题列表；若 LLM 调用失败则返回原始查询的单元素列表。
    """
    return IntentSplitMethod.split(query, conversation_context=conversation_context)


def classify_intent(
    query: str,
    conversation_context: Optional[str] = None,
) -> IntentResult:
    """对单条查询按流程图顺序执行意图检测，返回分类结果。

    检测顺序（per Intent-Classification-Workflow diagram）：
      1. SP-API Intent?   → sp_api
      2. UDS Intent?      → uds
      3. Amazon Business?  → amazon_docs
      4. 全部不匹配      → general

    Args:
        query: 单条待分类查询。
        conversation_context: 可选的对话历史，注入到每个检测步骤的 prompt {history} 占位符。

    Returns:
        IntentResult，包含 workflow, intent_name, confidence, source。
    """
    return ClassificationImplementMethod().detect(
        query, conversation_context=conversation_context
    )


# Framework: at most 3 intent clauses classified in parallel per batch.
_INTENT_CLASSIFY_PARALLEL_CHUNK = 3


def classify_intents_batch(
    intents: List[str],
    conversation_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对拆分后的意图列表分类，返回合并后的结果（顺序与输入中非空子句一致）。

    每条子句内部：keyword → vector → LLM 串行短路；子句之间按批并行，每批最多 3 条。

    Args:
        intents: split_intents() 返回的子问题列表。
        conversation_context: 可选的对话上下文。

    Returns:
        按输入顺序排列的字典列表，每项包含:
        - query: 子查询文本
        - workflow: 最终路由工作流
        - intent_name: 意图名称（LLM 返回）
        - confidence: 置信度等级
        - source: 匹配来源
        - required_fields: 空列表（当前无 registry 数据源）
        - clarification_template: 空字符串
    """
    # (original_index, stripped_query) preserves order across parallel batches.
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
    """校验已分类意图的必填字段，缺失时生成追问文本。

    Args:
        intents_with_meta: 包含 query, intent_name, required_fields,
            clarification_template 的字典列表。
        conversation_context: 可选的对话上下文。

    Returns:
        追问文本字符串，若所有字段已满足则返回 None。
    """
    return _IntentValidator.validate(intents_with_meta, conversation_context)
