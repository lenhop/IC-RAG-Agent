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
  _IntentSplitter    — LLM / 启发式多意图拆分
  _LLMIntentDetector — 基于 prompt + LLM 的串行意图检测器
  _IntentValidator   — 必填字段校验 & 追问生成

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
from dataclasses import dataclass, field
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
    """

    intent_name: str
    workflow: str
    confidence: str = "medium"
    source: str = "llm"
    required_fields: List[str] = field(default_factory=list)
    clarification_template: str = ""


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


class _IntentSplitter:
    """多意图拆分器：先尝试 LLM 拆分，失败时回退到启发式规则。"""

    _SPLIT_COMMA_QUESTION = re.compile(
        r",\s+(?=what|how|get|show|list|compare|when|which|where|who|why|tell|give|check|find|return|describe|explain)",
        flags=re.I,
    )

    @classmethod
    def split(cls, query: str) -> List[str]:
        """拆分用户查询为多个独立子问题。

        1. 先用 LLM（Ollama 或 DeepSeek）解析
        2. 失败则回退到启发式规则拆分
        """
        if not query or not query.strip():
            return []

        prompt_template = load_prompt("classification/intent_split_query")
        user_line = f"Input: {query.strip()}"
        text = ""

        if (
            _RuntimeConfig.get_intent_split_backend() == "deepseek"
            and (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        ):
            try:
                text = DeepSeekChat().complete(
                    prompt_template,
                    user_line,
                    max_tokens=512,
                )
            except Exception as exc:
                logger.warning("Intent split DeepSeek failed: %s; heuristic fallback", exc)
                return cls._fallback_split(query)
        else:
            prompt = f"{prompt_template}\n\n{user_line}"
            try:
                text = OllamaClient().generate(prompt, empty_fallback="")
            except Exception as exc:
                logger.warning("Intent split LLM call failed: %s; trying heuristic fallback", exc)
                return cls._fallback_split(query)

        if not text:
            return cls._fallback_split(query)

        raw = cls._strip_markdown_fences(text)
        parsed = cls._parse_json_response(raw)
        if parsed is None:
            return cls._fallback_split(query)

        intents = parsed.get("intents")
        if not isinstance(intents, list) or not intents:
            return cls._fallback_split(query)

        result = cls._dedupe_intents(intents)
        if not result:
            return cls._fallback_split(query)

        # 若 LLM 只返回 1 项但查询含逗号/and，尝试启发式补充
        if len(result) == 1 and ("," in query or " and " in query.lower()):
            heuristic = cls._heuristic_split_multi_intent(query)
            if len(heuristic) >= 2:
                logger.info(
                    "Intent split: LLM returned 1 item; using heuristic split (%d clauses)",
                    len(heuristic),
                )
                return heuristic

        cls._log_split_result(query, result)
        return result

    # ── 启发式拆分 ──

    @classmethod
    def _heuristic_split_multi_intent(cls, query: str) -> List[str]:
        if not query or not query.strip():
            return []

        parts = cls._SPLIT_COMMA_QUESTION.split(query.strip())
        expanded: List[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            sub_parts = re.split(
                r"\s+and\s+(?=what|how|get|show|list|compare|when|which|where|who|why|tell|give|check|find|return|describe|explain)",
                part,
                flags=re.I,
            )
            for sub_part in sub_parts:
                cleaned = sub_part.strip().rstrip(",").strip()
                if cleaned and len(cleaned) > 2:
                    expanded.append(cleaned)

        result: List[str] = []
        for segment in expanded:
            if segment.isdigit() and len(segment) == 4 and result:
                result[-1] = f"{result[-1]}, {segment}"
            else:
                result.append(segment)
        return result

    @classmethod
    def _fallback_split(cls, query: str) -> List[str]:
        stripped = (query or "").strip()
        if not stripped:
            return []
        if "," in stripped or " and " in stripped.lower():
            heuristic = cls._heuristic_split_multi_intent(stripped)
            if len(heuristic) >= 2:
                return heuristic
        return [stripped]

    # ── JSON 解析工具 ──

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return raw

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, object]]:
        try:
            return json.loads(raw)
        except ValueError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except ValueError:
                    logger.warning("Intent split JSON parse failed; heuristic fallback")
                    return None
            logger.warning("Intent split no JSON found; heuristic fallback")
            return None

    @staticmethod
    def _dedupe_intents(intents: List[object]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for item in intents:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            lowered = cleaned.lower()
            if cleaned and lowered not in seen:
                seen.add(lowered)
                result.append(cleaned)
        return result

    @staticmethod
    def _log_split_result(query: str, result: List[str]) -> None:
        if not _gateway_logger:
            return
        try:
            _gateway_logger.log_runtime(
                event_name="intent_split_completed",
                stage="intent_classification",
                message="split_intents completed",
                status="success",
                workflow="intent_classification",
                query_raw=query,
                intent_list=result,
                metadata={"intent_count": len(result)},
            )
        except Exception:
            pass


class _LLMIntentDetector:
    """基于 prompt + LLM 的串行意图检测器。

    按照流程图顺序依次执行检测步骤：
      1. SP-API Intent?   → sp_api_prompts.md
      2. UDS Intent?      → uds_prompts.md
      3. Amazon Business?  → amazon_prompts.md
      4. 全部不匹配      → general

    每一步将关键词和示例句子写在 prompt 中，喂给 LLM 做 yes/no 判断。
    一旦匹配即返回，不继续后续步骤。
    """

    @classmethod
    def detect(cls, query: str) -> IntentResult:
        """对单条查询按流程图顺序检测意图类型。

        Args:
            query: 待分类的子查询。

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

        # ── 按流程图顺序逐步检测 ──
        for step in _DETECTION_STEPS:
            try:
                result = cls._run_detection_step(
                    query=stripped,
                    prompt_name=step["prompt_name"],
                    target_workflow=step["workflow"],
                )
                if result is not None:
                    cls._log_detection_result(stripped, result)
                    return result
            except Exception as exc:
                logger.warning(
                    "Intent detection step '%s' failed: %s; continuing to next step",
                    step["prompt_name"],
                    exc,
                )
                continue

        # ── 全部不匹配 → Mark as General ──
        general_result = IntentResult(
            intent_name="general",
            workflow="general",
            confidence="low",
            source="llm",
        )
        cls._log_detection_result(stripped, general_result)
        return general_result

    @classmethod
    def _run_detection_step(
        cls,
        query: str,
        prompt_name: str,
        target_workflow: str,
    ) -> Optional[IntentResult]:
        """执行单个检测步骤：加载 prompt → 替换 {query} → 调用 LLM → 解析结果。

        Args:
            query: 用户查询。
            prompt_name: prompt 文件路径（如 'classification/sp_api_prompts'）。
            target_workflow: 匹配成功时的 workflow 名称。

        Returns:
            IntentResult if matched, None if not matched.
        """
        prompt_template = load_prompt(prompt_name)
        prompt_text = prompt_template.replace("{query}", query)

        # ── 调用 LLM ──
        response_text = cls._call_llm(prompt_text)
        if not response_text:
            return None

        # ── 解析 JSON 响应 ──
        parsed = cls._parse_llm_response(response_text)
        if parsed is None:
            return None

        match_value = parsed.get("match")
        if match_value is not True:
            return None

        # ── 匹配成功：提取 intent_name 和 confidence ──
        intent_name = parsed.get("intent_name", target_workflow) or target_workflow
        confidence = parsed.get("confidence", "medium") or "medium"

        return IntentResult(
            intent_name=intent_name,
            workflow=target_workflow,
            confidence=confidence,
            source="llm",
        )

    @classmethod
    def _call_llm(cls, prompt: str) -> str:
        """调用 LLM 后端（Ollama 或 DeepSeek）执行 prompt。"""
        backend = _RuntimeConfig.get_intent_detect_backend()

        if backend == "deepseek" and (os.getenv("DEEPSEEK_API_KEY") or "").strip():
            try:
                return DeepSeekChat().complete(
                    system_prompt="You are an intent classifier. Output ONLY valid JSON.",
                    user_message=prompt,
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


def split_intents(query: str) -> List[str]:
    """将重写后的查询拆分为单意图子问题列表。

    Args:
        query: 经过 query rewrite 的用户查询文本。

    Returns:
        拆分后的子问题列表；若无法拆分则返回原始查询的单元素列表。
    """
    return _IntentSplitter.split(query)


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
        conversation_context: 可选的对话上下文（当前保留，未传入 LLM）。

    Returns:
        IntentResult，包含 workflow, intent_name, confidence, source。
    """
    return _LLMIntentDetector.detect(query)


def classify_intents_batch(
    intents: List[str],
    conversation_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对拆分后的意图列表逐一分类，返回合并后的结果。

    完整实现流程图中 Intent Classification 框内的：
      loop through intent list → SP-API / UDS / Amazon Business / General → Merge

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
    results: List[Dict[str, Any]] = []

    for intent in intents:
        q = (intent or "").strip()
        if not q:
            continue

        # ── 按流程图顺序检测意图 ──
        classification = _LLMIntentDetector.detect(q)

        results.append({
            "query": q,
            "workflow": classification.workflow,
            "intent_name": classification.intent_name,
            "confidence": classification.confidence,
            "source": classification.source,
            "required_fields": classification.required_fields,
            "clarification_template": classification.clarification_template,
        })

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
