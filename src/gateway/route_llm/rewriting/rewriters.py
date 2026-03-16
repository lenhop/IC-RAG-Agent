"""
Gateway 查询改写模块 (rewriters)

职责：
  1. 对用户原始查询进行 normalize → LLM 改写 → 后处理
  2. 根据请求/环境变量选择 workflow（手动 > LLM > 启发式）

支持后端：
  - Ollama（本地）：通过 OllamaClient 调用 /api/generate
  - DeepSeek（远程）：通过 DeepSeekChat 调用 OpenAI 兼容 API

设计原则：
  - 类方法 (@classmethod) 保持无状态
  - LLM 调用失败时抛异常，公共 API 层捕获并 fallback
  - 改写阶段只做：规范化、上下文补全、清晰度改写，不做意图拆分
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional, Tuple

from ...schemas import QueryRequest
from ...prompt_loader import load_prompt
from ...message import ConversationHistoryHandler
from src.logger import get_logger_facade
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from ..routing_heuristics import (
    apply_docs_preference as _apply_docs_preference,
    normalize_query,
    route_workflow_heuristic as _route_workflow_heuristic,
)

logger = logging.getLogger(__name__)

_gateway_logger = None
try:
    _gateway_logger = get_logger_facade()
except Exception:
    _gateway_logger = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. RewriteResponseProcessor — LLM 输出后处理工具集
# ─────────────────────────────────────────────────────────────────────────────


class RewriteResponseProcessor:
    """LLM 改写输出的后处理工具集。

    所有方法均为 @staticmethod / @classmethod，无实例状态。
    处理流程：strip_echoed_context → enforce_responsibility → normalize
    """

    @staticmethod
    def strip_echoed_context(response: str, fallback_query: str) -> str:
        """检测 LLM 是否回显了上下文/系统痕迹，若是则返回 fallback。"""
        if not response or not response.strip():
            return fallback_query
        r_lower = response.strip().lower()
        echo_patterns = (
            "normalize: completed",
            "integrate short-term memory",
            "rewrite backend",
            "rewrite time",
            "intent classification",
            "rewrite-only test mode",
            "user:",
            "assistant:",
        )
        if any(p in r_lower for p in echo_patterns):
            logger.debug("Rewrite output contains echoed context; using fallback query")
            return fallback_query
        return response.strip()

    @staticmethod
    def collapse_to_single_sentence(text: str) -> str:
        """将多行文本折叠为单行（改写输出必须是单句）。"""
        if not text or not text.strip():
            return text or ""
        line = re.sub(r"\s*\n+\s*", " ", text.strip())
        line = re.sub(r"\s+", " ", line).strip()
        return line if line else text.strip()

    @staticmethod
    def apply_normalization_fixes(text: str) -> str:
        """文本规范化：修正常见拼写错误、去除口语填充词、统一小写。"""
        if not text or not text.strip():
            return text or ""
        s = text.strip()
        # 合并重复标点
        s = re.sub(r"\?+", "?", s)
        s = re.sub(r"!+", "!", s)
        # 常见拼写修正
        replacements = [
            (r"\bwat\b", "what"),
            (r"\binvetory\b", "inventory"),
            (r"\binvnetory\b", "inventory"),
            (r"\btehm\b", "them"),
            (r"\bpls\b", "please"),
            (r"\bthx\b", "thanks"),
            (r"\bu\b", "you"),
            (r"\bur\b", "your"),
        ]
        for pattern, repl in replacements:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        # 去除开头的 "hey" 和结尾的 "thx/thanks"
        s = re.sub(r"^\s*hey\s*,?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*(thx|thanks)\s*[.!?]*\s*$", "", s, flags=re.IGNORECASE).strip()
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else text.strip()

    @staticmethod
    def is_rewrite_responsibility_compliant(text: str) -> bool:
        """检查改写输出是否符合 Rewriting Responsibility 约束。

        不合规的情况：空文本、包含代码围栏、JSON 结构、对话角色标记、编号/列表格式。
        """
        candidate = (text or "").strip()
        if not candidate:
            return False
        lower_text = candidate.lower()
        if "```" in candidate or candidate.startswith("{") or candidate.startswith("["):
            return False
        if '"intents"' in lower_text or '"task_groups"' in lower_text or '"workflow"' in lower_text:
            return False
        if "user:" in lower_text or "assistant:" in lower_text:
            return False
        if re.search(r"\b1\.\s+\S+.*\b2\.\s+\S+", candidate):
            return False
        if re.search(r"(?:^|\s)-\s+\S+.*(?:\s-\s+\S+)", candidate):
            return False
        return True

    @classmethod
    def enforce_rewrite_responsibility(cls, rewritten_text: str, fallback_query: str) -> str:
        """强制执行改写职责约束：不合规时回退到原始查询。"""
        collapsed = cls.collapse_to_single_sentence(rewritten_text)
        if cls.is_rewrite_responsibility_compliant(collapsed):
            return collapsed
        logger.warning(
            "Rewrite output violates rewriting responsibility; fallback to normalized original query"
        )
        return cls.collapse_to_single_sentence(fallback_query)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _RewriteLLM — LLM 改写调用（内部类）
# ─────────────────────────────────────────────────────────────────────────────


class _RewriteLLM:
    """LLM 改写核心类。

    加载 rewrite_prompt.md，根据 backend 调用 Ollama 或 DeepSeek，
    对返回结果执行后处理链。失败时抛出 RuntimeError。
    """

    REWRITE_PROMPT = load_prompt("rewriting/rewrite_prompt")

    @classmethod
    def rewrite_with_context(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
        backend: str = "ollama",
        model: Optional[str] = None,
    ) -> str:
        """带会话上下文的查询改写。

        流程：拼接上下文 → 调用 LLM → strip_echoed_context → enforce_responsibility → normalize
        失败时抛出 RuntimeError（不静默回退）。
        """
        if not query or not query.strip():
            return query

        effective_backend = RouterEnvConfig.get_rewrite_backend(backend)
        logger.info("Rewrite with context: backend=%s query_len=%d", effective_backend, len(query.strip()))

        if conversation_context and conversation_context.strip():
            prompt_prefix = (
                f"Conversation context (recent turns):\n{conversation_context.strip()}\n\n"
                f"Current query to rewrite: {query.strip()}\n\n"
                "CRITICAL: Output ONLY the rewritten query. Do NOT repeat the context, "
                "'user:', 'assistant:', or any trace (Normalize, Rewrite Backend, etc.)."
            )
        else:
            prompt_prefix = query.strip()

        raw: str
        if effective_backend == "ollama":
            raw = cls._call_ollama(prompt_prefix, query.strip(), model)
        elif effective_backend == "deepseek":
            raw = cls._call_deepseek(prompt_prefix, query.strip(), model)
        else:
            logger.error("rewrite_with_context: unknown backend %s", effective_backend)
            raise ValueError(f"Unknown rewrite backend {effective_backend}; must be 'ollama' or 'deepseek'")

        cleaned = RewriteResponseProcessor.strip_echoed_context(raw, query.strip())
        out = RewriteResponseProcessor.enforce_rewrite_responsibility(cleaned, query.strip())
        normalized = RewriteResponseProcessor.apply_normalization_fixes(out)

        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="rewriter_with_context_done",
                    stage="rewriter",
                    message="rewrite_with_context completed",
                    status="success",
                    workflow="rewrite",
                    query_raw=query.strip(),
                    query_rewritten=normalized,
                    metadata={"backend": effective_backend},
                )
            except Exception:
                pass

        logger.debug("Rewrite completed: %d chars", len(normalized))
        return normalized

    @classmethod
    def _call_ollama(
        cls,
        prompt_content: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """通过 OllamaClient 调用本地 Ollama 进行改写。"""
        logger.debug("Ollama rewrite via OllamaClient")
        prompt = f"{cls.REWRITE_PROMPT}\n\nUser question: {prompt_content}"
        return OllamaClient().generate(
            prompt,
            model=model,
            empty_fallback=fallback_query,
        )

    @classmethod
    def _call_deepseek(
        cls,
        prompt_content: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """通过 DeepSeekChat 调用远程 DeepSeek API 进行改写。失败时抛出 RuntimeError。"""
        logger.debug("DeepSeek rewrite via DeepSeekChat (stage=rewrite)")
        try:
            out = DeepSeekChat().complete(
                cls.REWRITE_PROMPT,
                prompt_content,
                model_override=model,
            )
            return out if out else fallback_query
        except Exception as exc:
            logger.error("DeepSeek rewrite failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek rewrite failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# 3. 向后兼容公共 API（委托到内部类，捕获异常并 fallback）
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_PROMPT = _RewriteLLM.REWRITE_PROMPT


def rewrite_with_context(
    query: str,
    conversation_context: Optional[str] = None,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> str:
    """带上下文的查询改写（公共 API）。失败时返回原始查询。"""
    try:
        return _RewriteLLM.rewrite_with_context(
            query,
            conversation_context=conversation_context,
            backend=backend,
            model=model,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("rewrite_with_context failed; returning original query: %s", exc)
        return (query or "").strip() or ""


def rewrite_with_ollama(query: str, model: Optional[str] = None) -> str:
    """使用 Ollama 改写查询（向后兼容）。失败时返回原始查询。"""
    if not query or not query.strip():
        return query
    prompt = f"{REWRITE_PROMPT}\n\nUser question: {query.strip()}"
    try:
        return _RewriteLLM._call_ollama(prompt, query.strip(), model)
    except (RuntimeError, ValueError):
        logger.warning("rewrite_with_ollama failed; returning original query")
        return query.strip()


def rewrite_with_deepseek(query: str, model: Optional[str] = None) -> str:
    """使用 DeepSeek 改写查询（向后兼容）。失败时返回原始查询。"""
    if not query or not query.strip():
        return query
    prompt = query.strip()
    try:
        return _RewriteLLM._call_deepseek(prompt, query.strip(), model)
    except (RuntimeError, ValueError):
        logger.warning("rewrite_with_deepseek failed; returning original query")
        return query.strip()


# 测试兼容别名
_enforce_rewrite_responsibility = RewriteResponseProcessor.enforce_rewrite_responsibility
_strip_echoed_context_from_rewrite = RewriteResponseProcessor.strip_echoed_context


# ─────────────────────────────────────────────────────────────────────────────
# 4. RouterEnvConfig — 路由/改写环境变量配置
# ─────────────────────────────────────────────────────────────────────────────


class RouterEnvConfig:
    """路由与改写相关的环境变量读取。

    集中管理 GATEWAY_ROUTE_LLM_ENABLED、GATEWAY_ROUTE_LLM_THRESHOLD、
    GATEWAY_REWRITE_MEMORY_ROUNDS、GATEWAY_REWRITE_BACKEND、GATEWAY_ROUTE_BACKEND。
    """

    @staticmethod
    def route_llm_enabled() -> bool:
        """读取 GATEWAY_ROUTE_LLM_ENABLED，判断是否启用 LLM 路由。"""
        return os.getenv("GATEWAY_ROUTE_LLM_ENABLED", "false").strip().lower() in (
            "1", "true", "yes", "on",
        )

    @staticmethod
    def route_llm_threshold() -> float:
        """读取 GATEWAY_ROUTE_LLM_THRESHOLD（LLM 路由置信度阈值，默认 0.8）。"""
        try:
            return float(os.getenv("GATEWAY_ROUTE_LLM_THRESHOLD", "0.8"))
        except (ValueError, TypeError):
            logger.warning("GATEWAY_ROUTE_LLM_THRESHOLD invalid; using 0.8")
            return 0.8

    @staticmethod
    def get_memory_rounds() -> int:
        """读取 GATEWAY_REWRITE_MEMORY_ROUNDS（改写时加载的会话历史轮数，默认 3）。"""
        try:
            return max(1, int(os.getenv("GATEWAY_REWRITE_MEMORY_ROUNDS", "3")))
        except (ValueError, TypeError):
            logger.warning("GATEWAY_REWRITE_MEMORY_ROUNDS invalid; using 3")
            return 3

    @staticmethod
    def get_rewrite_backend(request_backend: Optional[str]) -> str:
        """解析改写后端：优先用请求参数，否则读 GATEWAY_REWRITE_BACKEND（默认 ollama）。"""
        backend = (request_backend or "").strip().lower()
        if backend:
            return backend
        return (os.getenv("GATEWAY_REWRITE_BACKEND", "ollama") or "ollama").strip().lower()

    @staticmethod
    def get_route_backend(request: QueryRequest) -> str:
        """解析路由后端：优先用请求参数，否则读 GATEWAY_ROUTE_BACKEND（默认 ollama）。"""
        backend = (getattr(request, "route_backend", None) or "").strip().lower()
        if backend:
            return backend
        return (os.getenv("GATEWAY_ROUTE_BACKEND", "ollama") or "ollama").strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. _RewriteRouter — 改写流水线编排 + Workflow 路由（内部类）
# ─────────────────────────────────────────────────────────────────────────────


class _RewriteRouter:
    """改写流水线编排与 workflow 路由。

    - rewrite_query: normalize → 加载会话历史 → rewrite_with_context → 可选意图分类
    - route_workflow: 手动指定 > LLM 路由 > 启发式路由
    """

    @classmethod
    def rewrite_query(
        cls,
        request: QueryRequest,
        gateway_memory: Optional[Any] = None,
        conversation_context: Optional[str] = None,
    ) -> Tuple[str, Optional[List[str]], int, int]:
        """执行完整的查询改写流水线。

        返回: (改写后查询, 意图列表, 使用的记忆轮数, 记忆文本长度)
        """
        normalized = normalize_query(request.query or "")
        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="router_rewrite_start",
                    stage="router_rewrite",
                    message="rewrite_query started",
                    status="started",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    workflow=request.workflow,
                    query_raw=request.query or "",
                    query_rewritten=normalized,
                )
            except Exception:
                pass

        if not normalized or not normalized.strip():
            logger.debug("rewrite_query: empty query, early exit")
            return ("", None, 0, 0)

        if not request.rewrite_enable:
            logger.debug("rewrite_query: rewrite disabled, returning normalized")
            return (normalized, None, 0, 0)

        backend = RouterEnvConfig.get_rewrite_backend(request.rewrite_backend)
        logger.info("rewrite_query: backend=%s session=%s", backend, request.session_id)

        # 加载会话历史作为改写上下文
        memory_rounds_used = 0
        memory_text_length = 0
        memory_context: Optional[str] = None
        last_n = RouterEnvConfig.get_memory_rounds()
        sid = (request.session_id or "").strip()
        if sid:
            res = ConversationHistoryHandler.get_session_history(
                gateway_memory, sid, last_n=last_n
            )
            history = res.get("history", [])
        else:
            history = []
        if history:
            memory_context = ConversationHistoryHandler.format_history_for_llm_markdown(history)
            memory_rounds_used = len(history)
            logger.debug("rewrite_query: loaded %d memory rounds", memory_rounds_used)

        conversation_context = ConversationHistoryHandler.merge_context_strings(
            conversation_context, memory_context
        )
        if conversation_context and conversation_context.strip():
            memory_text_length = len(conversation_context)
            if memory_rounds_used <= 0:
                memory_rounds_used = ConversationHistoryHandler.count_context_rounds(
                    conversation_context
                )

        # 调用 LLM 改写
        optimized_query = rewrite_with_context(
            normalized,
            conversation_context=conversation_context,
            backend=backend,
        )

        # 可选：改写阶段意图分类（当前默认关闭）
        intents: Optional[List[str]] = None
        if intent_classification_enabled():
            try:
                intents_payload = rewrite_intents_only(optimized_query)
                if isinstance(intents_payload, dict):
                    intents_raw = intents_payload.get("intents")
                else:
                    intents_raw = intents_payload
                if isinstance(intents_raw, list):
                    intents = [str(item).strip() for item in intents_raw if str(item).strip()] or None
            except Exception as exc:
                logger.warning("rewrite_query: intent classification failed: %s", exc)
                intents = None

        if _gateway_logger:
            try:
                _gateway_logger.log_runtime(
                    event_name="router_rewrite_done",
                    stage="router_rewrite",
                    message="rewrite_query completed",
                    status="success",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    workflow=request.workflow,
                    query_raw=request.query or "",
                    query_rewritten=optimized_query,
                    latency_ms=None,
                    metadata={
                        "memory_rounds_used": memory_rounds_used,
                        "memory_text_length": memory_text_length,
                    },
                )
            except Exception:
                pass

        logger.debug("rewrite_query: done, query_len=%d", len(optimized_query))
        return (optimized_query, intents, memory_rounds_used, memory_text_length)

    @classmethod
    def route_workflow(
        cls,
        query: str,
        request: QueryRequest,
    ) -> tuple[str, float, str, str | None, float | None]:
        """选择 workflow 及路由置信度。

        优先级：手动指定 > LLM 路由（需启用且置信度达标）> 启发式规则。
        返回: (workflow, confidence, method, backend, llm_confidence)
        """
        explicit = (request.workflow or "auto").strip().lower()
        if explicit != "auto":
            logger.debug("route_workflow: manual workflow=%s", explicit)
            return explicit, 1.0, "manual", None, None

        backend = RouterEnvConfig.get_route_backend(request)
        if _route_llm_enabled():
            from src.gateway import route_llm as route_pkg

            llm_wf, llm_conf = route_pkg.route_with_llm(query or "", backend)
            llm_wf = _apply_docs_preference(query or "", llm_wf)
            if llm_conf >= RouterEnvConfig.route_llm_threshold():
                logger.debug("route_workflow: llm workflow=%s conf=%.2f", llm_wf, llm_conf)
                return llm_wf, llm_conf, "llm", backend, llm_conf

        wf, conf = _route_workflow_heuristic(query or "")
        wf = _apply_docs_preference(query or "", wf)
        logger.debug("route_workflow: heuristic workflow=%s conf=%.2f", wf, conf)
        return wf, conf, "heuristic", None, None

    @staticmethod
    def _intent_classification_enabled() -> bool:
        """改写阶段的意图分类是否启用（当前硬编码 False，意图分类在 classification 子包中执行）。"""
        return False

    @staticmethod
    def _rewrite_intents_only(query: str):
        """对改写后的查询进行意图分类（预留接口，当前默认关闭）。"""
        try:
            from ..classification import split_intents
            intents = split_intents(query)
            if intents:
                return {"intents": intents}
        except Exception as exc:
            logger.debug("rewrite_intents_only failed: %s", exc)
            return None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. 公共 API — 改写流水线 + 路由（委托到内部类）
# ─────────────────────────────────────────────────────────────────────────────


def rewrite_query(
    request: QueryRequest,
    gateway_memory: Optional[Any] = None,
    conversation_context: Optional[str] = None,
) -> Tuple[str, Optional[List[str]], int, int]:
    """执行查询改写流水线（公共入口）。"""
    return _RewriteRouter.rewrite_query(
        request,
        gateway_memory=gateway_memory,
        conversation_context=conversation_context,
    )


def route_workflow(
    query: str, request: QueryRequest
) -> tuple[str, float, str, str | None, float | None]:
    """选择 workflow 及路由置信度（公共入口）。"""
    return _RewriteRouter.route_workflow(query, request)


def route_with_llm(query: str, backend: str = "ollama") -> tuple[str, float]:
    """LLM 路由占位钩子。返回安全默认值，确保启发式路由保持权威。"""
    _ = backend
    return "general", 0.0


def intent_classification_enabled() -> bool:
    """改写阶段的意图分类是否启用。"""
    return _RewriteRouter._intent_classification_enabled()


def rewrite_intents_only(query: str):
    """对改写后的查询进行意图分类。"""
    return _RewriteRouter._rewrite_intents_only(query)


def _route_llm_enabled() -> bool:
    """LLM 路由是否启用。"""
    return RouterEnvConfig.route_llm_enabled()


__all__ = [
    "REWRITE_PROMPT",
    "rewrite_with_context",
    "rewrite_with_ollama",
    "rewrite_with_deepseek",
    "rewrite_query",
    "route_workflow",
    "route_with_llm",
    "intent_classification_enabled",
    "rewrite_intents_only",
    "_route_llm_enabled",
]
