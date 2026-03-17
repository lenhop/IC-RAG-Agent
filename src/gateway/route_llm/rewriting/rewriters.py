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
from typing import Any, Optional, Tuple

from ...schemas import QueryRequest
from ...prompt_loader import load_prompt
from ...message import ConversationHistoryHandler
from src.logger import get_logger_facade
from src.llm.call_deepseek import DeepSeekChat
from src.llm.call_ollama import OllamaClient
from ..routing_heuristics import (
    normalize_query,
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
    def _render_rewrite_prompt(
        cls,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> str:
        """将 rewrite_prompt.md 中的 {history}/{query} 占位符替换为实际值。"""
        history_value = (conversation_context or "").strip()
        query_value = (query or "").strip()
        return (
            cls.REWRITE_PROMPT
            .replace("{history}", history_value)
            .replace("{query}", query_value)
        )

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

        rendered_prompt = cls._render_rewrite_prompt(
            query=query,
            conversation_context=conversation_context,
        )

        raw: str
        if effective_backend == "ollama":
            raw = cls._call_ollama(rendered_prompt, query.strip(), model)
        elif effective_backend == "deepseek":
            raw = cls._call_deepseek(rendered_prompt, query.strip(), model)
        else:
            logger.error("rewrite_with_context: unknown backend %s", effective_backend)
            raise ValueError(f"Unknown rewrite backend {effective_backend}; must be 'ollama' or 'deepseek'")

        cleaned = RewriteResponseProcessor.strip_echoed_context(raw, query.strip())
        out = RewriteResponseProcessor.enforce_rewrite_responsibility(cleaned, query.strip())
        normalized = RewriteResponseProcessor.apply_normalization_fixes(out)

        if _gateway_logger:
            try:
                # 记录单次 LLM 改写调用完成事件，便于按 backend 追踪效果与问题定位。
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
        rendered_prompt: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """通过 OllamaClient 调用本地 Ollama 进行改写。"""
        logger.debug("Ollama rewrite via OllamaClient")
        return OllamaClient().generate(
            rendered_prompt,
            model=model,
            empty_fallback=fallback_query,
        )

    @classmethod
    def _call_deepseek(
        cls,
        rendered_prompt: str,
        fallback_query: str,
        model: Optional[str] = None,
    ) -> str:
        """通过 DeepSeekChat 调用远程 DeepSeek API 进行改写。失败时抛出 RuntimeError。"""
        logger.debug("DeepSeek rewrite via DeepSeekChat (stage=rewrite)")
        try:
            out = DeepSeekChat().complete(
                "You are a query rewriting engine. Follow the user prompt strictly.",
                rendered_prompt,
                model_override=model,
            )
            return out if out else fallback_query
        except Exception as exc:
            logger.error("DeepSeek rewrite failed: %s", exc, exc_info=True)
            raise RuntimeError(f"DeepSeek rewrite failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# 3. RouterEnvConfig — 路由/改写环境变量配置
# ─────────────────────────────────────────────────────────────────────────────


class RouterEnvConfig:
    """路由与改写相关的环境变量读取。

    集中管理 GATEWAY_REWRITE_MEMORY_ROUNDS、GATEWAY_REWRITE_BACKEND。
    """

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. _RewriteRouter — 改写流水线编排 + Workflow 路由（内部类）
# ─────────────────────────────────────────────────────────────────────────────


class _RewriteRouter:
    """改写流水线编排与 workflow 路由。

    - rewrite_query: normalize → 加载会话历史 → _RewriteLLM.rewrite_with_context
    - route_workflow: 纯 LLM 路由（通过 route_with_llm 实现）
    """

    @classmethod
    def rewrite_query(
        cls,
        request: QueryRequest,
        gateway_memory: Optional[Any] = None,
        conversation_context: Optional[str] = None,
    ) -> Tuple[str, None, int, int]:
        """执行完整的查询改写流水线。

        要求: request.session_id 必须存在且非空，否则抛出 ValueError。

        返回: (改写后查询, None, 使用的记忆轮数, 记忆文本长度)
        """
        # 先对原始查询做轻量规范化，统一后续改写输入。
        normalized = normalize_query(request.query or "")
        if _gateway_logger:
            try:
                # 记录 rewrite 流水线开始，方便观察请求是否进入改写阶段。
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

        # 空查询直接短路，避免无意义的上下文加载和 LLM 调用。
        if not normalized or not normalized.strip():
            logger.debug("rewrite_query: empty query, early exit")
            return ("", None, 0, 0)

        # 调用方显式关闭改写时，仅返回规范化结果。
        if not request.rewrite_enable:
            logger.debug("rewrite_query: rewrite disabled, returning normalized")
            return (normalized, None, 0, 0)

        backend = RouterEnvConfig.get_rewrite_backend(request.rewrite_backend)
        logger.info("rewrite_query: backend=%s session=%s", backend, request.session_id)

        # 改写阶段强依赖 session_id；缺失时直接报错，避免产生无会话上下文的半有效请求。
        memory_rounds_used = 0
        memory_text_length = 0
        memory_context: Optional[str] = None
        last_n = RouterEnvConfig.get_memory_rounds()
        sid = (request.session_id or "").strip()
        if not sid:
            logger.error("rewrite_query: session_id is required")
            raise ValueError("session_id is required for rewrite_query")

        # 读取最近 N 轮会话历史，为代词解析和上下文补全提供依据。
        res = ConversationHistoryHandler.get_session_history(
            gateway_memory, sid, last_n=last_n
        )
        history = res.get("history", [])
        if history:
            memory_context = ConversationHistoryHandler.format_history_for_llm_markdown(history)
            memory_rounds_used = len(history)
            logger.debug("rewrite_query: loaded %d memory rounds", memory_rounds_used)

        # 将外部传入上下文与 Redis 中读取到的会话历史合并成单一改写上下文。
        conversation_context = ConversationHistoryHandler.merge_context_strings(
            conversation_context, memory_context
        )
        if conversation_context and conversation_context.strip():
            memory_text_length = len(conversation_context)
            if memory_rounds_used <= 0:
                memory_rounds_used = ConversationHistoryHandler.count_context_rounds(
                    conversation_context
                )

        # 调用改写器生成最终查询；此阶段只做改写，不做意图拆分。
        optimized_query = _RewriteLLM.rewrite_with_context(
            normalized,
            conversation_context=conversation_context,
            backend=backend,
        )

        if _gateway_logger:
            try:
                # 记录 rewrite 流水线完成，并附带记忆使用情况供排障和效果分析。
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

        # 返回改写结果以及记忆使用情况，供上层日志和调试面板使用。
        logger.debug("rewrite_query: done, query_len=%d", len(optimized_query))
        return (optimized_query, None, memory_rounds_used, memory_text_length)

    @classmethod
    def route_workflow(
        cls,
        query: str,
        request: QueryRequest,
    ) -> tuple[str, float, str, str | None, float | None]:
        """通过 LLM 选择 workflow 及置信度。

        返回: (workflow, confidence, method, backend, llm_confidence)
        """
        backend = (getattr(request, "route_backend", None) or "").strip().lower() or \
            (os.getenv("GATEWAY_ROUTE_BACKEND", "ollama") or "ollama").strip().lower()

        from src.gateway import route_llm as route_pkg
        llm_wf, llm_conf = route_pkg.route_with_llm(query or "", backend)
        logger.debug("route_workflow: llm workflow=%s conf=%.2f", llm_wf, llm_conf)
        return llm_wf, llm_conf, "llm", backend, llm_conf


# ─────────────────────────────────────────────────────────────────────────────
# 5. 公共 API — 改写 + 路由统一入口
# ─────────────────────────────────────────────────────────────────────────────


def rewrite_and_route(
    request: QueryRequest,
    gateway_memory: Optional[Any] = None,
    conversation_context: Optional[str] = None,
    route_query: Optional[str] = None,
    enable_routing: bool = True,
    rewritten_query: Optional[str] = None,
    memory_rounds: int = 0,
    memory_text_length: int = 0,
) -> tuple[str, None, int, int, str | None, float | None, str | None, str | None, float | None]:
    """统一入口：rewrite + optional route。

    调用约定：
      1) 默认先执行 rewrite，再按需执行 route；
      2) 若调用方已提供 rewritten_query，则跳过 rewrite（沿用传入值）；
      3) 若 enable_routing=False，仅返回 rewrite 相关结果。

    返回:
      (
        rewritten_query,
        None,
        memory_rounds,
        memory_text_length,
        workflow,
        routing_confidence,
        route_source,
        route_backend,
        route_llm_confidence,
      )
    """
    # 保留调用方可能已预计算的改写结果与记忆统计，避免重复改写。
    current_rewritten = rewritten_query
    rounds_used = memory_rounds
    text_length = memory_text_length

    # 仅在未提供 rewritten_query 时触发完整 rewrite 流水线。
    if current_rewritten is None:
        current_rewritten, _, rounds_used, text_length = _RewriteRouter.rewrite_query(
            request,
            gateway_memory=gateway_memory,
            conversation_context=conversation_context,
        )

    # rewrite-only 模式：路由字段统一置空，保持返回结构稳定。
    if not enable_routing:
        return (
            current_rewritten,
            None,
            rounds_used,
            text_length,
            None,
            None,
            None,
            None,
            None,
        )

    # 路由输入优先级：显式 route_query > 改写结果 > 空串。
    route_input = (route_query or current_rewritten or "").strip()
    workflow, routing_confidence, route_source, route_backend, route_llm_confidence = _RewriteRouter.route_workflow(
        route_input,
        request,
    )

    # 返回统一的 rewrite+route 元组，供 API 层按位解包使用。
    return (
        current_rewritten,
        None,
        rounds_used,
        text_length,
        workflow,
        routing_confidence,
        route_source,
        route_backend,
        route_llm_confidence,
    )


def route_with_llm(query: str, backend: str = "ollama") -> tuple[str, float]:
    """LLM 路由占位钩子。返回安全默认值，确保启发式路由保持权威。"""
    _ = backend
    return "general", 0.0


__all__ = [
    "route_with_llm",
    "rewrite_and_route",
]
