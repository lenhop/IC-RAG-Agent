"""
Route LLM clarification: detect ambiguous queries before rewriting.

Clarification workflow (clarification 澄清流程):
  1. Skip: empty query, documentation/policy questions
  2. Heuristic: pattern-match inventory/order/fees/sales without required identifiers
     - If match + no context: return clarification (optional LLM question)
     - If match + context: call LLM to decide (user may have provided info)
  3. Full LLM: when heuristic does not match, LLM decides needs_clarification

For Amazon seller queries, detects when the user query is ambiguous or lacks
critical information (ASIN, Order ID, date range, time period, store, SKU,
marketplace). The LLM returns a clarification question instead of executing.
Uses Ollama only (DeepSeek removed to avoid data leakage).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import requests

from ...prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# Reuse env defaults from rewriters
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REWRITE_TIMEOUT = 10

# Prompts loaded from route_llm/clarification/*.txt (cached after first access)
CLARIFICATION_PROMPT = load_prompt("clarification/clarification_detect_ambiguity")
_GENERATE_QUESTION_PROMPT = load_prompt("clarification/clarification_generate_question")

def _is_concrete_documentation_query(query: str) -> bool:
    """
    [Step 1a] 文档/政策类问题跳过澄清。
    Return True for documentation, policy, compliance, requirements questions.
    These are self-contained conceptual questions and do NOT need clarification.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = [
        r"documentation\s+requirements",
        r"product\s+compliance",
        r"safety\s+documentation",
        r"policy\s+on",
        r"what\s+are\s+.*\s+requirements",
        r"what\s+does\s+amazon",
        r"guidelines",
        r"business\s+rules",
        r"compliance\s+and\s+safety",
        r"requirements\s+for",
    ]
    return any(re.search(p, q) for p in patterns)


# [Step 2] 启发式规则: 命中主题但缺少必要标识符时需澄清
# (topic_pattern, has_required_pattern, fallback_question when LLM fails)
# 规则: 库存/订单/费用/销售 需对应 ASIN/OrderID/日期/费用类型 等
_HEURISTIC_AMBIGUOUS = [
    (
        r"\b(inventory|stock)\b",
        r"\b(ASIN|B0[0-9A-Z]{8}|store|SKU|marketplace)\b",
        "Which store, ASIN, or SKU do you want inventory for?",
    ),
    (
        r"\b(order|orders)\s+(status|info|details)?\b|\b(check|get)\s+(my\s+)?order\b",
        r"\d{3}-\d{7}-\d{7}|order\s*[iI][dD]",
        "Please provide the Order ID (e.g. 112-1234567-8901234) to check order status.",
    ),
    (
        r"\b(fees?|charges?|breakdown)\b",
        r"\b(FBA|storage|referral|last\s+(month|quarter|year)|Q[1-4]|20\d{2}|20\d{6})\b|\b\d{4}-\d{2}-\d{2}\b",
        "Which fees do you mean? (FBA, storage, or referral) And for which time period?",
    ),
    (
        r"\b(sales?|trends?|metrics?)\b",
        r"\b(last\s+(month|quarter|year)|Q[1-4]|20\d{2}|20\d{6})\b|\b\d{4}-\d{2}-\d{2}\b|january|february|march|april|may|june|july|august|september|october|november|december\b",
        "Which date range or time period do you want the data for?",
    ),
]


def _build_user_input(query: str, conversation_context: Optional[str] = None) -> str:
    """[辅助] 构建 LLM 输入: 可选对话历史 + 当前查询"""
    parts = []
    if conversation_context and conversation_context.strip():
        parts.append(f"Conversation history:\n{conversation_context.strip()}\n")
    parts.append(f"User query: {query.strip()}")
    return "\n".join(parts)


def _generate_clarification_question_ollama(
    query: str, conversation_context: Optional[str] = None
) -> Optional[str]:
    """[Step 2b] 启发式命中时，调用 Ollama 生成澄清问题。使用 clarification_generate_question.txt"""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_timeout()
    user_input = _build_user_input(query, conversation_context)
    payload = {
        "model": mdl,
        "prompt": f"{_GENERATE_QUESTION_PROMPT}{user_input}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama generate-question failed: %s", exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        text = (data.get("response") or "").strip()
    except (ValueError, TypeError):
        return None
    if not text:
        return None
    raw = _strip_markdown_fences(text)
    candidate = _extract_first_json_object(raw)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
        q = parsed.get("clarification_question")
        if isinstance(q, str) and q.strip():
            return q.strip()
    except (ValueError, TypeError):
        pass
    return None


# def _heuristic_needs_clarification(query: str) -> Optional[dict]:
#     """
#     [Step 2a] 启发式匹配: 检查是否命中 库存/订单/费用/销售 且缺少必要标识符。
#     命中则返回 {needs_clarification, clarification_question}，否则 None。
#     """
#     q = (query or "").strip()
#     if not q:
#         return None
#     for topic_pattern, has_required, question in _HEURISTIC_AMBIGUOUS:
#         if not re.search(topic_pattern, q, re.IGNORECASE):
#             continue
#         if has_required is None:
#             return {"needs_clarification": True, "clarification_question": question}
#         if not re.search(has_required, q, re.IGNORECASE):
#             return {"needs_clarification": True, "clarification_question": question}
#     return None


def _get_timeout() -> int:
    """[辅助] 读取 GATEWAY_REWRITE_TIMEOUT，默认 10 秒"""
    try:
        return int(os.getenv("GATEWAY_REWRITE_TIMEOUT", str(DEFAULT_REWRITE_TIMEOUT)))
    except ValueError:
        return DEFAULT_REWRITE_TIMEOUT


def _strip_markdown_fences(text: str) -> str:
    """[辅助] 去除 LLM 输出中的 ``` 代码块标记"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_first_json_object(text: str) -> Optional[str]:
    """[辅助] 从文本中提取第一个完整的 {...} JSON 对象"""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _call_clarification_ollama(
    query: str, conversation_context: Optional[str] = None
) -> str:
    """[Step 3] 调用 Ollama 做歧义检测。使用 clarification_detect_ambiguity.txt"""
    url = os.getenv("GATEWAY_REWRITE_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    mdl = os.getenv("GATEWAY_REWRITE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    timeout = _get_timeout()
    user_input = _build_user_input(query, conversation_context)
    payload = {
        "model": mdl,
        "prompt": f"{CLARIFICATION_PROMPT}{user_input}",
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
        logger.warning("Ollama clarification check failed: %s", exc)
        return ""
    if resp.status_code != 200:
        logger.warning("Ollama clarification HTTP %s", resp.status_code)
        return ""
    try:
        data = resp.json()
        return (data.get("response") or "").strip()
    except (ValueError, TypeError):
        return ""


# DeepSeek removed to avoid data leakage; use Ollama only for clarification
# def _call_clarification_deepseek(
#     query: str, conversation_context: Optional[str] = None
# ) -> str:
#     """[Step 3] 调用 DeepSeek 做歧义检测。使用 clarification_detect_ambiguity.txt"""
#     api_key = os.getenv("DEEPSEEK_API_KEY")
#     if not api_key or not api_key.strip():
#         logger.warning("DEEPSEEK_API_KEY not set; cannot call DeepSeek clarification")
#         return ""
#     mdl = os.getenv("GATEWAY_REWRITE_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
#     timeout = _get_timeout()
#     try:
#         from openai import OpenAI
#     except ImportError:
#         logger.warning("openai client not installed; cannot call DeepSeek clarification")
#         return ""
#     client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
#     user_input = _build_user_input(query, conversation_context)
#     messages = [
#         {"role": "system", "content": CLARIFICATION_PROMPT},
#         {"role": "user", "content": user_input},
#     ]
#     try:
#         response = client.chat.completions.create(
#             model=mdl,
#             messages=messages,
#             max_tokens=256,
#             temperature=0.3,
#         )
#         choice = response.choices[0] if response.choices else None
#         if choice and choice.message and choice.message.content:
#             return (choice.message.content or "").strip()
#     except Exception as exc:
#         logger.warning("DeepSeek clarification failed: %s", exc)
#     return ""


def check_ambiguity(
    query: str,
    backend: Optional[str] = None,
    conversation_context: Optional[str] = None,
) -> dict:
    """
    [入口] 澄清流程主入口。判断用户查询是否需澄清。

    Flow:
      1. 空查询/文档类 -> 直接返回 needs_clarification=False
      2. 启发式命中 -> 有上下文则问 LLM 决定，否则直接返回澄清
      3. 启发式未命中 -> 调用 LLM 做歧义检测

    Args:
        query: Raw user query text.
        backend: Optional backend override; uses GATEWAY_REWRITE_BACKEND env if None.
        conversation_context: Optional formatted conversation history (last 3 rounds)
            from Redis. When provided, the LLM considers prior turns so it won't
            ask for info the user already supplied in recent conversation.

    Returns:
        {"needs_clarification": True, "clarification_question": "..."} when ambiguous,
        or {"needs_clarification": False} when clear. On LLM failure, returns
        {"needs_clarification": False} to allow normal flow to proceed.
    """
    # [Step 1] 空查询直接返回
    if not query or not query.strip():
        return {"needs_clarification": False}

    # [Step 1a] 文档/政策/合规类问题跳过澄清
    if _is_concrete_documentation_query(query):
        return {"needs_clarification": False}

        # [Step 2] 启发式路径 (已注释，仅用 LLM 做歧义检测)
    # heuristic_result = _heuristic_needs_clarification(query)
    # if heuristic_result is not None:
    #     fallback_question = heuristic_result.get("clarification_question", "")
    #     effective_backend = (backend or "").strip().lower() or os.getenv(
    #         "GATEWAY_REWRITE_BACKEND", "ollama"
    #     ).strip().lower()
    #     if conversation_context:
    #         if effective_backend == "ollama":
    #             text = _call_clarification_ollama(query.strip(), conversation_context)
    #         elif effective_backend == "deepseek":
    #             text = _call_clarification_deepseek(query.strip(), conversation_context)
    #         else:
    #             return {"needs_clarification": True, "clarification_question": fallback_question}
    #         if text and text.strip():
    #             raw = _strip_markdown_fences(text)
    #             parsed = None
    #             try:
    #                 parsed = json.loads(raw)
    #             except ValueError:
    #                 candidate = _extract_first_json_object(raw)
    #                 if candidate:
    #                     try:
    #                         parsed = json.loads(candidate)
    #                     except ValueError:
    #                         parsed = None
    #             if isinstance(parsed, dict) and not parsed.get("needs_clarification"):
    #                 return {"needs_clarification": False}
    #         if effective_backend == "ollama":
    #             llm_question = _generate_clarification_question_ollama(
    #                 query.strip(), conversation_context
    #             )
    #             if llm_question:
    #                 return {"needs_clarification": True, "clarification_question": llm_question}
    #         return {"needs_clarification": True, "clarification_question": fallback_question}
    #     if effective_backend == "ollama":
    #         llm_question = _generate_clarification_question_ollama(query.strip())
    #         if llm_question:
    #             return {"needs_clarification": True, "clarification_question": llm_question}
    #     return {"needs_clarification": True, "clarification_question": fallback_question}

    # [Step 3] 调用 LLM 做歧义检测 (clarification_detect_ambiguity.txt)
    effective_backend = (backend or "").strip().lower() or os.getenv(
        "GATEWAY_REWRITE_BACKEND", "ollama"
    ).strip().lower()

    # DeepSeek removed to avoid data leakage; Ollama only
    if effective_backend != "ollama":
        logger.warning("check_ambiguity: backend %s not supported (Ollama only); skipping", effective_backend)
        return {"needs_clarification": False}

    if conversation_context:
        text = _call_clarification_ollama(query.strip(), conversation_context)
    else:
        text = _call_clarification_ollama(query.strip())

    if not text or not text.strip():
        return {"needs_clarification": False}

    # [Step 3] 解析 LLM 返回的 JSON
    raw = _strip_markdown_fences(text)
    parsed = None
    try:
        parsed = json.loads(raw)
    except ValueError:
        candidate = _extract_first_json_object(raw)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except ValueError:
                parsed = None

    if not isinstance(parsed, dict):
        return {"needs_clarification": False}

    needs = parsed.get("needs_clarification")
    if not needs:
        return {"needs_clarification": False}

    question = parsed.get("clarification_question")
    if not isinstance(question, str) or not question.strip():
        return {"needs_clarification": False}

    return {
        "needs_clarification": True,
        "clarification_question": question.strip(),
    }
