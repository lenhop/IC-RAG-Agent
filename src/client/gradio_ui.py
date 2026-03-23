"""
Gradio chat UI for unified gateway client.

Layout: controls (scale 1) left, chat (scale 3) right.
Left column: Workflow, Rewriting Enable, Rewrite backend, Session, Status.
Right column: ChatInterface with fill_height and autoscroll.
Uses GatewayClient.query_sync with mock mode when gateway unavailable.
"""

from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from uuid import uuid4

import gradio as gr

# Load .env from project root so UNIFIED_CHAT_* and skip-login credentials are available
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from .api_client import GatewayClient
from src.llm.chat_backend_policy import resolve_chat_backend

logger = logging.getLogger(__name__)

# Session-scoped pending query for clarification follow-up (client-side merge).
# Key: session_id, Value: pending_query. Cleared when user provides follow-up or clears session.
_pending_queries: Dict[str, str] = {}
_pending_lock = threading.Lock()


def _get_pending_query(session_id: Optional[str]) -> Optional[str]:
    """Return and remove pending query for session if any."""
    if not session_id or not str(session_id).strip():
        return None
    with _pending_lock:
        return _pending_queries.pop(session_id.strip(), None)


def _set_pending_query(session_id: Optional[str], pending_query: str) -> None:
    """Store pending query for clarification follow-up."""
    if not session_id or not str(session_id).strip() or not pending_query or not pending_query.strip():
        return
    with _pending_lock:
        _pending_queries[session_id.strip()] = pending_query.strip()


def _clear_all_pending() -> None:
    """Clear all pending queries (e.g., when user clears session)."""
    with _pending_lock:
        _pending_queries.clear()


def _format_query_as_bullets(query: str, min_length: int = 60) -> Optional[str]:
    """
    Split long or multi-part query into bullet points for display.
    Client-side fallback when gateway does not return rewritten_query_display.
    Handles numbered lists, bullet points, comma/semicolon separation.
    """
    if not query:
        return None

    # Check for numbered list (1. ... 2. ...)
    numbered = re.split(r"\s*\d+\.\s+", query.strip())
    numbered = [p.strip() for p in numbered if p.strip()]
    if len(numbered) >= 2:
        return "\n".join(f"    - {c}" for c in numbered)

    # Check for bullet points (- ... or • ...)
    bullet_items = re.split(r"\s*[-•]\s+", query.strip())
    bullet_items = [p.strip() for p in bullet_items if p.strip()]
    if len(bullet_items) >= 2:
        return "\n".join(f"    - {c}" for c in bullet_items)

    if len(query) < min_length:
        return None

    # Split by comma (avoid dates like "Q3 and Q4 2025") or semicolon or "and" before question words.
    parts = re.split(
        r",\s*(?!\s*\d{4}\b)|;\s*|\s+and\s+(?=how|what|which|when|show|get|check)",
        query.strip(),
        flags=re.I,
    )
    clauses = []
    for p in parts:
        s = p.strip()
        if s.lower().startswith("and "):
            s = s[4:].strip()
        # Strip trailing "assistant" (LLM echo artifact).
        if s and s.lower().endswith(" assistant"):
            s = s[:-9].strip()
        if s and len(s) > 3:
            clauses.append(s)
    if len(clauses) < 2:
        return None
    return "\n".join(f"    - {c}" for c in clauses)


def _format_dispatcher_section_html(
    plan: Optional[Any],
    task_results: Optional[List[Any]],
    debug: Optional[Dict[str, Any]],
    response_workflow: Optional[str] = None,
) -> str:
    """
    Render section 4 (Dispatcher) as HTML matching Route LLM trace cards.

    Shows plan build time, worker execution time, plan summary, and per-task metrics.

    Args:
        plan: Rewrite plan dict from API (plan_type, task_groups), or None.
        task_results: List of task result dicts (workflow, status, duration_ms, query, ...).
        debug: Response debug dict; may include plan_build_ms, dispatch_execute_ms.
        response_workflow: Top-level ``workflow`` from ``/query`` (e.g. ``rewrite_only``).

    Returns:
        HTML string for Gradio Markdown, or empty string when nothing to show.
    """
    try:
        debug = debug or {}
        plan_build = debug.get("plan_build_ms")
        dispatch_ms = debug.get("dispatch_execute_ms")
        raw_tasks = task_results or []

        plan_type = "single_domain"
        planned_task_count = 0
        parallel_groups = 0
        if plan and isinstance(plan, dict):
            plan_type = str(plan.get("plan_type") or "single_domain")
            for g in plan.get("task_groups") or []:
                if isinstance(g, dict):
                    parallel_groups += 1
                    planned_task_count += len(g.get("tasks") or [])

        completed = 0
        failed = 0
        skipped = 0
        normalized_tasks: List[Dict[str, Any]] = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            st = str(t.get("status") or "").lower()
            if st == "completed":
                completed += 1
            elif st == "failed":
                failed += 1
            elif st == "skipped":
                skipped += 1
            normalized_tasks.append(t)

        wf_top = str(response_workflow or "").strip().lower()

        # Skip empty card when API omitted dispatcher metrics and there is no task data.
        if (
            plan_build is None
            and dispatch_ms is None
            and not normalized_tasks
            and planned_task_count == 0
            and wf_top != "rewrite_only"
        ):
            return ""

        lines: List[str] = [
            "<div style=\"border: 1px solid #d1d5db; border-radius: 10px; "
            "padding: 10px 12px; margin: 8px 0; background-color: #f9fafb;\">",
            "<h4 style=\"margin: 0 0 8px 0;\">4. Dispatcher</h4>",
            "<ul style=\"margin: 0; padding-left: 20px;\">",
        ]
        # Gateway route-only: /query returns before workers; explain missing timings.
        if wf_top == "rewrite_only" and dispatch_ms is None:
            lines.append(
                "<li><strong>Worker execution:</strong> skipped (gateway rewrite-only / route-only; "
                "no dispatcher run)</li>"
            )
        if plan_build is not None:
            try:
                lines.append(f"<li>Plan build: {int(plan_build)} ms</li>")
            except (TypeError, ValueError):
                lines.append(f"<li>Plan build: {html.escape(str(plan_build))}</li>")
        if dispatch_ms is not None:
            try:
                lines.append(f"<li>Execute plan (workers): {int(dispatch_ms)} ms</li>")
            except (TypeError, ValueError):
                lines.append(
                    f"<li>Execute plan (workers): {html.escape(str(dispatch_ms))}</li>"
                )
        lines.append(f"<li>Plan type: {html.escape(plan_type)}</li>")
        lines.append(
            f"<li>Task groups: {parallel_groups} | Planned tasks: {planned_task_count}</li>"
        )
        lines.append(
            f"<li>Results: <strong>{completed}</strong> completed, "
            f"{failed} failed, {skipped} skipped</li>"
        )
        for i, tr in enumerate(normalized_tasks, 1):
            wf = html.escape(str(tr.get("workflow") or "—"))
            st = html.escape(str(tr.get("status") or "—"))
            try:
                ms = int(tr.get("duration_ms") or 0)
            except (TypeError, ValueError):
                ms = 0
            q = (tr.get("query") or "").strip()
            if len(q) > 120:
                q = q[:117] + "..."
            q_esc = html.escape(q)
            tid = html.escape(str(tr.get("task_id") or f"task_{i}"))
            err = tr.get("error")
            err_part = f" | error: {html.escape(str(err))}" if err else ""
            lines.append(
                f"<li>Task {i} (<code>{tid}</code>): workflow <code>{wf}</code> | "
                f"status {st} | {ms} ms | query: {q_esc}{err_part}</li>"
            )
        lines.extend(["</ul>", "</div>"])
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Dispatcher section HTML failed (non-fatal): %s", exc)
        return ""


def _to_single_line(text: str) -> str:
    """Normalize any model text into a single display-safe line."""
    if not text:
        return ""
    one_line = re.sub(r"\s+", " ", text).strip()
    # Avoid markdown code-span breakage when model outputs backticks.
    return one_line.replace("`", "'")


def _format_intent_classification_lines(
    intent_details_list: List[Dict[str, Any]],
    intents_list: List[str],
    workflows_list: List[str],
    classification_time_ms: Optional[int] = None,
) -> List[str]:
    """
    Build HTML list block for intent classification (renders as bullet points).
    """
    lines: List[str] = []
    if not (intent_details_list or intents_list or workflows_list):
        return lines

    lines.append("<h4 style=\"margin: 0 0 8px 0;\">3. Classification</h4>")
    lines.append("<p style=\"margin: 4px 0;\"><strong>Intent classification list:</strong></p>")
    if intent_details_list:
        for detail in intent_details_list:
            intent_text = (detail.get("intent") or detail.get("query") or "").strip()
            if not intent_text:
                continue
            final_wf = (detail.get("workflow") or "general").strip() or "general"
            keyword_wf = (detail.get("keyword") or "").strip()
            vector_wf = (detail.get("vector") or "").strip()
            confidence = (detail.get("confidence") or "").strip()
            source = (detail.get("source") or "").strip()
            intent_ms = detail.get("intent_elapsed_ms")
            step_timings = detail.get("step_timings") or []
            lines.append(f"<p style=\"margin: 4px 0 2px 0;\">{intent_text}</p>")
            extra_parts: List[str] = [f"Workflow: {final_wf}"]
            if intent_ms is not None:
                extra_parts.append(f"Use time: {intent_ms} ms")
            if step_timings:
                step_strs = [f"{s.get('workflow', '?')}: {s.get('ms', 0)} ms" for s in step_timings]
                extra_parts.append(f"LLM calls: {len(step_timings)} ({', '.join(step_strs)})")
            if keyword_wf or vector_wf:
                extra_parts.insert(0, f"Keyword: {keyword_wf or '—'}, Vector: {vector_wf or '—'}")
            if confidence:
                extra_parts.append(f"Confidence: {confidence}")
            if source:
                extra_parts.append(f"Source: {source}")
            lines.append(
                f"<div style=\"margin: 2px 0 8px 0; padding: 6px 10px; background-color: #e5e7eb; border-radius: 6px; color: #4b5563;\">"
                + ", ".join(extra_parts)
                + "</div>"
            )
    else:
        lines.append("<ul style=\"margin: 0; padding-left: 20px;\">")
        for item in intents_list:
            line = (item or "").strip()
            if line:
                lines.append(f"<li>{line}</li>")
        lines.append("</ul>")

    lines.append("<ul style=\"margin: 8px 0 0 0; padding-left: 20px;\">")
    if workflows_list:
        workflows_str = ", ".join(workflows_list)
        lines.append(f"<li><strong>Intent classification result:</strong> {workflows_str}</li>")
    if classification_time_ms is not None:
        lines.append(f"<li><strong>Classification time:</strong> {classification_time_ms} ms</li>")
    if intent_details_list:
        total_calls = sum(
            len(d.get("step_timings") or [])
            for d in intent_details_list
        )
        if total_calls > 0:
            lines.append(f"<li><strong>Classification LLM calls:</strong> {total_calls}</li>")
    lines.append("</ul>")
    return lines


# Keys copied from final /rewrite (or stream ``complete``) into UI state for cards 1–3.
REWRITE_PREVIEW_MERGE_KEYS: tuple[str, ...] = (
    "clarification_time_ms",
    "clarification_backend",
    "clarification_status",
    "memory_rounds",
    "memory_text_length",
    "rewritten_query",
    "rewrite_time_ms",
    "rewrite_backend",
    "intents",
    "intent_details",
    "workflows",
    "classification_time_ms",
)


def _format_rewrite_plan_markdown(rewrite_result: Dict[str, Any]) -> str:
    """
    Build optional markdown appendix when rewrite preview includes an execution plan.

    Args:
        rewrite_result: Parsed /rewrite (or stream ``complete``) payload.

    Returns:
        Markdown fragment or empty string when no plan tasks exist.
    """
    try:
        plan = rewrite_result.get("plan")
        if not plan or not isinstance(plan, dict):
            return ""
        plan_type = plan.get("plan_type", "single_domain")
        task_groups = plan.get("task_groups") or []
        tasks_flat: List[Dict[str, Any]] = []
        for g in task_groups:
            tasks_flat.extend(g.get("tasks") or [])
        if not tasks_flat:
            return ""
        lines = [f"\n\n**Plan** (`{plan_type}`):\n"]
        for i, t in enumerate(tasks_flat, 1):
            wf = t.get("workflow", "general")
            q = (t.get("query") or "").strip()
            lines.append(f"\n{i}. `{wf}`: {q}")
        return "".join(lines)
    except Exception as exc:
        logger.warning("Plan markdown formatting failed (non-fatal): %s", exc)
        return ""


def _build_rewrite_preview_html_staged(
    state: Dict[str, Any],
    phases_done: int,
    original_query: str,
    rewrite_backend_fallback: str,
) -> str:
    """
    Assemble cumulative HTML for rewrite preview cards (clarification / rewrite / classification).

    Args:
        state: Merged fields from stream payloads (and optional final rewrite_response).
        phases_done: Number of stages to include: 1, 2, or 3.
        original_query: Raw user text (fallback before rewritten_query exists).
        rewrite_backend_fallback: UI/env backend label if API omits rewrite_backend.

    Returns:
        HTML string for Gradio chat (markdown with embedded HTML).
    """
    if phases_done < 1:
        return ""

    memory_rounds = int(state.get("memory_rounds") or 0)
    memory_text_len = int(state.get("memory_text_length") or 0)
    clarification_status = str(state.get("clarification_status") or "—")
    clarification_backend_value = _display_clarification_backend(
        state.get("clarification_backend")
    )
    clarification_time_raw = state.get("clarification_time_ms")
    clarification_time_display = (
        f"{int(clarification_time_raw)} ms"
        if clarification_time_raw is not None
        else "N/A"
    )

    lines_parts: List[str] = [
        "<div style=\"border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; margin: 8px 0;\">",
        "<h4 style=\"margin: 0 0 8px 0;\">1. Clarification</h4>",
        "<ul style=\"margin: 0; padding-left: 20px;\">",
        f"<li>Integrate historical conversations: {memory_rounds} rounds (text length: {memory_text_len} chars)</li>",
        f"<li>Clarification backend: {clarification_backend_value}</li>",
        f"<li>Clarification: {clarification_status}</li>",
        f"<li>Clarification time: {clarification_time_display}</li>",
        "</ul>",
        "</div>",
        "",
    ]

    if phases_done >= 2:
        routed = _to_single_line(
            str(state.get("rewritten_query") or original_query)
        )
        try:
            rewrite_ms = int(state.get("rewrite_time_ms") or 0)
        except (TypeError, ValueError):
            rewrite_ms = 0
        rewrite_backend_value = _display_rewrite_backend(
            state.get("rewrite_backend") or rewrite_backend_fallback
        )
        lines_parts.extend(
            [
                "<div style=\"border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; margin: 8px 0;\">",
                "<h4 style=\"margin: 0 0 8px 0;\">2. Rewritten</h4>",
                "<ul style=\"margin: 0; padding-left: 20px;\">",
                f"<li>Integrate short-term memory: {memory_rounds} rounds (text length: {memory_text_len} chars)</li>",
                "<li>Normalize: Completed</li>",
                f"<li>Rewritten query: {routed}</li>",
                f"<li>Rewrite backend: {rewrite_backend_value}</li>",
                f"<li>Rewrite time: {rewrite_ms} ms</li>",
                "</ul>",
                "</div>",
                "",
            ]
        )

    if phases_done >= 3:
        classification_ms = state.get("classification_time_ms")
        if classification_ms is not None:
            try:
                classification_ms = int(classification_ms)
            except (TypeError, ValueError):
                classification_ms = None
        intents_list = list(state.get("intents") or [])
        intent_details_list = list(state.get("intent_details") or [])
        workflows_list = list(state.get("workflows") or [])
        lines_parts.append(
            "<div style=\"border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; margin: 8px 0;\">"
        )
        lines_parts.extend(
            _format_intent_classification_lines(
                intent_details_list=intent_details_list,
                intents_list=intents_list,
                workflows_list=workflows_list,
                classification_time_ms=classification_ms,
            )
        )
        lines_parts.append("</div>")

    return "\n".join(lines_parts)


# Config from env (after load_dotenv above so .env is applied)
GATEWAY_API_URL = os.environ.get("GATEWAY_API_URL", "").rstrip("/")
GATEWAY_MOCK = os.environ.get("GATEWAY_MOCK", "").lower() in ("true", "1", "yes")
GRADIO_PORT = int(os.environ.get("UNIFIED_CHAT_GRADIO_PORT", "7862"))
REWRITE_ENABLE_DEFAULT = os.environ.get(
    "UNIFIED_CHAT_REWRITE_ENABLE", "true"
).lower() in ("true", "1", "yes")


def _gateway_rewrite_backend_from_env() -> str:
    """
    Effective rewrite-stage chat backend (same precedence as gateway ``chat_backend_policy``).

    Returns:
        ``ollama`` or ``deepseek``.
    """
    try:
        return resolve_chat_backend("rewrite")
    except Exception as exc:
        logger.warning("rewrite backend from policy failed: %s; using deepseek", exc)
        return "deepseek"


def _gateway_clarification_backend_from_env() -> str:
    """
    Effective clarification chat backend (same precedence as gateway ``chat_backend_policy``).

    Returns:
        ``ollama`` or ``deepseek``.
    """
    try:
        return resolve_chat_backend("clarification")
    except Exception as exc:
        logger.warning("clarification backend from policy failed: %s; using deepseek", exc)
        return "deepseek"


def _display_clarification_backend(api_value: Any) -> str:
    """Prefer API-reported backend; if absent, show .env-configured clarification backend."""
    if api_value is not None and str(api_value).strip():
        return str(api_value).strip()
    return _gateway_clarification_backend_from_env()


def _display_rewrite_backend(api_value: Any) -> str:
    """Prefer API-reported backend; if absent, show .env-configured rewrite backend."""
    if api_value is not None and str(api_value).strip():
        return str(api_value).strip()
    return _gateway_rewrite_backend_from_env()


# Default rewrite_backend sent on /rewrite and /query: must match gateway .env (no separate UI var).
REWRITE_BACKEND_DEFAULT = _gateway_rewrite_backend_from_env()
# Bumped when chat client behavior changes. Shown in Status panel so operators can confirm UI reload.
UNIFIED_CHAT_CLIENT_BUILD = "trace-amazon-docs-chroma-count-2026-03-18"
SKIP_LOGIN = os.environ.get("UNIFIED_CHAT_SKIP_LOGIN", "").lower() in ("true", "1", "yes")
# Default credentials when SKIP_LOGIN: auto sign-in so token/user_id are set (e.g. for history).
DEFAULT_SKIP_LOGIN_USER = os.environ.get("UNIFIED_CHAT_SKIP_LOGIN_USER", "lenhe")
DEFAULT_SKIP_LOGIN_PASSWORD = os.environ.get("UNIFIED_CHAT_SKIP_LOGIN_PASSWORD", "juvale`123")

# Workflow options: (label, value) for Dropdown
WORKFLOW_CHOICES = [
    ("Auto", "auto"),
    ("General", "general"),
    ("Amazon Docs", "amazon_docs"),
    ("IC Docs", "ic_docs"),
    ("SP-API", "sp_api"),
    ("UDS", "uds"),
]


def _do_signin(user_name: str, password: str) -> tuple[Optional[str], Optional[Dict], str, bool, bool]:
    """
    Sign in handler. Returns (token, user_info, message, show_chat, show_login).
    """
    if not user_name or not user_name.strip():
        return None, None, "Please enter user name.", False, True
    if not password or not password.strip():
        return None, None, "Please enter password.", False, True
    client = GatewayClient(base_url=GATEWAY_API_URL or None)
    result = client.signin_sync(user_name=user_name.strip(), password=password)
    err = result.get("error")
    if err:
        return None, None, f"Sign in failed: {err}", False, True
    token = result.get("access_token")
    user = result.get("user") or {}
    return token, user, f"Signed in as {user.get('user_name', user_name)}", True, False


def _do_register(user_name: str, password: str, email: str) -> tuple[Optional[str], Optional[Dict], str, bool, bool]:
    """
    Register handler. Returns (token, user_info, message, show_chat, show_login).
    After register we auto sign-in to get token.
    """
    if not user_name or not user_name.strip():
        return None, None, "Please enter user name.", False, True
    if not password or not password.strip():
        return None, None, "Please enter password.", False, True
    client = GatewayClient(base_url=GATEWAY_API_URL or None)
    reg_result = client.register_sync(
        user_name=user_name.strip(),
        password=password,
        email=(email or "").strip() or None,
    )
    err = reg_result.get("error")
    if err:
        return None, None, f"Registration failed: {err}", False, True
    # Auto sign-in after register
    signin_result = client.signin_sync(user_name=user_name.strip(), password=password)
    signin_err = signin_result.get("error")
    if signin_err:
        return None, None, f"Registered but sign in failed: {signin_err}", False, True
    token = signin_result.get("access_token")
    user = signin_result.get("user") or {}
    return token, user, f"Registered and signed in as {user.get('user_name', user_name)}", True, False


def _do_signout() -> tuple[None, None, dict, dict]:
    """Sign out handler. Returns (None, None, login_visible, chat_visible)."""
    client = GatewayClient(base_url=GATEWAY_API_URL or None)
    client.signout_sync()
    return None, None, gr.update(visible=True), gr.update(visible=False)


def _auto_signin_if_skip_login(
    session_id: Optional[str] = None,
    existing_chat_history: Optional[List[Dict[str, str]]] = None,
) -> tuple:
    """
    When SKIP_LOGIN is True, sign in with default credentials and return state updates.
    Used on demo load so token/user_info and history are set without showing login.

    On every full browser refresh, Gradio runs this once (not an empty chat query). It calls
    the gateway sign-in API and may fetch session history; until it returns, the UI can show
    a loading/processing state.

    Args:
        session_id: Current Session ID state (same as left panel). Used for history fetch.
        existing_chat_history: Current ChatInterface history state.

    Returns:
        Tuple for demo.load outputs: token, user, login_panel, chat_panel, signin_msg,
        user_md, chatbot update, chatbot_state.
    """
    if not SKIP_LOGIN:
        return (
            None,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
            gr.update(),
            [],
        )
    token, user, msg, show_chat, show_login = _do_signin(
        DEFAULT_SKIP_LOGIN_USER, DEFAULT_SKIP_LOGIN_PASSWORD
    )
    user_md = (
        f"**UserName:** {user.get('user_name', '')}\n**Role:** {user.get('role', 'general')}"
        if user else ""
    )
    chat_history: List[Dict[str, str]] = []
    if token and show_chat:
        client = GatewayClient(base_url=GATEWAY_API_URL or None)
        try:
            sid = (session_id or "").strip() or str(uuid4())
            hist_result = client.get_session_history_sync(sid, last_n=3)
            if not hist_result.get("error"):
                raw = hist_result.get("history", [])
                for h in raw:
                    q, a = h.get("query", ""), h.get("answer", "")
                    if q or a:
                        chat_history.append({"role": "user", "content": q})
                        chat_history.append({"role": "assistant", "content": a})
        except Exception as e:
            logger.debug("Auto sign-in: load history failed (non-fatal): %s", e)
    # In skip-login mode always show chat panel (never show login form).
    login_vis = False if SKIP_LOGIN else show_login
    chat_vis = True if SKIP_LOGIN else show_chat
    # Important: on websocket reconnect, demo.load can fire again. If history API returns empty
    # (or temporarily fails), do not wipe the user's current chat content.
    has_fetched_history = len(chat_history) > 0
    chatbot_update = gr.update(value=chat_history) if has_fetched_history else gr.update()
    chatbot_state_value = chat_history if has_fetched_history else (existing_chat_history or [])
    return (
        token,
        user,
        gr.update(visible=login_vis),
        gr.update(visible=chat_vis),
        msg or "",
        user_md,
        chatbot_update,
        chatbot_state_value,
    )


def _get_gateway_status() -> str:
    """Return human-readable gateway status for sidebar."""
    client_line = f"\n**UI client build:** `{UNIFIED_CHAT_CLIENT_BUILD}` (always calls `/query` after preview)"
    if not GATEWAY_API_URL or GATEWAY_MOCK:
        return "**Status:** Mock mode (no gateway)" + client_line
    return f"**Status:** Gateway at {GATEWAY_API_URL}" + client_line


def _normalize_rewrite_backend(value: str) -> str:
    """
    Normalize rewrite backend from env/UI.

    Args:
        value: Candidate backend value.

    Returns:
        Valid backend value: "ollama" or "deepseek".
    """
    normalized = (value or "").strip().lower()
    if normalized in ("ollama", "local", "deepseek", "ds"):
        if normalized in ("deepseek", "ds"):
            return "deepseek"
        return "ollama"
    return "deepseek"


def _amazon_docs_chroma_source_count(result: Dict[str, Any]) -> Optional[int]:
    """
    Count Chroma-backed source rows for amazon_docs worker task(s) in a /query response.

    RAG maps each similarity-gated Chroma hit to one ``sources`` entry
    (see ChromaRetriever.hits_to_sources). For ``hybrid`` plans, only tasks whose
    ``workflow`` is ``amazon_docs`` are counted.

    Args:
        result: Parsed gateway JSON (dict) including ``task_results`` and/or ``sources``.

    Returns:
        Chunk count when amazon_docs ran; None when this query did not involve amazon_docs.
    """
    try:
        task_results = result.get("task_results") or []
        count_amazon = 0
        saw_amazon_task = False
        for tr in task_results:
            if not isinstance(tr, dict):
                continue
            w = str(tr.get("workflow") or "").strip().lower()
            if w != "amazon_docs":
                continue
            saw_amazon_task = True
            if str(tr.get("status") or "").strip().lower() == "completed":
                src = tr.get("sources")
                if isinstance(src, list):
                    count_amazon += len(src)
        if saw_amazon_task:
            return count_amazon
        wf = str(result.get("workflow") or "").strip().lower()
        if wf == "amazon_docs":
            sources = result.get("sources")
            if isinstance(sources, list):
                return len(sources)
            return 0
        return None
    except (TypeError, ValueError) as exc:
        logger.warning("amazon_docs Chroma count failed (non-fatal): %s", exc)
        return None


def _chat_handler(
    message: str,
    history: List[Tuple[str, str]],
    workflow: str,
    rewrite_preview: bool,
    rewrite_backend: str,
    session_id: str,
    auth_token: Optional[str] = None,
    user_info: Optional[Dict[str, Any]] = None,
) -> Union[str, Iterator[str]]:
    """
    Chat callback: stream rewrite progress and final answer.

    Args:
        message: User message.
        history: Chat history (unused; ChatInterface manages it).
        workflow: Selected workflow (auto|general|amazon_docs|ic_docs|sp_api|uds).
        rewrite_preview: When True, call /rewrite first to show unified rewrite preview.
        rewrite_backend: Rewrite LLM backend: "ollama" or "deepseek".
        session_id: Session UUID for multi-turn context.
        auth_token: Optional JWT for protected gateway.
        user_info: Optional user dict with user_id for user-scoped history.

    Returns:
        Iterator yielding intermediate/final messages, or a plain string for
        immediate validation errors.
    """
    if not message or not message.strip():
        yield "Please enter a question."
        return

    client = GatewayClient(base_url=GATEWAY_API_URL or None)
    user_id = (user_info.get("user_id") or "").strip() if user_info else None
    raw_query = message.strip()

    # Merge with pending query when this is a clarification follow-up
    pending = _get_pending_query(session_id)
    if pending:
        raw_query = f"{pending} {raw_query}".strip()
        routed_query = raw_query
    else:
        routed_query = raw_query
    rewrite_ms: Optional[int] = None
    rewrite_backend_value = "none"
    rewrite_message: Optional[str] = None
    # Sum of clarification + rewrite + classification (ms) for UI trace after full query.
    route_llm_total_ms: Optional[int] = None

    # Optional first call: /rewrite/stream yields NDJSON so cards 1–3 appear as each stage finishes.
    rewrite_result: Dict[str, Any] = {}
    ui_state: Dict[str, Any] = {}
    if rewrite_preview:
        rb_fallback = (rewrite_backend or "").strip()
        for event in client.rewrite_stream_sync(
            query=raw_query,
            rewrite_backend=rb_fallback or None,
            session_id=session_id or None,
            user_id=user_id,
            token=auth_token,
        ):
            step = event.get("step")
            if step == "clarification":
                ui_state.update(event.get("payload") or {})
                rewrite_message = _build_rewrite_preview_html_staged(
                    ui_state, 1, raw_query, rb_fallback
                )
                yield rewrite_message
            elif step == "rewrite":
                ui_state.update(event.get("payload") or {})
                rewrite_message = _build_rewrite_preview_html_staged(
                    ui_state, 2, raw_query, rb_fallback
                )
                yield rewrite_message
            elif step == "classification":
                ui_state.update(event.get("payload") or {})
                rewrite_message = _build_rewrite_preview_html_staged(
                    ui_state, 3, raw_query, rb_fallback
                )
                yield rewrite_message
            elif step == "complete":
                rewrite_result = event.get("rewrite_response") or {}
                if (
                    not rewrite_result.get("error")
                    and not rewrite_result.get("clarification_required")
                ):
                    # Always rebuild cards from the final payload. Proxies or clients may only
                    # deliver this event (no progressive NDJSON), or drops earlier lines.
                    for key in REWRITE_PREVIEW_MERGE_KEYS:
                        if key in rewrite_result and rewrite_result[key] is not None:
                            ui_state[key] = rewrite_result[key]
                    merged_preview = (
                        _build_rewrite_preview_html_staged(
                            ui_state, 3, raw_query, rb_fallback
                        )
                        + _format_rewrite_plan_markdown(rewrite_result)
                    )
                    # Avoid a duplicate Gradio update when progressive events already matched.
                    prev_preview = rewrite_message
                    rewrite_message = merged_preview
                    if merged_preview != prev_preview:
                        yield merged_preview

        # Stream returned no terminal payload (empty body, non-NDJSON, parse errors).
        if rewrite_preview and not rewrite_result:
            sync_fallback = client.rewrite_sync(
                query=raw_query,
                rewrite_backend=rb_fallback or None,
                session_id=session_id or None,
                user_id=user_id,
                token=auth_token,
            )
            if isinstance(sync_fallback, dict) and sync_fallback:
                rewrite_result = sync_fallback
                if (
                    not sync_fallback.get("error")
                    and not sync_fallback.get("clarification_required")
                ):
                    for key in REWRITE_PREVIEW_MERGE_KEYS:
                        if key in sync_fallback and sync_fallback[key] is not None:
                            ui_state[key] = sync_fallback[key]
                    rewrite_message = (
                        _build_rewrite_preview_html_staged(
                            ui_state, 3, raw_query, rb_fallback
                        )
                        + _format_rewrite_plan_markdown(sync_fallback)
                    )
                    yield rewrite_message

        rewrite_error = rewrite_result.get("error")
        if rewrite_error:
            err_text = (
                "Rewrite failed; continuing with original query.\n"
                f"Error: {rewrite_error}"
            )
            # Keep any progressive cards already shown before the failure.
            yield f"{rewrite_message}\n\n{err_text}" if rewrite_message else err_text
        elif rewrite_result.get("clarification_required"):
            # Clarification needed: display question and store pending for follow-up
            clarification_question = (
                rewrite_result.get("clarification_question") or "Please provide more details."
            )
            clarification_backend = _display_clarification_backend(
                rewrite_result.get("clarification_backend")
            )
            pending_query = rewrite_result.get("pending_query") or raw_query
            _set_pending_query(session_id, pending_query)
            yield (
                f"**Clarification needed ({clarification_backend}):**\n\n"
                f"{clarification_question}\n\n"
                "---\n"
                "Reply with the missing details (e.g. store, ASIN, date range). "
                "Your follow-up will be merged with the original query."
            )
            return
        elif rewrite_result:
            routed_query = _to_single_line(
                str(rewrite_result.get("rewritten_query") or raw_query)
            )
            rewrite_ms = int(rewrite_result.get("rewrite_time_ms") or 0)
            rewrite_backend_value = _display_rewrite_backend(
                rewrite_result.get("rewrite_backend") or rewrite_backend
            )
            clarification_time_raw = rewrite_result.get("clarification_time_ms")
            classification_ms = rewrite_result.get("classification_time_ms")
            clar_ms_int = 0
            if clarification_time_raw is not None:
                try:
                    clar_ms_int = int(clarification_time_raw)
                except (TypeError, ValueError):
                    clar_ms_int = 0
            rewrite_ms_int = 0
            if rewrite_ms is not None:
                try:
                    rewrite_ms_int = int(rewrite_ms)
                except (TypeError, ValueError):
                    rewrite_ms_int = 0
            class_ms_int = 0
            if classification_ms is not None:
                try:
                    class_ms_int = int(classification_ms)
                except (TypeError, ValueError):
                    class_ms_int = 0
            route_llm_total_ms = clar_ms_int + rewrite_ms_int + class_ms_int
            if not rewrite_message:
                rewrite_message = _build_rewrite_preview_html_staged(
                    ui_state, 3, raw_query, rb_fallback
                ) + _format_rewrite_plan_markdown(rewrite_result)
                yield rewrite_message

    # Full query: always call /query so planner + dispatcher run (section 4 card when gateway executes workers).
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.query_sync,
            query=routed_query,
            workflow=workflow or "auto",
            rewrite_backend=(rewrite_backend or "").strip() or None,
            session_id=session_id or None,
            user_id=user_id,
            token=auth_token,
        )
        elapsed_seconds = 0
        while not future.done():
            time.sleep(1)
            elapsed_seconds += 1
            if elapsed_seconds == 1 or elapsed_seconds % 10 == 0:
                if rewrite_message:
                    yield (
                        f"{rewrite_message}\n\n"
                        f"Processing routed query... `{elapsed_seconds}s` elapsed."
                    )
                else:
                    yield f"Processing query... `{elapsed_seconds}s` elapsed."

        result: Dict[str, Any] = future.result()

    backend_error = result.get("error")
    if backend_error:
        yield f"Error: {backend_error}"
        return

    # Handle clarification response: store pending_query and display question
    if result.get("clarification_required"):
        clarification_question = result.get("clarification_question") or "Please provide more details."
        clarification_backend = _display_clarification_backend(
            result.get("clarification_backend")
        )
        pending_query = result.get("pending_query") or routed_query
        _set_pending_query(session_id, pending_query)
        yield (
            f"**Clarification needed ({clarification_backend}):** {clarification_question}\n\n"
            "Please provide the requested information in your next message."
        )
        return

    answer = result.get("answer", "")
    if not answer:
        yield "No response from gateway."
        return

    debug = result.get("debug") or {}
    route_source = str(debug.get("route_source") or "unknown")
    route_conf = result.get("routing_confidence")

    trace_lines = [
        "",
        "---",
        "Trace",
        f"- Routed Input: `{routed_query}`",
        f"- Rewrite: `{rewrite_backend_value}` in `{rewrite_ms if rewrite_ms is not None else 0} ms`",
        f"- Route Source: `{route_source}` | Confidence: `{route_conf}`",
    ]
    if route_llm_total_ms is not None:
        trace_lines.append(
            f"- Route LLM total (Clarification + Rewritten + Classification): `{route_llm_total_ms} ms`"
        )
    chroma_amazon_n = _amazon_docs_chroma_source_count(result)
    if chroma_amazon_n is not None:
        trace_lines.append(
            f"- Chroma (amazon_docs) returned chunks: `{chroma_amazon_n}`"
        )
    pb = debug.get("plan_build_ms")
    de = debug.get("dispatch_execute_ms")
    if pb is not None or de is not None:
        trace_lines.append(
            f"- Dispatcher: plan build `{pb} ms` | workers `{de} ms`"
        )

    dispatcher_html = _format_dispatcher_section_html(
        result.get("plan"),
        result.get("task_results"),
        debug,
        response_workflow=str(result.get("workflow") or ""),
    )
    # Final chunk must include Route LLM HTML (1–4); streaming replaces prior assistant text.
    answer_and_trace = f"{answer}\n" + "\n".join(trace_lines)
    final_parts: List[str] = []
    if rewrite_message:
        final_parts.append(rewrite_message)
    if dispatcher_html:
        final_parts.append(dispatcher_html)
    final_parts.append(answer_and_trace)
    yield "\n\n".join(final_parts)


# CSS to make chat dialog and input box taller, keep input visible at bottom
CHAT_DIALOG_CSS = """
/* Chat column: fixed height, flex column, messages scroll, input stays visible at bottom */
#ic_chat_column { height: 95vh !important; max-height: 95vh !important; display: flex !important; flex-direction: column !important; overflow: hidden !important; }
#ic_chat_column > div { flex: 1 !important; min-height: 0 !important; overflow: hidden !important; display: flex !important; flex-direction: column !important; }
#ic_chat_column .contain { flex: 1 !important; min-height: 0 !important; overflow: hidden !important; display: flex !important; flex-direction: column !important; }
/* Chatbot/messages: flex-grow to fill space, scroll internally, never push input off */
#ic_chatbot { flex: 1 1 auto !important; min-height: 0 !important; overflow-y: auto !important; max-height: none !important; }
.chatbot { flex: 1 1 auto !important; min-height: 0 !important; overflow-y: auto !important; }
[data-testid="chatbot"] { flex: 1 1 auto !important; min-height: 0 !important; overflow-y: auto !important; }
.gr-chat { flex: 1 !important; min-height: 0 !important; display: flex !important; flex-direction: column !important; overflow: hidden !important; }
.gr-chat .scroll-hide, .gr-chat [class*="scroll"] { flex: 1 1 auto !important; min-height: 0 !important; overflow-y: auto !important; }
/* Input area: flex-shrink 0 so it stays visible at bottom, never covered */
#ic_chat_column .gr-box, #ic_chat_column form, .gr-chat .gr-box, .gr-chat form { flex-shrink: 0 !important; }
/* Input box: taller for multi-line typing */
#ic_chat_column textarea { min-height: 150px !important; height: 150px !important; flex-shrink: 0 !important; }
.gr-chat textarea { min-height: 150px !important; height: 150px !important; flex-shrink: 0 !important; }
/* Workflow radio: single column, left-aligned, no extra gap */
#workflow_radio { flex: 0 0 auto !important; }
#workflow_radio .wrap { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio .contain { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio > div { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio [class*="wrap"] { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
/* Left column: pack content at top, no stretch */
#ic_controls_column { justify-content: flex-start !important; align-items: flex-start !important; }
/* Sign out button: top-right, compact */
#ic_signout_row { justify-content: flex-end !important; padding: 0 4px !important; margin-bottom: 4px !important; }
#ic_signout_row button { max-width: 120px !important; }
"""

# Keep JS hook as no-op to preserve launch signature without forcing scroll.
CHAT_AUTOSCROLL_JS = ""


def create_demo() -> gr.Blocks:
    """
    Build and return the Gradio Blocks demo.

    Returns:
        Configured gr.Blocks instance (not launched).
    """
    # Initial session ID
    initial_session_id = str(uuid4())

    with gr.Blocks(
        title="IC-RAG-Agent Chat",
        theme=gr.themes.Soft(),
        fill_height=True,
        css=CHAT_DIALOG_CSS,
    ) as demo:
        gr.Markdown("# IC-RAG-Agent Chat")

        session_id_state = gr.State(value=initial_session_id)
        auth_token_state = gr.State(value=None)
        user_info_state = gr.State(value=None)

        # Login panel: visible when not logged in (hidden when UNIFIED_CHAT_SKIP_LOGIN=true).
        with gr.Column(visible=not SKIP_LOGIN, elem_id="ic_login_panel") as login_panel:
            with gr.Tabs():
                with gr.Tab("Sign In"):
                    signin_user = gr.Textbox(label="User Name", placeholder="Enter user name")
                    signin_pass = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                    signin_btn = gr.Button("Sign In", variant="primary")
                    signin_msg = gr.Markdown("")
                with gr.Tab("Register"):
                    reg_user = gr.Textbox(label="User Name", placeholder="Enter user name")
                    reg_pass = gr.Textbox(label="Password", type="password", placeholder="Min 8 chars, 1 letter, 1 digit")
                    reg_email = gr.Textbox(label="Email (optional)", placeholder="Enter email")
                    reg_btn = gr.Button("Register", variant="primary")
                    reg_msg = gr.Markdown("")

        # Chat panel: visible when logged in (or when SKIP_LOGIN for dev).
        with gr.Column(visible=SKIP_LOGIN, elem_id="ic_chat_panel") as chat_panel:
            with gr.Row(elem_id="ic_signout_row"):
                signout_btn = gr.Button("Sign Out", variant="secondary", size="sm")
            with gr.Row():
                with gr.Column(scale=1, elem_id="ic_controls_column"):
                    gr.Markdown("### Workflow")
                    workflow_radio = gr.Dropdown(
                        choices=WORKFLOW_CHOICES,
                        value="auto",
                        label="Workflow",
                        info="auto|general|amazon_docs|ic_docs|sp_api|uds",
                    )
                    # Optional /rewrite preview before full query; backend from env default.
                    rewrite_preview_checkbox = gr.State(value=True)
                    rewrite_backend_dropdown = gr.State(value=_normalize_rewrite_backend(REWRITE_BACKEND_DEFAULT))
                    gr.Markdown("### User")
                    user_display = gr.Markdown("", elem_id="ic_user_display", line_breaks=True)
                    gr.Markdown("### Session")
                    session_display = gr.Textbox(
                        label="Session ID",
                        value=initial_session_id,
                        interactive=False,
                    )
                    clear_btn = gr.Button("Clear Session", variant="secondary")
                    gr.Markdown("### Status")
                    health_status = gr.Markdown(_get_gateway_status())

                with gr.Column(scale=3, elem_id="ic_chat_column"):
                    chatbot = gr.Chatbot(value=[], elem_id="ic_chatbot")
                    chat = gr.ChatInterface(
                        fn=_chat_handler,
                        chatbot=chatbot,
                        title="",
                        additional_inputs=[
                            workflow_radio,
                            rewrite_preview_checkbox,
                            rewrite_backend_dropdown,
                            session_id_state,
                            auth_token_state,
                            user_info_state,
                        ],
                        fill_height=True,
                        autoscroll=False,
                    )

        def on_signin(su: str, sp: str, session_id: str):
            token, user, msg, show_chat, show_login = _do_signin(su, sp)
            user_md = (
                f"**UserName:** {user.get('user_name', '')}\n**Role:** {user.get('role', 'general')}"
                if user else ""
            )
            chat_history: List[Dict[str, str]] = []
            if token and show_chat and (session_id or "").strip():
                client = GatewayClient(base_url=GATEWAY_API_URL or None)
                hist_result = client.get_session_history_sync((session_id or "").strip(), last_n=3)
                if not hist_result.get("error"):
                    raw = hist_result.get("history", [])
                    for h in raw:
                        q, a = h.get("query", ""), h.get("answer", "")
                        if q or a:
                            chat_history.append({"role": "user", "content": q})
                            chat_history.append({"role": "assistant", "content": a})
            # Update both chatbot display and ChatInterface internal state so history
            # persists after the next user message (submit uses chatbot_state, not chatbot).
            return (
                token,
                user,
                gr.update(visible=show_login),
                gr.update(visible=show_chat),
                msg,
                user_md,
                gr.update(value=chat_history),
                chat_history,
            )

        def on_register(ru: str, rp: str, re: str, session_id: str):
            token, user, msg, show_chat, show_login = _do_register(ru, rp, re)
            user_md = (
                f"**UserName:** {user.get('user_name', '')}\n**Role:** {user.get('role', 'general')}"
                if user else ""
            )
            chat_history: List[Dict[str, str]] = []
            if token and show_chat and (session_id or "").strip():
                client = GatewayClient(base_url=GATEWAY_API_URL or None)
                hist_result = client.get_session_history_sync((session_id or "").strip(), last_n=3)
                if not hist_result.get("error"):
                    raw = hist_result.get("history", [])
                    for h in raw:
                        q, a = h.get("query", ""), h.get("answer", "")
                        if q or a:
                            chat_history.append({"role": "user", "content": q})
                            chat_history.append({"role": "assistant", "content": a})
            # Update both chatbot display and ChatInterface internal state.
            return (
                token,
                user,
                gr.update(visible=show_login),
                gr.update(visible=show_chat),
                msg,
                user_md,
                gr.update(value=chat_history),
                chat_history,
            )

        def on_signout():
            token, user, login_vis, chat_vis = _do_signout()
            return token, user, login_vis, chat_vis

        signin_btn.click(
            fn=on_signin,
            inputs=[signin_user, signin_pass, session_id_state],
            outputs=[
                auth_token_state,
                user_info_state,
                login_panel,
                chat_panel,
                signin_msg,
                user_display,
                chatbot,
                chat.chatbot_state,
            ],
        )

        # When SKIP_LOGIN: on load, sign in with default user so token/user_id and history are set.
        # queue=False avoids tying the ChatInterface "processing" indicator to this slow I/O on refresh.
        demo.load(
            fn=_auto_signin_if_skip_login,
            inputs=[session_id_state, chat.chatbot_state],
            outputs=[
                auth_token_state,
                user_info_state,
                login_panel,
                chat_panel,
                signin_msg,
                user_display,
                chatbot,
                chat.chatbot_state,
            ],
            queue=False,
        )
        reg_btn.click(
            fn=on_register,
            inputs=[reg_user, reg_pass, reg_email, session_id_state],
            outputs=[
                auth_token_state,
                user_info_state,
                login_panel,
                chat_panel,
                reg_msg,
                user_display,
                chatbot,
                chat.chatbot_state,
            ],
        )
        signout_btn.click(
            fn=on_signout,
            inputs=[],
            outputs=[auth_token_state, user_info_state, login_panel, chat_panel],
        )

        # Clear Session: update state and display with new UUID; clear pending clarification
        def on_clear_session():
            _clear_all_pending()
            new_id = str(uuid4())
            return new_id, new_id

        clear_btn.click(
            fn=on_clear_session,
            outputs=[session_id_state, session_display],
        )

    return demo


def launch(server_name: str = "0.0.0.0", server_port: Optional[int] = None) -> None:
    """
    Create demo and launch Gradio server.

    Args:
        server_name: Bind address (default 0.0.0.0).
        server_port: Port (default from UNIFIED_CHAT_GRADIO_PORT env).
    """
    port = server_port or GRADIO_PORT
    demo = create_demo()
    print(f"Starting IC-RAG-Agent Chat at http://localhost:{port} (bind {server_name})")
    print(f"Gateway: {GATEWAY_API_URL or '(mock mode)'}")
    print(f"UI client build: {UNIFIED_CHAT_CLIENT_BUILD} (if chat shows old 'Rewrite-only test mode', restart UI)")
    demo.launch(server_name=server_name, server_port=port, share=False, js=CHAT_AUTOSCROLL_JS)


def main() -> None:
    """Entry point for direct execution."""
    launch()


if __name__ == "__main__":
    main()
