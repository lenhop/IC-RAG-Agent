"""
Gradio chat UI for unified gateway client.

Layout: controls (scale 1) left, chat (scale 3) right.
Left column: Workflow, Rewriting Enable, Rewrite backend, Session, Status.
Right column: ChatInterface with fill_height and autoscroll.
Uses GatewayClient.query_sync with mock mode when gateway unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from uuid import uuid4

import gradio as gr

from .api_client import GatewayClient

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


# Config from env
GATEWAY_API_URL = os.environ.get("GATEWAY_API_URL", "").rstrip("/")
GATEWAY_MOCK = os.environ.get("GATEWAY_MOCK", "").lower() in ("true", "1", "yes")
GRADIO_PORT = int(os.environ.get("UNIFIED_CHAT_GRADIO_PORT", "7862"))
REWRITE_ENABLE_DEFAULT = os.environ.get(
    "UNIFIED_CHAT_REWRITE_ENABLE", "true"
).lower() in ("true", "1", "yes")
REWRITE_BACKEND_DEFAULT = (
    os.environ.get("UNIFIED_CHAT_REWRITE_BACKEND", "ollama").strip().lower()
)
REWRITE_ONLY_TEST_MODE = (
    os.environ.get("UNIFIED_CHAT_REWRITE_ONLY_MODE", "").lower() in ("true", "1", "yes")
    or os.environ.get("GATEWAY_REWRITE_ONLY_MODE", "").lower() in ("true", "1", "yes")
)

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


def _get_gateway_status() -> str:
    """Return human-readable gateway status for sidebar."""
    if not GATEWAY_API_URL or GATEWAY_MOCK:
        return "**Status:** Mock mode (no gateway)"
    if REWRITE_ONLY_TEST_MODE:
        return f"**Status:** Gateway at {GATEWAY_API_URL} (rewrite-only test mode)"
    return f"**Status:** Gateway at {GATEWAY_API_URL}"


def _normalize_rewrite_backend(value: str) -> str:
    """
    Normalize rewrite backend from env/UI.

    Args:
        value: Candidate backend value.

    Returns:
        Valid backend value: "ollama" or "deepseek".
    """
    normalized = (value or "").strip().lower()
    if normalized in ("ollama", "deepseek"):
        return normalized
    return "ollama"


def _chat_handler(
    message: str,
    history: List[Tuple[str, str]],
    workflow: str,
    rewrite_enable: bool,
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
        rewrite_enable: Whether query rewriting is enabled.
        rewrite_backend: Rewrite backend when enabled: "ollama" or "deepseek".
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

    if rewrite_enable:
        rewrite_result = client.rewrite_sync(
            query=raw_query,
            rewrite_enable=True,
            rewrite_backend=(rewrite_backend or "").strip() or None,
            session_id=session_id or None,
            user_id=user_id,
            token=auth_token,
        )
        rewrite_error = rewrite_result.get("error")
        if rewrite_error:
            yield (
                "Rewrite failed; continuing with original query.\n"
                f"Error: {rewrite_error}"
            )
        elif rewrite_result.get("clarification_required"):
            # Clarification needed: display question and store pending for follow-up
            clarification_question = (
                rewrite_result.get("clarification_question") or "Please provide more details."
            )
            pending_query = rewrite_result.get("pending_query") or raw_query
            _set_pending_query(session_id, pending_query)
            yield (
                "**Clarification needed:**\n\n"
                f"{clarification_question}\n\n"
                "---\n"
                "Reply with the missing details (e.g. store, ASIN, date range). "
                "Your follow-up will be merged with the original query."
            )
            return
        else:
            routed_query = str(rewrite_result.get("rewritten_query") or raw_query).strip()
            # Use bullet-point display: gateway's rewritten_query_display or client-side fallback.
            gateway_bullets = rewrite_result.get("rewritten_query_display")
            client_bullets = _format_query_as_bullets(routed_query, min_length=60)
            display_query = gateway_bullets or client_bullets or routed_query
            has_bullets = bool(gateway_bullets or client_bullets)
            rewrite_ms = int(rewrite_result.get("rewrite_time_ms") or 0)
            rewrite_backend_value = str(
                rewrite_result.get("rewrite_backend")
                or (rewrite_backend or "ollama")
            )
            memory_rounds = int(rewrite_result.get("memory_rounds") or 0)
            memory_text_len = int(rewrite_result.get("memory_text_length") or 0)
            rewritten_len = int(rewrite_result.get("rewritten_query_length") or 0)
            workflows_list = rewrite_result.get("workflows") or []

            q_display = (
                f"- Rewritten Query: (text length: {rewritten_len} chars)\n{display_query}"
                if has_bullets
                else f"- Rewritten Query: (text length: {rewritten_len} chars) `{routed_query}`"
            )
            lines_parts: List[str] = [
                "- Normalize: Completed",
                f"- Integrate short-term memory: {memory_rounds} rounds (text length: {memory_text_len} chars)",
                q_display,
                f"- Rewrite Backend: `{rewrite_backend_value}`",
                f"- Rewrite Time: `{rewrite_ms} ms`",
            ]
            if workflows_list:
                workflows_str = ", ".join(workflows_list)
                lines_parts.append(f"- Intent classification: {workflows_str}")
            rewrite_message = "\n".join(lines_parts)
            # Include execution plan when available (planner mode)
            plan = rewrite_result.get("plan")
            if plan and isinstance(plan, dict):
                plan_type = plan.get("plan_type", "single_domain")
                task_groups = plan.get("task_groups") or []
                tasks_flat: List[Dict[str, Any]] = []
                for g in task_groups:
                    tasks_flat.extend(g.get("tasks") or [])
                if tasks_flat:
                    rewrite_message += f"\n\n**Plan** (`{plan_type}`):\n"
                    for i, t in enumerate(tasks_flat, 1):
                        wf = t.get("workflow", "general")
                        q = (t.get("query") or "").strip()
                        rewrite_message += f"\n{i}. `{wf}`: {q}"
            yield rewrite_message

    if REWRITE_ONLY_TEST_MODE:
        # Quick-test mode: stop after rewriting and skip all routing/downstream calls.
        if rewrite_message:
            yield (
                f"{rewrite_message}\n\n"
                "---\n"
                "**Rewrite-only test mode.** Route LLM and downstream services are skipped."
            )
        else:
            yield (
                "Rewrite-only test mode enabled, but rewriting is disabled.\n"
                "No route/downstream call was executed."
            )
        return

    # Query using rewritten text; disable rewrite to avoid double rewriting.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.query_sync,
            query=routed_query,
            workflow=workflow or "auto",
            rewrite_enable=False,
            rewrite_backend=None,
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
        pending_query = result.get("pending_query") or routed_query
        _set_pending_query(session_id, pending_query)
        yield (
            f"**Clarification needed:** {clarification_question}\n\n"
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
    yield f"{answer}\n" + "\n".join(trace_lines)


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
"""

# JavaScript to auto-scroll chat to bottom when new messages arrive or on submit
CHAT_AUTOSCROLL_JS = """
(function() {
    function scrollChatToBottom() {
        var selectors = ['#ic_chatbot', '[aria-label="chatbot conversation"]', '.gr-chat .scroll-hide', '#ic_chat_column [class*="scroll"]'];
        selectors.forEach(function(sel) {
            var el = document.querySelector(sel);
            if (el) {
                el.scrollTop = el.scrollHeight;
                var children = el.querySelectorAll('[style*="overflow"]');
                for (var i = 0; i < children.length; i++) {
                    children[i].scrollTop = children[i].scrollHeight;
                }
            }
        });
    }
    function attachObserver() {
        var root = document.getElementById('ic_chatbot') || document.getElementById('ic_chat_column') || document.querySelector('.gr-chat');
        if (root && !root._icScrollObserved) {
            root._icScrollObserved = true;
            var obs = new MutationObserver(scrollChatToBottom);
            obs.observe(root, { childList: true, subtree: true });
            scrollChatToBottom();
        }
    }
    function init() {
        attachObserver();
        setTimeout(attachObserver, 800);
        setTimeout(attachObserver, 2000);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""


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
        gr.Markdown(
            "Unified Chat UI - Sign in or register, then select workflow and type your question. "
            "Uses mock mode when gateway is unavailable."
        )

        session_id_state = gr.State(value=initial_session_id)
        auth_token_state = gr.State(value=None)
        user_info_state = gr.State(value=None)

        # Login panel: visible when not logged in
        with gr.Column(visible=True, elem_id="ic_login_panel") as login_panel:
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

        # Chat panel: visible when logged in
        with gr.Column(visible=False, elem_id="ic_chat_panel") as chat_panel:
            signout_btn = gr.Button("Sign Out", variant="secondary")
            with gr.Row():
                with gr.Column(scale=1, elem_id="ic_controls_column"):
                    gr.Markdown("### Workflow")
                    workflow_radio = gr.Dropdown(
                        choices=WORKFLOW_CHOICES,
                        value="auto",
                        label="Workflow",
                        info="auto|general|amazon_docs|ic_docs|sp_api|uds",
                    )
                    rewrite_checkbox = gr.Checkbox(
                        value=REWRITE_ENABLE_DEFAULT,
                        label="Rewriting Enable",
                        info="Enable query rewriting",
                    )
                    rewrite_backend_dropdown = gr.Dropdown(
                        choices=[("Local (Ollama)", "ollama"), ("DeepSeek", "deepseek")],
                        value=_normalize_rewrite_backend(REWRITE_BACKEND_DEFAULT),
                        label="Rewrite backend",
                        info="Local (Ollama) or DeepSeek when rewriting enabled",
                        interactive=REWRITE_ENABLE_DEFAULT,
                    )
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
                            rewrite_checkbox,
                            rewrite_backend_dropdown,
                            session_id_state,
                            auth_token_state,
                            user_info_state,
                        ],
                        fill_height=True,
                        autoscroll=True,
                    )

        def on_signin(su: str, sp: str):
            token, user, msg, show_chat, show_login = _do_signin(su, sp)
            user_md = (
                f"**UserName:** {user.get('user_name', '')}\n**Role:** {user.get('role', 'general')}"
                if user else ""
            )
            chat_history: List[Dict[str, str]] = []
            if token and show_chat:
                client = GatewayClient(base_url=GATEWAY_API_URL or None)
                hist_result = client.get_user_history_sync(token, last_n=5)
                if not hist_result.get("error"):
                    raw = hist_result.get("history", [])
                    for h in raw:
                        q, a = h.get("query", ""), h.get("answer", "")
                        if q or a:
                            chat_history.append({"role": "user", "content": q})
                            chat_history.append({"role": "assistant", "content": a})
            return (
                token,
                user,
                gr.update(visible=show_login),
                gr.update(visible=show_chat),
                msg,
                user_md,
                gr.update(value=chat_history),
            )

        def on_register(ru: str, rp: str, re: str):
            token, user, msg, show_chat, show_login = _do_register(ru, rp, re)
            user_md = (
                f"**UserName:** {user.get('user_name', '')}\n**Role:** {user.get('role', 'general')}"
                if user else ""
            )
            chat_history: List[Dict[str, str]] = []
            if token and show_chat:
                client = GatewayClient(base_url=GATEWAY_API_URL or None)
                hist_result = client.get_user_history_sync(token, last_n=5)
                if not hist_result.get("error"):
                    raw = hist_result.get("history", [])
                    for h in raw:
                        q, a = h.get("query", ""), h.get("answer", "")
                        if q or a:
                            chat_history.append({"role": "user", "content": q})
                            chat_history.append({"role": "assistant", "content": a})
            return (
                token,
                user,
                gr.update(visible=show_login),
                gr.update(visible=show_chat),
                msg,
                user_md,
                gr.update(value=chat_history),
            )

        def on_signout():
            token, user, login_vis, chat_vis = _do_signout()
            return token, user, login_vis, chat_vis

        signin_btn.click(
            fn=on_signin,
            inputs=[signin_user, signin_pass],
            outputs=[
                auth_token_state,
                user_info_state,
                login_panel,
                chat_panel,
                signin_msg,
                user_display,
                chatbot,
            ],
        )
        reg_btn.click(
            fn=on_register,
            inputs=[reg_user, reg_pass, reg_email],
            outputs=[
                auth_token_state,
                user_info_state,
                login_panel,
                chat_panel,
                reg_msg,
                user_display,
                chatbot,
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

        # Keep backend selector disabled when rewriting is disabled.
        rewrite_checkbox.change(
            fn=lambda enabled: gr.update(interactive=bool(enabled)),
            inputs=[rewrite_checkbox],
            outputs=[rewrite_backend_dropdown],
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
    demo.launch(server_name=server_name, server_port=port, share=False, js=CHAT_AUTOSCROLL_JS)


def main() -> None:
    """Entry point for direct execution."""
    launch()


if __name__ == "__main__":
    main()
