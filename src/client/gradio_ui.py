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

    Returns:
        Iterator yielding intermediate/final messages, or a plain string for
        immediate validation errors.
    """
    if not message or not message.strip():
        yield "Please enter a question."
        return

    client = GatewayClient(base_url=GATEWAY_API_URL or None)
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
            rewrite_ms = int(rewrite_result.get("rewrite_time_ms") or 0)
            rewrite_backend_value = str(
                rewrite_result.get("rewrite_backend")
                or (rewrite_backend or "ollama")
            )
            rewrite_message = (
                "Rewrite completed.\n\n"
                f"- Rewritten Query: `{routed_query}`\n"
                f"- Rewrite Backend: `{rewrite_backend_value}`\n"
                f"- Rewrite Time: `{rewrite_ms} ms`"
            )
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


# CSS to make chat dialog and input box taller
CHAT_DIALOG_CSS = """
/* Chat dialog: fill most of viewport */
#ic_chat_column { min-height: 95vh !important; display: flex !important; flex-direction: column !important; }
#ic_chat_column .contain { flex: 1 !important; min-height: 0 !important; }
.chatbot { min-height: 95vh !important; flex: 1 !important; }
[data-testid="chatbot"] { min-height: 95vh !important; }
.gr-chat { min-height: 95vh !important; }
/* Input box: taller for multi-line typing */
#ic_chat_column textarea { min-height: 150px !important; height: 150px !important; }
.gr-chat textarea { min-height: 150px !important; height: 150px !important; }
.chatbot textarea { min-height: 150px !important; height: 150px !important; }
/* Workflow radio: single column, left-aligned, no extra gap */
#workflow_radio { flex: 0 0 auto !important; }
#workflow_radio .wrap { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio .contain { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio > div { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
#workflow_radio [class*="wrap"] { display: flex !important; flex-direction: column !important; flex-wrap: nowrap !important; align-items: flex-start !important; justify-content: flex-start !important; }
/* Left column: pack content at top, no stretch */
#ic_controls_column { justify-content: flex-start !important; align-items: flex-start !important; }
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
            "Unified Chat UI - Select workflow and type your question. "
            "Uses mock mode when gateway is unavailable."
        )

        session_id_state = gr.State(value=initial_session_id)

        # Row: controls left (scale 1), chat right (scale 3).
        # Left column: Workflow, Rewriting, Session, Status.
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
                chat = gr.ChatInterface(
                    fn=_chat_handler,
                    title="",
                    additional_inputs=[
                        workflow_radio,
                        rewrite_checkbox,
                        rewrite_backend_dropdown,
                        session_id_state,
                    ],
                    fill_height=True,
                    autoscroll=True,
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
    demo.launch(server_name=server_name, server_port=port, share=False)


def main() -> None:
    """Entry point for direct execution."""
    launch()


if __name__ == "__main__":
    main()
