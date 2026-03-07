"""
Gradio chat UI for unified gateway client.

Layout: chat (scale 3) left, sidebar (scale 1) right.
Sidebar: workflow Radio, Rewriting Enable checkbox, session state, health status.
Uses GatewayClient.query_sync with mock mode when gateway unavailable.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr

from .api_client import GatewayClient

logger = logging.getLogger(__name__)

# Config from env
GATEWAY_API_URL = os.environ.get("GATEWAY_API_URL", "").rstrip("/")
GATEWAY_MOCK = os.environ.get("GATEWAY_MOCK", "").lower() in ("true", "1", "yes")
GRADIO_PORT = int(os.environ.get("UNIFIED_CHAT_GRADIO_PORT", "7862"))

# Workflow options: (label, value) for Radio
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
    return f"**Status:** Gateway at {GATEWAY_API_URL}"


def _chat_handler(
    message: str,
    history: List[Tuple[str, str]],
    workflow: str,
    rewrite_enable: bool,
    rewrite_backend: str,
    session_id: str,
) -> str:
    """
    Chat callback: call GatewayClient.query_sync and return answer or error.

    Args:
        message: User message.
        history: Chat history (unused; ChatInterface manages it).
        workflow: Selected workflow (auto|general|amazon_docs|ic_docs|sp_api|uds).
        rewrite_enable: Whether query rewriting is enabled.
        rewrite_backend: Rewrite backend when enabled: "ollama" or "deepseek".
        session_id: Session UUID for multi-turn context.

    Returns:
        Answer text or error message string.
    """
    if not message or not message.strip():
        return "Please enter a question."

    client = GatewayClient(base_url=GATEWAY_API_URL or None)
    result: Dict[str, Any] = client.query_sync(
        query=message.strip(),
        workflow=workflow or "auto",
        rewrite_enable=bool(rewrite_enable),
        rewrite_backend=(rewrite_backend or "").strip() or None,
        session_id=session_id or None,
    )

    if "error" in result:
        return f"Error: {result['error']}"

    answer = result.get("answer", "")
    if not answer:
        return "No response from gateway."

    return answer


def create_demo() -> gr.Blocks:
    """
    Build and return the Gradio Blocks demo.

    Returns:
        Configured gr.Blocks instance (not launched).
    """
    # Initial session ID
    initial_session_id = str(uuid4())

    with gr.Blocks(title="IC-RAG-Agent Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# IC-RAG-Agent Chat")
        gr.Markdown(
            "Unified Chat UI - Select workflow and type your question. "
            "Uses mock mode when gateway is unavailable."
        )

        session_id_state = gr.State(value=initial_session_id)

        # Workflow and rewrite at top (must exist before ChatInterface).
        # Row: chat left (scale 3), sidebar right (scale 1).
        workflow_radio = gr.Radio(
            choices=WORKFLOW_CHOICES,
            value="auto",
            label="Workflow",
            info="auto|general|amazon_docs|ic_docs|sp_api|uds",
        )
        rewrite_checkbox = gr.Checkbox(
            value=True,
            label="Rewriting Enable",
            info="Enable query rewriting",
        )
        rewrite_backend_dropdown = gr.Dropdown(
            choices=[("Local (Ollama)", "ollama"), ("DeepSeek", "deepseek")],
            value="ollama",
            label="Rewrite backend",
            info="Local (Ollama) or DeepSeek when rewriting enabled",
        )

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.ChatInterface(
                    fn=_chat_handler,
                    title="",
                    additional_inputs=[
                        workflow_radio,
                        rewrite_checkbox,
                        rewrite_backend_dropdown,
                        session_id_state,
                    ],
                    additional_inputs_accordion=gr.Accordion(
                        label="Options",
                        open=False,
                    ),
                )

            with gr.Column(scale=1):
                gr.Markdown("### Session")
                session_display = gr.Textbox(
                    label="Session ID",
                    value=initial_session_id,
                    interactive=False,
                )
                clear_btn = gr.Button("Clear Session", variant="secondary")
                gr.Markdown("### Status")
                health_status = gr.Markdown(_get_gateway_status())

        # Clear Session: update state and display with new UUID
        def on_clear_session():
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
    demo.launch(server_name=server_name, server_port=port, share=False)


def main() -> None:
    """Entry point for direct execution."""
    launch()


if __name__ == "__main__":
    main()
