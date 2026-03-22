#!/usr/bin/env python3
"""
Gradio chat UI for SP-API Agent.

Calls the SP-API REST API (POST /api/v1/seller/query or /query/stream)
and displays agent responses. Session ID persists across turns.

Usage:
  python scripts/run_sp_api_gradio.py
  SP_API_URL=http://127.0.0.1:8000 python scripts/run_sp_api_gradio.py

Requires the SP-API Agent API to be running (e.g. uvicorn src.agent.sp_api.app:app --port 8003).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

# Path setup: project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import requests

# Config from env
SP_API_URL = os.getenv("SP_API_URL", "http://127.0.0.1:8000").rstrip("/")
SP_API_GRADIO_PORT = int(os.getenv("SP_API_GRADIO_PORT", "7861"))
SP_API_QUERY_TIMEOUT = int(os.getenv("SP_API_QUERY_TIMEOUT", "120"))

# Session state (single-user; cleared on Clear Session)
_session_id: str = ""


def _get_or_create_session_id() -> str:
    """Return current session ID or create new one."""
    global _session_id
    if not _session_id:
        _session_id = str(uuid4())
    return _session_id


def _clear_session_id() -> str:
    """Clear server session, generate new ID, return new ID."""
    global _session_id
    old_id = _session_id
    _session_id = str(uuid4())
    if old_id:
        try:
            requests.delete(
                f"{SP_API_URL}/api/v1/seller/session/{old_id}",
                timeout=10,
            )
        except Exception:
            pass
    return _session_id


def _check_health() -> tuple[bool, str]:
    """GET /api/v1/health. Returns (ok, message)."""
    try:
        r = requests.get(f"{SP_API_URL}/api/v1/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return True, f"Status: {data.get('status', 'ok')}"
        return False, f"Health check failed: {r.status_code}"
    except requests.ConnectionError:
        return False, "API not reachable. Is it running?"
    except Exception as e:
        return False, str(e)


def _fetch_tools() -> list[dict]:
    """GET /api/v1/seller/tools. Returns list of tool dicts or empty on error."""
    try:
        r = requests.get(f"{SP_API_URL}/api/v1/seller/tools", timeout=5)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def _query_sync(query: str, session_id: str) -> str:
    """POST /api/v1/seller/query. Returns response text or error message."""
    payload = {"query": query, "session_id": session_id}
    try:
        r = requests.post(
            f"{SP_API_URL}/api/v1/seller/query",
            json=payload,
            timeout=SP_API_QUERY_TIMEOUT,
        )
    except requests.ConnectionError:
        return "Cannot connect to SP-API. Is it running? Start with: uvicorn src.agent.sp_api.app:app --port 8003"
    except requests.Timeout:
        return "Query timed out. The server took too long to respond."
    except requests.RequestException as e:
        return f"Request error: {e}"

    if r.status_code == 503:
        return "Agent not initialized. Check API logs."
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.json().get("message", r.text))
        except Exception:
            detail = r.text
        return f"Error {r.status_code}: {detail}"

    try:
        data = r.json()
    except Exception:
        return "Invalid response from API."

    response = data.get("response", "")
    iterations = data.get("iterations", 0)
    if iterations > 0:
        response += f"\n\n---\nIterations: {iterations}"
    return response


def _query_stream_generator(query: str, session_id: str):
    """
    POST /api/v1/seller/query/stream, consume SSE, yield accumulated response.
    Falls back to sync if stream fails.
    """
    payload = {"query": query, "session_id": session_id}
    accumulated = []
    try:
        r = requests.post(
            f"{SP_API_URL}/api/v1/seller/query/stream",
            json=payload,
            timeout=SP_API_QUERY_TIMEOUT,
            stream=True,
        )
    except requests.ConnectionError:
        yield _query_sync(query, session_id)
        return
    except requests.Timeout:
        yield "Query timed out."
        return
    except requests.RequestException as e:
        yield f"Request error: {e}"
        return

    if r.status_code != 200:
        try:
            yield _query_sync(query, session_id)
        except Exception:
            yield f"Stream error {r.status_code}. Try sync."
        return

    final_response = ""
    try:
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                try:
                    chunk = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = chunk.get("type")
                if t == "thought":
                    content = chunk.get("content", "")
                    if content:
                        accumulated.append(f"_Thought:_ {content[:200]}...")
                        yield "\n\n".join(accumulated)
                elif t == "observation":
                    content = chunk.get("content", "")
                    if content:
                        accumulated.append(f"_Observation:_ {content[:200]}...")
                        yield "\n\n".join(accumulated)
                elif t == "final":
                    final_response = chunk.get("response", "")
                    break
        result = "\n\n".join(accumulated) + "\n\n---\n" + final_response if accumulated else (final_response or "No response.")
        yield result
    except Exception as e:
        yield f"Stream parse error: {e}. Falling back to sync."
        yield _query_sync(query, session_id)


def main() -> None:
    """Launch Gradio chat UI."""
    import gradio as gr

    ok, health_msg = _check_health()
    tools_list = _fetch_tools()
    tool_names = [t.get("name", "?") for t in tools_list] if tools_list else ["(API not reachable)"]
    tools_md = "\n".join(f"- {n}" for n in tool_names) if tool_names else "No tools loaded."

    def chat_fn(message: str, history: list) -> str:
        """Chat callback: call API with persisted session ID."""
        session_id = _get_or_create_session_id()
        return _query_sync(message, session_id)

    def chat_stream_fn(message: str, history: list):
        """Streaming chat: yield chunks from SSE."""
        session_id = _get_or_create_session_id()
        for chunk in _query_stream_generator(message, session_id):
            yield chunk

    # Use streaming (generator); falls back to sync on connection error
    use_stream = True
    chat_handler = chat_stream_fn if use_stream else chat_fn

    with gr.Blocks(title="SP-API Agent Chat") as demo:
        gr.Markdown("# SP-API Agent Chat")
        gr.Markdown("Chat with the Seller Operations Agent. Queries catalog, inventory, orders, and more via Amazon SP-API.")

        session_id_state = gr.State(_get_or_create_session_id())

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.ChatInterface(
                    fn=chat_handler,
                    title="",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Sidebar")
                health_status = gr.Markdown(
                    f"**Health:** {'OK' if ok else 'Offline'}\n{health_msg}"
                )
                gr.Markdown("### Available Tools")
                tools_display = gr.Markdown(tools_md)
                session_display = gr.Textbox(
                    label="Session ID",
                    value=_get_or_create_session_id(),
                    interactive=False,
                )
                clear_btn = gr.Button("Clear Session", variant="secondary")

        def on_clear():
            new_id = _clear_session_id()
            return new_id, new_id

        clear_btn.click(
            fn=on_clear,
            outputs=[session_id_state, session_display],
        )

        # Refresh health and tools on load
        def refresh_sidebar():
            ok_r, msg_r = _check_health()
            tools_r = _fetch_tools()
            names_r = [t.get("name", "?") for t in tools_r] if tools_r else ["(API not reachable)"]
            tools_md_r = "\n".join(f"- {n}" for n in names_r)
            return (
                f"**Health:** {'OK' if ok_r else 'Offline'}\n{msg_r}",
                tools_md_r,
            )

        demo.load(
            fn=refresh_sidebar,
            outputs=[health_status, tools_display],
        )

    print(f"Starting SP-API Gradio at http://localhost:{SP_API_GRADIO_PORT} (bind 0.0.0.0)")
    print(f"SP-API URL: {SP_API_URL}")
    demo.launch(server_name="0.0.0.0", server_port=SP_API_GRADIO_PORT, share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
