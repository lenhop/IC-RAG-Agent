#!/usr/bin/env python3
"""
Gradio chat UI for RAG.

Calls the RAG REST API (POST /query) and displays answers with source attribution.
Requires the RAG API to be running: ./scripts/run_rag_api.sh

Usage:
  python scripts/run_rag_gradio.py
  RAG_API_URL=http://localhost:9000 python scripts/run_rag_gradio.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

# Config from .env
RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
RAG_GRADIO_PORT = int(os.getenv("RAG_GRADIO_PORT", "7860"))
QUERY_TIMEOUT = int(os.getenv("RAG_QUERY_TIMEOUT", "120"))


def _call_rag_api(question: str, mode: str) -> str:
    """
    POST /query to RAG API. Returns formatted answer + source, or error message.
    """
    url = f"{RAG_API_URL.rstrip('/')}/query"
    payload = {"question": question, "mode": mode}

    try:
        r = requests.post(url, json=payload, timeout=QUERY_TIMEOUT)
    except requests.ConnectionError:
        return (
            "Cannot connect to RAG API. Is it running? "
            "Start with: ./scripts/run_rag_api.sh"
        )
    except requests.Timeout:
        return "Query timed out. The server took too long to respond."
    except requests.RequestException as e:
        return f"Request error: {e}"

    if r.status_code == 503:
        return "Server busy. Try again later."
    if r.status_code == 504:
        return "Query timed out."
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return f"Error {r.status_code}: {detail}"

    try:
        data = r.json()
    except Exception:
        return "Invalid response from API."

    answer = data.get("answer", "")
    source = data.get("source", "")
    sources = data.get("sources", [])
    selected_mode = data.get("selected_mode")

    # Format: answer + source + optional source list + selected_mode when auto
    if not answer:
        return answer

    parts = [answer]
    if source:
        parts.append(f"\n\n---\nSource: {source}")
    if selected_mode:
        parts.append(f"\nAuto selected: {selected_mode}")
    if sources:
        lines = [f"  {i}. {s.get('file', '?')} (page {s.get('page', '?')})" for i, s in enumerate(sources, 1)]
        parts.append("\n" + "\n".join(lines))

    return "".join(parts)


def main() -> None:
    """Launch Gradio chat UI."""
    import gradio as gr

    def chat_fn(message: str, history: list, mode: str) -> str:
        """Chat callback: call API and return formatted response."""
        return _call_rag_api(message, mode)

    mode_radio = gr.Radio(
        choices=["documents", "general", "hybrid", "auto"],
        value="auto",
        label="Answer mode",
        info="documents=only from ingested docs; general=LLM only; hybrid=both; auto=sequential classifier (keywords + distance threshold)",
    )

    demo = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[mode_radio],
        additional_inputs_accordion=gr.Accordion(label="Answer mode", open=True),
        title="RAG Chat",
        description="Ask questions. Requires RAG API running at " + RAG_API_URL,
    )

    print(f"Starting Gradio at http://127.0.0.1:{RAG_GRADIO_PORT}")
    print(f"RAG API URL: {RAG_API_URL}")
    demo.launch(server_port=RAG_GRADIO_PORT, share=False)


if __name__ == "__main__":
    main()
