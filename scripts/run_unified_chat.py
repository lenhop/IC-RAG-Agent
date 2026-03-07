#!/usr/bin/env python3
"""
Gradio chat UI launcher for unified gateway client.

Launches the unified chat interface that communicates with the gateway API.
Supports mock mode when gateway is unavailable.

Usage:
  python scripts/run_unified_chat.py
  GATEWAY_API_URL=http://127.0.0.1:8001 python scripts/run_unified_chat.py
  GATEWAY_MOCK=true python scripts/run_unified_chat.py

Requires gradio and the src.client module.
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

# Config from env
GATEWAY_API_URL = os.getenv("GATEWAY_API_URL", "").rstrip("/")
GATEWAY_MOCK = os.getenv("GATEWAY_MOCK", "").lower() in ("true", "1", "yes")
CLIENT_GRADIO_PORT = int(os.getenv("CLIENT_GRADIO_PORT", "7862"))

# Import and launch UI
from src.client.gradio_ui import launch


def main():
    """Start unified chat UI."""
    # Print startup message
    gateway_status = "Mock mode (no gateway)" if (not GATEWAY_API_URL or GATEWAY_MOCK) else f"Gateway: {GATEWAY_API_URL}"
    print(f"Starting Unified Chat Gradio at http://localhost:{CLIENT_GRADIO_PORT} (bind 0.0.0.0)")
    print(f"Gateway: {gateway_status}")
    
    # Launch the UI with configured port
    launch(server_name="0.0.0.0", server_port=CLIENT_GRADIO_PORT)


if __name__ == "__main__":
    main()
