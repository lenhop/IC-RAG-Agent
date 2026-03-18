#!/usr/bin/env python3
"""
Gateway FastAPI server launcher.

Runs the unified query gateway API using Uvicorn.
Loads .env configuration and supports configurable port.

Usage:
  python scripts/run_gateway.py
  GATEWAY_PORT=8001 python scripts/run_gateway.py

Requires: pip install fastapi uvicorn python-dotenv
"""

from __future__ import annotations

import logging
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

# Ensure gateway and route_llm loggers output [Perf] and other INFO to stderr
# (project_stack.sh redirects stderr to gateway.log)
logging.getLogger("src.gateway").setLevel(logging.INFO)

# Config from env
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
GATEWAY_LOG_LEVEL = os.getenv("GATEWAY_LOG_LEVEL", "info")
GATEWAY_WORKERS = int(os.getenv("GATEWAY_WORKERS", "1"))
GATEWAY_RELOAD = os.getenv("GATEWAY_RELOAD", "false").lower() in ("true", "1", "yes")

if __name__ == "__main__":
    import uvicorn

    print(f"Starting Gateway API at http://{GATEWAY_HOST}:{GATEWAY_PORT}")
    print(f"Log level: {GATEWAY_LOG_LEVEL}, Workers: {GATEWAY_WORKERS}, Reload: {GATEWAY_RELOAD}")

    uvicorn.run(
        "src.gateway.api.api:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level=GATEWAY_LOG_LEVEL,
        workers=GATEWAY_WORKERS,
        reload=GATEWAY_RELOAD,
    )