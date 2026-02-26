#!/usr/bin/env bash
# Run Gradio chat UI for RAG (Phase 4).
# Requires RAG API to be running: ./scripts/run_rag_api.sh
#
# Usage:
#   ./scripts/run_rag_gradio.sh
#   RAG_API_URL=http://localhost:9000 ./scripts/run_rag_gradio.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load Gradio vars from .env (optional)
if [[ -f .env ]]; then
  RAG_API_URL="${RAG_API_URL:-$(grep -E '^RAG_API_URL=' .env 2>/dev/null | cut -d= -f2- || true)}"
  RAG_GRADIO_PORT="${RAG_GRADIO_PORT:-$(grep -E '^RAG_GRADIO_PORT=' .env 2>/dev/null | cut -d= -f2- || true)}"
fi

RAG_API_URL="${RAG_API_URL:-http://127.0.0.1:8000}"
RAG_GRADIO_PORT="${RAG_GRADIO_PORT:-7860}"

export RAG_API_URL RAG_GRADIO_PORT

echo "Starting Gradio chat UI at http://127.0.0.1:${RAG_GRADIO_PORT}"
echo "RAG API URL: ${RAG_API_URL}"
exec python scripts/run_rag_gradio.py
