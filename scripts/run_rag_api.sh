#!/usr/bin/env bash
# Run RAG REST API with Uvicorn.
# Concurrency and reliability for 10+ users.
# Loads .env from project root. Requires: pip install fastapi uvicorn python-dotenv
#
# Usage:
#   ./scripts/run_rag_api.sh
#   RAG_API_PORT=9000 ./scripts/run_rag_api.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load RAG API vars from .env (optional; rag_api.py also loads .env)
if [[ -f .env ]]; then
  RAG_API_HOST="${RAG_API_HOST:-$(grep -E '^RAG_API_HOST=' .env 2>/dev/null | cut -d= -f2- || true)}"
  RAG_API_PORT="${RAG_API_PORT:-$(grep -E '^RAG_API_PORT=' .env 2>/dev/null | cut -d= -f2- || true)}"
  RAG_LIMIT_CONCURRENCY="${RAG_LIMIT_CONCURRENCY:-$(grep -E '^RAG_LIMIT_CONCURRENCY=' .env 2>/dev/null | cut -d= -f2- || true)}"
  RAG_TIMEOUT_KEEP_ALIVE="${RAG_TIMEOUT_KEEP_ALIVE:-$(grep -E '^RAG_TIMEOUT_KEEP_ALIVE=' .env 2>/dev/null | cut -d= -f2- || true)}"
fi

HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"
# Uvicorn concurrency - allow 20 connections (10 users + buffer)
LIMIT_CONCURRENCY="${RAG_LIMIT_CONCURRENCY:-20}"
# Keep-alive timeout (seconds) for idle connections
TIMEOUT_KEEP_ALIVE="${RAG_TIMEOUT_KEEP_ALIVE:-30}"

echo "Starting RAG API at http://${HOST}:${PORT} (limit-concurrency=${LIMIT_CONCURRENCY})"
exec python -m uvicorn src.rag.rag_api:app \
  --host "$HOST" \
  --port "$PORT" \
  --limit-concurrency "$LIMIT_CONCURRENCY" \
  --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE"
