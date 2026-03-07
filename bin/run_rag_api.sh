#!/usr/bin/env bash
# Run RAG REST API with Uvicorn.
#
# Usage:
#   ./bin/run_rag_api.sh
#   RAG_API_PORT=9000 ./bin/run_rag_api.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  RAG_API_HOST="${RAG_API_HOST:-$(awk -F= '/^RAG_API_HOST=/{print $2}' .env | tail -1 || true)}"
  RAG_API_PORT="${RAG_API_PORT:-$(awk -F= '/^RAG_API_PORT=/{print $2}' .env | tail -1 || true)}"
  RAG_LIMIT_CONCURRENCY="${RAG_LIMIT_CONCURRENCY:-$(awk -F= '/^RAG_LIMIT_CONCURRENCY=/{print $2}' .env | tail -1 || true)}"
  RAG_TIMEOUT_KEEP_ALIVE="${RAG_TIMEOUT_KEEP_ALIVE:-$(awk -F= '/^RAG_TIMEOUT_KEEP_ALIVE=/{print $2}' .env | tail -1 || true)}"
fi

HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"
LIMIT_CONCURRENCY="${RAG_LIMIT_CONCURRENCY:-20}"
TIMEOUT_KEEP_ALIVE="${RAG_TIMEOUT_KEEP_ALIVE:-30}"

echo "Starting RAG API at http://${HOST}:${PORT} (limit-concurrency=${LIMIT_CONCURRENCY})"
exec python -m uvicorn src.rag.rag_api:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --limit-concurrency "${LIMIT_CONCURRENCY}" \
  --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE}"
