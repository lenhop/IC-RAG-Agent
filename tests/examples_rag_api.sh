#!/usr/bin/env bash
# RAG API usage examples (curl).
# Prerequisite: Start the API with ./scripts/run_rag_api.sh
#
# Usage:
#   ./tests/examples_rag_api.sh
#   RAG_API_URL=http://localhost:9000 ./tests/examples_rag_api.sh

set -e

BASE_URL="${RAG_API_URL:-http://127.0.0.1:8000}"

echo "=== RAG API Examples (base: $BASE_URL) ==="
echo ""

echo "--- 1. Health check (GET /health) ---"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

echo "--- 2. Query with hybrid mode (POST /query) ---"
curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "mode": "hybrid"}' \
  | python3 -m json.tool
echo ""
echo ""

echo "--- 3. Query with documents-only mode ---"
curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the main topic.", "mode": "documents"}' \
  | python3 -m json.tool
echo ""
echo ""

echo "--- 4. Query with general-knowledge mode (no retrieval) ---"
curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2 + 2?", "mode": "general"}' \
  | python3 -m json.tool
echo ""
echo ""

echo "--- 5. Query with default mode (hybrid when omitted) ---"
curl -s -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}' \
  | python3 -m json.tool
echo ""
echo ""

echo "Done."
