#!/usr/bin/env bash
# Load vector_intent_registry.csv into local ChromaDB intent_registry collection.
# Requires: Ollama (GATEWAY_REWRITE_OLLAMA_URL) with GATEWAY_INTENT_EMBEDDING_MODEL.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
exec python scripts/load_to_chroma/load_intent_registry_to_chroma.py "$@"
