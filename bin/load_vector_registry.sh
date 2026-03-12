#!/usr/bin/env bash
# Load vector_intent_registry.csv into ChromaDB (local or ECS).
# Requires: Ollama running with all-minilm model for embeddings.
#
# Local:   ./bin/load_vector_registry.sh --local
# Auto:    ./bin/load_vector_registry.sh   (uses CHA_HOST from .env if set; falls back to local on failure)
# Remote:  ./bin/load_vector_registry.sh --remote   (requires CHA_HOST)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
exec python scripts/load_vector_registry.py "$@"
