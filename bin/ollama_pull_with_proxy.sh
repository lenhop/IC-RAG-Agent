#!/usr/bin/env bash
#
# Pull Ollama models via proxy (useful for restricted network environments).
#
# Usage:
#   ./bin/ollama_pull_with_proxy.sh
#   ./bin/ollama_pull_with_proxy.sh qwen3:1.7b
#

set -euo pipefail

DEFAULT_MODELS=(all-minilm llama3.2 qwen3:1.7b)
PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"

if pgrep -x ollama >/dev/null 2>&1; then
  pkill -f ollama || true
  sleep 2
fi

echo "Using proxy: ${PROXY}"
HTTPS_PROXY="${PROXY}" ollama serve &
OLLAMA_PID=$!
sleep 3

pull_model() {
  local model="$1"
  if HTTPS_PROXY="${PROXY}" ollama pull "${model}"; then
    echo "Pulled model: ${model}"
    return 0
  fi
  echo "Failed to pull model: ${model}"
  return 1
}

if [[ -n "${1:-}" ]]; then
  pull_model "$1" || {
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 1
  }
else
  failed=0
  for model in "${DEFAULT_MODELS[@]}"; do
    pull_model "${model}" || failed=1
  done
  if [[ "${failed}" -eq 1 ]]; then
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 1
  fi
fi

echo "Done."
