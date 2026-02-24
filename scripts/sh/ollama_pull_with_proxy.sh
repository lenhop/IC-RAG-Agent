#!/usr/bin/env bash
#
# Pull Ollama models via proxy (fixes registry.ollama.ai i/o timeout in China).
# Default: downloads all-minilm, llama3.2, qwen3:1.7b when run with no arguments.
#
# Usage:
#   ./scripts/sh/ollama_pull_with_proxy.sh                    # Downloads all default models
#   ./scripts/sh/ollama_pull_with_proxy.sh [model_name]       # e.g. llama3.2, qwen3:1.7b
#
# Requires: proxy on 127.0.0.1:7890 (Clash/V2Ray default). Override with HTTPS_PROXY.
#
# If Ollama is already running from the app, quit it first (Cmd+Q), then run this script.
# Or set HTTPS_PROXY before launching Ollama app (launchctl setenv HTTPS_PROXY "http://127.0.0.1:7890").
#

set -e

# Default models: all-minilm (embedding), llama3.2 (LLM), qwen3:1.7b (LLM)
DEFAULT_MODELS=(all-minilm llama3.2 qwen3:1.7b)
PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"

# Stop existing Ollama if running (so we can start with proxy)
if pgrep -x ollama >/dev/null 2>&1; then
    echo "[OK] Stopping existing Ollama..."
    pkill -f ollama 2>/dev/null || true
    sleep 2
fi

# Start Ollama with proxy in background
echo "[OK] Using proxy: $PROXY"
echo "[OK] Starting Ollama with proxy..."
HTTPS_PROXY="$PROXY" ollama serve &
OLLAMA_PID=$!
sleep 3

pull_model() {
    local model="$1"
    echo "[OK] Pulling $model..."
    if HTTPS_PROXY="$PROXY" ollama pull "$model"; then
        echo "[OK] Model $model downloaded successfully"
        return 0
    else
        echo "[WARN] Failed to pull $model"
        return 1
    fi
}

if [ -n "${1:-}" ]; then
    # Single model specified
    if pull_model "$1"; then
        echo "[OK] Done."
    else
        kill $OLLAMA_PID 2>/dev/null || true
        echo "[FAIL] Pull failed. Ensure proxy is running (e.g. Clash on port 7890)"
        exit 1
    fi
else
    # Download all default models
    echo "[OK] No model specified, downloading defaults: ${DEFAULT_MODELS[*]}"
    failed=0
    for model in "${DEFAULT_MODELS[@]}"; do
        pull_model "$model" || failed=1
    done
    if [ $failed -eq 1 ]; then
        kill $OLLAMA_PID 2>/dev/null || true
        echo "[FAIL] One or more models failed. Ensure proxy is running (e.g. Clash on port 7890)"
        exit 1
    fi
    echo "[OK] All default models downloaded."
fi

# Note: ollama serve keeps running in background. User can stop it with: pkill -f ollama
