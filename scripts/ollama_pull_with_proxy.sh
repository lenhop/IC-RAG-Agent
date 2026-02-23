#!/usr/bin/env bash
#
# Pull Ollama models via proxy (fixes registry.ollama.ai i/o timeout in China).
#
# Usage:
#   ./scripts/ollama_pull_with_proxy.sh [model_name]
#   ./scripts/ollama_pull_with_proxy.sh all-minilm
#
# Requires: proxy on 127.0.0.1:7890 (Clash/V2Ray default). Override with HTTPS_PROXY.
#
# If Ollama is already running from the app, quit it first (Cmd+Q), then run this script.
# Or set HTTPS_PROXY before launching Ollama app (launchctl setenv HTTPS_PROXY "http://127.0.0.1:7890").
#

set -e

PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
MODEL="${1:-all-minilm}"

echo "[OK] Using proxy: $PROXY"
echo "[OK] Pulling model: $MODEL"

# Stop existing Ollama if running (so we can start with proxy)
if pgrep -x ollama >/dev/null 2>&1; then
    echo "[OK] Stopping existing Ollama..."
    pkill -f ollama 2>/dev/null || true
    sleep 2
fi

# Start Ollama with proxy in background
echo "[OK] Starting Ollama with proxy..."
HTTPS_PROXY="$PROXY" ollama serve &
OLLAMA_PID=$!
sleep 3

# Pull model
echo "[OK] Pulling $MODEL..."
if HTTPS_PROXY="$PROXY" ollama pull "$MODEL"; then
    echo "[OK] Model $MODEL downloaded successfully"
else
    kill $OLLAMA_PID 2>/dev/null || true
    echo "[FAIL] Pull failed. Ensure proxy is running (e.g. Clash on port 7890)"
    exit 1
fi

# Note: ollama serve keeps running in background. User can stop it with: pkill -f ollama
