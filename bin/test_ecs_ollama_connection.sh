#!/bin/bash
# Test connectivity from local machine to Ollama running on ECS.
#
# ECS Ollama is not exposed on the public internet (port 11434 blocked by security group).
# Use SSH tunnel to connect: local port -> ECS localhost:11434
#
# Usage:
#   ./scripts/test_ecs_ollama_connection.sh           # Uses tunnel port 11435
#   ./scripts/test_ecs_ollama_connection.sh 11436    # Custom tunnel port
#
# Prerequisites: ssh len must work

set -e

TUNNEL_PORT="${1:-11435}"
TUNNEL_PID=""
ECS_HOST="len"

cleanup() {
    if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
        kill "${TUNNEL_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== ECS Ollama Connection Test ==="
echo ""

# Step 1: Check if direct connection works (requires security group rule for 11434)
ECS_IP=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${ECS_HOST}" 'curl -s ifconfig.me' 2>/dev/null || echo "")
if [[ -n "${ECS_IP}" ]]; then
    echo "[1] ECS public IP: ${ECS_IP}"
    if curl -s --connect-timeout 10 "http://${ECS_IP}:11434/api/tags" >/dev/null 2>&1; then
        echo "    Direct connection OK (port 11434 open)"
        OLLAMA_URL="http://${ECS_IP}:11434"
    else
        echo "    Direct connection failed (port 11434 not open - use SSH tunnel)"
        OLLAMA_URL=""
    fi
else
    echo "[1] Could not get ECS IP"
    OLLAMA_URL=""
fi
echo ""

# Step 2: Establish SSH tunnel if direct connection failed
if [[ -z "${OLLAMA_URL}" ]]; then
    echo "[2] Creating SSH tunnel: localhost:${TUNNEL_PORT} -> ${ECS_HOST}:11434"
    ssh -f -N -L "${TUNNEL_PORT}:localhost:11434" "${ECS_HOST}" 2>/dev/null || {
        echo "    Tunnel may already exist. Testing..."
    }
    sleep 1
    OLLAMA_URL="http://localhost:${TUNNEL_PORT}"
fi
echo ""

# Step 3: Test /api/tags
echo "[3] Testing /api/tags..."
TAGS=$(curl -s --connect-timeout 10 "${OLLAMA_URL}/api/tags" 2>/dev/null || echo "")
if [[ -z "${TAGS}" ]]; then
    echo "    FAIL: Could not reach Ollama at ${OLLAMA_URL}"
    exit 1
fi

MODEL_COUNT=$(echo "${TAGS}" | grep -o '"name":' | wc -l | tr -d ' ')
echo "    OK: Connected. Models available: ${MODEL_COUNT}"
echo "${TAGS}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for m in d.get('models', [])[:5]:
        print(f\"      - {m.get('name', '?')} ({m.get('details', {}).get('parameter_size', '?')})\")
except: pass
" 2>/dev/null || echo "      (raw response received)"
echo ""

# Step 4: Optional quick generate test (skip by default - slow on CPU)
if [[ "${TEST_GENERATE:-0}" == "1" ]]; then
    echo "[4] Testing /api/generate (stream=false)..."
    RESP=$(curl -s -X POST "${OLLAMA_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3:1.7b","prompt":"Reply with exactly: OK","stream":false}' \
        --max-time 60 2>/dev/null || echo "")
    if echo "${RESP}" | grep -q '"response"'; then
        echo "    OK: Generate succeeded"
    else
        echo "    WARN: Generate test inconclusive (${#RESP} bytes)"
    fi
else
    echo "[4] Skipping generate test (set TEST_GENERATE=1 to run)"
fi
echo ""

echo "=== Result: ECS Ollama reachable from local ==="
echo ""
if [[ -n "${ECS_IP}" ]] && curl -s --connect-timeout 2 "http://${ECS_IP}:11434/api/tags" >/dev/null 2>&1; then
    echo "To use ECS Ollama from gateway (direct connection):"
    echo "  GATEWAY_REWRITE_OLLAMA_URL=http://${ECS_IP}:11434/api/generate"
    echo "  GATEWAY_REWRITE_OLLAMA_MODEL=qwen3:1.7b"
else
    echo "To use ECS Ollama from gateway (via SSH tunnel):"
    echo "  1. Start tunnel: ssh -f -N -L ${TUNNEL_PORT}:localhost:11434 ${ECS_HOST}"
    echo "  2. Set env:     GATEWAY_REWRITE_OLLAMA_URL=http://localhost:${TUNNEL_PORT}/api/generate"
    echo "  3. Set env:     GATEWAY_REWRITE_OLLAMA_MODEL=qwen3:1.7b"
fi
echo ""
