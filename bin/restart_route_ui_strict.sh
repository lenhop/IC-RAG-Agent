#!/usr/bin/env bash
# One-click restart (route-only + UI) with strict gateway health check.
#
# Usage:
#   ./bin/restart_route_ui_strict.sh
#
# Behavior:
#   1) Calls ./bin/restart_route_ui.sh (route-only + UI + no-login)
#   2) Waits until http://127.0.0.1:8000/health returns 200
#   3) If not healthy, prints port/listener status and last gateway.log tail
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GATEWAY_HEALTH_URL="http://127.0.0.1:8000/health"
GATEWAY_PORT="8000"
MAX_WAIT_SECONDS="90"
SLEEP_SECONDS="1"

echo "[1/3] Restarting route-only + UI..."
"${PROJECT_ROOT}/bin/restart_route_ui.sh"

echo "[2/3] Waiting gateway healthy: ${GATEWAY_HEALTH_URL}"
start_ts="$(date +%s)"

while true; do
  if curl -fsS "${GATEWAY_HEALTH_URL}" >/dev/null 2>&1; then
    echo "[OK] Gateway is healthy: ${GATEWAY_HEALTH_URL}"
    break
  fi

  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if [[ "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
    echo "[FAIL] Gateway not healthy within ${MAX_WAIT_SECONDS}s"
    echo "--- Port listener check (port ${GATEWAY_PORT}) ---"
    lsof -nP -iTCP:${GATEWAY_PORT} -sTCP:LISTEN 2>/dev/null || true
    echo "--- Gateway log tail ---"
    if [[ -f "${PROJECT_ROOT}/logs/runtime/gateway.log" ]]; then
      tail -n 120 "${PROJECT_ROOT}/logs/runtime/gateway.log" || true
    else
      echo "No logs/runtime/gateway.log found"
    fi
    exit 1
  fi

  sleep "${SLEEP_SECONDS}"
done

echo "[3/3] Done. You can retry your rewrite/UI now."
