#!/usr/bin/env bash
# Start/restart the local IC-RAG-Agent service stack.
#
# Default managed services and ports:
# - gateway: 8000
# - uds:     8001
# - rag:     8002
# - sp_api:  8003
# Optional:
# - ui:      7862
#
# Usage:
#   ./bin/project_stack.sh start
#   ./bin/project_stack.sh restart
#   ./bin/project_stack.sh stop
#   ./bin/project_stack.sh status
#   ./bin/project_stack.sh start --with-ui

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Source .env if present so gateway inherits GATEWAY_* vars (e.g. ECS Ollama URL).
# Skip if .env causes errors (e.g. malformed API secrets); gateway falls back to defaults.
if [[ -f .env ]]; then
  set +e
  set -a
  source .env 2>/dev/null
  set +a
  set -e
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/opt/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/opt/miniconda3/bin/python"
else
  PYTHON_BIN="python"
fi
LOG_DIR="${PROJECT_ROOT}/logs/runtime"
PID_DIR="${PROJECT_ROOT}/.runtime"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

WITH_UI="false"
REWRITE_ONLY_MODE="false"
NO_LOGIN="false"

print_usage() {
  cat <<'EOF'
Manage local IC-RAG-Agent stack.

Commands:
  start              Start all backend services
  restart            Restart all backend services
  stop               Stop all managed services
  status             Show managed services status

Options:
  --with-ui          Also run unified chat UI on port 7862
  --route-only       Route LLM only: gateway runs clarification + rewrite + intents (no plan in rewrite endpoint);
                     no downstream workers (uds/rag/sp_api). Use for quick testing.
  --rewrite-only     Alias for --route-only (deprecated, use --route-only)
  --no-login         With --with-ui: skip login and show chat directly (dev convenience).

Examples:
  ./bin/project_stack.sh start
  ./bin/project_stack.sh restart --with-ui
  ./bin/project_stack.sh restart --route-only --with-ui --no-login
  ./bin/project_stack.sh status
EOF
}

services() {
  # Space-delimited list for bash 3 compatibility.
  if [[ "${REWRITE_ONLY_MODE}" == "true" ]]; then
    if [[ "${WITH_UI}" == "true" ]]; then
      echo "gateway ui"
    else
      echo "gateway"
    fi
    return 0
  fi

  if [[ "${WITH_UI}" == "true" ]]; then
    echo "gateway uds rag sp_api ui"
  else
    echo "gateway uds rag sp_api"
  fi
}

service_health_attempts() {
  case "$1" in
    # RAG can take longer to initialize local models/vector store.
    rag) echo "180" ;;
    # UI boot can be slower when dependencies are cold-starting.
    ui) echo "90" ;;
    *) echo "30" ;;
  esac
}

service_port() {
  case "$1" in
    gateway) echo "8000" ;;
    uds) echo "8001" ;;
    rag) echo "8002" ;;
    sp_api) echo "8003" ;;
    ui) echo "7862" ;;
    *) echo "" ;;
  esac
}

service_pid_file() {
  echo "${PID_DIR}/$1.pid"
}

service_log_file() {
  echo "${LOG_DIR}/$1.log"
}

service_health_url() {
  case "$1" in
    gateway) echo "http://127.0.0.1:8000/health" ;;
    uds) echo "http://127.0.0.1:8001/health" ;;
    rag) echo "http://127.0.0.1:8002/health" ;;
    sp_api) echo "http://127.0.0.1:8003/api/v1/health" ;;
    ui) echo "http://127.0.0.1:7862/" ;;
    *) echo "" ;;
  esac
}

service_start_cmd() {
  case "$1" in
    gateway)
      local rag_url="${RAG_API_URL:-http://127.0.0.1:8002}"
      local uds_url="${UDS_API_URL:-http://127.0.0.1:8001}"
      local sp_url="${SP_API_URL:-http://127.0.0.1:8003}"
      local rewrite_backend="${GATEWAY_REWRITE_BACKEND:-ollama}"
      local clarification_backend="${GATEWAY_CLARIFICATION_BACKEND:-${rewrite_backend}}"
      local intent_split_backend="${GATEWAY_INTENT_SPLIT_BACKEND:-${rewrite_backend}}"
      local intent_detect_backend="${GATEWAY_INTENT_DETECT_BACKEND:-${rewrite_backend}}"
      local ollama_url="${GATEWAY_REWRITE_OLLAMA_URL:-http://localhost:11434/api/generate}"
      local ollama_model="${GATEWAY_REWRITE_OLLAMA_MODEL:-qwen3:1.7b}"
      local route_url="${GATEWAY_ROUTE_LLM_OLLAMA_URL:-http://localhost:11434}"
      local route_model="${GATEWAY_ROUTE_LLM_OLLAMA_MODEL:-qwen3:1.7b}"
      if [[ "${REWRITE_ONLY_MODE}" == "true" ]]; then
        echo "RAG_API_URL=${rag_url} UDS_API_URL=${uds_url} SP_API_URL=${sp_url} GATEWAY_REWRITE_ONLY_MODE=true GATEWAY_REWRITE_PLANNER_ENABLED=true GATEWAY_CLARIFICATION_ENABLED=true GATEWAY_REWRITE_BACKEND=${rewrite_backend} GATEWAY_CLARIFICATION_BACKEND=${clarification_backend} GATEWAY_INTENT_SPLIT_BACKEND=${intent_split_backend} GATEWAY_INTENT_DETECT_BACKEND=${intent_detect_backend} GATEWAY_REWRITE_OLLAMA_URL=${ollama_url} GATEWAY_REWRITE_OLLAMA_MODEL=${ollama_model} GATEWAY_ROUTE_LLM_OLLAMA_URL=${route_url} GATEWAY_ROUTE_LLM_OLLAMA_MODEL=${route_model} GATEWAY_PORT=8000 ${PYTHON_BIN} scripts/run_gateway.py"
      else
        echo "RAG_API_URL=${rag_url} UDS_API_URL=${uds_url} SP_API_URL=${sp_url} GATEWAY_REWRITE_PLANNER_ENABLED=true GATEWAY_CLARIFICATION_ENABLED=true GATEWAY_REWRITE_BACKEND=${rewrite_backend} GATEWAY_CLARIFICATION_BACKEND=${clarification_backend} GATEWAY_INTENT_SPLIT_BACKEND=${intent_split_backend} GATEWAY_INTENT_DETECT_BACKEND=${intent_detect_backend} GATEWAY_REWRITE_OLLAMA_URL=${ollama_url} GATEWAY_REWRITE_OLLAMA_MODEL=${ollama_model} GATEWAY_ROUTE_LLM_OLLAMA_URL=${route_url} GATEWAY_ROUTE_LLM_OLLAMA_MODEL=${route_model} GATEWAY_PORT=8000 ${PYTHON_BIN} scripts/run_gateway.py"
      fi
      ;;
    uds)
      echo "UDS_LLM_PROVIDER=${UDS_LLM_PROVIDER:-ollama} ${PYTHON_BIN} -m uvicorn src.uds.api:app --host 0.0.0.0 --port 8001"
      ;;
    rag)
      # Force RAG to 8002 to avoid collision with gateway on 8000.
      echo "RAG_API_PORT=8002 ./bin/run_rag_api.sh"
      ;;
    sp_api)
      echo "${PYTHON_BIN} -m uvicorn src.sp_api.fast_api:app --host 0.0.0.0 --port 8003"
      ;;
    ui)
      local no_login_env=""
      local ui_rewrite_backend="${UNIFIED_CHAT_REWRITE_BACKEND:-${GATEWAY_REWRITE_BACKEND:-ollama}}"
      [[ "${NO_LOGIN}" == "true" ]] && no_login_env="UNIFIED_CHAT_SKIP_LOGIN=true "
      if [[ "${REWRITE_ONLY_MODE}" == "true" ]]; then
        echo "GATEWAY_API_URL=http://127.0.0.1:8000 GATEWAY_MOCK=false ${no_login_env}UNIFIED_CHAT_REWRITE_ONLY_MODE=true UNIFIED_CHAT_REWRITE_ENABLE=true UNIFIED_CHAT_REWRITE_BACKEND=${ui_rewrite_backend} UNIFIED_CHAT_GRADIO_PORT=7862 ${PYTHON_BIN} scripts/run_unified_chat.py"
      else
        echo "GATEWAY_API_URL=http://127.0.0.1:8000 GATEWAY_MOCK=false ${no_login_env}UNIFIED_CHAT_GRADIO_PORT=7862 ${PYTHON_BIN} scripts/run_unified_chat.py"
      fi
      ;;
    *)
      echo ""
      ;;
  esac
}

is_pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

kill_port_listener() {
  local port="$1"
  if [[ -z "${port}" ]]; then
    return 0
  fi
  # Collect PIDs listening on the target port (if any).
  local pids
  pids="$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill ${pids} >/dev/null 2>&1 || true
    sleep 1
    pids="$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -9 ${pids} >/dev/null 2>&1 || true
    fi
  fi
}

wait_for_health() {
  local service="$1"
  local url max_attempts pid_file pid
  url="$(service_health_url "${service}")"
  max_attempts="$(service_health_attempts "${service}")"
  pid_file="$(service_pid_file "${service}")"
  local i=1
  while [[ "${i}" -le "${max_attempts}" ]]; do
    # Fail fast: if managed process already died, don't wait for full timeout.
    if [[ -f "${pid_file}" ]]; then
      pid="$(tr -d '[:space:]' < "${pid_file}")"
      if ! is_pid_alive "${pid}"; then
        echo "[warn] ${service} process exited before becoming healthy"
        return 1
      fi
    fi

    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[ok] ${service} healthy (${url})"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "[warn] ${service} health check timed out (${url})"
  return 1
}

check_python_module() {
  local module_name="$1"
  "${PYTHON_BIN}" -c "import ${module_name}" >/dev/null 2>&1
}

validate_optional_services() {
  if [[ "${WITH_UI}" == "true" ]]; then
    if ! check_python_module "gradio"; then
      echo "[warn] UI requested (--with-ui) but Python module 'gradio' is not installed."
      echo "[warn] Skipping UI startup. Install with: ${PYTHON_BIN} -m pip install gradio"
      WITH_UI="false"
    fi
  fi
}

start_service() {
  local service="$1"
  local port pid_file log_file cmd existing_pid
  port="$(service_port "${service}")"
  pid_file="$(service_pid_file "${service}")"
  log_file="$(service_log_file "${service}")"
  cmd="$(service_start_cmd "${service}")"

  if [[ -f "${pid_file}" ]]; then
    existing_pid="$(tr -d '[:space:]' < "${pid_file}")"
    if is_pid_alive "${existing_pid}"; then
      echo "[skip] ${service} already running (pid=${existing_pid}, port=${port})"
      return 0
    fi
    rm -f "${pid_file}"
  fi

  # Ensure target port is free before startup.
  kill_port_listener "${port}"

  echo "[start] ${service} on port ${port}"
  nohup bash -lc "${cmd}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_file}"
  sleep 1
  if ! is_pid_alive "${pid}"; then
    echo "[error] ${service} failed to start. Check log: ${log_file}"
    return 1
  fi
}

stop_service() {
  local service="$1"
  local pid_file port pid
  pid_file="$(service_pid_file "${service}")"
  port="$(service_port "${service}")"

  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -d '[:space:]' < "${pid_file}")"
    if is_pid_alive "${pid}"; then
      echo "[stop] ${service} pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 1
      if is_pid_alive "${pid}"; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    fi
    rm -f "${pid_file}"
  fi

  # Clean up any stray listener on the managed port.
  kill_port_listener "${port}"
}

status_service() {
  local service="$1"
  local pid_file port pid health log_file
  pid_file="$(service_pid_file "${service}")"
  port="$(service_port "${service}")"
  log_file="$(service_log_file "${service}")"

  if [[ -f "${pid_file}" ]]; then
    pid="$(tr -d '[:space:]' < "${pid_file}")"
    if is_pid_alive "${pid}"; then
      health="unreachable"
      if curl -fsS "$(service_health_url "${service}")" >/dev/null 2>&1; then
        health="healthy"
      fi
      echo "[running] ${service} pid=${pid} port=${port} health=${health}"
      return 0
    fi
    echo "[stale]   ${service} pid file exists but process is dead"
    if [[ -f "${log_file}" ]]; then
      echo "          last log lines (${log_file}):"
      tail -n 5 "${log_file}" | sed 's/^/          /'
    fi
    return 0
  fi

  if lsof -tiTCP:${port} -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[running] ${service} port=${port} (not managed by this script)"
  else
    echo "[stopped] ${service}"
  fi
}

do_start() {
  local service
  validate_optional_services
  for service in $(services); do
    start_service "${service}"
  done

  echo ""
  echo "Waiting for services to become healthy..."
  for service in $(services); do
    wait_for_health "${service}" || true
  done

  echo ""
  do_status
  echo ""
  echo "Logs: ${LOG_DIR}"
}

do_stop() {
  local service
  # Stop in reverse dependency order.
  if [[ "${WITH_UI}" == "true" ]]; then
    stop_service "ui"
  fi
  stop_service "sp_api"
  stop_service "rag"
  stop_service "uds"
  stop_service "gateway"
}

do_restart() {
  do_stop
  do_start
}

do_status() {
  local service
  for service in $(services); do
    status_service "${service}"
  done
}

main() {
  local command="${1:-}"
  shift || true

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-ui)
        WITH_UI="true"
        ;;
      --route-only|--rewrite-only)
        REWRITE_ONLY_MODE="true"
        ;;
      --no-login)
        NO_LOGIN="true"
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        print_usage
        exit 1
        ;;
    esac
    shift
  done

  case "${command}" in
    start) do_start ;;
    restart) do_restart ;;
    stop) do_stop ;;
    status) do_status ;;
    ""|help|--help|-h)
      print_usage
      ;;
    *)
      echo "Unknown command: ${command}"
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
