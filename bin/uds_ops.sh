#!/usr/bin/env bash
# Unified operations command for UDS Agent deployment and maintenance.
#
# Usage:
#   ./bin/uds_ops.sh status
#   ./bin/uds_ops.sh logs
#   ./bin/uds_ops.sh deploy [version]
#   ./bin/uds_ops.sh rollback <version>
#   ./bin/uds_ops.sh install-service
#   ./bin/uds_ops.sh setup-server
#   ./bin/uds_ops.sh setup-nginx

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_APP_DIR="/opt/uds-agent"
REMOTE_COMPOSE_FILE="docker-compose.prod.yml"

print_help() {
  cat <<'EOF'
UDS operations helper

Commands:
  status                 Show remote service status
  logs                   Tail remote service logs
  deploy [version]       Deploy version (default: latest)
  rollback <version>     Roll back to a version
  install-service        Install local systemd service from docker/uds-agent.service
  setup-server           Setup ECS host dependencies (docker/compose/nginx)
  setup-nginx            Install Nginx reverse proxy config
EOF
}

remote_status() {
  ssh len <<EOF
set -e
cd "${REMOTE_APP_DIR}"
docker-compose -f "${REMOTE_COMPOSE_FILE}" ps
EOF
}

health_check() {
  local ecs_host
  ecs_host="$(ssh len 'curl -s ifconfig.me')"
  echo "Checking health endpoint: http://${ecs_host}/health"
  if ! curl -fsS "http://${ecs_host}/health" >/dev/null; then
    echo "Health check failed. Run:"
    echo "  ./bin/uds_ops.sh logs"
    echo "  ./bin/uds_ops.sh status"
    exit 1
  fi
  echo "Health check passed."
}

do_deploy() {
  local version="${1:-latest}"
  echo "Deploying version: ${version}"

  ssh len <<EOF
set -e
cd "${REMOTE_APP_DIR}"
export VERSION="${version}"
docker-compose -f "${REMOTE_COMPOSE_FILE}" pull
docker-compose -f "${REMOTE_COMPOSE_FILE}" down
docker-compose -f "${REMOTE_COMPOSE_FILE}" up -d
docker-compose -f "${REMOTE_COMPOSE_FILE}" ps
EOF

  sleep 10
  health_check
  echo "Deployment completed."
}

do_rollback() {
  local version="${1:-}"
  if [[ -z "${version}" ]]; then
    echo "Usage: ./bin/uds_ops.sh rollback <version>"
    exit 1
  fi

  echo "Rolling back to version: ${version}"
  ssh len <<EOF
set -e
cd "${REMOTE_APP_DIR}"
export VERSION="${version}"
docker-compose -f "${REMOTE_COMPOSE_FILE}" pull
docker-compose -f "${REMOTE_COMPOSE_FILE}" down
docker-compose -f "${REMOTE_COMPOSE_FILE}" up -d
docker-compose -f "${REMOTE_COMPOSE_FILE}" ps
EOF

  sleep 10
  health_check
  echo "Rollback completed."
}

do_logs() {
  ssh len <<EOF
set -e
cd "${REMOTE_APP_DIR}"
docker-compose -f "${REMOTE_COMPOSE_FILE}" logs -f uds-agent --tail=100
EOF
}

do_install_service() {
  cd "${PROJECT_ROOT}"
  sudo cp docker/uds-agent.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable uds-agent
  sudo systemctl restart uds-agent
  sudo systemctl status uds-agent
}

do_setup_server() {
  if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "${USER}"
  fi

  if ! command -v docker-compose >/dev/null 2>&1; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
  fi

  if ! command -v nginx >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y nginx
  fi

  sudo mkdir -p /opt/uds-agent /var/log/uds-agent
  sudo chown -R "${USER}:${USER}" /opt/uds-agent /var/log/uds-agent

  echo "Server setup completed."
}

do_setup_nginx() {
  cd "${PROJECT_ROOT}"
  sudo cp docker/nginx/uds-agent.conf /etc/nginx/sites-available/uds-agent
  sudo ln -sf /etc/nginx/sites-available/uds-agent /etc/nginx/sites-enabled/uds-agent
  sudo nginx -t
  sudo systemctl reload nginx
  echo "Nginx setup completed."
}

main() {
  local command="${1:-help}"
  shift || true

  case "${command}" in
    status) remote_status ;;
    logs) do_logs ;;
    deploy) do_deploy "${1:-latest}" ;;
    rollback) do_rollback "${1:-}" ;;
    install-service) do_install_service ;;
    setup-server) do_setup_server ;;
    setup-nginx) do_setup_nginx ;;
    help|--help|-h) print_help ;;
    *)
      echo "Unknown command: ${command}"
      print_help
      exit 1
      ;;
  esac
}

main "$@"
