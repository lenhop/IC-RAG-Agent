#!/usr/bin/env bash
# Shortcut: restart stack with route-only + UI and skip login (no auth for quick dev testing).
# Usage: ./bin/restart_route_ui.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/project_stack.sh" restart --route-only --with-ui --no-login
