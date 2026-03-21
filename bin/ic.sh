#!/usr/bin/env bash
# Short alias for project_stack.sh — same commands and options.
# See: ./bin/project_stack.sh --help
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/project_stack.sh" "$@"
