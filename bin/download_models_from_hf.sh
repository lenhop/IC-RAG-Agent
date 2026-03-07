#!/usr/bin/env bash
#
# Download Qwen3-1.7B-GGUF and all-MiniLM-L6-v2 from HuggingFace.
#
# Usage:
#   ./bin/download_models_from_hf.sh [all|qwen3|minilm]
#   ./bin/download_models_from_hf.sh
#   ./bin/download_models_from_hf.sh qwen3
#
# Requires: huggingface_hub (pip install huggingface_hub).
# Proxy: Set HTTPS_PROXY for China (for example: export HTTPS_PROXY=http://127.0.0.1:7890).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-${PROJECT_ROOT}/models}"

QWEN3_GGUF_REPO="Qwen/Qwen3-1.7B-GGUF"
MINILM_REPO="sentence-transformers/all-MiniLM-L6-v2"

download_repo() {
  local repo_id="$1"
  local local_name="$2"
  local local_dir="${MODELS_DIR}/${local_name}"

  echo "Downloading: ${repo_id}"
  echo "Target: ${local_dir}"

  mkdir -p "${local_dir}"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" \
      --local-dir "${local_dir}" \
      --local-dir-use-symlinks False \
      --resume-download
  else
    echo "huggingface-cli not found, using Python fallback."
    python3 <<PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${repo_id}",
    local_dir="${local_dir}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PYEOF
  fi
}

main() {
  local target="${1:-all}"

  case "${target}" in
    all)
      download_repo "${QWEN3_GGUF_REPO}" "Qwen3-1.7B-GGUF"
      download_repo "${MINILM_REPO}" "all-MiniLM-L6-v2"
      ;;
    qwen3)
      download_repo "${QWEN3_GGUF_REPO}" "Qwen3-1.7B-GGUF"
      ;;
    minilm)
      download_repo "${MINILM_REPO}" "all-MiniLM-L6-v2"
      ;;
    *)
      echo "Usage: $0 [all|qwen3|minilm]"
      exit 1
      ;;
  esac

  echo "Download complete."
}

main "$@"
