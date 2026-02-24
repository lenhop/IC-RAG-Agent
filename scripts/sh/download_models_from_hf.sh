#!/usr/bin/env bash
#
# Download Qwen3-1.7B-GGUF and all-MiniLM-L6-v2 from HuggingFace.
#
# Usage:
#   ./scripts/sh/download_models_from_hf.sh [all|qwen3|minilm]
#   ./scripts/sh/download_models_from_hf.sh
#   ./scripts/sh/download_models_from_hf.sh qwen3
#
# Requires: huggingface_hub (pip install huggingface_hub).
# Proxy: Set HTTPS_PROXY for China (e.g. export HTTPS_PROXY=http://127.0.0.1:7890).
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"

# HuggingFace repo IDs
QWEN3_GGUF_REPO="Qwen/Qwen3-1.7B-GGUF"
MINILM_REPO="sentence-transformers/all-MiniLM-L6-v2"

download_repo() {
    local repo_id="$1"
    local local_name="$2"
    local local_dir="$MODELS_DIR/$local_name"

    echo "=============================================="
    echo "Downloading: $repo_id"
    echo "Target: $local_dir"
    echo "=============================================="

    mkdir -p "$local_dir"

    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo_id" \
            --local-dir "$local_dir" \
            --local-dir-use-symlinks False \
            --resume-download
    else
        echo "[WARN] huggingface-cli not found. Using Python..."
        python3 << PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$repo_id",
    local_dir="$local_dir",
    local_dir_use_symlinks=False,
    resume_download=True
)
PYEOF
    fi

    echo "[OK] Downloaded: $local_name"
    echo ""
}

main() {
    local target="${1:-all}"

    echo "[OK] Models directory: $MODELS_DIR"
    if [ -n "${HTTPS_PROXY:-}" ]; then
        echo "[OK] Using proxy: $HTTPS_PROXY"
    fi
    echo ""

    case "$target" in
        all)
            download_repo "$QWEN3_GGUF_REPO" "Qwen3-1.7B-GGUF"
            download_repo "$MINILM_REPO" "all-MiniLM-L6-v2"
            ;;
        qwen3)
            download_repo "$QWEN3_GGUF_REPO" "Qwen3-1.7B-GGUF"
            ;;
        minilm)
            download_repo "$MINILM_REPO" "all-MiniLM-L6-v2"
            ;;
        *)
            echo "Usage: $0 [all|qwen3|minilm]"
            echo "  all    - Download both Qwen3-1.7B-GGUF and all-MiniLM-L6-v2 (default)"
            echo "  qwen3  - Download Qwen3-1.7B-GGUF only (~1.8GB)"
            echo "  minilm - Download all-MiniLM-L6-v2 only (~80MB)"
            exit 1
            ;;
    esac

    echo "=============================================="
    echo "[OK] Download complete."
    echo "=============================================="
}

main "$@"
