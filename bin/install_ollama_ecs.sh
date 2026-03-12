#!/bin/bash
# Install Ollama on Alibaba ECS (Linux amd64)
# Run on the ECS: bash install_ollama_ecs.sh
# Or: ssh len 'bash -s' < bin/install_ollama_ecs.sh

set -e

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.17.7}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
TMP_DIR="/tmp/ollama-install-$$"
ARCH="linux-amd64"
ASSET="ollama-${ARCH}.tar.zst"
GITHUB_URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${ASSET}"

# Mirrors for China (try in order if direct GitHub fails)
MIRRORS=(
  "https://ghproxy.com/${GITHUB_URL}"
  "https://ghproxy.net/${GITHUB_URL}"
  "https://mirror.ghproxy.com/${GITHUB_URL}"
  "${GITHUB_URL}"
)

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TMP_DIR}"
cd "${TMP_DIR}"

# Use pre-downloaded file from manual scp if present
if [[ -f "/tmp/${ASSET}" ]]; then
  echo "[1/4] Using pre-downloaded file: /tmp/${ASSET}"
  cp "/tmp/${ASSET}" .
else
  echo "[1/4] Downloading Ollama ${OLLAMA_VERSION}..."
  for url in "${MIRRORS[@]}"; do
    echo "  Trying: ${url}"
    if curl -fsSL --connect-timeout 30 --max-time 300 -o "${ASSET}" "${url}"; then
      echo "  Download OK"
      break
    else
      rm -f "${ASSET}"
    fi
  done

  if [[ ! -f "${ASSET}" ]]; then
    echo "ERROR: Download failed from all sources."
    echo "Manual option: Download from https://github.com/ollama/ollama/releases"
    echo "  Then: scp ollama-linux-amd64.tar.zst len:/tmp/"
    echo "  Then re-run this script"
    exit 1
  fi
fi

echo "[2/4] Extracting..."
zstd -d -f "${ASSET}" -o ollama.tar
tar -xf ollama.tar

echo "[3/4] Installing to ${INSTALL_DIR}..."
sudo cp -f bin/ollama "${INSTALL_DIR}/ollama"
sudo chmod +x "${INSTALL_DIR}/ollama"

echo "[4/4] Creating systemd service..."
sudo tee /etc/systemd/system/ollama.service > /dev/null << 'SVC'
[Unit]
Description=Ollama LLM Server
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"
User=root

[Install]
WantedBy=default.target
SVC

sudo systemctl daemon-reload
sudo systemctl enable ollama

echo ""
echo "Ollama installed. Start with:"
echo "  sudo systemctl start ollama"
echo "  ollama pull qwen3:1.7b"
echo ""
echo "Or run in foreground: ollama serve"
