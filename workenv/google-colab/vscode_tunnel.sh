#!/usr/bin/env bash
# ==============================================================================
# DervGen Workspace: Visual Studio Code Tunnel Setup for Google Colab
# Connect directly from your local desktop VS Code / Antigravity to Colab GPU!
# ==============================================================================

set -euo pipefail

TUNNEL_NAME="${1:-synapse-colab-gpu}"

echo "=========================================================="
echo " Setting up VS Code Remote Tunnel: ${TUNNEL_NAME}"
echo "=========================================================="

if ! command -v code &>/dev/null; then
  echo "Downloading VS Code CLI for Linux x64..."
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  curl -fsSL --proto '=https' --tlsv1.2 \
    'https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64' \
    --output "${TMP_DIR}/vscode_cli.tar.gz"
  tar -xf "${TMP_DIR}/vscode_cli.tar.gz" -C /usr/local/bin
  echo "VS Code CLI installed to /usr/local/bin/code"
fi

echo "Starting VS Code Tunnel..."
echo "Follow the on-screen instructions to log in via GitHub / Microsoft."
echo "Once logged in, open the printed link in your local browser or VS Code."
code tunnel --accept-server-license-terms --name "${TUNNEL_NAME}"
