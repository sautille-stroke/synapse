#!/usr/bin/env bash
# ==============================================================================
# DervGen Workspace: Tailscale & Tailscale SSH Setup for Google Colab
# Connect directly from your local desktop / terminal to Colab GPU via Tailscale!
# ==============================================================================

set -euo pipefail

# Prefer passing auth key via TAILSCALE_AUTHKEY env var (never argv) to avoid
# exposing secrets in /proc/<pid>/cmdline.
# Usage:
#   TAILSCALE_AUTHKEY=... bash workenv/google-colab/tailscale.sh [hostname]
HOSTNAME="${1:-${TAILSCALE_HOSTNAME:-synapse-colab-gpu}}"
AUTH_KEY="${TAILSCALE_AUTHKEY:-}"

echo "=========================================================="
echo " Setting up Tailscale & Tailscale SSH: ${HOSTNAME}"
echo "=========================================================="

# 1. 确保 TUN 设备存在（Colab Docker 环境容错）
if [ ! -c /dev/net/tun ]; then
  echo "Creating /dev/net/tun device..."
  mkdir -p /dev/net
  mknod /dev/net/tun c 10 200 2>/dev/null || true
  chmod 600 /dev/net/tun 2>/dev/null || true
fi

# 2. 安装 Tailscale CLI & Daemon（若未安装）
if ! command -v tailscale &>/dev/null || ! command -v tailscaled &>/dev/null; then
  echo "Installing Tailscale..."
  curl -fsSL --proto '=https' --tlsv1.2 https://tailscale.com/install.sh | sh
  echo "Tailscale installed successfully."
fi

# 3. 准备运行目录并清理残留坏套接字
mkdir -p /var/run/tailscale /var/lib/tailscale

# 4. 检查 tailscaled 守护进程是否已在运行
if ! pgrep -x tailscaled >/dev/null; then
  rm -f /var/run/tailscale/tailscaled.sock
  echo "Starting tailscaled daemon in background..."
  nohup setsid tailscaled \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    >/tmp/tailscaled.log 2>&1 </dev/null &
  sleep 2

  # 若内核 TUN 启动失败，回退至 userspace-networking 容错模式
  if ! pgrep -x tailscaled >/dev/null; then
    rm -f /var/run/tailscale/tailscaled.sock
    echo "Standard TUN start failed; falling back to userspace-networking mode..."
    nohup setsid tailscaled \
      --tun=userspace-networking --socks5-server=localhost:1055 \
      --state=/var/lib/tailscale/tailscaled.state \
      --socket=/var/run/tailscale/tailscaled.sock \
      >/tmp/tailscaled.log 2>&1 </dev/null &
    sleep 2
  fi
fi

# 等待 tailscaled 存活且套接字就绪
for _ in {1..10}; do
  if pgrep -x tailscaled >/dev/null && [ -S /var/run/tailscale/tailscaled.sock ]; then
    break
  fi
  sleep 1
done

if ! pgrep -x tailscaled >/dev/null || [ ! -S /var/run/tailscale/tailscaled.sock ]; then
  echo "❌ Error: tailscaled failed to start. Log output:"
  cat /tmp/tailscaled.log || true
  exit 1
fi

echo "✅ tailscaled daemon is active."

# 5. 执行 tailscale up（使用 0600 临时密钥文件避免泄露至进程命令行）
UP_ARGS=("--hostname=${HOSTNAME}" "--ssh" "--accept-routes")
KEY_FILE=""
cleanup() {
  if [ -n "${KEY_FILE}" ] && [ -f "${KEY_FILE}" ]; then
    rm -f "${KEY_FILE}"
  fi
}
trap cleanup EXIT

if [ -n "${AUTH_KEY}" ]; then
  echo "Authenticating via TAILSCALE_AUTHKEY..."
  KEY_FILE="$(mktemp)"
  chmod 600 "${KEY_FILE}"
  printf '%s' "${AUTH_KEY}" > "${KEY_FILE}"
  UP_ARGS+=("--auth-key=file:${KEY_FILE}")
else
  echo "⚠️ No TAILSCALE_AUTHKEY provided. Interactive login URL will be generated."
  echo "👉 For unattended setups, store your Ephemeral Auth Key in Colab Secrets as 'TAILSCALE_AUTHKEY'."
fi

# 接入 Tailnet
tailscale up "${UP_ARGS[@]}"

# 6. 显示连接信息
echo "=========================================================="
echo "🎉 Tailscale is online!"
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "unknown")
echo "  • Node Hostname : ${HOSTNAME}"
echo "  • Tailscale IP  : ${TAILSCALE_IP}"
echo ""
echo "💻 How to connect from your local computer:"
echo "  1) Native Terminal / PowerShell:"
echo "     ssh root@${HOSTNAME}"
echo "     # 或使用 IP: ssh root@${TAILSCALE_IP}"
echo ""
echo "  2) VS Code Remote-SSH:"
echo "     Host ${HOSTNAME}"
echo "       HostName ${HOSTNAME}"
echo "       User root"
echo "=========================================================="
