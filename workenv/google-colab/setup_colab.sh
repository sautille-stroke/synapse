#!/usr/bin/env bash
# ==============================================================================
# Turnkey Environment Setup Script for Synapse on Google Colab (Linux / CUDA)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo ">>> [1/6] Detecting System and GPU Hardware..."
echo "======================================================================"
uname -a
python3 --version

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️ WARNING: nvidia-smi not found! Ensure Colab Runtime is set to GPU (T4 / L4 / A100)."
fi

echo ""
echo "======================================================================"
echo ">>> [2/6] Installing System Binaries & OpenEXR C++ Libraries..."
echo "======================================================================"
sudo apt-get update -y -qq
sudo apt-get install -y -qq \
    libopenexr-dev \
    openexr \
    libglib2.0-0 \
    ffmpeg \
    tar \
    gzip
# libgl1 is the modern package name; fall back to libgl1-mesa-glx on older distros
sudo apt-get install -y libgl1 2>/dev/null || sudo apt-get install -y libgl1-mesa-glx

# Optional: shallow-clone dervgen/aether for native geometry compilation
SYNAPSE_ROOT="$(pwd)"
if [ ! -d "${SYNAPSE_ROOT}/../aether" ]; then
  echo "[setup_colab] Cloning dervgen/aether (shallow) for native geometry build..."
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    AUTH_B64="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 | tr -d '\n')"
    git -c "http.extraheader=Authorization: Basic ${AUTH_B64}" \
      clone --depth 1 https://github.com/dervgen/aether.git "${SYNAPSE_ROOT}/../aether" || \
      echo "[setup_colab] WARNING: Could not clone dervgen/aether. Operating in pure Python Oracle geometry mode."
  else
    git clone --depth 1 https://github.com/dervgen/aether.git "${SYNAPSE_ROOT}/../aether" || \
      echo "[setup_colab] WARNING: Could not clone dervgen/aether (set GITHUB_TOKEN for private repo). Operating in pure Python Oracle geometry mode."
  fi
fi

# Install Bazelisk if bazel/bazelisk not found (required for native kernel build)
if ! command -v bazel &>/dev/null && ! command -v bazelisk &>/dev/null; then
  echo "[setup_colab] Installing Bazelisk..."
  BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
  curl -fsSL "${BAZELISK_URL}" -o /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
  echo "[setup_colab] ✅ Bazelisk installed as /usr/local/bin/bazel"
fi

echo ""
echo "======================================================================"
echo ">>> [3/6] Installing Synapse Python SDK and Core Dependencies..."
echo "======================================================================"
# Upgrade pip, wheel, setuptools
python3 -m pip install --upgrade --quiet pip setuptools wheel

# Install Synapse in editable mode with GPU, image, metrics, mesh, and training extras
if [ -f "pyproject.toml" ]; then
    python3 -m pip install --quiet -e ".[image,metrics,gpu,train,mesh,dev]"
else
    echo "❌ Error: pyproject.toml not found. Please run this script from the synapse repository root."
    exit 1
fi

# Install modern diffusers, accelerate, and huggingface_hub for Wan 2.1 inference
python3 -m pip install --quiet \
    "diffusers>=0.30.0" \
    "accelerate>=0.33.0" \
    "transformers>=4.44.0" \
    "sentencepiece>=0.2.0" \
    "webdataset>=0.2.86" \
    "huggingface_hub>=0.24.0" \
    "pybind11>=2.12.0"

# Install optional Traceboard SDK (prefer sibling aether source, fallback to PyPI)
if [ -d "${SYNAPSE_ROOT}/../aether/traceboard/python" ]; then
    python3 -m pip install --quiet -e "${SYNAPSE_ROOT}/../aether/traceboard/python"
else
    python3 -m pip install --quiet "aether-traceboard>=0.1.0" || \
        echo "⚠️ Optional aether-traceboard unavailable; continuing in graceful-degradation mode."
fi

# Build and install native C++20 geometry kernel (@aether//python:aether_geo)
echo ">>> Building native aether_geo extension kernel..."
if python3 scripts/build_native.py --output-dir python; then
    python3 -c "from synapse.bridge.aether_bridge import require_aether_geo; require_aether_geo(); print('✅ aether_geo native extension OK')"
else
    echo "⚠️ Native compilation skipped or failed; operating in pure Python Oracle geometry mode."
fi

echo ""
echo "======================================================================"
echo ">>> [4/6] Verifying Python & C Extensions Import Integrity..."
echo "======================================================================"
python3 -c "
import numpy as np
import pydantic
import OpenEXR
import cv2
import torch
import synapse
from synapse.fgpp import FGPPFrame
from synapse.adapters import WanAdapter, CausalWindowPlan

print(f'✅ Synapse version: {synapse.__version__}')
print(f'✅ PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
    print(f'   bfloat16 supported: {torch.cuda.is_bf16_supported()}')
print('✅ OpenEXR & OpenCV modules initialized cleanly.')
"

echo ""
echo "======================================================================"
echo ">>> [5/6] Preparing Directory Layouts & Mount Anchors..."
echo "======================================================================"
mkdir -p /content/datasets/synapse
mkdir -p /content/synapse_checkpoints

DERVGEN_DRIVE_ROOT="${DERVGEN_DRIVE_ROOT:-/content/drive/MyDrive/dervgen}"
USER_ID="${COLAB_USER:-yanglei}"
JOB_DATE="$(date +%F)"
if [ -d "/content/drive/MyDrive" ]; then
    # SSOT V2.0: All experiment output under /dervgen/work/{user}/synapse/{date}/{job_id}/
    mkdir -p "${DERVGEN_DRIVE_ROOT}/work/${USER_ID}/synapse/${JOB_DATE}/e1-e4/checkpoints"
    echo "✅ Google Drive SSOT V2.0 ready: ${DERVGEN_DRIVE_ROOT}/work/${USER_ID}/synapse/${JOB_DATE}/e1-e4"
else
    echo "ℹ️ Note: /content/drive/MyDrive not mounted yet. Checkpoints will fall back to /content/synapse_checkpoints"
fi

echo ""
echo ">>> Staging Starter Benchmark Datasets (RFC 0063 ~250MB suite)..."
# SSOT V2.0: Benchmark datasets are staged exclusively to fast NVMe local disk (/content/datasets/benchmarks),
# NEVER to Drive FUSE mount (prevents I/O bottlenecks and Drive quota pollution).
BENCH_TARGET="/content/datasets/benchmarks"
mkdir -p "${BENCH_TARGET}"
python3 scripts/prepare_benchmark_assets.py --tier starter_dev --allow-synthetic-smoke --target-dir "${BENCH_TARGET}"

echo ""
echo "======================================================================"
echo ">>> [6/6] Executing Synapse GPU Environment Auto-Detection..."
echo "======================================================================"
python3 scripts/colab_runner.py --check-env

echo ""
echo "======================================================================"
echo "🎉 Colab Environment Setup Complete! Ready for experiments E1-E4."
echo "======================================================================"

