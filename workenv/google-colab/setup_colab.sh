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
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    tar \
    gzip

echo ""
echo "======================================================================"
echo ">>> [3/6] Installing Synapse Python SDK and Core Dependencies..."
echo "======================================================================"
# Upgrade pip, wheel, setuptools
python3 -m pip install --upgrade --quiet pip setuptools wheel

# Install Synapse in editable mode with GPU, image, metrics, and training extras
# Note: [mesh] (trimesh) and [jit] (numba) omitted to avoid Colab LLVM conflicts;
# install them manually if needed: pip install -e ".[mesh,jit]"
if [ -f "pyproject.toml" ]; then
    python3 -m pip install --quiet -e ".[image,metrics,gpu,train]"
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
    "webdataset>=0.2.86"

# Build and install native C++20 geometry kernel (_c_geometry)
echo ">>> Building native _c_geometry extension kernel..."
python3 scripts/build_native.py
python3 -c "from synapse._c_geometry import validate_mesh_geometry; print('native OK')"

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
if [ -d "/content/drive/MyDrive" ]; then
    mkdir -p "${DERVGEN_DRIVE_ROOT}/work/${USER_ID}/synapse/$(date +%F)/e1-e4/checkpoints"
    mkdir -p "${DERVGEN_DRIVE_ROOT}/data/video/base_models"
    mkdir -p "${DERVGEN_DRIVE_ROOT}/data/common/foundations"
    # Backward compatibility
    mkdir -p "${DERVGEN_DRIVE_ROOT}/data/models/video_gen"
    echo "✅ Google Drive SSOT V2.1 ready: ${DERVGEN_DRIVE_ROOT}/work/${USER_ID}/synapse/$(date +%F)/e1-e4"
else
    echo "ℹ️ Note: /content/drive/MyDrive not mounted yet. Checkpoints will fall back to /content/synapse_checkpoints"
fi

echo ""
echo ">>> Staging Starter Benchmark Datasets (RFC 0063 ~250MB suite)..."
python3 scripts/prepare_benchmark_assets.py --tier starter --target-dir "${DERVGEN_DRIVE_ROOT}/data/benchmarks" || \
python3 scripts/prepare_benchmark_assets.py --tier starter --target-dir /content/datasets/benchmarks

echo ""
echo "======================================================================"
echo ">>> [6/6] Executing Synapse GPU Environment Auto-Detection..."
echo "======================================================================"
python3 scripts/colab_runner.py --check-env

echo ""
echo "======================================================================"
echo "🎉 Colab Environment Setup Complete! Ready for experiments E1-E4."
echo "======================================================================"
