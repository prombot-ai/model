#!/usr/bin/env bash
# setup_environment.sh
# Prepares an Ubuntu system to run Qwen3.5-27B with TensorRT-LLM on 4x RTX 4090.
#
# Tested on: Ubuntu 22.04 LTS, NVIDIA driver 550+, CUDA 12.8, Python 3.10
#
# Usage:
#   bash scripts/setup_environment.sh

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-python3}"
TENSORRT_LLM_VERSION="${TENSORRT_LLM_VERSION:-0.20.0}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
CUDA_VERSION="${CUDA_VERSION:-cu128}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_VERSION}}"

PIP_COMMON_ARGS=(
    --index-url "${PIP_INDEX_URL}"
    --trusted-host "${PIP_TRUSTED_HOST}"
    --default-timeout "${PIP_DEFAULT_TIMEOUT}"
    --retries "${PIP_RETRIES}"
)

echo "=== Qwen3.5-27B TensorRT-LLM Setup ==="
echo "TensorRT-LLM : ${TENSORRT_LLM_VERSION}"
echo "PyTorch      : ${TORCH_VERSION}+${CUDA_VERSION}"
echo "PyPI mirror   : ${PIP_INDEX_URL}"
echo ""

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "--- Installing system packages ---"
sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    libopenmpi-dev \
    openmpi-bin \
    wget \
    curl \
    build-essential

git lfs install

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------
echo "--- Creating Python virtual environment (venv) ---"
if [[ ! -d "venv" ]]; then
    "${PYTHON_VERSION}" -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
python -m pip install --upgrade "${PIP_COMMON_ARGS[@]}" pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. PyTorch (CUDA build)
# ---------------------------------------------------------------------------
echo "--- Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION} ---"
python -m pip install \
    "torch==${TORCH_VERSION}" \
    torchvision \
    torchaudio \
    --index-url "${PYTORCH_INDEX_URL}" \
    --trusted-host "$(printf '%s' "${PYTORCH_INDEX_URL}" | sed -E 's#https?://([^/]+)/?.*#\1#')" \
    --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
    --retries "${PIP_RETRIES}"

# ---------------------------------------------------------------------------
# 4. TensorRT-LLM
# ---------------------------------------------------------------------------
echo "--- Installing TensorRT-LLM ${TENSORRT_LLM_VERSION} ---"
python -m pip install "${PIP_COMMON_ARGS[@]}" "tensorrt_llm==${TENSORRT_LLM_VERSION}"

# ---------------------------------------------------------------------------
# 5. Additional inference tooling
# ---------------------------------------------------------------------------
echo "--- Installing additional dependencies ---"
python -m pip install "${PIP_COMMON_ARGS[@]}" \
    huggingface_hub \
    transformers \
    accelerate \
    sentencepiece \
    openai

# ---------------------------------------------------------------------------
# 6. Verify GPU visibility
# ---------------------------------------------------------------------------
echo "--- Checking NVIDIA GPUs ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected ${GPU_COUNT} GPU(s)"
    if [[ "${GPU_COUNT}" -lt 4 ]]; then
        echo "WARNING: Expected 4 GPUs for TP=4, found ${GPU_COUNT}."
    fi
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment with:"
echo "  source venv/bin/activate"
