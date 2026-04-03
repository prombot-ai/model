#!/usr/bin/env bash
# setup_environment.sh
# Prepares a Linux system to run Gemma 4 26B (MoE) with TensorRT-LLM on 4x RTX 4090.
#
# Gemma 4 26B A4B is a Mixture-of-Experts model:
#   - 26B total parameters (all must be loaded into VRAM)
#   - 4B parameters active per token
#   - fp16 footprint ~48 GB  →  fits comfortably on 4× RTX 4090 (96 GB total)
#
# Tested on: Ubuntu 22.04 LTS / CentOS Stream 9, NVIDIA driver 550+, CUDA 12.8,
# Python 3.10+
#
# Usage:
#   bash scripts/gemma4-26b/setup_environment.sh

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

OS_ID=""
OS_VERSION_ID=""
OS_FAMILY=""

PIP_COMMON_ARGS=(
    --index-url "${PIP_INDEX_URL}"
    --trusted-host "${PIP_TRUSTED_HOST}"
    --default-timeout "${PIP_DEFAULT_TIMEOUT}"
    --retries "${PIP_RETRIES}"
)

detect_os() {
    if [[ ! -r /etc/os-release ]]; then
        echo "ERROR: /etc/os-release not found; unable to detect the operating system." >&2
        exit 1
    fi

    # shellcheck disable=SC1091
    source /etc/os-release

    OS_ID="${ID:-}"
    OS_VERSION_ID="${VERSION_ID:-}"

    case "${OS_ID}" in
        ubuntu|debian)
            OS_FAMILY="debian"
            ;;
        centos|rhel|rocky|almalinux)
            OS_FAMILY="rhel"
            ;;
        *)
            echo "ERROR: Unsupported distribution '${OS_ID}'. Supported families: Ubuntu/Debian and CentOS/RHEL 9-compatible systems." >&2
            exit 1
            ;;
    esac
}

install_system_packages() {
    echo "--- Installing system packages ---"
    echo "Detected OS  : ${OS_ID} ${OS_VERSION_ID}"

    case "${OS_FAMILY}" in
        debian)
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
            ;;
        rhel)
            if ! command -v dnf &>/dev/null; then
                echo "ERROR: dnf package manager not found. Please use CentOS/RHEL 8+." >&2
                exit 1
            fi
            sudo dnf install -y epel-release || true
            sudo dnf install -y \
                python3 \
                python3-pip \
                git \
                git-lfs \
                openmpi \
                openmpi-devel \
                wget \
                curl \
                gcc \
                gcc-c++ \
                make
            # Add OpenMPI to PATH on RHEL/CentOS
            MPI_MODULE_FILE="/etc/profile.d/openmpi.sh"
            if [[ ! -f "${MPI_MODULE_FILE}" ]]; then
                echo 'export PATH=/usr/lib64/openmpi/bin:${PATH:-}' | sudo tee "${MPI_MODULE_FILE}"
                echo 'export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}' | sudo tee -a "${MPI_MODULE_FILE}"
                # shellcheck disable=SC1090
                source "${MPI_MODULE_FILE}" || true
            fi
            ;;
    esac

    # Ensure git-lfs is initialised (required to download model weights)
    git lfs install
}

check_cuda() {
    echo "--- Checking CUDA installation ---"
    if ! command -v nvcc &>/dev/null && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
        echo "WARNING: nvcc not found. Ensure CUDA ${CUDA_VERSION} toolkit is installed." >&2
        echo "  Ubuntu: https://developer.nvidia.com/cuda-downloads"
        echo "  CentOS: https://developer.nvidia.com/cuda-downloads"
    else
        NVCC_BIN="$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)"
        echo "nvcc found: $("${NVCC_BIN}" --version | head -1)"
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers (550+) are installed." >&2
    else
        echo "GPU info:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | \
            awk '{print "  GPU " NR-1 ": " $0}'
    fi
}

setup_venv() {
    echo "--- Creating Python virtual environment (venv) ---"
    if [[ ! -d "venv" ]]; then
        "${PYTHON_VERSION}" -m venv venv
    else
        echo "venv already exists, skipping creation."
    fi
    PYTHON_VERSION="$(pwd)/venv/bin/python"

    # Bootstrap pip if absent — common on CentOS when the venv was created
    # before python3-pip was installed, or when RHEL strips pip from the venv.
    if ! "${PYTHON_VERSION}" -m pip --version &>/dev/null; then
        echo "pip not found in venv; bootstrapping via ensurepip..."
        "${PYTHON_VERSION}" -m ensurepip --upgrade
    fi
}

upgrade_pip() {
    echo "--- Upgrading pip / setuptools / wheel ---"
    "${PYTHON_VERSION}" -m pip install \
        "${PIP_COMMON_ARGS[@]}" \
        --upgrade pip setuptools wheel
}

install_torch() {
    echo "--- Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION} ---"
    "${PYTHON_VERSION}" -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        --extra-index-url "${PIP_INDEX_URL}" \
        --trusted-host "${PIP_TRUSTED_HOST}" \
        --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
        --retries "${PIP_RETRIES}" \
        "torch==${TORCH_VERSION}"
}

install_tensorrt_llm() {
    echo "--- Installing TensorRT-LLM ${TENSORRT_LLM_VERSION} ---"
    # tensorrt_llm ships pre-built wheels; only Linux x86_64 is officially supported.
    "${PYTHON_VERSION}" -m pip install \
        "${PIP_COMMON_ARGS[@]}" \
        "tensorrt_llm==${TENSORRT_LLM_VERSION}"
}

install_huggingface_tools() {
    echo "--- Installing Hugging Face tooling ---"
    # huggingface_hub provides the huggingface-cli download command used by download_model.sh.
    # Gemma 4 is a gated model on Hugging Face — you must supply HF_TOKEN at runtime.
    "${PYTHON_VERSION}" -m pip install \
        "${PIP_COMMON_ARGS[@]}" \
        huggingface_hub \
        transformers \
        accelerate \
        sentencepiece \
        protobuf
}

print_summary() {
    echo ""
    echo "========================================"
    echo " Environment setup complete"
    echo "========================================"
    echo ""
    echo "Installed:"
    "${PYTHON_VERSION}" -m pip show torch tensorrt_llm huggingface_hub 2>/dev/null | \
        grep -E '^(Name|Version):' | paste - - | awk '{print "  " $0}'
    echo ""
    echo "Next steps:"
    echo "  1. Accept the Gemma 4 licence on Hugging Face:"
    echo "     https://huggingface.co/google/gemma-4-26b-it"
    echo "  2. Set your HF token:  export HF_TOKEN=hf_xxxx"
    echo "  3. Download the model: bash scripts/gemma4-26b/download_model.sh"
    echo "  4. Convert checkpoint: bash scripts/gemma4-26b/convert_checkpoint.sh"
    echo "  5. Build TRT engine:   bash scripts/gemma4-26b/build_engine.sh"
    echo "  6. Start server:       bash scripts/gemma4-26b/serve.sh"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
detect_os
install_system_packages
check_cuda
setup_venv
upgrade_pip
install_torch
install_tensorrt_llm
install_huggingface_tools
print_summary
