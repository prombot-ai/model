#!/usr/bin/env bash
# convert_checkpoint.sh
# Converts the Hugging Face Qwen3.5-27B checkpoint into TensorRT-LLM format
# with Tensor Parallelism = 4 (one shard per RTX 4090).
#
# Environment variables:
#   MODEL_DIR          path to the downloaded HF model      (default: ./models/Qwen3.5-27B)
#   CKPT_DIR           output directory for converted ckpt  (default: ./checkpoints/qwen3.5-27b-tp4)
#   TRTLLM_REPO        path to a cloned TensorRT-LLM repo   (default: ./TensorRT-LLM)
#   TP_SIZE            tensor parallel degree               (default: 4)
#   DTYPE              compute dtype: float16 | bfloat16    (default: bfloat16)
#
# Usage:
#   bash scripts/convert_checkpoint.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/Qwen3.5-27B}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/qwen3.5-27b-tp4}"
TRTLLM_REPO="${TRTLLM_REPO:-./TensorRT-LLM}"
TP_SIZE="${TP_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
TRTLLM_ZIP_URL="${TRTLLM_ZIP_URL:-https://github.com/NVIDIA/TensorRT-LLM/archive/refs/heads/main.zip}"

CONVERT_SCRIPT="${TRTLLM_REPO}/examples/models/core/qwen/convert_checkpoint.py"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

download_trtllm_zip() {
    local zip_file extract_dir extracted_root

    zip_file="$(mktemp --suffix=.zip)"
    extract_dir="$(mktemp -d)"

    cleanup_download_artifacts() {
        rm -f "${zip_file}"
        rm -rf "${extract_dir}"
    }
    trap cleanup_download_artifacts RETURN

    echo "--- Downloading TensorRT-LLM source ZIP ---"
    if command_exists curl; then
        curl -L --fail --retry 3 --connect-timeout 20 -o "${zip_file}" "${TRTLLM_ZIP_URL}"
    elif command_exists wget; then
        wget -O "${zip_file}" "${TRTLLM_ZIP_URL}"
    else
        echo "ERROR: Neither curl nor wget is available to download TensorRT-LLM." >&2
        exit 1
    fi

    if ! command_exists unzip; then
        echo "ERROR: unzip is required to extract TensorRT-LLM source ZIP." >&2
        exit 1
    fi

    unzip -q "${zip_file}" -d "${extract_dir}"
    extracted_root="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"

    if [[ -z "${extracted_root}" ]]; then
        echo "ERROR: Failed to locate extracted TensorRT-LLM source directory." >&2
        exit 1
    fi

    rm -rf "${TRTLLM_REPO}"
    mv "${extracted_root}" "${TRTLLM_REPO}"
}

echo "=== Converting Qwen3.5-27B checkpoint for TensorRT-LLM ==="
echo "Source HF model  : ${MODEL_DIR}"
echo "Output ckpt dir  : ${CKPT_DIR}"
echo "Tensor parallelism: ${TP_SIZE}"
echo "Dtype            : ${DTYPE}"
echo ""

# ---------------------------------------------------------------------------
# Clone TensorRT-LLM if not already present
# ---------------------------------------------------------------------------
if [[ ! -d "${TRTLLM_REPO}" ]]; then
    echo "--- Cloning TensorRT-LLM ---"
    git config --global http.version HTTP/1.1
    git config --global http.postBuffer 524288000
    if ! git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "${TRTLLM_REPO}"; then
        echo "git clone failed or timed out; falling back to GitHub ZIP download."
        rm -rf "${TRTLLM_REPO}"
        download_trtllm_zip
    fi
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: Conversion script not found at ${CONVERT_SCRIPT}"
    echo "  Please update TRTLLM_REPO to point to a valid TensorRT-LLM checkout."
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# ---------------------------------------------------------------------------
# Run the conversion
# ---------------------------------------------------------------------------
echo "--- Running checkpoint conversion ---"
python3 "${CONVERT_SCRIPT}" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${CKPT_DIR}" \
    --dtype "${DTYPE}" \
    --tp_size "${TP_SIZE}"

echo ""
echo "=== Checkpoint conversion complete ==="
echo "Converted checkpoint saved to: ${CKPT_DIR}"
