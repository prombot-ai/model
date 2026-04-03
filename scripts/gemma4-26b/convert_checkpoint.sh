#!/usr/bin/env bash
# convert_checkpoint.sh
# Converts the Hugging Face Gemma 4 26B (MoE) checkpoint into TensorRT-LLM format.
#
# Gemma 4 26B A4B architecture notes:
#   - Mixture-of-Experts: 26B total parameters, 4B active per token
#   - All 26B parameters must reside in VRAM (routing happens at inference time)
#   - Tensor Parallelism (TP=4) shards the model across 4x RTX 4090 GPUs
#
# Environment variables:
#   MODEL_DIR          path to the downloaded HF model       (default: ./models/gemma-4-26b-it)
#   CKPT_DIR           output directory for converted ckpt   (default: ./checkpoints/gemma-4-26b-tp4)
#   TRTLLM_REPO        path to a cloned TensorRT-LLM repo    (default: ./TensorRT-LLM)
#   TP_SIZE            tensor parallel degree                (default: 4)
#   EP_SIZE            expert parallel degree for MoE layers (default: 1)
#                      Use EP_SIZE=4 to shard experts across GPUs instead of replicating.
#   DTYPE              compute dtype: float16 | bfloat16     (default: bfloat16)
#
# Usage:
#   bash scripts/gemma4-26b/convert_checkpoint.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/gemma-4-26b-it}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/gemma-4-26b-tp4}"
TRTLLM_REPO="${TRTLLM_REPO:-./TensorRT-LLM}"
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
TRTLLM_ZIP_URL="${TRTLLM_ZIP_URL:-https://github.com/NVIDIA/TensorRT-LLM/archive/refs/heads/main.zip}"

CONVERT_SCRIPT="${TRTLLM_REPO}/examples/models/core/gemma/convert_checkpoint.py"

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

echo "=== Converting Gemma 4 26B (MoE) checkpoint for TensorRT-LLM ==="
echo "Source HF model   : ${MODEL_DIR}"
echo "Output ckpt dir   : ${CKPT_DIR}"
echo "Tensor parallelism: ${TP_SIZE}"
echo "Expert parallelism: ${EP_SIZE}"
echo "Dtype             : ${DTYPE}"
echo ""

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}" >&2
    echo "  Run download_model.sh first." >&2
    exit 1
fi

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
    echo "ERROR: Conversion script not found at ${CONVERT_SCRIPT}" >&2
    echo ""
    echo "  Possible reasons:" >&2
    echo "    - Your TensorRT-LLM clone is outdated (TRT-LLM 0.20.0+ required for Gemma 4)" >&2
    echo "    - The path changed in a newer release; check:" >&2
    echo "      ${TRTLLM_REPO}/examples/models/core/gemma/" >&2
    echo ""
    echo "  To update the repo:" >&2
    echo "    cd ${TRTLLM_REPO} && git pull" >&2
    exit 1
fi

mkdir -p "${CKPT_DIR}"

# ---------------------------------------------------------------------------
# Run the conversion
#
# Key flags:
#   --tp_size  : shards model weights across TP_SIZE GPUs (tensor parallelism)
#   --ep_size  : distributes MoE experts across EP_SIZE GPUs (expert parallelism)
#                TP and EP must satisfy: TP_SIZE * EP_SIZE <= num_gpus
#                With 4 GPUs and defaults (TP=4, EP=1), all GPUs share tensor-parallel shards.
# ---------------------------------------------------------------------------
echo "--- Running checkpoint conversion ---"

CONVERT_ARGS=(
    --model_dir  "${MODEL_DIR}"
    --output_dir "${CKPT_DIR}"
    --dtype      "${DTYPE}"
    --tp_size    "${TP_SIZE}"
)

# Only pass ep_size when explicitly using expert parallelism
if [[ "${EP_SIZE}" -gt 1 ]]; then
    CONVERT_ARGS+=(--ep_size "${EP_SIZE}")
fi

python3 "${CONVERT_SCRIPT}" "${CONVERT_ARGS[@]}"

echo ""
echo "=== Checkpoint conversion complete ==="
echo "Converted checkpoint saved to: ${CKPT_DIR}"
echo ""
echo "Next step: bash scripts/gemma4-26b/build_engine.sh"
