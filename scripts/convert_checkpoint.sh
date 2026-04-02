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

CONVERT_SCRIPT="${TRTLLM_REPO}/examples/models/core/qwen/convert_checkpoint.py"

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
    git clone https://github.com/NVIDIA/TensorRT-LLM.git "${TRTLLM_REPO}"
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
