#!/usr/bin/env bash
# serve.sh
# Launches an OpenAI-compatible HTTP server for Qwen3.5-27B using TensorRT-LLM.
# Runs across 4x RTX 4090 GPUs with tensor parallelism = 4.
#
# Environment variables:
#   MODEL_DIR          path to the HF model (used as the model ID in API responses)
#                      (default: ./models/Qwen3.5-27B)
#   ENGINE_DIR         TensorRT engine directory    (default: ./engines/qwen3.5-27b-tp4)
#   HOST               bind address                (default: 0.0.0.0)
#   PORT               HTTP port                   (default: 8000)
#   TP_SIZE            tensor parallel degree       (default: 4)
#   MAX_BATCH_SIZE     max concurrent requests      (default: 8)
#   MAX_NUM_TOKENS     max total tokens in flight   (default: 8192)
#   CUDA_VISIBLE_DEVICES  GPUs to use              (default: 0,1,2,3)
#
# Usage:
#   bash scripts/serve.sh
#   # then query the API at http://localhost:8000/v1/chat/completions

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/Qwen3.5-27B}"
ENGINE_DIR="${ENGINE_DIR:-./engines/qwen3.5-27b-tp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== Starting Qwen3.5-27B server ==="
echo "Engine dir  : ${ENGINE_DIR}"
echo "Listen      : ${HOST}:${PORT}"
echo "TP size     : ${TP_SIZE}"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES}"
echo ""

# trtllm-serve supports two operational modes:
#   1. Engine mode  – pass a pre-built engine directory
#   2. HF model mode – pass a Hugging Face model ID / local path (slower startup)
#
# We prefer the faster engine mode when the engine directory exists.
if [[ -d "${ENGINE_DIR}" ]] && [[ -n "$(ls -A "${ENGINE_DIR}" 2>/dev/null)" ]]; then
    echo "Using pre-built TensorRT engine from ${ENGINE_DIR}"
    trtllm-serve "${ENGINE_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --tp_size "${TP_SIZE}" \
        --pp_size 1 \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_num_tokens "${MAX_NUM_TOKENS}" \
        --tokenizer "${MODEL_DIR}"
else
    echo "Engine directory empty or missing; falling back to HF model mode."
    echo "NOTE: This performs JIT compilation on first run and is much slower."
    trtllm-serve "${MODEL_DIR}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --tp_size "${TP_SIZE}" \
        --pp_size 1 \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_num_tokens "${MAX_NUM_TOKENS}"
fi
