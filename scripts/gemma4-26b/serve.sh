#!/usr/bin/env bash
# serve.sh
# Launches an OpenAI-compatible HTTP server for Gemma 4 26B (MoE) using TensorRT-LLM.
# Runs across 4x RTX 4090 GPUs with tensor parallelism = 4.
#
# Environment variables:
#   MODEL_DIR          path to the HF model (used as tokenizer source and model ID)
#                      (default: ./models/gemma-4-26b-it)
#   ENGINE_DIR         TensorRT engine directory     (default: ./engines/gemma-4-26b-tp4)
#   HOST               bind address                 (default: 0.0.0.0)
#   PORT               HTTP port                    (default: 8000)
#   TP_SIZE            tensor parallel degree        (default: 4)
#   MAX_BATCH_SIZE     max concurrent requests       (default: 4)
#   MAX_NUM_TOKENS     max total tokens in flight    (default: 16384)
#   CUDA_VISIBLE_DEVICES  GPUs to use               (default: 0,1,2,3)
#
# Usage:
#   bash scripts/gemma4-26b/serve.sh
#   # then query the API at http://localhost:8000/v1/chat/completions

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/gemma-4-26b-it}"
ENGINE_DIR="${ENGINE_DIR:-./engines/gemma-4-26b-tp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-16384}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== Starting Gemma 4 26B A4B (MoE) server ==="
echo "Engine dir  : ${ENGINE_DIR}"
echo "Tokenizer   : ${MODEL_DIR}"
echo "Listen      : ${HOST}:${PORT}"
echo "TP size     : ${TP_SIZE}"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES}"
echo ""

# ---------------------------------------------------------------------------
# Validate that either the engine or the HF model directory exists
# ---------------------------------------------------------------------------
if [[ ! -d "${ENGINE_DIR}" ]] || [[ -z "$(ls -A "${ENGINE_DIR}" 2>/dev/null)" ]]; then
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "ERROR: Neither engine directory '${ENGINE_DIR}' nor model directory '${MODEL_DIR}' found." >&2
        echo "  To use engine mode: run build_engine.sh first." >&2
        echo "  To use HF model mode: run download_model.sh first." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# trtllm-serve supports two operational modes:
#   1. Engine mode  – pass a pre-built engine directory (fast startup, recommended)
#   2. HF model mode – pass a Hugging Face model path  (JIT compilation, slower)
#
# Engine mode is preferred. HF model mode is provided as a fallback.
# ---------------------------------------------------------------------------
if [[ -d "${ENGINE_DIR}" ]] && [[ -n "$(ls -A "${ENGINE_DIR}" 2>/dev/null)" ]]; then
    echo "Using pre-built TensorRT engine from ${ENGINE_DIR}"
    trtllm-serve "${ENGINE_DIR}" \
        --host           "${HOST}" \
        --port           "${PORT}" \
        --tp_size        "${TP_SIZE}" \
        --pp_size        1 \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_num_tokens "${MAX_NUM_TOKENS}" \
        --tokenizer      "${MODEL_DIR}"
else
    echo "Engine directory empty or missing; falling back to HF model mode."
    echo "NOTE: First-run JIT compilation for a 26B MoE model may take 30+ minutes."
    echo "      Build the engine with build_engine.sh for production use."
    trtllm-serve "${MODEL_DIR}" \
        --host           "${HOST}" \
        --port           "${PORT}" \
        --tp_size        "${TP_SIZE}" \
        --pp_size        1 \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_num_tokens "${MAX_NUM_TOKENS}"
fi
