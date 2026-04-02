#!/usr/bin/env bash
# build_engine.sh
# Builds the TensorRT engine from the converted Qwen3.5-27B checkpoint.
# Optimised for 4x NVIDIA RTX 4090 (Ada Lovelace, 24 GB VRAM each).
#
# Environment variables:
#   CKPT_DIR           converted checkpoint directory      (default: ./checkpoints/qwen3.5-27b-tp4)
#   ENGINE_DIR         output directory for TRT engines    (default: ./engines/qwen3.5-27b-tp4)
#   TRTLLM_REPO        path to a cloned TensorRT-LLM repo  (default: ./TensorRT-LLM)
#   TP_SIZE            tensor parallel degree              (default: 4)
#   MAX_BATCH_SIZE     maximum concurrent sequences        (default: 8)
#   MAX_INPUT_LEN      maximum input token length          (default: 4096)
#   MAX_OUTPUT_LEN     maximum output token length         (default: 2048)
#   MAX_NUM_TOKENS     max total tokens in flight          (default: 8192)
#
# Usage:
#   bash scripts/build_engine.sh

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoints/qwen3.5-27b-tp4}"
ENGINE_DIR="${ENGINE_DIR:-./engines/qwen3.5-27b-tp4}"
TRTLLM_REPO="${TRTLLM_REPO:-./TensorRT-LLM}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-4096}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-2048}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"

echo "=== Building TensorRT-LLM engine for Qwen3.5-27B (TP=${TP_SIZE}) ==="
echo "Checkpoint dir  : ${CKPT_DIR}"
echo "Engine dir      : ${ENGINE_DIR}"
echo "Max batch size  : ${MAX_BATCH_SIZE}"
echo "Max input len   : ${MAX_INPUT_LEN}"
echo "Max output len  : ${MAX_OUTPUT_LEN}"
echo ""

mkdir -p "${ENGINE_DIR}"

# trtllm-build is installed by the tensorrt_llm Python package.
# Confirm it is on PATH; fall back to the example script if needed.
if command -v trtllm-build &>/dev/null; then
    BUILD_CMD="trtllm-build"
else
    BUILD_SCRIPT="${TRTLLM_REPO}/examples/trtllm_build.py"
    if [[ ! -f "${BUILD_SCRIPT}" ]]; then
        echo "ERROR: trtllm-build not found and fallback script missing at ${BUILD_SCRIPT}"
        echo "  Make sure tensorrt_llm is installed: pip install tensorrt_llm"
        exit 1
    fi
    BUILD_CMD="python3 ${BUILD_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# Build the engine
# RTX 4090 supports FP8 compute but stores weights in BF16; we use BF16 here
# for maximum accuracy. Use --use_fp8_context_fmha for extra speed if desired.
# ---------------------------------------------------------------------------
${BUILD_CMD} \
    --checkpoint_dir "${CKPT_DIR}" \
    --output_dir "${ENGINE_DIR}" \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --workers "${TP_SIZE}" \
    --tp_size "${TP_SIZE}" \
    --pp_size 1 \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len "${MAX_INPUT_LEN}" \
    --max_output_len "${MAX_OUTPUT_LEN}" \
    --max_num_tokens "${MAX_NUM_TOKENS}" \
    --use_paged_context_fmha enable

echo ""
echo "=== Engine build complete ==="
echo "TensorRT engines saved to: ${ENGINE_DIR}"
echo ""
echo "Tip: engine files are GPU-specific. If you change hardware, rebuild the engine."
