#!/usr/bin/env bash
# build_engine.sh
# Builds the TensorRT engine from the converted Gemma 4 26B (MoE) checkpoint.
# Optimised for 4x NVIDIA RTX 4090 (Ada Lovelace, 24 GB VRAM each = 96 GB total).
#
# Gemma 4 26B A4B (MoE) memory budget on 4x RTX 4090:
#   - Static weights (bf16):  ~48 GB  →  ~12 GB per GPU after TP=4 sharding
#   - Remaining per GPU:      ~12 GB  →  available for KV cache + activations
#   - MAX_INPUT_LEN=8192 is a practical default; reduce if you hit OOM.
#
# Environment variables:
#   CKPT_DIR           converted checkpoint directory       (default: ./checkpoints/gemma-4-26b-tp4)
#   ENGINE_DIR         output directory for TRT engines     (default: ./engines/gemma-4-26b-tp4)
#   TRTLLM_REPO        path to a cloned TensorRT-LLM repo   (default: ./TensorRT-LLM)
#   TP_SIZE            tensor parallel degree               (default: 4)
#   MAX_BATCH_SIZE     maximum concurrent sequences         (default: 4)
#   MAX_INPUT_LEN      maximum input token length           (default: 8192)
#   MAX_OUTPUT_LEN     maximum output token length          (default: 2048)
#   MAX_NUM_TOKENS     max total tokens in flight           (default: 16384)
#   DTYPE              plugin dtype: float16 | bfloat16     (default: bfloat16)
#
# Usage:
#   bash scripts/gemma4-26b/build_engine.sh

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoints/gemma-4-26b-tp4}"
ENGINE_DIR="${ENGINE_DIR:-./engines/gemma-4-26b-tp4}"
TRTLLM_REPO="${TRTLLM_REPO:-./TensorRT-LLM}"
TP_SIZE="${TP_SIZE:-4}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-8192}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-2048}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-16384}"
DTYPE="${DTYPE:-bfloat16}"

echo "=== Building TensorRT-LLM engine for Gemma 4 26B A4B (TP=${TP_SIZE}) ==="
echo "Checkpoint dir  : ${CKPT_DIR}"
echo "Engine dir      : ${ENGINE_DIR}"
echo "Plugin dtype    : ${DTYPE}"
echo "Max batch size  : ${MAX_BATCH_SIZE}"
echo "Max input len   : ${MAX_INPUT_LEN}"
echo "Max output len  : ${MAX_OUTPUT_LEN}"
echo "Max num tokens  : ${MAX_NUM_TOKENS}"
echo ""

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_DIR}" >&2
    echo "  Run convert_checkpoint.sh first." >&2
    exit 1
fi

mkdir -p "${ENGINE_DIR}"

# ---------------------------------------------------------------------------
# Locate trtllm-build binary
# trtllm-build is installed by the tensorrt_llm Python package.
# Fall back to the upstream script when the command is not on PATH.
# ---------------------------------------------------------------------------
if command -v trtllm-build &>/dev/null; then
    BUILD_CMD="trtllm-build"
else
    BUILD_SCRIPT="${TRTLLM_REPO}/examples/trtllm_build.py"
    if [[ ! -f "${BUILD_SCRIPT}" ]]; then
        echo "ERROR: trtllm-build not found and fallback script missing at ${BUILD_SCRIPT}" >&2
        echo "  Ensure tensorrt_llm is installed: pip install tensorrt_llm" >&2
        exit 1
    fi
    BUILD_CMD="python3 ${BUILD_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# Build the engine
#
# Plugin notes for Gemma 4 26B A4B (MoE):
#   --gemm_plugin          : accelerates dense linear projections (attention, MLP)
#   --gpt_attention_plugin : uses optimised fused MHA / GQA kernels for Gemma attention
#   --moe_plugin           : REQUIRED for Mixture-of-Experts layers; uses fused expert
#                            GEMM + routing kernel for efficient sparse execution
#   --use_paged_context_fmha enable
#                          : paged KV-cache for longer contexts and higher throughput
#
# RTX 4090 supports FP8 compute (Ada Lovelace).  BF16 is used here for accuracy.
# To unlock extra speed at a slight accuracy cost, add:
#   --use_fp8_context_fmha enable
# ---------------------------------------------------------------------------
echo "--- Running trtllm-build ---"

${BUILD_CMD} \
    --checkpoint_dir "${CKPT_DIR}" \
    --output_dir     "${ENGINE_DIR}" \
    --gemm_plugin           "${DTYPE}" \
    --gpt_attention_plugin  "${DTYPE}" \
    --moe_plugin            "${DTYPE}" \
    --workers        "${TP_SIZE}" \
    --tp_size        "${TP_SIZE}" \
    --pp_size        1 \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len  "${MAX_INPUT_LEN}" \
    --max_output_len "${MAX_OUTPUT_LEN}" \
    --max_num_tokens "${MAX_NUM_TOKENS}" \
    --use_paged_context_fmha enable

echo ""
echo "=== Engine build complete ==="
echo "TensorRT engines saved to: ${ENGINE_DIR}"
echo ""
echo "Tips:"
echo "  - Engine files are GPU-specific. Rebuild if you change hardware."
echo "  - To increase context length, raise MAX_INPUT_LEN and MAX_NUM_TOKENS"
echo "    (monitor VRAM usage — KV cache grows linearly with sequence length)."
echo "  - For higher throughput, try MAX_BATCH_SIZE=8 and MAX_NUM_TOKENS=32768"
echo "    after verifying memory headroom with nvidia-smi."
echo ""
echo "Next step: bash scripts/gemma4-26b/serve.sh"
