#!/usr/bin/env bash
# download_model.sh
# Downloads the Google Gemma 4 26B IT model weights from Hugging Face Hub.
#
# Gemma 4 26B is a GATED model — you must:
#   1. Accept the licence at https://huggingface.co/google/gemma-4-26b-it
#   2. Generate a Hugging Face access token at https://huggingface.co/settings/tokens
#   3. Set the token via:  export HF_TOKEN=hf_xxxx
#
# Environment variables:
#   HF_TOKEN        Hugging Face access token (REQUIRED for Gemma 4)
#   MODEL_DIR       local directory to download the model into   (default: ./models/gemma-4-26b-it)
#   HF_MODEL_ID     Hugging Face model repository ID             (default: google/gemma-4-26b-it)
#   HF_ENDPOINT     Hugging Face endpoint or mirror              (default: https://hf-mirror.com)
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/gemma4-26b/download_model.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/gemma-4-26b-it}"
HF_MODEL_ID="${HF_MODEL_ID:-google/gemma-4-26b-it}"
HF_TOKEN="${HF_TOKEN:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_ENDPOINT="${HF_ENDPOINT%/}"

export HF_ENDPOINT
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"

# ---------------------------------------------------------------------------
# Validate token — Gemma 4 is a gated model
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "  Gemma 4 26B is a gated model. Please:" >&2
    echo "    1. Accept the licence at https://huggingface.co/google/gemma-4-26b-it" >&2
    echo "    2. Create a token at https://huggingface.co/settings/tokens" >&2
    echo "    3. Re-run with:  HF_TOKEN=hf_xxxx bash scripts/gemma4-26b/download_model.sh" >&2
    exit 1
fi

echo "=== Downloading ${HF_MODEL_ID} ==="
echo "Destination : ${MODEL_DIR}"
echo "HF endpoint : ${HF_ENDPOINT}"
echo ""
echo "NOTE: Gemma 4 26B A4B (MoE) model files are ~48 GB (fp16)."
echo "      Ensure sufficient disk space before proceeding."
echo ""

mkdir -p "${MODEL_DIR}"

# Prefer huggingface-cli when available (supports resume and local cache)
if command -v huggingface-cli &>/dev/null; then
    echo "Using huggingface-cli …"
    huggingface-cli download \
        "${HF_MODEL_ID}" \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False \
        --token "${HF_TOKEN}"
else
    # Fallback: git-lfs clone
    echo "huggingface-cli not found; falling back to git clone …"
    if [[ -n "${HF_TOKEN}" ]]; then
        BARE_HOST="${HF_ENDPOINT#https://}"
        CLONE_URL="https://user:${HF_TOKEN}@${BARE_HOST}/${HF_MODEL_ID}"
    else
        CLONE_URL="${HF_ENDPOINT}/${HF_MODEL_ID}"
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone "${CLONE_URL}" "${MODEL_DIR}"
    cd "${MODEL_DIR}"
    git lfs pull
    cd -
fi

echo ""
echo "=== Download complete ==="
echo "Model saved to: ${MODEL_DIR}"
echo ""
echo "Next step: bash scripts/gemma4-26b/convert_checkpoint.sh"
