#!/usr/bin/env bash
# download_model.sh
# Downloads the Qwen/Qwen3.5-27B model weights from Hugging Face Hub.
#
# Environment variables:
#   HF_TOKEN        (optional) Hugging Face access token for private/gated models
#   MODEL_DIR       local directory to download the model into  (default: ./models/Qwen3.5-27B)
#   HF_MODEL_ID     Hugging Face model repository ID            (default: Qwen/Qwen3.5-27B)
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/download_model.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models/Qwen3.5-27B}"
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.5-27B}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== Downloading ${HF_MODEL_ID} ==="
echo "Destination : ${MODEL_DIR}"
echo ""

mkdir -p "${MODEL_DIR}"

# Prefer huggingface-cli when available (uses local cache by default)
if command -v huggingface-cli &>/dev/null; then
    echo "Using huggingface-cli …"
    HF_CLI_ARGS=(
        download
        "${HF_MODEL_ID}"
        --local-dir "${MODEL_DIR}"
        --local-dir-use-symlinks False
    )
    if [[ -n "${HF_TOKEN}" ]]; then
        HF_CLI_ARGS+=(--token "${HF_TOKEN}")
    fi
    huggingface-cli "${HF_CLI_ARGS[@]}"
else
    # Fallback: git-lfs clone
    echo "huggingface-cli not found; falling back to git clone …"
    if [[ -n "${HF_TOKEN}" ]]; then
        CLONE_URL="https://user:${HF_TOKEN}@huggingface.co/${HF_MODEL_ID}"
    else
        CLONE_URL="https://huggingface.co/${HF_MODEL_ID}"
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone "${CLONE_URL}" "${MODEL_DIR}"
    cd "${MODEL_DIR}"
    git lfs pull
    cd -
fi

echo ""
echo "=== Download complete ==="
echo "Model saved to: ${MODEL_DIR}"
