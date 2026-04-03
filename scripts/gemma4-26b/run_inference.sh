#!/usr/bin/env bash
# run_inference.sh
# Sends a test prompt to the running Gemma 4 26B server and prints the response.
# The server must already be running via serve.sh.
#
# Environment variables:
#   HOST    server host   (default: 127.0.0.1)
#   PORT    server port   (default: 8000)
#   PROMPT  test prompt   (default: "Explain the Mixture-of-Experts architecture in simple terms.")
#
# Usage:
#   bash scripts/gemma4-26b/run_inference.sh
#   PROMPT="Write a Python function to sort a list." bash scripts/gemma4-26b/run_inference.sh

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Explain the Mixture-of-Experts architecture in simple terms.}"

BASE_URL="http://${HOST}:${PORT}/v1"

echo "=== Gemma 4 26B Inference Test ==="
echo "Server : ${BASE_URL}"
echo "Prompt : ${PROMPT}"
echo ""

# ---------------------------------------------------------------------------
# Check server availability
# ---------------------------------------------------------------------------
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
    echo "ERROR: Cannot reach server at ${BASE_URL}." >&2
    echo "  Start the server first: bash scripts/gemma4-26b/serve.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Non-streaming chat completion via OpenAI-compatible API
#
# Gemma 4 instruction-tuned models support system prompts natively.
# Adjust temperature, max_tokens, and top_p as needed.
# ---------------------------------------------------------------------------
RESPONSE=$(curl -sf "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"gemma-4-26b-it\",
        \"messages\": [
            {\"role\": \"system\", \"content\": \"You are a helpful AI assistant.\"},
            {\"role\": \"user\",   \"content\": \"${PROMPT}\"}
        ],
        \"max_tokens\": 512,
        \"temperature\": 0.7,
        \"top_p\": 0.9
    }")

echo "--- Response ---"
echo "${RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
choice = data['choices'][0]
print(choice['message']['content'])
print()
print('Finish reason :', choice['finish_reason'])
usage = data.get('usage', {})
print('Tokens used   :', usage)
"
