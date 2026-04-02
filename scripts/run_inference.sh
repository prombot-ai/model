#!/usr/bin/env bash
# run_inference.sh
# Sends a test prompt to the running Qwen3.5-27B TensorRT-LLM server
# and prints the response. The server must already be running via serve.sh.
#
# Environment variables:
#   HOST    server host  (default: 127.0.0.1)
#   PORT    server port  (default: 8000)
#   PROMPT  test prompt  (default: "Hello, what can you do?")
#
# Usage:
#   bash scripts/run_inference.sh
#   PROMPT="Write a haiku about GPUs." bash scripts/run_inference.sh

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-Hello, what can you do?}"

BASE_URL="http://${HOST}:${PORT}/v1"

echo "=== Qwen3.5-27B Inference Test ==="
echo "Server : ${BASE_URL}"
echo "Prompt : ${PROMPT}"
echo ""

# ---------------------------------------------------------------------------
# Check server availability
# ---------------------------------------------------------------------------
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
    echo "ERROR: Cannot reach server at ${BASE_URL}."
    echo "  Start the server first: bash scripts/serve.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Non-streaming chat completion via OpenAI-compatible API
# ---------------------------------------------------------------------------
RESPONSE=$(curl -sf "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"Qwen3.5-27B\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"${PROMPT}\"}
        ],
        \"max_tokens\": 512,
        \"temperature\": 0.7
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
