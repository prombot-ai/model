# model

Deployment scripts for running large language models in production.

---

## Qwen3.5-27B on 4× RTX 4090 with TensorRT-LLM (Ubuntu or CentOS 9)

End-to-end guide for deploying [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) using
[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on an Ubuntu 22.04 or CentOS Stream 9
host with four RTX 4090 GPUs.

### Hardware requirements

| Component | Minimum |
|-----------|---------|
| GPUs | 4× NVIDIA RTX 4090 (24 GB VRAM each, 96 GB total) |
| System RAM | 64 GB |
| Disk | 120 GB free (model weights ≈ 55 GB in BF16) |
| OS | Ubuntu 22.04 LTS or CentOS Stream 9 |
| NVIDIA driver | 550 or later |
| CUDA | 12.8 |

### Software requirements

- Python 3.10+
- PyTorch 2.7.0 (CUDA 12.8 build)
- TensorRT-LLM 0.20.0
- `git`, `git-lfs`, OpenMPI development headers

---

### Quick-start

All scripts live in `scripts/qwen3.5-27b/`. Run them in the order shown below.

#### 1 — Set up the environment

```bash
bash scripts/qwen3.5-27b/setup_environment.sh
source venv/bin/activate
```

The script auto-detects Ubuntu/Debian vs CentOS/RHEL-style distributions, installs the
matching system packages with `apt-get` or `dnf`, creates a Python virtual environment,
and installs PyTorch and TensorRT-LLM.

By default, Python packages are installed from the Tsinghua PyPI mirror to avoid timeouts
from mainland China. You can override the mirror if needed:

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
PIP_TRUSTED_HOST=mirrors.aliyun.com \
bash scripts/qwen3.5-27b/setup_environment.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PIP_INDEX_URL` | `https://pypi.tuna.tsinghua.edu.cn/simple` | Mirror used for PyPI packages such as `wheel` and `tensorrt_llm` |
| `PIP_TRUSTED_HOST` | `pypi.tuna.tsinghua.edu.cn` | Host passed to pip for the mirror |
| `PIP_DEFAULT_TIMEOUT` | `60` | pip network timeout in seconds |
| `PIP_RETRIES` | `10` | pip retry count |
| `PYTORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu128` | Wheel index for PyTorch CUDA builds |

#### 2 — Download the model

```bash
# Optional: set HF_TOKEN if the model requires authentication
export HF_TOKEN=hf_your_token_here

HF_ENDPOINT=https://hf-mirror.com bash scripts/qwen3.5-27b/download_model.sh
```

The downloader defaults to the China mirror `https://hf-mirror.com` to avoid direct
connectivity issues to `huggingface.co`. To use the official endpoint instead:

```bash
HF_ENDPOINT=https://huggingface.co bash scripts/qwen3.5-27b/download_model.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models/Qwen3.5-27B` | Local path for downloaded weights |
| `HF_MODEL_ID` | `Qwen/Qwen3.5-27B` | Hugging Face model repository |
| `HF_TOKEN` | *(empty)* | HF access token for gated models |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face API/download endpoint or mirror |

#### 3 — Convert the checkpoint

Converts the Hugging Face weights into TensorRT-LLM format with **tensor parallelism = 4**
(one shard per GPU).

```bash
bash scripts/qwen3.5-27b/convert_checkpoint.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models/Qwen3.5-27B` | Source HF model directory |
| `CKPT_DIR` | `./checkpoints/qwen3.5-27b-tp4` | Output directory |
| `TRTLLM_REPO` | `./TensorRT-LLM` | TensorRT-LLM repository (auto-cloned if absent) |
| `TP_SIZE` | `4` | Tensor parallel degree |
| `DTYPE` | `bfloat16` | Compute dtype (`float16` or `bfloat16`) |

#### 4 — Build the TensorRT engine

Compiles the converted checkpoint into an optimised engine for the RTX 4090 (Ada Lovelace).

```bash
bash scripts/qwen3.5-27b/build_engine.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CKPT_DIR` | `./checkpoints/qwen3.5-27b-tp4` | Converted checkpoint |
| `ENGINE_DIR` | `./engines/qwen3.5-27b-tp4` | Output engine directory |
| `TP_SIZE` | `4` | Tensor parallel degree |
| `MAX_BATCH_SIZE` | `8` | Maximum concurrent sequences |
| `MAX_INPUT_LEN` | `4096` | Maximum input tokens |
| `MAX_OUTPUT_LEN` | `2048` | Maximum generated tokens |
| `MAX_NUM_TOKENS` | `8192` | Maximum tokens in flight |

> **Note:** TensorRT engines are GPU-specific. Rebuild if you change hardware.

#### 5 — Start the server

Launches an OpenAI-compatible HTTP API on port 8000.

```bash
bash scripts/qwen3.5-27b/serve.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_DIR` | `./engines/qwen3.5-27b-tp4` | Pre-built engine (used if present) |
| `MODEL_DIR` | `./models/Qwen3.5-27B` | HF model path (tokenizer + fallback) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | HTTP port |
| `TP_SIZE` | `4` | Tensor parallel degree |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPUs to use |

#### 6 — Test inference

```bash
bash scripts/qwen3.5-27b/run_inference.sh

# Custom prompt
PROMPT="Explain tensor parallelism in one paragraph." bash scripts/qwen3.5-27b/run_inference.sh
```

The script calls the `/v1/chat/completions` endpoint and prints the model response.

---

### OpenAI-compatible API

Once the server is running, you can query it with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="na")

response = client.chat.completions.create(
    model="Qwen3.5-27B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| OOM during engine build | `MAX_NUM_TOKENS` too high | Reduce `MAX_NUM_TOKENS` or `MAX_BATCH_SIZE` |
| `trtllm-build` not found | Package not installed | Run `pip install tensorrt_llm` in the venv |
| GPU count warning | Fewer than 4 GPUs visible | Check `CUDA_VISIBLE_DEVICES`; confirm `nvidia-smi` shows all GPUs |
| Slow inter-GPU bandwidth | RTX 4090s on PCIe without NVLink bridge | Performance is normal; NVLink bridges improve bandwidth for 2-GPU pairs |
| HF download fails | Gated model or blocked official endpoint | Provide `HF_TOKEN`, or keep the default `HF_ENDPOINT=https://hf-mirror.com` |

---

### Directory layout

```
scripts/
└── qwen3.5-27b/
  ├── setup_environment.sh    # Install system & Python dependencies
  ├── download_model.sh       # Download Qwen/Qwen3.5-27B from Hugging Face
  ├── convert_checkpoint.sh   # Convert HF checkpoint → TRT-LLM format (TP=4)
  ├── build_engine.sh         # Compile optimised TensorRT engine
  ├── serve.sh                # Start OpenAI-compatible server
  └── run_inference.sh        # Send a test prompt and print the response

models/                     # Downloaded HF weights (created by download_model.sh)
checkpoints/                # TRT-LLM converted checkpoints
engines/                    # Compiled TensorRT engines
TensorRT-LLM/               # Cloned TRT-LLM repo (created by convert_checkpoint.sh)
venv/                       # Python virtual environment
```
