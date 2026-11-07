# qwen-engine: Bare-Metal CUDA & Python Inference Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-purple.svg)](https://en.cppreference.com/w/cpp/20)
[![Wheels CI](https://github.com/Areen-09/qwen-engine/actions/workflows/wheels.yml/badge.svg)](https://github.com/Areen-09/qwen-engine/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance C++20 and custom CUDA inference engine and Python package built from scratch for `Qwen2.5-1.5B-Instruct`. Designed to run at maximum memory bandwidth with zero external math runtime libraries—**no cuBLAS, cuDNN, CUTLASS, or PyTorch** required at runtime.

The compiled Python extension statically bundles the CUDA runtime (`CUDA::cudart_static`), allowing end users to run high-throughput LLM inference on any system with NVIDIA drivers (`>= 450.80.02`), without requiring the CUDA Toolkit to be installed.

---

## Key Highlights

- **Zero Math Runtime Dependencies:** Handcrafted CUDA kernels for RMSNorm, RoPE, Grouped-Query Attention (GQA), W4A16 GEMV, SwiGLU, and fused device sampling.
- **Ultra-Low Memory Footprint (<1 GB VRAM):**
  - **W4A16 (Group 128):** ~943 MB total discrete VRAM for weights, activations, and KV cache. Runs comfortably on 4 GB laptop GPUs (e.g. RTX 3050).
  - **FP16 Baseline:** ~3,234 MB total discrete VRAM.
- **Zero-Update Device-Driven CUDA Graphs:** Forward pass and sampling reduction are captured into a single CUDA graph. Sequence positions and token IDs are updated in device global memory via a `DeviceControlBlock`, avoiding host graph updates.
- **Tied Embedding Quantization (120 MB vs 467 MB):** The tied `151936 x 1536` vocabulary table is symmetrically quantized to W4A16. Row dequantization during input lookup takes `<2 µs`, cutting `lm_head` memory bandwidth by 74%.
- **GQA 6:1 Split-K FlashDecoding:** 2D grid `(NUM_KV_HEADS=2, FLASH_DECODE_SPLITS=16)` saturates all SMs, caching Query heads in shared memory once and streaming KV cache tokens with 6x reduced bandwidth.
- **Pythonic High-Level API:** Full Python package (`qwen-engine`) featuring synchronous/asynchronous token streaming, transactional session rollback, HuggingFace auto-downloading, and native exception handling.

---

## Python Package Quickstart

### Installation

From repository source:
```bash
pip install .
```

Or for development (editable mode):
```bash
pip install -e . --no-build-isolation
```

*Requirements: NVIDIA GPU (Compute Capability 7.5 to 9.0), NVIDIA driver, and Python 3.9+.*

---

### Basic Generation

```python
from qwen_engine import Engine, SamplingParams

# Load local weights or download from HuggingFace cache
engine = Engine.from_pretrained(
    model_path="models/qwen25_1.5b_w4a16.qwenbin",
    tokenizer_path="models/tokenizer.bin",
    device_id=0,
)

# Generate completion
response = engine.generate(
    prompt="Explain quantum entanglement in two simple sentences.",
    sampling_params=SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128),
)
print(response)
```

---

### Real-Time Synchronous Streaming

```python
from qwen_engine import Engine, SamplingParams

engine = Engine.from_pretrained("models/qwen25_1.5b_w4a16.qwenbin", "models/tokenizer.bin")

params = SamplingParams(temperature=0.8, top_k=50, max_tokens=256)
for token_text in engine.stream("Write a short haiku about GPUs:", sampling_params=params):
    print(token_text, end="", flush=True)
print()
```

---

### Real-Time Asynchronous Streaming (FastAPI / Web Serving)

The native engine releases the Python GIL during decode iterations, allowing concurrent execution:

```python
import asyncio
from qwen_engine import Engine, SamplingParams

async def stream_chat():
    engine = Engine.from_pretrained("models/qwen25_1.5b_w4a16.qwenbin", "models/tokenizer.bin")
    
    async for token in engine.stream_async("Tell me three fun facts about space:"):
        print(token, end="", flush=True)
    print()

asyncio.run(stream_chat())
```

---

### Multi-Turn Chat & Transactional Rollback

`Engine` retains KV cache state across calls. If generation is interrupted or rejected, you can roll back to any sequence checkpoint without recomputing prefill:

```python
from qwen_engine import Engine

engine = Engine.from_pretrained("models/qwen25_1.5b_w4a16.qwenbin", "models/tokenizer.bin")

messages = [
    {"role": "system", "content": "You are a concise programming assistant."},
    {"role": "user", "content": "What is RAII in C++?"}
]

# 1. Format ChatML prompt
prompt = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# 2. Checkpoint session position before generation
checkpoint = engine.current_seq_pos

try:
    for chunk in engine.stream(prompt):
        print(chunk, end="", flush=True)
except KeyboardInterrupt:
    # 3. Rollback KV cache to checkpoint cleanly
    engine.rollback_to(checkpoint)
    print("\n[Generation cancelled; KV cache rolled back]")
```

---

## C++ Engine & Standalone Tools

For maximum throughput and standalone deployment without Python, you can build the native C++ binaries directly:

### 1. Build via CMake

```powershell
# Configure release build with Ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build binaries
cmake --build build --config Release --parallel
```

Binaries produced:
- `build/cli_bench.exe`: Microbenchmark harness for prefill and decode throughput.
- `build/qwen_server.exe`: Standalone lightweight HTTP/SSE chat server.
- `build/test_kernels.exe`: Kernel correctness verification suite.

### 2. Export Weights from HuggingFace

```powershell
# Export FP16 model and tokenizer
python tools/export_qwen25.py --mode fp16 --out models/qwen25_1.5b_fp16.qwenbin --tokenizer-out models/tokenizer.bin

# Export W4A16 quantized model
python tools/export_qwen25.py --mode w4a16 --out models/qwen25_1.5b_w4a16.qwenbin --tokenizer-out models/tokenizer.bin
```

### 3. Run Deterministic CLI Benchmark

```powershell
./build/cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 10 --runs 50
```

### 4. Run the Zero-dGPU-VRAM Web UI

```powershell
./build/qwen_server.exe --model models/qwen25_1.5b_w4a16.qwenbin --port 8080
```
Open `http://127.0.0.1:8080` in your browser. The single-page HTML frontend streams responses via Server-Sent Events (SSE).

---

## Testing & Verification

Run the Python binding test suite (covers tokenization, ChatML, multi-byte UTF-8 streaming, greedy generation, rollback, session leases, and exceptions):

```bash
pytest tests/test_python_bindings.py -v
```

Run native CUDA kernel parity tests against PyTorch golden references:

```powershell
./build/test_kernels.exe --golden models/golden_ref.bin --w4a16 models/qwen25_1.5b_w4a16.qwenbin
```

---

## Architecture Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **`NUM_LAYERS`** | `28` | Transformer decoder layers |
| **`HIDDEN_SIZE`** | `1536` | Residual hidden dimension ($d$) |
| **`INTERMEDIATE_SIZE`** | `8960` | SwiGLU MLP intermediate dimension |
| **`NUM_Q_HEADS`** | `12` | Query attention heads |
| **`NUM_KV_HEADS`** | `2` | Key/Value attention heads (6:1 GQA) |
| **`HEAD_DIM`** | `128` | Dimension per attention head |
| **`VOCAB_SIZE`** | `151936` | Padded vocabulary size |
| **`MAX_SEQ_LEN`** | `4096` | Maximum KV cache context length |
| **`QUANT_GROUP_SIZE`** | `128` | Symmetric scale grouping for W4A16 |
| **`PREFILL_MAX_CHUNK`** | `256` | Outer prefill chunk size (WDDM TDR safe) |

---

## Documentation

For an in-depth guide on advanced Python streaming, exception handling, custom tokenization, and architecture deep dives, see:
📖 **[Walkthrough Guide: Using qwen-engine](docs/walkthrough_qwen_engine.md)**

---

## License

This project is licensed under the [MIT License](LICENSE).
