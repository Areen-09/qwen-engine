# qwen-engine: Bare-Metal CUDA & Python Inference Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-purple.svg)](https://en.cppreference.com/w/cpp/20)
[![Wheels CI](https://github.com/Areen-09/qwen-engine/actions/workflows/wheels.yml/badge.svg)](https://github.com/Areen-09/qwen-engine/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance C++20 and custom CUDA inference engine and Python package built from scratch for the **Qwen** model family, including:
- **Qwen 2.5-1.5B:** Dense Grouped-Query Attention (GQA) transformer.
- **Qwen 3.5-4B:** Hybrid linear attention model interleaving Gated DeltaNet and GQA in a 3:1 ratio (75% KV cache reduction).
- **Qwen 3.8-27B:** Dual-GPU hybrid model distributing Gated DeltaNet and GQA across 2 GPUs using 2-way Tensor Parallelism ($TP=2$) over PCIe.

Designed to run at maximum memory bandwidth with zero external math runtime libraries—**no cuBLAS, cuDNN, CUTLASS, or PyTorch** required at runtime. The compiled Python extension statically bundles the CUDA runtime (`CUDA::cudart_static`), allowing end users to run high-throughput LLM inference on any system with NVIDIA drivers (`>= 450.80.02`), without requiring the CUDA Toolkit to be installed.

---

## Supported Models Matrix

| Model | Parameters | Hidden Size ($H$) | Intermediate Size ($I$) | Layer Topology | Quantization | Disk Footprint | Peak VRAM | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5-1.5B** | 1.54B | 1536 | 8960 | 28 GQA Layers | W4A16 (G128) | **796 MB** | **~943 MB** | Single GPU (`cuda:0`) / RTX 3050 (4 GB) / Tesla T4 |
| **Qwen 3.5-4B** | ~3.8B | 2560 | 10240 | 32 Layers (24 DeltaNet : 8 GQA) | W4A16 (G128) | **~2.4 GB** | **~2.8 GB** | Single GPU (`cuda:0`) / Tesla T4 (16 GB) |
| **Qwen 3.8-27B** | ~27B | 5120 | 27648 | 64 Layers (48 DeltaNet : 16 GQA) | W4A16 Dual-Rank ($TP=2$) | **~7.1 GB / rank** | **~7.8 GB / GPU** | Dual GPU (`cuda:0` & `cuda:1`) / 2x Tesla T4 (32 GB) |

For comprehensive architecture details, selective KV cache savings formulas, and memory layout proofs, see **[`docs/MODELS.md`](docs/MODELS.md)**. For the exact binary container format, see **[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md)**.

---

## Key Highlights

- **Zero Math Runtime Dependencies:** Handcrafted CUDA kernels for RMSNorm, RoPE, Grouped-Query Attention (GQA), Gated DeltaNet linear attention, W4A16 GEMV, SwiGLU, and fused device sampling.
- **Unified Hybrid Architecture (DeltaNet + GQA):** Native support for hybrid models alternating GQA attention and fused Gated DeltaNet sm_75 linear recurrence. Selective KV cache allocation cuts VRAM requirements by up to 75% on recurrent layers.
- **Cross-Platform Tensor Parallelism:** High-performance 2-way TP engine with zero external communication dependencies. Uses direct BAR1 P2P with CUDA event sync on Windows (with double-buffered staging fallback under WDDM) and NCCL on Linux.
- **Universal GGUF & HuggingFace Importer:** Direct ingestion from standard GGUF v2/v3 or PyTorch safetensors with on-the-fly warp-interleaved W4A16 quantization and dynamic `.qwenbin` v2 serialization.
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

### Basic Generation (Single GPU)

```python
from qwen_engine import Engine, SamplingParams

# Load local weights or download from HuggingFace Hub cache
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

### Multi-GPU Dual-Rank Inference (Kaggle 2x Tesla T4, TP=2)

```python
from qwen_engine import Engine, SamplingParams

# Seamlessly distribute Qwen 3.8-27B across GPU 0 and GPU 1
engine_27b = Engine.from_pretrained(
    "areen09/qwen-engine-models-qwenbin",
    filename="qwen3.8-27b-w4a16",
    device_ids=[0, 1],
)

prompt = "<|im_start|>user\nWrite a short poem about CUDA Tensor Cores.<|im_end|>\n<|im_start|>assistant\n"
for token in engine_27b.stream(prompt, sampling_params=SamplingParams(temperature=0.7, max_tokens=128)):
    print(token, end="", flush=True)
print()
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

### 1. Standalone Synthetic CUDA Kernel Suites (Zero File Dependencies)
Run all 5 mathematical kernel verification suites (RMSNorm, RoPE, dynamic GEMV/SwiGLU sweeps across $\{1.5B, 4B, 27B\}$, Gated DeltaNet recurrence, and depthwise Conv1d) in under 3 seconds without downloading any weight files:

```powershell
./build/test_kernels.exe
```

### 2. Full Golden Reference Parity Suite (Opt-In)
Verify end-to-end activation and token parity against PyTorch golden references:

```powershell
./build/test_kernels.exe --golden models/golden_ref.bin --w4a16 models/qwen25_1.5b_w4a16.qwenbin
```

### 3. Dynamic Hybrid Engine & Rollback Verification (Python Pytest)
Run dynamic topology parsing, selective KV cache allocation invariants, and deterministic prefix replay tests:

```bash
pytest tests/test_dynamic_engine.py tests/test_gguf_converter.py -v
```

---

## Documentation

- 📖 **[Model Architecture & Deployment Guide (`docs/MODELS.md`)](docs/MODELS.md):** Architectural topologies, memory consumption formulas, selective KV cache savings (75%), and multi-GPU topologies ($TP=2$).
- 📐 **[Technical Specification & Container Layout (`docs/SPECIFICATION.md`)](docs/SPECIFICATION.md):** `.qwenbin` v2 container specification, 216-byte `LayerDescriptor` binary format (`"<2I26Q"`), and dynamic memory arena offsets.

---

## License

This project is licensed under the [MIT License](LICENSE).
