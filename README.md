# qwen-engine: Bare-Metal CUDA Inference for Qwen 2.5-1.5B

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-purple.svg)](https://en.cppreference.com/w/cpp/20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A dedicated, zero-dependency C++20 and custom CUDA inference engine for **Qwen 2.5-1.5B-Instruct** in **W4A16 (Group-128)** precision, tuned for NVIDIA Ampere GPUs (**RTX 3050 Laptop GPU, 4 GB VRAM, SM 8.6**).

Zero external math runtime libraries—**no cuBLAS, cuDNN, CUTLASS, or PyTorch** required at runtime.

---

## Performance vs. `llama.cpp` (`Q4_K_M`, RTX 3050 Laptop GPU)

Measured on NVIDIA GeForce RTX 3050 Laptop GPU (4095 MiB VRAM, 192 GB/s peak bandwidth, Windows):

| Metric | `qwen-engine` (W4A16 G128) | `llama.cpp` (`Q4_K_M`, `-ngl 99 -fa 1`) | Speedup |
| :--- | ---: | ---: | ---: |
| **Prefill `p=128`** | **3,247.60 tok/s** (`39.41 ms` TTFT) | 2,136.33 tok/s | **1.52x** |
| **Prefill `p=512`** | **2,869.85 tok/s** (`178.41 ms` TTFT) | 2,587.90 tok/s | **1.11x** |
| **Prefill `p=1024`** | 2,356.60 tok/s (`434.52 ms` TTFT) | 2,561.69 tok/s | 0.92x |
| **Decode `tg128` (`p=512`)** | **177.24 tok/s** | 61.19 tok/s | **2.90x** |
| **Decode `tg256` (`p=128`)** | **178.17 tok/s** (`144.86 GB/s`, 75.5% peak) | 60.98 tok/s | **2.92x** |
| **Discrete VRAM Used** | **879.35 MB** (759.31 MB weights + 112.00 MB KV + 8.04 MB workspace) | ~1.04 GiB+ | — |

See **[`comparison.md`](comparison.md)** for full benchmark methodology, stage-by-stage profiling, and Phase 1 vs. Phase 2 comparisons, **[`docs/MODELS.md`](docs/MODELS.md)** for memory formulas, and **[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md)** for the `.qwenbin` binary container spec.

---

## Key Architecture Highlights

- **8-Lane Sub-Warp `uint4` W4A8 (`__dp4a`) Decode GEMV:** Coalesced 128-bit weight loads (`__ldg`) paired with on-the-fly Q8_1 quantized activations and warp-shuffle reductions.
- **Shared-Memory K/V-Tiled FlashAttention Prefill (`Br=16, Bc=32`):** Causal online softmax tiling in shared memory with zero register spilling, specialized for `HEAD_DIM=128` and `GQA_GROUP_SIZE=6`.
- **6:1 GQA Split-K FlashDecoding:** 2D grid `(NUM_KV_HEADS=2, FLASH_DECODE_SPLITS=16)` with single-barrier parallel cross-warp merge and fused Q8_1 epilogue for O-projection.
- **Zero-Update Device-Driven CUDA Graphs:** Autoregressive decode forward pass + sampling captured into a static `cudaGraphExec_t` driven by `DeviceControlBlock`.
- **Single-Arena VRAM Management:** `879.35 MB` total discrete VRAM (weights, `max_seq_len=4096` FP16 KV cache, and `8.04 MB` workspace).

---

## Quickstart

### 1. Interactive Notebook

Open [`qwen25_1.5b.ipynb`](qwen25_1.5b.ipynb) for a ready-to-run walkthrough of streaming chat, memory inspection, and throughput benchmarking.

### 2. Python API

```python
from qwen_engine import Engine

engine = Engine("models/qwen25_1.5b_w4a16.qwenbin", "models/tokenizer.bin", device_id=0)

for chunk in engine.stream(
    [{"role": "user", "content": "Explain CUDA Graphs in two sentences."}],
    max_tokens=128,
    temperature=0.0,
):
    print(chunk, end="", flush=True)
```

---

## Building & Testing (Windows)

```powershell
# Configure & compile Release binaries (SM 8.6 by default)
.\build.bat
.\build.bat compile

# Run CUDA kernel unit tests & golden reference parity suite
.\build\test_kernels.exe --golden

# Run CLI benchmark with per-stage CUDA event profiling
.\build\cli_bench.exe --model models\qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --profile-stages

# Run Python integration tests
pytest tests/
```
