# Custom CUDA Inference Engine for Qwen2.5-1.5B (RTX 3050 Laptop / sm_86)

A high-performance C++20 and custom CUDA inference engine built from scratch for `Qwen2.5-1.5B-Instruct` on NVIDIA RTX 3050 Laptop GPU (Ampere `sm_86`, 4 GB VRAM) under Windows WDDM.

The engine operates with **zero external math runtime libraries** (`cuBLAS`, `cuDNN`, `CUTLASS`, or `libtorch`). Every operation—RMSNorm, GEMV, SwiGLU, RoPE, Grouped-Query Attention (GQA), INT4 dequantization, and sampling—is implemented in custom CUDA kernels.

---

## Key Features

1. **Strict 4 GB VRAM Budget via Static Arena:**
   - Single `cudaMalloc` allocation at startup with startup headroom validation.
   - **Phase 1 (FP16 Baseline):** ~3,234 MB total discrete VRAM (fits comfortably within the ~3,600 MB usable WDDM budget).
   - **Phase 2 (W4A16 Group-128):** ~943 MB total discrete VRAM (leaves >2.6 GB free headroom).
2. **Zero-Update Device-Driven CUDA Graph:**
   - Captures the entire 28-layer transformer forward pass and sampling reduction into a single device-driven `cudaGraphExec_t`.
   - Sequence position and token IDs are updated directly in GPU global memory (`DeviceControlBlock`), eliminating host `cudaGraphExecUpdate` calls.
3. **Quantized Tied Embeddings (`120.3 MB` vs `466.8 MB`):**
   - The tied `151936 x 1536` token embedding table is symmetrically quantized to W4A16 (Group 128).
   - On-the-fly row dequantization during input embedding lookup takes `< 2 microseconds`, cutting per-token memory traffic on `lm_head` by 74%.
4. **GQA 6:1 Split-K FlashDecoding:**
   - Launches a 2D grid `(NUM_KV_HEADS=2, FLASH_DECODE_SPLITS=16)` (32 thread blocks) to saturate all 16 SMs on the RTX 3050.
   - Caches the 6 Query heads in shared memory once, streaming KV cache tokens with 6x reduced bandwidth.
5. **Stateful KV Prefix Caching & Chunked Prefill:**
   - Detects the longest common prefix across multi-turn chat sessions so only new prompt tokens undergo prefill.
   - Processes long prompts in outer host chunks (`PREFILL_MAX_CHUNK = 256`) to stay within WDDM TDR driver timeout limits.
6. **Zero-dGPU-VRAM Web UI:**
   - Single-file vanilla HTML/CSS/JS chat frontend rendered on the Intel integrated GPU (`0 MB` RTX 3050 VRAM consumed).
   - Real-time token streaming over Server-Sent Events (SSE).

---

## Prerequisites

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (or any Ampere `sm_86` GPU).
- **OS:** Windows 10/11 64-bit.
- **CUDA Toolkit:** CUDA 12.0+ or 13.0+ (`nvcc` on PATH).
- **C++ Compiler:** Microsoft Visual Studio (MSVC with C++20 support).
- **Build System:** CMake 3.26+ and Ninja or MSBuild.
- **Python:** Python 3.10+ with `uv` or `pip` (`torch`, `transformers`, `safetensors`).

---

## Quick Start

### 1. Offline Model Export & Golden Reference Dump

Using `uv` (recommended):
```powershell
# Sync Python dependencies
uv sync

# Export FP16 model and binary tokenizer
.venv\Scripts\python.exe tools/export_qwen25.py --mode fp16 --out models/qwen25_1.5b_fp16.qwenbin --tokenizer-out models/tokenizer.bin

# Export W4A16 model
.venv\Scripts\python.exe tools/export_qwen25.py --mode w4a16 --out models/qwen25_1.5b_w4a16.qwenbin --tokenizer-out models/tokenizer.bin

# Generate PyTorch golden reference dump for kernel testing
.venv\Scripts\python.exe tools/verify_reference.py --out models/golden_ref.bin
```

### 2. Build with CMake

```powershell
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets in parallel
cmake --build build --config Release --parallel
```

This generates three release binaries:
- `build/Release/test_kernels.exe`
- `build/Release/cli_bench.exe`
- `build/Release/qwen_server.exe`

### 3. Run Kernel Unit & Parity Verification Tests

```powershell
./build/Release/test_kernels.exe --golden models/golden_ref.bin --fp16 models/qwen25_1.5b_fp16.qwenbin --w4a16 models/qwen25_1.5b_w4a16.qwenbin
```

All 8 verification suites test:
1. `embed_lookup_fp16` (exact match) and `embed_lookup_w4a16` (cosine sim >= 0.998)
2. `rmsnorm_fp16` (max abs err <= 1e-3)
3. `qkv_bias_rope_kv_append` (max abs err <= 1e-3)
4. `gemv_fp16` & `gemv_dual_gate_up_swiglu_fp16` (max abs err <= 5e-3)
5. `gemv_w4a16` & `gemv_dual_gate_up_swiglu_w4a16` (cosine sim >= 0.995)
6. `gemm_prefill_fp16` & `gemm_prefill_w4a16` (max abs err <= 5e-3, cosine sim >= 0.995)
7. `attention_gqa_flash_decode` vs prefill attention (max abs err <= 2e-3)
8. Device argmax / Gumbel-Max sampling & 28-layer greedy token parity against PyTorch

### 4. Deterministic CLI Benchmarking

Benchmark W4A16 mode:
```powershell
./build/Release/cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 10 --runs 50
```

Benchmark FP16 mode:
```powershell
./build/Release/cli_bench.exe --model models/qwen25_1.5b_fp16.qwenbin --prompt-len 128 --gen-len 256 --warmup 10 --runs 50
```

### 5. Launch Local HTTP/SSE Chat Server

```powershell
./build/Release/qwen_server.exe --model models/qwen25_1.5b_w4a16.qwenbin --port 8080
```
Then open `http://127.0.0.1:8080` in your web browser. Ensure your browser runs on the Intel integrated GPU in Windows Graphics Settings to ensure 0 MB discrete RTX 3050 VRAM is used for UI rendering.

---

## Architecture Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `NUM_LAYERS` | `28` | Number of transformer decoder layers |
| `HIDDEN_SIZE` | `1536` | Residual stream dimension |
| `INTERMEDIATE_SIZE` | `8960` | SwiGLU MLP dimension |
| `NUM_Q_HEADS` | `12` | Query attention heads |
| `NUM_KV_HEADS` | `2` | Key/Value attention heads (6:1 GQA) |
| `HEAD_DIM` | `128` | Dimension per head |
| `VOCAB_SIZE` | `151936` | Padded vocabulary size |
| `MAX_SEQ_LEN` | `4096` | Fixed KV Cache token capacity |
| `QUANT_GROUP_SIZE` | `128` | Group size for symmetric W4A16 |
| `PREFILL_MAX_CHUNK`| `256` | Maximum outer prefill chunk |

