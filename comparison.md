# Benchmark Comparison: `qwen-engine` vs. `llama.cpp`

This report documents the head-to-head benchmark comparison between **`qwen-engine` (W4A16 Group-128)** and **`llama.cpp` (`Q4_K_M`, build `9a1d03c7c (12175)`)** for **Qwen 2.5-1.5B-Instruct** on a Windows laptop **NVIDIA GeForce RTX 3050 Laptop GPU**.

---

## 1. Test Environment & Commands

- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,095 MiB VRAM, Compute Capability 8.6 / Ampere, 192 GB/s theoretical peak bandwidth)
- **OS:** Windows (x64)

### Commands Executed
```powershell
# qwen-engine (W4A16 Group-128)
.\build\cli_bench.exe --model models\qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 10 --runs 50 --profile-stages
.\build\cli_bench.exe --model models\qwen25_1.5b_w4a16.qwenbin --prompt-len 512 --gen-len 128 --warmup 5 --runs 20
.\build\cli_bench.exe --model models\qwen25_1.5b_w4a16.qwenbin --prompt-len 1024 --gen-len 128 --warmup 5 --runs 10

# llama.cpp (Q4_K_M with FlashAttention enabled, all layers offloaded to GPU)
.\tools\llama_cpp\llama-bench.exe -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -fa 1 -p 128,512,1024 -n 128,256
```

---

## 2. Head-to-Head Results

| Benchmark | `qwen-engine` (W4A16 G128) | `llama.cpp` (`Q4_K_M`) | Speedup (`qwen-engine` / `llama.cpp`) |
| :--- | ---: | ---: | ---: |
| **Prefill `pp128`** | **3,247.60 tok/s** (`39.41 ms` TTFT) | 2,136.33 ± 36.99 tok/s | **1.52x faster** |
| **Prefill `pp512`** | **2,869.85 tok/s** (`178.41 ms` TTFT) | 2,587.90 ± 5.62 tok/s | **1.11x faster** |
| **Prefill `pp1024`** | 2,356.60 tok/s (`434.52 ms` TTFT) | **2,561.69 ± 26.13 tok/s** | 0.92x |
| **Decode `tg128` (`p=512`)** | **177.24 tok/s** (`p50: 5.61 ms`) | 61.19 ± 0.13 tok/s | **2.90x faster** |
| **Decode `tg256` (`p=128`)** | **178.17 tok/s** (`p50: 5.52 ms`, `144.86 GB/s`) | 60.98 ± 0.09 tok/s | **2.92x faster** |
| **Model / Discrete VRAM** | **879.35 MB** (`759.31 MB` weights + `112.00 MB` KV + `8.04 MB` workspace) | 1.04 GiB model size (+ KV/runtime) | — |

---

## 3. Optimization Impact (Phase 1 Baseline vs. Phase 2 Final on `p=128, n=256`)

| Metric | Phase 1 Baseline | Phase 2 Final | Improvement |
| :--- | ---: | ---: | ---: |
| **TTFT (`p=128`)** | `68.26 ms` | **`39.41 ms`** | **1.73x faster** |
| **Prefill Throughput (`p=128`)** | `1,875.05 tok/s` | **`3,247.60 tok/s`** | **+73.2%** |
| **GQA Attention Prefill (28x)** | `33.294 ms` | **`4.008 ms`** | **8.31x faster** |
| **Decode Throughput (`n=256`)** | `122.01 tok/s` | **`178.17 tok/s`** | **+46.0%** |
| **Effective Memory Bandwidth** | `117.16 GB/s` (61.0% peak) | **`144.86 GB/s` (75.5% peak)** | **+23.6%** |

---

## 4. Why `qwen-engine` Leads on Decode & Short/Medium Prefill

1. **8-Lane Sub-Warp `uint4` (`__ldg`) W4A8 `__dp4a` Decode GEMV:** Each 8-lane sub-warp computes 1 output row using coalesced 128-bit weight loads (`uint4`) and 3 warp-shuffle reductions, saturating `75.5%` of the RTX 3050's 192 GB/s memory bandwidth during autoregressive decode.
2. **Zero-Update CUDA Graph Capture:** All 28 layers, RMSNorms, RoPE, Split-K FlashDecoding, LM Head, and token sampling run inside a single `cudaGraphExec_t`, eliminating host kernel-launch overhead between tokens.
3. **Shared-Memory K/V-Tiled FlashAttention (`Br=16, Bc=32`):** Eliminated register spilling and redundant global KV cache reads during prefill, cutting 28-layer prefill attention latency by **8.3x**.
4. **Chunked Prefill Note (`pp1024`):** `PREFILL_MAX_CHUNK = 512` executes a 1024-token prompt as two 512-token chunks to keep workspace VRAM at `8.04 MB`, which accounts for the slight dip from `2,869.85 tok/s` (`p=512`) to `2,356.60 tok/s` (`p=1024`).
