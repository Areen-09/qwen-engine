# Custom CUDA Inference Engine for Qwen2.5-1.5B on RTX 3050 Laptop (4 GB VRAM)

*   **Author:** Areen-09
*   **Status:** Approved
*   **Date:** 2026-11-06

--------------------------------------------------------------------------------

## Overview

This document specifies the design of a custom C++20 and CUDA inference engine built from scratch for the `Qwen2.5-1.5B-Instruct` language model on an NVIDIA RTX 3050 Laptop GPU (Compute Capability `sm_86`, 16 Streaming Multiprocessors, 4 GB GDDR6 VRAM, 192 GB/s theoretical memory bandwidth) running under Windows WDDM with 16 GB system RAM and an Intel integrated GPU handling display output.

The engine operates with zero external runtime math or deep learning libraries (`cuBLAS`, `cuDNN`, `CUTLASS`, or PyTorch C++ runtime). Every operation—RMSNorm, matrix-vector and tiled prefill projections, QKV bias addition, NeoX split-half Rotary Positional Embeddings (RoPE), Grouped-Query Attention (GQA), SwiGLU activation, weight dequantization, and token sampling—is implemented in custom CUDA C++ kernels. The system progresses from a verified FP16 baseline (~3.24 GB total VRAM) captured in a single device-driven CUDA Graph (Phase 1) to a fused 4-bit weight-only quantized runtime (`W4A16`, group size 128, ~0.80 GB weight footprint) with quantized tied embeddings (Phase 2). It exposes both a deterministic CLI benchmark harness (`cli_bench`) to measure performance against `llama.cpp` and a local HTTP Server-Sent Events (SSE) chat server paired with a static HTML/JS interface running on the Intel integrated GPU.

## The Problem

Running a 1.5B parameter language model at maximum token generation speed on a 4 GB Windows laptop GPU faces four hardware and system bottlenecks that general-purpose frameworks do not solve together when building from first principles:

1.  **Strict 4 GB VRAM Ceiling under Windows WDDM:** Even when an Intel integrated GPU drives the laptop display, Windows WDDM reserves 300 to 450 MB of the 4,096 MB physical VRAM for OS page tables and CUDA context overhead, leaving 3,550 to 3,700 MB usable. In FP16, `Qwen2.5-1.5B-Instruct` requires 3,087.6 MB for weights alone (2,620.8 MB across 28 layers plus 466.75 MB for the tied `151936 x 1536` token embedding table). Dynamic allocations or duplicate `lm_head` allocations exceed available VRAM and force Windows WDDM to page memory over PCIe, reducing throughput by an order of magnitude.
2.  **Memory Bandwidth Saturation and the Vocabulary Bottleneck:** Single-stream autoregressive decoding (`batch_size = 1`) reads the entire weight matrix from global memory for every generated token, making throughput strictly limited by the RTX 3050 Laptop GPU's 192 GB/s memory bus (~160–170 GB/s sustained ceiling). Furthermore, because `Qwen2.5-1.5B` has a 151,936-token vocabulary, leaving the tied `embed_tokens`/`lm_head` matrix in FP16 (466.75 MB) even after quantizing the 28 transformer layers to INT4 (675.7 MB) causes the output projection alone to consume 41% of per-token memory traffic.
3.  **Windows WDDM Kernel Launch Latency:** An unfused 28-layer transformer forward pass executes ~200 separate kernel launches per token. On Windows WDDM, CPU-to-GPU submission costs 3 to 8 microseconds per launch, adding 1.0 to 1.6 milliseconds of dead time per token—up to 30% of the total token execution time in INT4 mode. Standard CUDA Graphs still incur CPU update overhead if host scalar arguments (`seq_pos`, `token_id`) are updated via `cudaGraphExecUpdate` on every step.
4.  **GQA Hardware Under-Utilization and UI VRAM Contention:** `Qwen2.5-1.5B` uses 12 query heads and only 2 key-value heads (6:1 GQA ratio); naive per-KV-head attention kernels use only 2 of the 16 Streaming Multiprocessors (12.5% occupancy). Meanwhile, opening a hardware-accelerated desktop chat UI on the discrete GPU steals 150 to 300 MB of VRAM, pushing Phase 1 FP16 execution over the 4 GB budget.

**Proposed Solution:** Hardcode `Qwen2.5-1.5B-Instruct` dimensions as compile-time constants (`constexpr`), allocate a single static VRAM arena at startup (sharing the tied embedding tensor and reserving a fixed 4,096-token KV cache of 114.69 MB / 112.0 MiB), and capture the entire 28-layer decode step into a zero-update `cudaGraphExec_t` driven by a GPU-resident Device Control Block (`d_token_id`, `d_seq_pos`). Implement FlashDecoding (Split-K across sequence length sharing 6 query heads per KV head), chunked multi-token prefill with stateful KV prefix caching, and Phase 2 W4A16 group-128 quantization across both the 28 layers and the tied `embed_tokens`/`lm_head` table, served via a CLI benchmark tool and a zero-dGPU-VRAM local HTTP/SSE static web chat UI.

## Goals

*   **Zero External Math Runtime Dependencies:** Implement 100% of runtime tensor operations in custom C++20/CUDA (`sm_86`) without linking `cuBLAS`, `cuDNN`, `CUTLASS`, or `libtorch`.
*   **Strict 4 GB VRAM Fit via Single-Allocation Arena:** Keep total discrete GPU VRAM usage at or below `3,300 MB` in Phase 1 (FP16) and at or below `1,050 MB` in Phase 2 (W4A16 INT4, including quantized tied `embed_tokens`/`lm_head` and a 4,096-token FP16 KV cache) via a single startup `cudaMalloc`.
*   **Mathematical Parity with PyTorch Reference:** Achieve greedy token-for-token output parity and layer-by-layer maximum absolute error within FP16 tolerance (`<= 5e-3`) against the official HuggingFace `Qwen/Qwen2.5-1.5B-Instruct` model in Phase 1, and retain coherent instruction-following with perplexity degradation `< 3%` in Phase 2 W4A16 mode.
*   **High Bus Utilization & Decode Speed vs. `llama.cpp`:** Achieve `>= 75%` of theoretical memory bandwidth (`>= 46 tokens/sec`) in Phase 1 FP16 decode and `>= 150 tokens/sec` in Phase 2 W4A16 decode at `batch_size = 1` (context length 128 to 1024), matching or exceeding `llama.cpp` (`llama-bench`) on the same RTX 3050 Laptop GPU.
*   **Zero-Update CUDA Graph Execution:** Execute each decode token step with a single `cudaGraphLaunch` call and zero host-side `cudaGraphExecUpdate` calls by reading and incrementing sequence position and token IDs inside GPU global memory.
*   **Interactive Multi-Turn Chat with Prefix Caching:** Provide a local HTTP/SSE server and a single-file vanilla HTML/CSS/JS chat frontend that runs on the Intel integrated GPU (`0 MB` discrete VRAM consumed by the UI) and reuses cached KV states across conversation turns.

## Non-Goals

*   **Multi-Model or Dynamic Architecture Support:** The engine will not parse arbitrary model architectures (e.g., Llama, Mistral, Gemma) or dynamic layer dimensions at runtime.
*   **Batched Multi-User Serving (`batch_size > 1` Decode):** Continuous batching, PagedAttention virtual memory paging, and concurrent multi-client decoding are out of scope; the engine targets single-user (`batch_size = 1`) local execution.
*   **Multi-GPU Execution or CPU Layer Offloading:** Tensor parallelism, pipeline parallelism, and hybrid CPU/GPU PCIe weight offloading are excluded.
*   **Training, Fine-Tuning, or LoRA Adapters:** Backpropagation, gradient computation, and runtime LoRA adapter merging are excluded.
*   **Heavyweight Desktop GUI Frameworks:** Electron, Qt, ImGui (DirectX/OpenGL), or React/WebGL frontends that allocate discrete GPU rendering contexts are excluded.

## Open Questions

*   *None.*

## Critical User Journeys (CUJs)

### CUJ 1: Offline Model Export & Kernel Verification

*   **Given**: A developer on Windows with Python 3, PyTorch, and the HuggingFace `Qwen/Qwen2.5-1.5B-Instruct` checkpoint in local cache.
*   **When**: The developer runs `python tools/export_qwen25.py --mode fp16 --out models/qwen25_1.5b_fp16.qwenbin`, `python tools/export_qwen25.py --mode w4a16 --out models/qwen25_1.5b_w4a16.qwenbin`, and `python tools/verify_reference.py --out models/golden_ref.bin`, followed by executing `./build/Release/test_kernels.exe` (which defaults to `--golden models/golden_ref.bin --fp16 models/qwen25_1.5b_fp16.qwenbin --w4a16 models/qwen25_1.5b_w4a16.qwenbin`).
*   **Then**: The exporter writes page-aligned `.qwenbin` weight files with warp-interleaved tensor layouts, `models/tokenizer.bin`, and a golden activation dump; `test_kernels.exe` verifies all custom CUDA kernels (embedding lookup, RMSNorm, QKV Bias + RoPE, GQA FlashDecoding & prefill attention, SwiGLU, FP16 GEMV, W4A16 GEMV, tiled prefill GEMM, and token sampling) plus 28-layer CUDA Graph replay against the golden reference and exits with code `0`.

### CUJ 2: Deterministic Head-to-Head CLI Benchmarking

*   **Given**: Compiled release binaries (`cli_bench.exe`) and exported `.qwenbin` model files on the RTX 3050 Laptop GPU with no competing discrete GPU workloads.
*   **When**: The user runs `cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 10 --runs 50`.
*   **Then**: The engine allocates a single static VRAM arena (`<= 1,050 MB` for W4A16 or `<= 3,300 MB` for FP16), runs warmup passes, executes 50 timed iterations via the captured CUDA Graph, and prints a structured table reporting Allocated VRAM (MB), Time-To-First-Token (TTFT ms), Decode Throughput (tokens/sec), Inter-Token Latency (`p50`, `p95`, `p99` ms), and Effective Memory Bandwidth (GB/s and percentage of the 192 GB/s hardware peak).

### CUJ 3: Multi-Turn Local Chat UI with Zero Discrete VRAM Overhead

*   **Given**: The user starts `qwen_server.exe --model models/qwen25_1.5b_w4a16.qwenbin --port 8080` and opens `http://127.0.0.1:8080` in a web browser running on the Intel integrated GPU.
*   **When**: The user submits a multi-turn conversation message in the browser UI.
*   **Then**: The browser streams generated tokens in real time over Server-Sent Events (SSE) with live tokens/sec and TTFT badges; the server matches the conversation history prefix in the static KV cache so only the newly appended user tokens undergo prefill; discrete GPU VRAM usage remains unchanged (`0 MB` added by the browser UI).

## Technical Plan (Architecture)

### 1. Hardcoded Model Constants & Static VRAM Arena Layout

All structural dimensions of `Qwen/Qwen2.5-1.5B-Instruct` are defined as compile-time `constexpr` constants in `./include/config.cuh` so that loop trip counts, warp reductions, and shared-memory allocations are resolved at compile time without runtime branching:

| Constant Name | Value | Architectural Role & Implementation Constraint |
| :--- | :--- | :--- |
| `NUM_LAYERS` | `28` | Number of identical transformer decoder layers |
| `HIDDEN_SIZE` | `1536` | Residual stream width (`12 * 128`, divisible by 256 for 2-group W4A16 warp loads) |
| `INTERMEDIATE_SIZE` | `8960` | SwiGLU MLP width (`70 * 128`, divisible by 256 for 2-group W4A16 warp loads) |
| `NUM_Q_HEADS` | `12` | Query attention heads (`12 * 128 = 1536`) |
| `NUM_KV_HEADS` | `2` | Key/Value attention heads (`2 * 128 = 256`; 6:1 GQA sharing ratio) |
| `HEAD_DIM` | `128` | Dimension per head (`32` warp lanes $\times$ `4` elements per lane) |
| `VOCAB_SIZE` | `151936` | Padded vocabulary size (`1187 * 128`, divisible by 128) |
| `RMS_NORM_EPS` | `1e-6f` | Variance floor in FP32 RMSNorm reduction |
| `ROPE_THETA` | `1000000.0f` | Base frequency for NeoX split-half rotary embeddings |
| `MAX_SEQ_LEN` | `4096` | Fixed capacity of the static KV cache |
| `QUANT_GROUP_SIZE` | `128` | Number of weights sharing one FP16 group scale (with fixed implicit zero-point 8) in Phase 2 |
| `PREFILL_MAX_CHUNK` | `256` | Maximum number of prompt tokens processed per outer prefill pass |
| `PREFILL_TILE_M` | `16` | Thread-block token row tile dimension during prefill |
| `PREFILL_TILE_K` | `128` | Thread-block reduction dimension tile during prefill (`4 KB` shared memory per `[16, 128]` tile) |

To prevent Windows WDDM from fragmenting VRAM or paging allocations to system RAM over PCIe, the engine performs **a single `cudaMalloc` call at startup** via a static `GpuArena` allocator (`./include/tensor.cuh`), partitioning the contiguous block into 256-byte-aligned regions:

```text
+-----------------------------------------------------------------------------------+
|                    SINGLE CONTIGUOUS GPU VRAM ARENA (cudaMalloc)                  |
+-----------------------------------+-----------------------+---------------+-------+
| 1. Model Weights Region           | 2. Static KV Cache    | 3. Workspace  | 4.DCB |
| - Phase 1 (FP16): 3,087.6 MB      | [2, 28, 2, 4096, 128] | Ping-pong act | 256 B |
|   * Tied embed/lm_head: 466.75 MB | FP16: 114.69 MB       | (chunk<=256), | GPU   |
|   * 28 Layers: 2,620.80 MB        | (112.0 MiB;           | QKV, MLP,     | state |
| - Phase 2 (W4A16): 796.0 MB       |  28 KB per token)     | Split-K accum,| struct|
|   * Tied embed/lm_head: 120.3 MB  |                       | Logits [151k] |       |
|   * 28 Layers: 675.7 MB           |                       | ~32.0 MB      |       |
+-----------------------------------+-----------------------+---------------+-------+
  Phase 1 Total: ~3,234 MB (Fits inside ~3,600 MB usable WDDM VRAM)
  Phase 2 Total: ~943 MB   (Leaves >2.6 GB free VRAM headroom)
```

### 2. End-to-End System Architecture & Data Flow

```text
  [HuggingFace Safetensors]
             |
             v  (Offline Python Export: tools/export_qwen25.py)
  [Page-Aligned .qwenbin + tokenizer.bin]  +  [Golden Reference: golden_ref.bin]
             |
             v  (Windows MapViewOfFile -> Single cudaMemcpy to GpuArena)
+-----------------------------------------------------------------------------------+
|                         C++20 / CUDA Engine (QwenEngine)                          |
|                                                                                   |
|  +---------------------------+      +------------------------------------------+  |
|  | Session KV Prefix Matcher | ---> | Prefill Path (Outer Chunk <= 256 tokens) |  |
|  | ChatML BPE Tokenizer;     |      | - Tiled GEMM (M_TILE=16, K_TILE=128)     |  |
|  | only prefills delta tokens|      | - Causal prefill GQA + KV cache populate |  |
|  +---------------------------+      +--------------------+---------------------+  |
|                                                          |                        |
|                                                          v                        |
|  +-----------------------------------------------------------------------------+  |
|  | Decode Path: Single Captured cudaGraphExec_t (Zero Host Updates per Token)  |  |
|  |                                                                             |  |
|  |  [DeviceControlBlock: d_token_id, d_seq_pos]                                |  |
|  |       |                                                                     |  |
|  |       v                                                                     |  |
|  |  (1) Embed Lookup (FP16 copy or on-the-fly 1-row W4A16 dequant)             |  |
|  |       |                                                                     |  |
|  |       v                                                                     |  |
|  |  +--- Layer l = 0 .. 27 -------------------------------------------------+  |  |
|  |  | (2) Fused RMSNorm (FP32 warp-shuffle sum-of-squares)                  |  |  |
|  |  | (3) QKV Projection (GEMV FP16 or Fused W4A16)                         |  |  |
|  |  | (4) Fused QKV Bias + NeoX Split-Half RoPE + KV Cache Write            |  |  |
|  |  | (5) GQA 6:1 Split-K FlashDecoding (2 KV heads x 16 sequence splits)   |  |  |
|  |  | (6) O-Proj + Residual Add (Fused inside output writeback)             |  |  |
|  |  | (7) Post-Attn RMSNorm                                                 |  |  |
|  |  | (8) Fused Dual Gate+Up Projection + Inline SwiGLU(gate, up)           |  |  |
|  |  | (9) Down-Proj + Residual Add                                          |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |       |                                                                     |  |
|  |       v                                                                     |  |
|  |  (10) Final RMSNorm -> Tied LM Head GEMV [151936, 1536]                     |  |
|  |       |                                                                     |  |
|  |       v                                                                     |  |
|  |  (11) Device Argmax / Gumbel-Max Sampling + Pinned Write + d_seq_pos++      |  |
|  +-----------------------------------------------------------------------------+  |
+-------------------+---------------------------------------------+-----------------+
                    |                                             |
                    v                                             v
   +---------------------------------+           +----------------------------------+
   | CLI Benchmark (cli_bench.exe)   |           | HTTP/SSE Server (qwen_server.exe)|
   | - Measures TTFT, ITL p50/p99,   |           | - Direct single-thread loop      |
   |   tok/s, and Effective GB/s %   |           | - Overlaps send(t-1) & Graph(t)  |
   | - Compares vs. llama-bench      |           | - Serves web/index.html on iGPU  |
   +---------------------------------+           +----------------------------------+
```

### 3. Custom Binary Weight Format (`.qwenbin`) & Warp-Interleaved W4A16 Layout

To avoid parsing JSON headers or uncoalesced HuggingFace tensor layouts at startup, `./tools/export_qwen25.py` serializes weights into a flat binary file (`.qwenbin`) alongside a binary BPE vocabulary/merges file (`tokenizer.bin`):
*   **Header (4,096 bytes, page-aligned):** Contains magic number `0x5157454E` (`"QWEN"`), format version `1`, quantization mode (`0 = FP16`, `1 = W4A16_SYM_G128`), architectural dimensions, and a deterministic byte-offset table for the tied embedding tensor and each of the 28 layer weight blocks. In both modes, `gate_proj` and `up_proj` rows are interleaved 1-to-1 (`[8960, 2, 1536]`) so a single fused kernel evaluates both projections simultaneously.
*   **Phase 1 Layout (FP16):** Row-major contiguous `half` arrays padded to 256-byte boundaries so every row starts on a 128-bit (`uint4`) cache-line boundary.
*   **Phase 2 Layout (W4A16 Group-128):**
    *   Every weight matrix of shape `[N, K]` is partitioned along the reduction dimension `K` into groups of `128` elements (`K / 128` groups per row; `12` groups for `K = 1536` and `70` groups for `K = 8960`).
    *   Each group is symmetrically quantized to 4-bit unsigned integers $q \in [0, 15]$ with a fixed implicit zero-point of $8$ and one FP16 scale $s = \max(|w|) / 7.0$, representing $w \approx (q - 8) \cdot s$.
    *   Each 128-element group is packed into `16` `uint32_t` words (`8` INT4 weights per `uint32_t` word). Across a 32-lane warp, lanes `0..15` load group $2g$ (`128` weights) and lanes `16..31` load group $2g + 1$ (`128` weights) in a single coalesced 128-byte warp transaction (`256` weights per warp iteration). Because both `HIDDEN_SIZE = 1536` (`6 * 256`) and `INTERMEDIATE_SIZE = 8960` (`35 * 256`) are exact multiples of `256`, every warp loop terminates with zero remainder and zero out-of-bounds tail reads.
    *   The tied `embed_tokens` / `lm_head` matrix (`[151936, 1536]`) is stored in this exact same W4A16 format (`116.7 MB` packed weights + `3.6 MB` FP16 group scales = `120.3 MB` total). During input embedding lookup for token $t$, a 1-block kernel reads row $t$ (`12` groups of 128) and dequantizes it to `[1536]` FP16 in `< 2 microseconds`; during `lm_head` projection, the W4A16 GEMV kernel streams the `120.3 MB` table using the same coalesced warp-interleaved access pattern.

### 4. Custom CUDA Kernel Design (`sm_86` Ampere)

1.  **Fused RMSNorm (`./src/kernels/rmsnorm_rope.cu`):**
    *   Launches `1` thread block of `256` threads (`8` warps) per token row (`1` block for batch-1 decode, `M` blocks for prefill).
    *   Each thread loads `6` elements (`3` `half2` pairs) from the `[1536]` input vector, accumulates $x_i^2$ in FP32 registers, reduces within each warp via `__shfl_down_sync`, and reduces across the 8 warps via an 8-element shared memory array. Threads then multiply cached register values by `rsqrtf(sum_sq / 1536.0f + 1e-6f) * gamma[i]` and write out coalesced `half2` vectors.
2.  **Vectorized FP16 GEMV & Fused W4A16 GEMV (`./src/kernels/gemv_fp16.cu`, `./src/kernels/gemv_w4a16.cu`):**
    *   **Shared-Memory Activation Caching:** The input activation vector (`1536` `half` elements = `3 KB`, or `8960` `half` elements = `17.5 KB` for `down_proj`) is cooperatively loaded into shared memory once per thread block so global memory traffic during matrix-vector multiplication consists solely of weight reads.
    *   **Warp-per-Row Reduction:** Each warp (`32` threads) computes `ROWS_PER_WARP = 4` output rows simultaneously, reusing the shared-memory activation broadcast across all 4 rows.
    *   **LUT-Free Register Dequantization (Phase 2):** Each thread unpacks its `uint32_t` word (8 INT4 weights) in registers into four `half2` pairs using bitwise extraction and FP16 exponent-bias affine conversion, multiplies by the group's FP16 scale, and accumulates against the shared-memory activation vector in FP32 registers before a 5-step `__shfl_down_sync` warp reduction.
    *   **Fused Residual Add & Fused Dual `Gate+Up+SwiGLU`:**
        *   `o_proj` and `down_proj` accept the residual stream pointer and add the residual value prior to global memory writeback, eliminating standalone elementwise addition kernels.
        *   `gate_proj` and `up_proj` are evaluated in a single fused kernel from the same shared-memory activation vector, applying $\text{SiLU}(g) \cdot u = \frac{g}{1 + e^{-g}} \cdot u$ in FP32 registers before writing a single `[8960]` FP16 output vector.
3.  **Fused QKV Bias + NeoX Split-Half RoPE + KV Cache Append (`./src/kernels/rmsnorm_rope.cu`):**
    *   Reads raw `q` (`[1536]`), `k` (`[256]`), and `v` (`[256]`) outputs, adds the mandatory `Qwen2.5` bias vectors (`b_q`, `b_k`, `b_v`), reads the sequence position directly from device memory, and applies NeoX split-half rotation where coordinate $i \in [0, 63]$ is paired with $i + 64$ inside each 128-dim head at frequency $\theta_i = 1000000^{-2i/128}$:
        $$\begin{bmatrix} x'_i \\ x'_{i+64} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} x_i \\ x_{i+64} \end{bmatrix}$$
    *   Writes rotated `k` and biased `v` directly into `K_cache` and `V_cache` at the clamped sequence slot.
4.  **GQA 6:1 Split-K FlashDecoding (`./src/kernels/attention_gqa.cu`):**
    *   To saturate all 16 SMs on the RTX 3050 despite having only 2 KV heads, the kernel launches a 2D grid of `(NUM_KV_HEADS = 2, FLASH_DECODE_SPLITS = 16)` (`32` thread blocks, `2` blocks per SM).
    *   Each thread block loads all `6` Query vectors (`6 x 128` `half` values) belonging to its KV head group into shared memory once, then streams its assigned sequence chunk `[split_start, split_end)` of `K_cache` and `V_cache`. Each `K` and `V` vector loaded from global memory is reused across all 6 Query heads, cutting KV cache bandwidth by 6x.
    *   Each split writes partial online-softmax statistics (`max_logit`, `sum_exp`, and unnormalized output vector) to the workspace Split-K accumulator buffer, and a 12-block reduction kernel combines the active splits into the final `[1536]` attention context vector.
5.  **Device Control Block, Single-Pass Sampling & Zero-Update CUDA Graph (`./src/engine.cu`, `./src/kernels/sampling.cu`):**
    *   A persistent `DeviceControlBlock` in GPU global memory (`./include/tensor.cuh`) holds the active `token_id`, `seq_pos`, `temperature`, and RNG state at a fixed device address, paired with a 4-byte page-locked host pointer `h_pinned_token` mapped via `cudaHostAlloc(..., cudaHostAllocMapped)`.
    *   During engine initialization, the 28-layer decode forward pass plus the final `sample_and_advance` kernel is captured once via `cudaStreamBeginCapture` / `cudaGraphInstantiate` into `decode_graph_exec_`.
    *   `sample_and_advance` performs a two-stage parallel `argmax` reduction over the `[151936]` logits (folding inline Gumbel-Max noise $\text{logit}_i / T - \log(-\log(U_i))$ into the reduction when `temperature > 0.0f`), writes the selected token ID to both `d_control->token_id` and `*h_pinned_token`, and increments `d_control->seq_pos += 1`. The host loop calls `cudaGraphLaunch(decode_graph_exec_, stream_)` and `cudaStreamSynchronize(stream_)` to read `*h_pinned_token` with zero `cudaGraphExecUpdate` calls.

### 5. Multi-Token Chunked Prefill & Session KV Prefix Caching

*   **Outer Chunking (`PREFILL_MAX_CHUNK = 256`) & Tiled Prefill Kernels (`./src/kernels/gemm_prefill.cu`):**
    *   Long prompts are processed in outer host-loop chunks of up to `PREFILL_MAX_CHUNK = 256` tokens so intermediate prefill buffers (`[256, 8960]` = `4.58 MB`) fit inside the static `32 MB` workspace and each chunk completes well inside the Windows WDDM timeout window.
    *   Within each linear layer, thread blocks tile the prompt chunk by `PREFILL_TILE_M = 16` tokens and the reduction dimension by `PREFILL_TILE_K = 128` elements, cooperatively caching a `[16, 128]` FP16 activation tile (`4 KB` shared memory) per loop iteration and multiplying each loaded weight tile against all 16 token rows in registers.
*   **Causal Prefill Attention:** Processes prompt queries against the KV cache up to `seq_pos + m` for each token $m \in [0, M-1]$.
*   **Stateful KV Prefix Matching (`./src/engine.cu`):** `QwenEngine` stores `std::vector<int32_t> cached_tokens_` representing the exact token sequence currently materialized in the GPU KV cache. When a new chat turn arrives with tokenized prompt `P`, the engine finds the longest common prefix length `L_prefix` between `cached_tokens_` and `P`. It copies `L_prefix` to `d_control->seq_pos` and runs prefill only on the suffix `P[L_prefix .. P.size() - 1]`, keeping multi-turn prefill latency under 30 ms for typical chat messages.

### 6. CLI Benchmark Harness, BPE/ChatML Tokenizer & Zero-dGPU-VRAM Local HTTP/SSE Server

*   **Standalone BPE Tokenizer & ChatML Formatter (`./src/tokenizer.cpp`):**
    *   Loads `tokenizer.bin` (exported by `./tools/export_qwen25.py`) containing the 151,646 vocabulary byte pieces, BPE merge ranks, and Qwen2.5 special token IDs (`<|endoftext|> = 151643`, `<|im_start|> = 151644`, `<|im_end|> = 151645`).
    *   Formats multi-turn chat histories into deterministic `<|im_start|>role\ncontent<|im_end|>\n` ChatML token sequences so prefix matching hits across turns, halts generation when token `151645` or `151643` is sampled, and buffers incomplete trailing UTF-8 lead/continuation bytes across token steps so multi-byte UTF-8 characters are only flushed over SSE once a complete codepoint is formed.
*   **Deterministic Benchmark CLI (`./src/cli_bench.cpp`):**
    *   Uses `cudaEventRecord` around prefill and each decode graph launch to record exact GPU hardware timestamps alongside `std::chrono::steady_clock` wall-clock timers.
    *   Queries `cudaMemGetInfo` before and after arena initialization to report exact VRAM consumption.
    *   Computes Effective Memory Bandwidth:
        $$\text{Effective BW (GB/s)} = \frac{\text{Model Weight Bytes} + \text{KV Cache Bytes Read per Token}}{\text{Median Inter-Token Latency (s)}}$$
        and outputs a side-by-side comparison table against `llama-bench`.
*   **Single-Threaded Synchronous HTTP/SSE Server & Static UI (`./src/main_server.cpp`, `./web/index.html`):**
    *   Because the engine targets single-user (`batch_size = 1`) local execution on `127.0.0.1`, `./src/main_server.cpp` runs a synchronous single-threaded `WinSock2` HTTP loop that invokes `QwenEngine` directly. During SSE streaming, the host overlaps `send()` of token $t-1$ over the loopback socket with the asynchronous `cudaGraphLaunch` of token $t$, achieving CPU/GPU overlap without multi-threaded ring buffers.
    *   `./web/index.html` uses pure HTML/CSS and standard DOM text updates (no `<canvas>`, no WebGL, no external JS dependencies) so any browser running on the laptop's Intel integrated GPU consumes `0 MB` of RTX 3050 VRAM.

## Alternatives Considered

### Alternative A: Delegating Linear Projections to `cuBLAS` / `cuBLASLt`

*   **Pros**: Provides pre-tuned NVIDIA Tensor Core (`mma.sync`) Matrix-Matrix Multiplication (GEMM) kernels for large-batch prompt prefill without writing custom assembly or register-tiling logic.
*   **Cons**: Violates the core objective of writing a self-contained CUDA inference engine from scratch; performs poorly on narrow `M = 1` Matrix-Vector Multiplications (GEMV) during autoregressive decoding; cannot fuse INT4 weight dequantization, QKV bias addition, residual stream addition, or dual `gate_proj + up_proj + SwiGLU` activation into the projection kernel, increasing memory bandwidth traffic and kernel launch count.
*   **Reason for Rejection**: Rejected by author: The project goal requires writing raw CUDA kernels from scratch with zero external math library dependencies, and custom fused GEMV kernels achieve lower latency and higher memory bus utilization for `batch_size = 1` decoding on the RTX 3050 Laptop GPU.

### Alternative B: Targeting a 2B+ Parameter Model in FP16

*   **Pros**: Higher parameter count can improve baseline reasoning and coding accuracy compared to a 1.5B parameter model.
*   **Cons**: A 2B parameter model in FP16 requires ~4.0 GB for model weights alone. Because Windows WDDM and the CUDA context reserve 300 to 450 MB of the 4,096 MB physical VRAM on the RTX 3050 Laptop GPU, weights plus KV cache and workspace exceed physical VRAM and spill into system RAM over PCIe, degrading token generation throughput by 5x to 10x.
*   **Reason for Rejection**: Rejected by author: `Qwen2.5-1.5B-Instruct` (`3,087.6 MB` FP16 weights with tied embeddings and `114.69 MB` 4,096-token GQA KV cache) fits strictly inside the usable 4 GB WDDM VRAM budget in both FP16 (`~3.24 GB`) and W4A16 (`~0.94 GB`).

### Alternative C: Starting Directly with W4A16 INT4 Without a Phase 1 FP16 Baseline

*   **Pros**: Reaches the lower-memory, higher-throughput quantized path sooner without writing a standalone FP16 GEMV kernel.
*   **Cons**: Conflates numerical errors from bitwise INT4 weight packing/unpacking and group scaling with structural transformer bugs (such as Qwen2.5 QKV bias handling, NeoX split-half RoPE indexing, or GQA 6:1 head mapping), making it difficult to verify exact mathematical parity against PyTorch.
*   **Reason for Rejection**: Rejected by author: Building and verifying the Phase 1 FP16 baseline first isolates transformer correctness and CUDA Graph capture against golden PyTorch reference tensors before introducing Phase 2 W4A16 bitwise quantization.

### Alternative D: Leaving the Tied `embed_tokens` / `lm_head` Matrix in FP16 During Phase 2

*   **Pros**: Avoids dequantizing a row of the embedding table during input token lookup and preserves exact FP16 precision on the final vocabulary projection.
*   **Cons**: Because `Qwen2.5-1.5B` has a 151,936-token vocabulary (`[151936, 1536]`), an FP16 `lm_head` occupies `466.75 MB`. When the 28 transformer layers are quantized to W4A16 (`675.7 MB`), reading an unquantized `466.75 MB` `lm_head` accounts for 41% of total per-token memory traffic (~2.8 ms per token at 165 GB/s), capping decode throughput at ~140 tokens/sec instead of ~190+ tokens/sec.
*   **Reason for Rejection**: Rejected by author: Quantizing the tied `embed_tokens` / `lm_head` table to W4A16 shrinks it from `466.75 MB` to `120.3 MB` (saving ~2.1 ms per token during `lm_head` GEMV), while on-the-fly dequantization of a single 1,536-element row during input embedding lookup takes less than 2 microseconds.

### Alternative E: Native Desktop GUI (Electron, Qt, or DirectX/OpenGL ImGui)

*   **Pros**: Packages the chat interface as a standalone desktop window without opening a web browser.
*   **Cons**: Hardware-accelerated desktop UI frameworks on Windows frequently allocate swapchains and compositing surfaces on the discrete GPU (consuming 150 to 300 MB of VRAM), which risks pushing the Phase 1 FP16 model over the 4 GB VRAM ceiling and introduces frame-pacing jitter on the WDDM hardware scheduler.
*   **Reason for Rejection**: Rejected by author: A single-file static HTML/CSS/JS frontend served over localhost and rendered in a browser running on the laptop's Intel integrated GPU consumes `0 MB` of discrete RTX 3050 VRAM.

--------------------------------------------------------------------------------

## Risks, Limits & Mitigations

1.  **Context Window Overflow (`seq_pos >= MAX_SEQ_LEN = 4096`) Inside the Zero-Update CUDA Graph:**
    *   **Risk:** Because the static `GpuArena` places the Workspace region immediately after the KV Cache region, incrementing `d_control->seq_pos` past `4095` during decode would cause the KV cache write kernel to overwrite the Workspace buffer in GPU global memory.
    *   **Mitigation:** Enforce host-side bounds checks in `QwenEngine::prefill()` and `QwenEngine::decode_step()` that halt generation (and return HTTP `400 Bad Request` or truncate in CLI mode) when `seq_pos >= MAX_SEQ_LEN`. Additionally, clamp the write/read index inside `qkv_bias_rope_kv_append` and `attention_gqa_flash_decode` to `min(pos, MAX_SEQ_LEN - 1)` so out-of-bounds global memory writes are impossible even if invoked at the boundary.
2.  **Windows WDDM Timeout Detection and Recovery (TDR) Driver Resets (>2 Seconds):**
    *   **Risk:** Windows WDDM resets the GPU display driver if a single CUDA command batch runs for more than 2 seconds without returning control to the OS scheduler.
    *   **Mitigation:** Partition long prompt prefills into outer chunks of `PREFILL_MAX_CHUNK = 256` tokens and call `cudaStreamSynchronize(stream_)` between chunks so each submission completes in `< 60 ms`.
3.  **Background WDDM VRAM Contention in Phase 1 FP16 (`~3,234 MB` Arena):**
    *   **Risk:** If another desktop application allocates >350 MB on the discrete RTX 3050 GPU while running in Phase 1 FP16 mode, Windows WDDM may fail `cudaMalloc` or silently page portions of the arena to system RAM over PCIe.
    *   **Mitigation:** Query `cudaMemGetInfo(&free_bytes, &total_bytes)` before constructing `GpuArena` and fail fast with an explicit diagnostic message reporting required vs. available free VRAM (and recommending Phase 2 W4A16 mode, which requires only `943 MB`).
4.  **Split-K Empty Partition NaN Hazard (`seq_len < FLASH_DECODE_SPLITS = 16`):**
    *   **Risk:** When the active context length is shorter than 16 tokens, trailing sequence splits have `split_start >= split_end`; uninitialized or zero-divided online-softmax accumulators will propagate `NaN` into the combined attention output.
    *   **Mitigation:** In `attention_gqa_flash_decode_stage1_kernel`, explicitly write `max_logit = -INFINITY` and `sum_exp = 0.0f` for inactive splits, and in `stage2` skip any split whose `sum_exp == 0.0f`.
5.  **W4A16 Outlier Quantization Degradation:**
    *   **Risk:** Naive per-tensor 4-bit quantization degrades accuracy on small 1.5B models with activation/weight outliers.
    *   **Mitigation:** Use fine-grained `QUANT_GROUP_SIZE = 128` symmetric FP16 group scales across all linear layers and the tied embedding table, validated by automated cosine similarity (`>= 0.995`) in `./tests/test_kernels.cu`.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance & Latency:**
    *   **Phase 1 (FP16) Decode Throughput:** `>= 46 tokens/sec` (`<= 21.7 ms` median inter-token latency) at `batch_size = 1`, representing `>= 75%` of the 192 GB/s theoretical memory bandwidth on the RTX 3050 Laptop GPU.
    *   **Phase 2 (W4A16) Decode Throughput:** `>= 150 tokens/sec` (`<= 6.6 ms` median inter-token latency) at `batch_size = 1`, matching or exceeding `llama.cpp` (`Q4_0` / `Q4_K_M`) on identical hardware.
    *   **Host Launch Overhead:** `0` calls to `cudaGraphExecUpdate` during autoregressive token generation; `1` `cudaGraphLaunch` call per token step.
*   **Memory Footprint & Isolation:**
    *   Single `cudaMalloc` arena allocation at startup; zero dynamic device allocations (`cudaMalloc` / `cudaFree`) during prefill or decode.
    *   Maximum discrete GPU VRAM allocation: `<= 3,300 MB` in Phase 1 (FP16) and `<= 1,050 MB` in Phase 2 (W4A16) at `MAX_SEQ_LEN = 4096`.
*   **Security & Privacy:**
    *   The HTTP/SSE server binds exclusively to `127.0.0.1` (loopback interface) by default and operates 100% offline with local model weights and zero external network calls.

--------------------------------------------------------------------------------

## Future Improvements

*   **INT8 (`Q8_0`) KV Cache Quantization:** Quantizing the static KV cache from FP16 to INT8 would halve the 4,096-token KV cache footprint from `114.69 MB` (`112.0 MiB`) to `57.34 MB` (`56.0 MiB`) and halve KV cache bandwidth at long sequence lengths.
*   **`__dp4a` Weight-INT4 / Activation-INT8 (`W4A8`) Decode Kernel:** Quantizing the input activation vector on the fly to INT8 inside shared memory would allow replacing FP16 `__hfma2` instructions with Ampere hardware integer dot-product instructions (`__dp4a`), reducing register pressure in the GEMV kernel.
*   **Speculative Decoding (Self-Speculation / N-Gram Prompt Lookup):** Using zero-parameter prompt n-gram lookup to draft 3 to 5 candidate continuation tokens verified in a single tiled forward pass could increase effective tokens/sec by 1.5x to 2x on repetitive code or structured text generation.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Phasing & Incremental Checkpoints

Implementation is partitioned into four sequential phases so that each phase leaves the repository in a clean, buildable, and testable state:

1.  **Phase 1A — Build System, Offline Exporters & FP16 Kernel Unit Tests:**
    *   **Files Created:** `#1 ./CMakeLists.txt`, `#3 ./tools/export_qwen25.py`, `#4 ./tools/verify_reference.py`, `#5 ./include/config.cuh`, `#6 ./include/tensor.cuh`, `#7 ./include/kernels.cuh`, `#10 ./src/kernels/rmsnorm_rope.cu`, `#11 ./src/kernels/gemv_fp16.cu`, `#13 ./src/kernels/gemm_prefill.cu` (FP16 path), `#14 ./src/kernels/attention_gqa.cu`, `#15 ./src/kernels/sampling.cu`, and `#21 ./tests/test_kernels.cu` (with Phase 2 W4A16 wrappers stubbed in `#12 ./src/kernels/gemv_w4a16.cu`).
    *   **Checkpoint Gate:** `cmake --build build --config Release --target test_kernels` compiles cleanly; `test_kernels.exe` passes isolated FP16 kernel tests for `embed_lookup_fp16`, `rmsnorm_fp16`, `qkv_bias_rope_kv_append`, `gemv_fp16`, `gemm_prefill_fp16`, `attention_gqa_flash_decode`, and `sample_and_advance`.
2.  **Phase 1B — `QwenEngine` Runtime, Zero-Update CUDA Graph & FP16 CLI Benchmark:**
    *   **Files Created:** `#8 ./include/engine.cuh`, `#9 ./include/tokenizer.hpp`, `#16 ./src/engine.cu`, `#17 ./src/tokenizer.cpp`, and `#18 ./src/cli_bench.cpp`.
    *   **Checkpoint Gate:** `test_kernels.exe` passes the 28-layer greedy token parity check between uncaptured stream execution and `cudaGraphExec_t` replay; `cli_bench.exe --model models/qwen25_1.5b_fp16.qwenbin` executes within `<= 3,300 MB` VRAM and achieves `>= 46 tokens/sec`.
3.  **Phase 2 — W4A16 Group-128 Quantized Kernels & Tied Embedding Quantization:**
    *   **Files Completed:** `#12 ./src/kernels/gemv_w4a16.cu` plus W4A16 paths in `#10 ./src/kernels/rmsnorm_rope.cu`, `#13 ./src/kernels/gemm_prefill.cu`, `#16 ./src/engine.cu`, and `#21 ./tests/test_kernels.cu`.
    *   **Checkpoint Gate:** `test_kernels.exe` passes all 8 verification suites (including W4A16 cosine similarity `>= 0.995` and `compute-sanitizer` `memcheck`/`initcheck`/`racecheck` with 0 errors); `cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin` executes within `<= 1,050 MB` VRAM and achieves `>= 150 tokens/sec`.
4.  **Phase 3 — Local HTTP/SSE Chat Server, Zero-dGPU-VRAM Web UI & Documentation:**
    *   **Files Created:** `#19 ./src/main_server.cpp`, `#20 ./web/index.html`, and `#2 ./README.md`.
    *   **Checkpoint Gate:** `qwen_server.exe` streams multi-turn ChatML responses over SSE to `http://127.0.0.1:8080`, hits the KV prefix cache on turn 2, and adds `0 MB` of discrete GPU VRAM when opened in a browser on the Intel iGPU.

### Target Files to Create/Modify

#### 1. `./CMakeLists.txt` (Create)

*   **Type / Scope**: CMake Build Configuration (`C++20` + `CUDA`)
*   **Rationale**: Establishes a reproducible Windows MSVC + NVCC build pipeline targeting Ampere `sm_86` (RTX 3050 Laptop GPU) without linking external math libraries (`cuBLAS`, `cuDNN`, or `CUTLASS`).
*   **Proposed Changes**:
    *   Set `CMAKE_CXX_STANDARD 20`, `CMAKE_CUDA_STANDARD 20`, and `CMAKE_CUDA_ARCHITECTURES 86`.
    *   Configure NVCC release flags (`-O3`, `--use_fast_math`, `-lineinfo`, `--expt-relaxed-constexpr`) and MSVC flags (`/O2`, `/W4`).
    *   Define static library target `qwen_core` compiling all kernel and runtime sources (`./src/kernels/rmsnorm_rope.cu`, `./src/kernels/gemv_fp16.cu`, `./src/kernels/gemv_w4a16.cu`, `./src/kernels/gemm_prefill.cu`, `./src/kernels/attention_gqa.cu`, `./src/kernels/sampling.cu`, `./src/engine.cu`, `./src/tokenizer.cpp`).
    *   Define three executable targets linked against `qwen_core` and `cudart_static` (plus `ws2_32` for Windows sockets on the server): `cli_bench` (`./src/cli_bench.cpp`), `qwen_server` (`./src/main_server.cpp`), and `test_kernels` (`./tests/test_kernels.cu`).

#### 2. `./README.md` (Create)

*   **Type / Scope**: Project Documentation
*   **Rationale**: Documents prerequisites, offline weight export commands, Windows Graphics Settings configuration to keep the browser UI on the Intel integrated GPU, and CLI benchmark reproduction steps against `llama.cpp`.
*   **Proposed Changes**:
    *   Document step-by-step commands for exporting FP16 and W4A16 `.qwenbin` files and `tokenizer.bin` via `./tools/export_qwen25.py`, generating golden test vectors via `./tools/verify_reference.py`, building with CMake, running `./tests/test_kernels.cu`, benchmarking via `./src/cli_bench.cpp`, and launching the HTTP/SSE chat server via `./src/main_server.cpp` with `./web/index.html`.

#### 3. `./tools/export_qwen25.py` (Create)

*   **Type / Scope**: Offline Python Weight Exporter, Group-128 Quantizer & Tokenizer Serializer
*   **Rationale**: Converts HuggingFace `Qwen/Qwen2.5-1.5B-Instruct` safetensors into a single 4,096-byte page-aligned binary file (`.qwenbin`) with warp-coalesced layouts and exports `tokenizer.bin` so the C++ engine can map weights and vocabulary directly without runtime JSON parsing or bit shuffling.
*   **Proposed Changes**:
    *   Accept CLI arguments `--model-id` (default `Qwen/Qwen2.5-1.5B-Instruct`), `--mode` (`fp16` or `w4a16`), `--out` (output `.qwenbin` path), and `--tokenizer-out` (default `models/tokenizer.bin`).
    *   Write the 4,096-byte `QWQwenHeader` struct (matching `./include/config.cuh`) followed by 4,096-byte-aligned tensor payloads. Verify that `model.embed_tokens.weight` and `lm_head.weight` are tied, exporting a single shared embedding table (`[151936, 1536]`) plus the 28 transformer layers (including `q_proj.bias`, `k_proj.bias`, and `v_proj.bias`) and final RMSNorm weight.
    *   Interleave `gate_proj` (`[8960, 1536]`) and `up_proj` (`[8960, 1536]`) 1-to-1 by row into `gate_up_proj` (`[8960, 2, 1536]`, where row $2r$ is `gate_proj[r]` and row $2r+1$ is `up_proj[r]`, along with their corresponding group scales in `w4a16` mode).
    *   **Exact `w4a16` Group-128 Packing Specification:**
        *   Partition each weight row along `K` into groups of `128` elements (`K / 128` groups per row). For each group $g$, compute FP16 scale $s = \max(|w|) / 7.0$ and quantize each weight $w_i$ ($i \in [0, 127]$) to $q_i = \text{clamp}(\text{round}(w_i / s) + 8, 0, 15) \in [0, 15]$.
        *   Partition the 128 quantized nibbles of group $g$ into `16` consecutive 8-element blocks indexed by half-warp lane $l \in [0, 15]$, where lane $l$ owns elements $k_{\text{local}} = 8l + m$ ($m \in [0, 7]$).
        *   Pack the 8 nibbles $q_{8l+0} \dots q_{8l+7}$ of lane $l$ into one `uint32_t` word with nibble order $(q_0, q_4, q_1, q_5, q_2, q_6, q_3, q_7)$ across bits `[3:0], [7:4], ..., [31:28]`:
            `packed_u32 = q0 | (q4 << 4) | (q1 << 8) | (q5 << 12) | (q2 << 16) | (q6 << 20) | (q3 << 24) | (q7 << 28)`
            so that `lop3.b32` masking with `0x000F000F` and right-shifts by `0, 4, 8, 12` extract pairs $(q_0, q_1), (q_4, q_5), (q_2, q_3), (q_6, q_7)$ directly into four `half2` registers.
        *   Across a 32-lane warp step $s_{\text{warp}} \in [0, K / 256 - 1]$, lanes `0..15` store the 16 `uint32_t` words of group $2 s_{\text{warp}}$ and lanes `16..31` store the 16 `uint32_t` words of group $2 s_{\text{warp}} + 1$, forming a contiguous 128-byte coalesced warp load per row.
    *   **Exact `tokenizer.bin` Binary Layout:** Write `TokenizerBinHeader` (`uint32_t magic = 0x51544F4B`, `uint32_t vocab_size = 151646`, `uint32_t num_merges`, `uint32_t eos_id = 151645`), followed by `vocab_size` sequential records of `[uint16_t byte_len, uint8_t utf8_bytes[byte_len], int32_t rank]`, followed by `num_merges` sequential records of `[int32_t left_id, int32_t right_id, int32_t merged_id]`.

#### 4. `./tools/verify_reference.py` (Create)

*   **Type / Scope**: Offline PyTorch Reference Activation & Logits Exporter
*   **Rationale**: Produces ground-truth intermediate tensors and 32 greedy reference tokens from the official HuggingFace PyTorch implementation on a fixed 8-token prompt so `./tests/test_kernels.cu` can validate each CUDA kernel in isolation before full-graph execution.
*   **Proposed Changes**:
    *   Accept `--model-id` (default `Qwen/Qwen2.5-1.5B-Instruct`) and `--out` (default `models/golden_ref.bin`).
    *   Use the fixed 8-token ChatML prompt `prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847]`.
    *   Write a 256-byte `GoldenRefHeader` (`uint32_t magic = 0x474F4C44`, `uint32_t prompt_len = 8`, `int32_t prompt_tokens[8]`, `int32_t greedy_tokens[32]`, `uint64_t tensor_offsets[13]`, zero-padded to 256 bytes) followed by 13 contiguous FP16 reference tensors for the 8-token prompt (plus FP32 final logits for the last prompt position):
        0. `embed_out` (`[8, 1536]` FP16)
        1. `layer0_input_norm` (`[8, 1536]` FP16)
        2. `layer0_q_raw` (`[8, 1536]` FP16, before bias/RoPE)
        3. `layer0_k_raw` (`[8, 256]` FP16, before bias/RoPE)
        4. `layer0_v_raw` (`[8, 256]` FP16, before bias)
        5. `layer0_q_rope` (`[8, 1536]` FP16, after bias + NeoX RoPE)
        6. `layer0_k_rope` (`[8, 256]` FP16, after bias + NeoX RoPE)
        7. `layer0_v_biased` (`[8, 256]` FP16, after bias)
        8. `layer0_attn_out` (`[8, 1536]` FP16)
        9. `layer0_post_norm` (`[8, 1536]` FP16)
        10. `layer0_swiglu_out` (`[8, 8960]` FP16)
        11. `layer0_mlp_out` (`[8, 1536]` FP16, after `down_proj` + residual add)
        12. `final_logits_last` (`[151936]` FP32, logits of token index 7 after Layer 27 and final RMSNorm)

#### 5. `./include/config.cuh` (Create)

*   **Type / Scope**: CUDA / C++20 Header (Compile-Time Model Constants & Binary Headers)
*   **Rationale**: Centralizes all `Qwen2.5-1.5B-Instruct` dimensions, `sm_86` hardware tile sizes, and binary file header structs (`QWQwenHeader`, `GoldenRefHeader`, `TokenizerBinHeader`) as `constexpr` and fixed-size POD structs.
*   **Proposed Changes**:
    *   Define `constexpr` constants: `HIDDEN_SIZE = 1536`, `INTERMEDIATE_SIZE = 8960`, `NUM_LAYERS = 28`, `NUM_Q_HEADS = 12`, `NUM_KV_HEADS = 2`, `GQA_GROUP_SIZE = 6`, `HEAD_DIM = 128`, `Q_DIM = 1536`, `KV_DIM = 256`, `VOCAB_SIZE = 151936`, `MAX_SEQ_LEN = 4096`, `RMS_NORM_EPS = 1e-6f`, `ROPE_THETA = 1000000.0f`, `QUANT_GROUP_SIZE = 128`, `PREFILL_MAX_CHUNK = 256`, `PREFILL_TILE_M = 16`, `PREFILL_TILE_K = 128`, and `FLASH_DECODE_SPLITS = 16`.
    *   Define `enum class QuantMode : uint32_t { FP16 = 0, W4A16_G128 = 1 };`.
    *   Define `struct alignas(4096) QWQwenHeader` containing:
        *   `uint32_t magic` (`0x5157454E`), `uint32_t version` (`1`), `uint32_t quant_mode`, `uint32_t vocab_size`, `uint32_t hidden_size`, `uint32_t intermediate_size`, `uint32_t num_layers`, `uint32_t num_q_heads`, `uint32_t num_kv_heads`, `uint32_t head_dim`, `uint32_t quant_group_size`, `uint32_t reserved0`
        *   `uint64_t embed_weight_offset`, `uint64_t embed_scale_offset`, `uint64_t final_norm_offset`
        *   `uint64_t layer_offsets[28][17]` (byte offsets inside `.qwenbin` for: `0: input_layernorm`, `1: q_weight`, `2: q_scale`, `3: q_bias`, `4: k_weight`, `5: k_scale`, `6: k_bias`, `7: v_weight`, `8: v_scale`, `9: v_bias`, `10: o_weight`, `11: o_scale`, `12: post_attention_layernorm`, `13: gate_up_weight`, `14: gate_up_scale`, `15: down_weight`, `16: down_scale`; scale offsets are `0` when `quant_mode == 0`)
        *   `uint8_t padding[216]` (ensuring `sizeof(QWQwenHeader) == 4096`).

#### 6. `./include/tensor.cuh` (Create)

*   **Type / Scope**: CUDA / C++20 Header (Static GPU Arena Allocator & Device Control Block)
*   **Rationale**: Enforces the single-`cudaMalloc` invariant with a startup `cudaMemGetInfo` pre-flight check to prevent Windows WDDM memory fragmentation.
*   **Proposed Changes**:
    *   Define `CUDA_CHECK(call)` macro reporting file, line, and `cudaGetErrorString`.
    *   Define `class GpuArena` with `explicit GpuArena(size_t total_bytes)` (queries `cudaMemGetInfo`, fails fast with diagnostic error if `free_bytes < total_bytes + (64ULL << 20)`, and calls `cudaMalloc` once), `void* allocate(size_t bytes, size_t alignment = 256)`, `size_t used_bytes() const`, `size_t capacity_bytes() const`, and `~GpuArena()`.
    *   Define `struct alignas(256) DeviceControlBlock { int32_t token_id; int32_t seq_pos; float temperature; uint64_t rng_seed; };`.

#### 7. `./include/kernels.cuh` (Create)

*   **Type / Scope**: CUDA Header (Kernel Host Launch Declarations)
*   **Rationale**: Declares host-callable wrapper functions with complete C++ parameter signatures for all custom CUDA kernels.
*   **Proposed Changes**:
    *   Declare the following host launch functions:
        *   `void launch_embed_lookup_fp16(const half* embed_table, const int32_t* d_token_ids, half* out_hidden, int num_tokens, cudaStream_t stream);`
        *   `void launch_embed_lookup_w4a16(const uint32_t* embed_qweight, const half* embed_scales, const int32_t* d_token_ids, half* out_hidden, int num_tokens, cudaStream_t stream);`
        *   `void launch_rmsnorm_fp16(const half* input, const half* gamma, half* output, int num_tokens, cudaStream_t stream);`
        *   `void launch_gemv_fp16(const half* weight, const half* input_x, const half* residual, half* output_y, int out_features, int in_features, cudaStream_t stream);`
        *   `void launch_gemv_dual_gate_up_swiglu_fp16(const half* gate_up_weight, const half* input_x, half* output_mlp, cudaStream_t stream);`
        *   `void launch_gemv_w4a16(const uint32_t* qweight, const half* scales, const half* input_x, const half* residual, half* output_y, int out_features, int in_features, cudaStream_t stream);`
        *   `void launch_gemv_dual_gate_up_swiglu_w4a16(const uint32_t* gate_up_qweight, const half* gate_up_scales, const half* input_x, half* output_mlp, cudaStream_t stream);`
        *   `void launch_gemm_prefill_fp16(const half* weight, const half* input_x, const half* residual, half* output_y, int M, int N, int K, cudaStream_t stream);`
        *   `void launch_gemm_prefill_dual_gate_up_swiglu_fp16(const half* gate_up_weight, const half* input_x, half* output_mlp, int M, cudaStream_t stream);`
        *   `void launch_gemm_prefill_w4a16(const uint32_t* qweight, const half* scales, const half* input_x, const half* residual, half* output_y, int M, int N, int K, cudaStream_t stream);`
        *   `void launch_gemm_prefill_dual_gate_up_swiglu_w4a16(const uint32_t* gate_up_qweight, const half* gate_up_scales, const half* input_x, half* output_mlp, int M, cudaStream_t stream);`
        *   `void launch_qkv_bias_rope_kv_append(half* q, half* k, half* v, const half* q_bias, const half* k_bias, const half* v_bias, half* layer_k_cache, half* layer_v_cache, const DeviceControlBlock* d_control, int num_tokens, cudaStream_t stream);`
        *   `void launch_attention_gqa_flash_decode(const half* q, const half* layer_k_cache, const half* layer_v_cache, float* splitk_partial_out, float* splitk_partial_lse, half* attn_out, const DeviceControlBlock* d_control, cudaStream_t stream);`
        *   `void launch_attention_gqa_prefill(const half* q, const half* layer_k_cache, const half* layer_v_cache, half* attn_out, const DeviceControlBlock* d_control, int num_tokens, cudaStream_t stream);`
        *   `void launch_sample_and_advance(const half* logits, float* sample_scratch_val, int32_t* sample_scratch_idx, DeviceControlBlock* d_control, int32_t* d_mapped_pinned_token, cudaStream_t stream);`

#### 8. `./include/engine.cuh` (Create)

*   **Type / Scope**: CUDA / C++20 Header (`QwenEngine` Runtime State, Workspace Sub-Buffers & Graph Orchestrator)
*   **Rationale**: Encapsulates the GPU arena, layer weight pointers, static KV cache, deterministic workspace sub-buffers, session prefix cache, and captured `cudaGraphExec_t` instance.
*   **Proposed Changes**:
    *   Define `LayerWeights` holding device pointers for `input_layernorm`, `q_weight`, `q_scale`, `q_bias`, `k_weight`, `k_scale`, `k_bias`, `v_weight`, `v_scale`, `v_bias`, `o_weight`, `o_scale`, `post_attention_layernorm`, `gate_up_weight`, `gate_up_scale`, `down_weight`, and `down_scale`.
    *   Define `struct WorkspaceBuffers` partitioning the `32 MB` workspace region for up to `PREFILL_MAX_CHUNK = 256` tokens (`total = ~10.2 MB`):
        *   `half* d_act_0` (`[256, 1536]` = `786,432 B`)
        *   `half* d_act_1` (`[256, 1536]` = `786,432 B`)
        *   `half* d_q` (`[256, 1536]` = `786,432 B`)
        *   `half* d_k` (`[256, 256]` = `131,072 B`)
        *   `half* d_v` (`[256, 256]` = `131,072 B`)
        *   `half* d_attn_out` (`[256, 1536]` = `786,432 B`)
        *   `half* d_mlp_act` (`[256, 8960]` = `4,587,520 B`)
        *   `float* d_splitk_out` (`[12, 16, 128]` = `98,304 B`)
        *   `float* d_splitk_lse` (`[12, 16, 2]` = `1,536 B`)
        *   `half* d_logits` (`[151936]` = `303,872 B`)
        *   `float* d_sample_val` (`[256]` = `1,024 B`) and `int32_t* d_sample_idx` (`[256]` = `1,024 B`)
        *   `int32_t* d_prompt_tokens` (`[4096]` = `16,384 B`)
    *   Define `struct ArenaStats { size_t weights_bytes; size_t kv_cache_bytes; size_t workspace_bytes; size_t total_allocated_bytes; size_t device_free_bytes; size_t device_total_bytes; QuantMode mode; };`.
    *   Define `class QwenEngine` with `QwenEngine(const std::string& qwenbin_path)`, `void reset_session()`, `int32_t prefill(const std::vector<int32_t>& prompt_tokens, float temperature = 0.0f)`, `int32_t decode_step()`, `int32_t current_seq_pos() const`, and `ArenaStats get_memory_stats() const`.

#### 9. `./include/tokenizer.hpp` (Create)

*   **Type / Scope**: C++20 Header (Byte-Level BPE Tokenizer, ChatML Formatter & Streaming UTF-8 Decoder)
*   **Rationale**: Tokenizes raw user strings, formats multi-turn conversations using Qwen2.5's `<|im_start|>` / `<|im_end|>` ChatML special tokens, and buffers incomplete UTF-8 multi-byte sequences during token-by-token streaming.
*   **Proposed Changes**:
    *   Define `struct ChatMessage { std::string role; std::string content; };`.
    *   Define Qwen2.5 special token IDs (`ENDOFTEXT_ID = 151643`, `IM_START_ID = 151644`, `IM_END_ID = 151645`).
    *   Define `class QwenTokenizer` with `explicit QwenTokenizer(const std::string& tokenizer_bin_path)`, `std::vector<int32_t> encode(const std::string& text) const`, `std::string decode_streaming(int32_t token_id, std::string& utf8_carry_buffer) const`, `std::vector<int32_t> format_chatml(const std::vector<ChatMessage>& messages) const`, and `static bool is_eos(int32_t token_id)`.

#### 10. `./src/kernels/rmsnorm_rope.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (Embedding Lookup, RMSNorm, QKV Biases, NeoX RoPE, KV Cache Write)
*   **Rationale**: Implements multi-token/single-token embedding lookup, fused RMSNorm, and fused QKV bias + NeoX split-half RoPE + KV cache insertion, reading `seq_pos` directly from `DeviceControlBlock` so decode requires zero host argument updates.
*   **Proposed Changes**:
    *   Implement `embed_lookup_fp16_kernel` and `embed_lookup_w4a16_kernel`: launches `num_tokens` thread blocks (`1` block reading `&d_control->token_id` in decode; `M` blocks reading `d_prompt_tokens` in prefill); in W4A16 mode, 192 threads (`6` warps) dequantize the `12` groups of 128 INT4 weights for `d_token_ids[blockIdx.x]` into `[1536]` FP16.
    *   Implement `rmsnorm_fp16_kernel` (`num_tokens` blocks of `256` threads, `half2` vectorized loads, FP32 sum-of-squares warp shuffle reduction, multiplying by `rsqrtf(mean_sq + 1e-6f) * gamma[i]`).
    *   Implement `qkv_bias_rope_kv_append_kernel`: for each token offset `m in [0, num_tokens - 1]`, computes `pos = min(d_control->seq_pos + m, MAX_SEQ_LEN - 1)`, adds `q_bias` (`[1536]`), `k_bias` (`[256]`), and `v_bias` (`[256]`), applies NeoX split-half rotation pairing index `i` (`0..63`) with `i + 64` inside each 128-dim head for all 12 Q heads and 2 K heads at position `pos`, and writes rotated `K` and biased `V` into `layer_k_cache` and `layer_v_cache` at slot `pos`.

#### 11. `./src/kernels/gemv_fp16.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (Phase 1 Vectorized FP16 GEMV & Fused SwiGLU)
*   **Rationale**: Achieves `>= 75%` of the RTX 3050 Laptop GPU's 192 GB/s memory bandwidth in Phase 1 by caching the input vector in shared memory and streaming 128-bit (`uint4`) weight chunks across warps.
*   **Proposed Changes**:
    *   Implement `gemv_fp16_kernel`: cooperatively loads input vector `x` (`1536` or `8960` `half` elements) into `__shared__ half s_x[]`, assigns `4` output rows per warp, loads 8 `half` weights per thread per step via `uint4`, accumulates in FP32 registers, reduces across the 32 warp lanes with `__shfl_down_sync`, and if `residual != nullptr` adds `residual[row]` before writing `out[row]`.
    *   Implement `gemv_dual_gate_up_swiglu_fp16_kernel`: loads `s_x` (`[1536]`) once into shared memory, evaluates interleaved row `2*r` (`gate_proj[r]`) and row `2*r + 1` (`up_proj[r]`) per warp, applies `silu(gate) * up` in FP32 registers, and writes `[8960]` FP16 output.

#### 12. `./src/kernels/gemv_w4a16.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (Phase 2 Fused W4A16 Group-128 Dequantize-GEMV)
*   **Rationale**: Reduces per-token weight memory traffic by 3.85x in Phase 2 (`0.80 GB` vs. `3.08 GB` in FP16, including `lm_head`) by streaming 4-bit warp-interleaved weights (`256` weights = `2` groups of 128 per 32-lane warp step) and unpacking them in registers without shared-memory bank conflicts.
*   **Proposed Changes**:
    *   Implement inline device helper `dequant_8x_int4_to_half2(uint32_t packed, half scale, half2 out[4])`: extracts four `half2` pairs `(q0, q1), (q2, q3), (q4, q5), (q6, q7)` from `packed` using `lop3.b32` with mask `0x000F000F` and exponent bias `0x64006400`, subtracts `0x64086408` (`1032.0`, combining the `1024.0` FP16 bias and the symmetric zero-point `8.0`) via `__hsub2`, and multiplies by `__half2half2(scale)` via `__hmul2`.
    *   Implement `gemv_w4a16_kernel`: caches input `x` in `__shared__ half s_x[]`; on each step $s_{\text{warp}} \in [0, K / 256 - 1]$, lane `lane_id` loads one `uint32_t` word (`8` INT4 weights for group $g = 2 s_{\text{warp}} + (\text{lane\_id} / 16)$ and local offset $8 \cdot (\text{lane\_id} \bmod 16)$), unpacks to four `half2` pairs, multiplies against the four matching `half2` pairs in `s_x`, accumulates in FP32 registers, performs a 5-step `__shfl_down_sync` reduction across all 32 lanes, and optionally adds `residual[row]`.
    *   Implement `gemv_dual_gate_up_swiglu_w4a16_kernel`: fuses W4A16 `gate_proj` (row $2r$) and `up_proj` (row $2r+1$) dot products with register-level `silu(gate) * up` activation.

#### 13. `./src/kernels/gemm_prefill.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (Multi-Token Tiled Prefill Projections & Fused SwiGLU)
*   **Rationale**: Processes prompt chunks (`M <= PREFILL_MAX_CHUNK = 256`) by tiling along both `PREFILL_TILE_M = 16` and `PREFILL_TILE_K = 128` (`4 KB` shared memory per tile), reusing loaded weights across 16 prompt tokens in registers without exceeding `sm_86` shared memory limits.
*   **Proposed Changes**:
    *   Implement `gemm_prefill_fp16_kernel` and `gemm_prefill_w4a16_kernel`: for each `K`-tile of `128` elements, threads cooperatively load `[PREFILL_TILE_M = 16, PREFILL_TILE_K = 128]` input activations (`4,096 B`) into `__shared__ half s_tile[16][128]`; warps load the weight row chunk (and group scale in W4A16 mode) once from global memory, accumulate dot products across all `m in [0..15]` in FP32 register arrays `float acc[16]`, reduce across warp lanes, and write out `output_y[m * N + col] + (residual ? residual[m * N + col] : 0)`.
    *   Implement `gemm_prefill_dual_gate_up_swiglu_fp16_kernel` and `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`: evaluate row $2r$ (`gate`) and row $2r+1$ (`up`) of `gate_up_weight` against `s_tile[16][128]`, accumulate `gate_acc[16]` and `up_acc[16]`, and write `silu(gate_acc[m]) * up_acc[m]` to `output_mlp[m * 8960 + r]`.

#### 14. `./src/kernels/attention_gqa.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (GQA 6:1 Split-K FlashDecoding & Causal Prefill Attention)
*   **Rationale**: Solves the 2-KV-head SM under-utilization problem on the 16-SM RTX 3050 by parallelizing across 16 sequence length splits while sharing each loaded KV head across all 6 grouped Query heads.
*   **Proposed Changes**:
    *   Implement `attention_gqa_flash_decode_stage1_kernel`: launched with grid `(NUM_KV_HEADS = 2, FLASH_DECODE_SPLITS = 16)`. Reads `seq_len = min(d_control->seq_pos + 1, MAX_SEQ_LEN)`. If `split_start >= split_end`, writes `max_logit = -INFINITY, sum_exp = 0.0f` to `splitk_partial_lse` and exits. Otherwise, loads the `6` Query vectors (`[6, 128]`) for the KV group into shared memory, streams `K_cache` and `V_cache` once per token in `[split_start, split_end)`, updates 6 online-softmax accumulators (`max_score`, `sum_exp`, `acc_vec[128]`), and writes partial outputs to `splitk_partial_out` and `splitk_partial_lse`.
    *   Implement `attention_gqa_flash_decode_stage2_kernel`: launched with grid `(NUM_Q_HEADS = 12, 1)`. Skips splits where `sum_exp == 0.0f`, rescales active splits by their log-sum-exp weights, and writes the combined `[1536]` FP16 attention output.
    *   Implement `attention_gqa_prefill_kernel`: computes causal multi-token attention for `num_tokens` prompt tokens during prefill against `K_cache` and `V_cache` up to `d_control->seq_pos + m`.

#### 15. `./src/kernels/sampling.cu` (Create)

*   **Type / Scope**: CUDA Kernel Implementation (Device-Side Argmax / Gumbel-Max Sampling & Graph State Advance)
*   **Rationale**: Keeps the entire decode step self-contained inside the CUDA Graph by selecting the next token on the GPU in a single two-stage parallel reduction, writing it to both `d_control->token_id` and mapped pinned memory `*h_pinned_token`, and incrementing `d_control->seq_pos`.
*   **Proposed Changes**:
    *   Implement `sample_stage1_kernel` (`256` blocks of `256` threads over `VOCAB_SIZE = 151936`): for each vocabulary index $i$, reads `val = (float)logits[i]`; if `d_control->temperature > 1e-5f`, computes uniform random $u \in (10^{-7}, 1 - 10^{-7})$ via inline stateless Philox-4x32-10 (`seed = d_control->rng_seed`, `subsequence = i`, `offset = d_control->seq_pos`) and perturbs `val = val / d_control->temperature - logf(-logf(u))` (Gumbel-Max trick); reduces `(max_val, argmax_idx)` across the block into `sample_scratch_val[blockIdx.x]` and `sample_scratch_idx[blockIdx.x]`.
    *   Implement `sample_stage2_and_advance_kernel` (`1` block of `256` threads): reduces the 256 block candidates to the global winning `token_id`, writes `token_id` to `d_control->token_id` and `*d_mapped_pinned_token`, and increments `d_control->seq_pos += 1` on thread 0.

#### 16. `./src/engine.cu` (Create)

*   **Type / Scope**: CUDA / C++20 Implementation (`QwenEngine` Loader, Arena Setup, Chunked Prefill & CUDA Graph Capture)
*   **Rationale**: Wires together memory-mapped `.qwenbin` loading, single-call `GpuArena` allocation, outer `PREFILL_MAX_CHUNK = 256` chunked prefill with longest-common-prefix KV session caching, and `cudaGraphExec_t` capture/launch.
*   **Proposed Changes**:
    *   Map `.qwenbin` via Windows `CreateFileMappingA` / `MapViewOfFile`, validate `QWQwenHeader`, compute exact arena size (`3,234 MB` for FP16 or `943 MB` for W4A16), construct `GpuArena`, and copy the aligned weight block to GPU memory in a single `cudaMemcpy`.
    *   Sub-allocate the static KV cache (`114.69 MB`), `WorkspaceBuffers` (`10.2 MB` inside the `32 MB` workspace region), `DeviceControlBlock` (`256 B`), and mapped host pinned token buffer (`cudaHostAlloc(..., cudaHostAllocMapped)`).
    *   Capture the 28-layer decode forward pass + final RMSNorm + tied `lm_head` GEMV + `launch_sample_and_advance` into `decode_graph_exec_` using `cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal)` and `cudaGraphInstantiate`.
    *   Implement `prefill(prompt_tokens, temperature)`: enforces `prompt_tokens.size() < MAX_SEQ_LEN`, computes longest common prefix `L_prefix` against `cached_tokens_`, loops `chunk_start` from `L_prefix` to `prompt_tokens.size()` in steps of `PREFILL_MAX_CHUNK = 256` (copying `d_control->seq_pos = chunk_start`, running the 28-layer prefill kernels for `M = chunk_len` tokens, and calling `cudaStreamSynchronize(stream_)` after each chunk), runs final RMSNorm + `lm_head` + `launch_sample_and_advance` on the final prompt token, updates `cached_tokens_`, and returns `*h_pinned_token_`.
    *   Implement `decode_step()`: verifies `seq_pos_ < MAX_SEQ_LEN`, calls `cudaGraphLaunch(decode_graph_exec_, stream_)` and `cudaStreamSynchronize(stream_)`, appends `*h_pinned_token_` to `cached_tokens_`, and returns `*h_pinned_token_`.

#### 17. `./src/tokenizer.cpp` (Create)

*   **Type / Scope**: C++20 Implementation (BPE Tokenizer, ChatML Formatter & UTF-8 Stream Buffer)
*   **Rationale**: Provides standalone text encoding/decoding, Qwen2.5 ChatML prompt templating, and multi-byte UTF-8 boundary checking.
*   **Proposed Changes**:
    *   Read `TokenizerBinHeader` (`magic == 0x51544F4B`), vocabulary byte pieces, and BPE merge pairs from `tokenizer.bin`.
    *   Implement byte-level BPE merge encoding with special token splitting (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`), `format_chatml()`, and `decode_streaming(token_id, utf8_carry_buffer)` (inspects trailing UTF-8 lead bytes `0xC0..0xF7` and holds back incomplete multi-byte sequences in `utf8_carry_buffer` until the continuation bytes arrive in subsequent tokens).

#### 18. `./src/cli_bench.cpp` (Create)

*   **Type / Scope**: C++20 Executable (Deterministic Benchmark Harness)
*   **Rationale**: Measures prefill and decode performance with hardware `cudaEvent_t` timers and calculates achieved memory bus utilization for comparison against `llama.cpp` (`llama-bench`).
*   **Proposed Changes**:
    *   Parse `--model` (required), `--tokenizer` (default `models/tokenizer.bin`), `--prompt-len` (default `128`), `--gen-len` (default `256`), `--warmup` (default `10`), `--runs` (default `50`), and `--prompt` (optional text prompt for greedy output inspection).
    *   Record `cudaMemGetInfo` delta, prefill TTFT (ms), per-token decode latencies across `runs * gen_len` steps, `p50`/`p95`/`p99` inter-token latency (ms), decode throughput (tokens/sec), and Effective Memory Bandwidth (`GB/s` and `%` of `192.0 GB/s`).
    *   Output a formatted Markdown/ASCII benchmark summary table.

#### 19. `./src/main_server.cpp` (Create)

*   **Type / Scope**: C++20 Executable (Single-Threaded Loopback HTTP/SSE Server)
*   **Rationale**: Serves the static chat frontend and streams generated tokens over SSE directly from the single-user `QwenEngine` instance while overlapping socket sends with asynchronous CUDA Graph launches.
*   **Proposed Changes**:
    *   Parse `--model` (required), `--tokenizer` (default `models/tokenizer.bin`), `--web-root` (default `./web`), and `--port` (default `8080`).
    *   Bind a `WinSock2` TCP listener exclusively to `127.0.0.1:<port>`.
    *   Serve `<web-root>/index.html` on `GET /`, session reset (`engine.reset_session()`) on `POST /api/reset`, and JSON telemetry on `GET /api/stats`:
        `{"mode":"W4A16","vram_used_mb":943.2,"vram_total_mb":4096.0,"cached_tokens":48,"effective_bw_gbs":158.4,"bw_utilization_pct":82.5}`
    *   Handle `POST /api/chat` accepting JSON request payload:
        `{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"temperature":0.0,"max_new_tokens":512}`
        Format `messages` via `tokenizer.format_chatml()`, run `engine.prefill()`, and stream SSE events (`Content-Type: text/event-stream`):
        `data: {"token":"...","is_eos":false,"ttft_ms":12.3,"decode_tps":162.4,"cached_tokens":48}\n\n`
        Within the decode loop, launch token step $t$ via `cudaGraphLaunch` asynchronously before formatting and calling `send()` for token $t-1$'s SSE frame, then synchronize `stream_` to read token $t$.

#### 20. `./web/index.html` (Create)

*   **Type / Scope**: Static HTML5 / CSS3 / Vanilla JavaScript Frontend
*   **Rationale**: Provides an interactive multi-turn chat UI with live performance telemetry that runs on the laptop's Intel integrated GPU without allocating discrete GPU VRAM.
*   **Proposed Changes**:
    *   Implement a single-file chat interface using standard DOM elements (zero `<canvas>`, zero WebGL, zero external CDN dependencies).
    *   Stream responses from `POST /api/chat` via `fetch` `ReadableStream`, parsing `data: {...}\n\n` SSE frames and displaying real-time generated text alongside a telemetry bar showing Model Mode (`FP16` / `W4A16`), Allocated dGPU VRAM (`MB`), Cached Prefix Tokens, TTFT (`ms`), Live Decode Speed (`tok/s`), and Achieved Memory Bandwidth (`GB/s` & `%` of `192 GB/s`).

#### 21. `./tests/test_kernels.cu` (Create)

*   **Type / Scope**: CUDA / C++20 Unit & Integration Test Suite
*   **Rationale**: Verifies 100% of custom CUDA kernels against the golden PyTorch reference tensors from `./tools/verify_reference.py` and validates zero-update CUDA Graph parity.
*   **Proposed Changes**:
    *   Parse `--golden` (default `models/golden_ref.bin`), `--fp16` (default `models/qwen25_1.5b_fp16.qwenbin`), and `--w4a16` (default `models/qwen25_1.5b_w4a16.qwenbin`).
    *   Execute 8 isolated and end-to-end verification suites:
        1. `embed_lookup_fp16` & `embed_lookup_w4a16` (`M = 1` and `M = 8` against `embed_out`; FP16 `max_abs_err == 0.0f`, W4A16 cosine similarity `>= 0.998`)
        2. `rmsnorm_fp16` (`max_abs_err <= 1e-3` against `layer0_input_norm` and `layer0_post_norm`)
        3. `qkv_bias_rope_kv_append` (`max_abs_err <= 1e-3` against `layer0_q_rope`, `layer0_k_rope`, and `layer0_v_biased`)
        4. `gemv_fp16` & `gemv_dual_gate_up_swiglu_fp16` (`max_abs_err <= 5e-3` against `layer0_q_raw`, `layer0_swiglu_out`, and `layer0_mlp_out`)
        5. `gemv_w4a16` & `gemv_dual_gate_up_swiglu_w4a16` (relative cosine similarity `>= 0.995` against FP16 reference)
        6. `gemm_prefill_fp16`, `gemm_prefill_w4a16`, `gemm_prefill_dual_gate_up_swiglu_fp16`, & `gemm_prefill_dual_gate_up_swiglu_w4a16` (`M = 8` prompt batch against `golden_ref.bin`; FP16 `max_abs_err <= 5e-3`, W4A16 cosine similarity `>= 0.995`)
        7. `attention_gqa_flash_decode` vs. `attention_gqa_prefill` (`max_abs_err <= 2e-3` against `layer0_attn_out` and across synthetic sequence lengths `1`, `7` [empty-split guard check], `17`, `256`, and `1024`)
        8. `sample_and_advance` (`argmax` exact match on `final_logits_last` + Gumbel-Max reproducibility under fixed `rng_seed`) and end-to-end 28-layer greedy token match between uncaptured stream execution, captured `cudaGraphExec_t` execution, and `GoldenRefHeader::greedy_tokens[32]`.

### Verification Plan

*   **Compilation/Build**:
    *   `cmake -B build -DCMAKE_BUILD_TYPE=Release`
    *   `cmake --build build --config Release --parallel` (must compile with zero errors under MSVC + NVCC `sm_86`).
*   **Unit Tests**:
    *   `python tools/export_qwen25.py --mode fp16 --out models/qwen25_1.5b_fp16.qwenbin --tokenizer-out models/tokenizer.bin`
    *   `python tools/export_qwen25.py --mode w4a16 --out models/qwen25_1.5b_w4a16.qwenbin --tokenizer-out models/tokenizer.bin`
    *   `python tools/verify_reference.py --out models/golden_ref.bin`
    *   `./build/Release/test_kernels.exe --golden models/golden_ref.bin --fp16 models/qwen25_1.5b_fp16.qwenbin --w4a16 models/qwen25_1.5b_w4a16.qwenbin` (all 8 kernel and graph parity suites must pass with exit code `0`).
*   **Static Analysis & Linting**:
    *   Build with NVCC `-Xcompiler "/W4"` and run `compute-sanitizer --tool memcheck ./build/Release/test_kernels.exe`, `compute-sanitizer --tool initcheck ./build/Release/test_kernels.exe`, and `compute-sanitizer --tool racecheck ./build/Release/test_kernels.exe` to verify zero out-of-bounds global/shared memory accesses, zero uninitialized device reads, and zero shared-memory race conditions.
*   **Test Coverage Requirement**:
    *   100% of custom CUDA kernels (`embed_lookup_fp16`, `embed_lookup_w4a16`, `rmsnorm_fp16`, `qkv_bias_rope_kv_append`, `gemv_fp16`, `gemv_w4a16`, `gemv_dual_gate_up_swiglu_fp16`, `gemv_dual_gate_up_swiglu_w4a16`, `gemm_prefill_fp16`, `gemm_prefill_w4a16`, `gemm_prefill_dual_gate_up_swiglu_fp16`, `gemm_prefill_dual_gate_up_swiglu_w4a16`, `attention_gqa_flash_decode`, `attention_gqa_prefill`, `sample_and_advance`) and CUDA Graph capture/replay verified by suites 1–8 of `./tests/test_kernels.cu`.
*   **E2E/Integration**:
    *   Run `./build/Release/cli_bench.exe --model models/qwen25_1.5b_fp16.qwenbin --prompt-len 128 --gen-len 256` and verify VRAM `<= 3,300 MB` and decode throughput `>= 46 tokens/sec` (`>= 75%` of 192 GB/s).
    *   Run `./build/Release/cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256` and verify VRAM `<= 1,050 MB` and decode throughput `>= 150 tokens/sec`, comparing directly against `llama-bench -m qwen2.5-1.5b-instruct-q4_0.gguf -p 128 -n 256`.
    *   Start `./build/Release/qwen_server.exe --model models/qwen25_1.5b_w4a16.qwenbin --port 8080`, send two consecutive multi-turn chat requests to `http://127.0.0.1:8080/api/chat`, and verify SSE token streaming, KV prefix cache hit on turn 2, and `0 MB` additional discrete GPU VRAM allocation when opening `http://127.0.0.1:8080` in the browser.

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
