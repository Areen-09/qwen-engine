# Outperforming llama.cpp in Prefill and Decode on RTX 3050 Laptop (sm_86)

*   **Author:** Areen-09
*   **Status:** Draft
*   **Date:** 2026-11-07

--------------------------------------------------------------------------------

## Overview

This document specifies the architectural and kernel upgrades required for `qwen25_cuda_engine` (`Qwen2.5-1.5B-Instruct` on an NVIDIA GeForce RTX 3050 Laptop GPU, Ampere `sm_86`, 16 Streaming Multiprocessors, 4 GB GDDR6 at 192 GB/s theoretical bandwidth) to outperform `llama.cpp` (`GGUF Q4_K_M`, build `v12175` with CUDA 13.5 backend) across both prompt prefill (Time-To-First-Token) and single-stream autoregressive token generation (`batch_size = 1`).

The engine already operates with a smaller memory footprint than `llama.cpp` (`879.34 MB` total discrete GPU VRAM and `0 MB` host CPU weight offload vs. `951.36 MB` GPU VRAM plus `125.19 MB` host CPU buffer in `llama.cpp`), zero external math runtime libraries (`cuBLAS`, `cuDNN`, `CUTLASS`), a single static `GpuArena` allocation, and a zero-update captured CUDA Graph (`cudaGraphExec_t`). However, empirical benchmarks (`../../comparison.md`) show that `qwen25_cuda_engine` currently trails `llama.cpp` in two areas:
1.  **Prompt Prefill (`pp128` TTFT):** `1,436.67 ms` (`89.1 tok/s`) vs. `33.35 ms` (`3,838.0 tok/s`) in `llama.cpp` — a 43x latency gap.
2.  **Single-Stream Decode (`tg256`, `batch_size = 1`):** `116.50 tok/s` (`8.58 ms/tok`, `93.21 GB/s`, `48.55%` memory bus utilization) vs. `125.58 tok/s` (`7.96 ms/tok`, `100.46 GB/s`, `52.32%` memory bus utilization) in `llama.cpp` — a 7.8% throughput gap.

This design closes both gaps without linking external libraries, without altering the `.qwenbin` v1 model file format on disk, and without increasing the `879.34 MB` static GPU arena allocation.

## The Problem

Running `Qwen2.5-1.5B-Instruct` in 4-bit weight-only quantized mode (`W4A16_G128`) on a 16-SM RTX 3050 Laptop GPU under Windows WDDM currently hits four bottlenecks in `qwen25_cuda_engine`:

1.  **Scalar Column-per-Block Prefill Matrix Multiplication (`1,436.67 ms` for 128 tokens):**
    In `../../src/kernels/gemm_prefill.cu`, `gemm_prefill_w4a16_kernel` and `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel` launch a 2D grid of `dim3(N, M / 16)` where each 128-thread block computes only a single output column (`BN = 1`) across 16 token rows (`BM = 16`). For a 128-token prompt (`8` row tiles), the kernel streams the entire 580 MB transformer weight buffer from GDDR6 eight separate times (`~4.64 GB` of weight traffic), reloads the `[16, K]` activation tile from global/L2 memory `14,080` times per layer across linear projections, unpacks individual 4-bit nibbles via a per-thread `switch (nibble_idx)` branch (`unpack_single_int4`), and executes two `__syncthreads()` barriers per 128-element reduction step without using Ampere Tensor Cores. In addition, `QwenEngine::prefill()` (`../../src/engine.cu`) executes a blocking `cudaStreamSynchronize(stream_)` inside the chunk loop before launching the final token projection.
2.  **ALU Instruction Bottleneck and 1-Row-per-Warp Activation Reloads in W4 Decode (`116.50 tok/s`):**
    While the engine's FP16 decode kernel achieves `143.22 GB/s` (`74.60%` of the 192 GB/s memory bus), the W4A16 decode kernel in `../../src/kernels/gemv_w4a16.cu` drops to `93.21 GB/s` (`48.55%`). For every packed `uint32_t` weight word (8 INT4 weights), `unpack_8x_int4` executes 48+ scalar instructions (8 bit shifts, 8 bitwise masks, 8 `int-to-float` conversions, 8 FP32 zero-point subtractions, 8 FP32 scale multiplications, 8 `__half2float` activation conversions, and 8 FP32 multiply-adds). Furthermore, each warp computes only 1 output row (`4` rows per 128-thread block), forcing `down_proj` (`K = 8960, N = 1536`) to reload its `17.92 KB` activation vector into shared memory `384` times per layer (`192.6 MB` of redundant L2 traffic across 28 layers) and forcing `lm_head` (`N = 151936, K = 1536`) to reload its activation vector `37,984` times (`116.7 MB` of redundant L2 traffic).
3.  **Unfused Q, K, and V Projections and Uncovered Quantization Boundaries:**
    In `../../src/engine.cu`, `execute_decode_pass()` and `execute_prefill_chunk()` launch `q_proj` (`1536` rows), `k_proj` (`256` rows), and `v_proj` (`256` rows) as three separate kernels per layer (`56` extra kernel launches per token), reading the normalized input activation three times and under-filling the 16 SMs on the narrow 256-row K and V grids. Moreover, switching decode projections to hardware `__dp4a` (`W4A8` with `Q8_1` quantized activations) requires producing `Q8_1` inputs not only after `RMSNorm` (`QKV`, `Gate+Up`, `LM Head`), but also before `O-Proj` (fed by FlashDecoding Stage 2) and `Down-Proj` (fed by `gemv_dual_gate_up_swiglu`), which together account for `30.8%` of layer weight traffic (`225.7 MB`).
4.  **Inner-Loop Thread-Block Barriers in Grouped-Query Attention & Missing Stage Telemetry:**
    In `../../src/kernels/attention_gqa.cu`, both `attention_gqa_flash_decode_stage1_kernel` and `attention_gqa_prefill_kernel` assign 128 threads (`4` warps) to a single 128-element head (`1` scalar `half` per thread) and execute two `__syncthreads()` barriers inside the inner token loop `for (int t = ...)` on every cached token. In prefill, the kernel also launches 12 separate query-head blocks per token, reading the same KV cache vectors 6 times per GQA group. Finally, `../../src/cli_bench.cpp` measures only end-to-end prefill and decode wall time, providing no operator-level latency breakdown or prefill TFLOPS telemetry to isolate bottlenecks.

**Proposed Solution:**
Upgrade the decode path to a 4-row-per-warp `W4A8` (`__dp4a`) integer dot-product architecture backed by complete `Q8_1` activation producer coverage (`RMSNorm`, `FlashDecoding Stage 2`, and a pre-`Down-Proj` L2 quantizer) carved from unused buffer capacity inside the existing workspace arena, and merge `Q`, `K`, and `V` projections into a single virtual-row kernel launch (`QKVProjectionView`). Upgrade the prefill path in `../../src/kernels/gemm_prefill.cu` to a 2D-tiled (`BM = 64, BN = 64, BK = 32`) Ampere Tensor Core (`nvcuda::wmma` `m16n16k16` with FP32 accumulators) template architecture that dequantizes 4-bit weights into shared memory on the fly. Restructure `../../src/kernels/attention_gqa.cu` so a single warp (`32` lanes, `4` `half` elements per lane in registers) processes each token with zero inner-loop `__syncthreads()` barriers, and add per-stage profiling (`--profile-stages`) and prefill TFLOPS reporting to `../../src/cli_bench.cpp`.

## Goals

*   **Exceed `llama.cpp` in Single-Stream Decode Throughput (`batch_size = 1`):** Achieve `> 140.0 tokens/sec` (`< 7.14 ms` median inter-token latency, `>= 55%` effective memory bus utilization) on `qwen25_1.5b_w4a16.qwenbin` at context length 128 to 384 (`tg256`) on the RTX 3050 Laptop GPU, surpassing `llama.cpp` (`125.58 tokens/sec`).
*   **Exceed `llama.cpp` in Prompt Prefill Speed (`pp128` TTFT):** Achieve `< 30.0 ms` Time-To-First-Token (`> 4,260 tokens/sec`) for a 128-token prompt (`pp128`) on `qwen25_1.5b_w4a16.qwenbin`, surpassing `llama.cpp` (`33.35 ms` / `3,838 tokens/sec`).
*   **Zero External Math Libraries:** Implement all Tensor Core (`nvcuda::wmma`), integer dot-product (`__dp4a`), quantization, norm, and attention kernels in self-contained C++20 and CUDA (`sm_86`) without linking `cuBLAS`, `cuBLASLt`, `cuDNN`, or `CUTLASS`.
*   **Zero-Migration `.qwenbin` v1 & Static `GpuArena` Preservation:** Maintain 100% binary compatibility with existing `.qwenbin` v1 files (`magic = 0x5157454E, version = 1`) and keep total `GpuArena` allocation and `arena_->used_bytes()` unchanged (`879.34 MB` for W4A16, carving `14 KB` (`14,336` bytes) of `Q8_1` scratch pointers inside the existing workspace region alongside the `23.97 MB` of unused workspace slack).
*   **Zero-Update CUDA Graph & Numerical Verification Parity:** Preserve single-launch zero-update `cudaGraphExec_t` decode execution and pass all 8 verification suites in `../../tests/test_kernels.cu` (`max_abs_err <= 5e-3` for FP16 kernels, `cosine_similarity >= 0.990` for W4 linear projections, `>= 0.985` for W4 SwiGLU, and deterministic greedy token verification).
*   **Operator-Level Telemetry & Launch Diagnostics:** Provide a `--profile-stages` mode in `../../src/cli_bench.cpp` reporting per-operator prefill/decode latency, effective GB/s, and prefill TFLOPS, paired with synchronous `CUDA_KERNEL_CHECK()` launch diagnostics.

## Non-Goals

*   **External BLAS or Template Library Integration:** Linking `cuBLAS`, `cuBLASLt`, `CUTLASS`, or `TensorRT` is excluded.
*   **Modifying the `.qwenbin` v1 File Format or Re-Exporting Models:** Altering `QWQwenHeader`, reordering tensors on disk, or requiring users to re-run `../../tools/export_qwen25.py` is excluded.
*   **Multi-Sequence Dynamic Batching (`batch_size > 1` Decode):** Continuous batching and PagedAttention remain out of scope.
*   **INT8 KV Cache Quantization (`Q8_0` KV Cache):** Modifying the contiguous FP16 `[NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]` KV cache layout is excluded to avoid short-context dequantization overhead and preserve prefix-cache compatibility.
*   **Tensor Core FlashAttention-2 Rewrite for Prefill:** Because linear projections account for over 95% of `pp128` prefill runtime and warp-per-token register GQA attention with 6-head sharing completes in under `0.2 ms`, a Tensor Core FlashAttention-2 kernel is excluded.

## Open Questions

*   *None.* All architectural and compatibility boundaries were resolved during the `/egm-brainstorm` phase (`../../concepts_brief.md`).

## Critical User Journeys (CUJs)

### CUJ 1: Drop-In Verification of Existing `.qwenbin` v1 Models & Golden Reference Suite

*   **Given**: An existing exported `models/qwen25_1.5b_fp16.qwenbin`, `models/qwen25_1.5b_w4a16.qwenbin`, and `models/golden_ref.bin` on the RTX 3050 Laptop GPU without re-running Python export scripts.
*   **When**: The developer builds the `Release` targets and executes `./build/Release/test_kernels.exe`.
*   **Then**: `test_kernels.exe` loads the unmodified `.qwenbin` v1 files, validates all isolated and end-to-end FP16 and W4A8/W4A16 kernels (including `M = 8` Tensor Core prefill, fused `RMSNorm + Q8_1` round-trip accuracy, 4-row-per-warp `__dp4a` GEMV, fused `QKVProjectionView`, and 28-layer CUDA Graph replay) against `golden_ref.bin`, prints cosine similarity and max absolute error metrics, and exits with code `0`.

### CUJ 2: Head-to-Head Benchmark Verification Against `llama.cpp` (`cli_bench.exe`)

*   **Given**: Compiled `./build/Release/cli_bench.exe` and `models/qwen25_1.5b_w4a16.qwenbin` on the RTX 3050 Laptop GPU.
*   **When**: The user runs `./build/Release/cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 5 --runs 10 --profile-stages`.
*   **Then**: The engine reports the exact same `879.34 MB` static arena footprint, completes 128-token prefill in `< 30.0 ms` (`> 4,260 tok/s`, beating `llama.cpp`'s `33.35 ms`), executes single-stream decode at `> 140.0 tok/s` (`< 7.14 ms/tok`, beating `llama.cpp`'s `125.58 tok/s`), and outputs a per-stage breakdown of prefill/decode operator latencies, achieved GB/s, and prefill TFLOPS.

## Technical Plan (Architecture)

### 1. Static `GpuArena` Sub-Allocation for `Q8_1` Activations (Zero Total VRAM Increase)

In `../../src/engine.cu`, `QwenEngine::setup_arena()` reserves `workspace_bytes_ = 32ULL << 20` (`32.00 MB`, `33,554,432` bytes), of which existing `WorkspaceBuffers` sub-allocations (`d_act_0`, `d_act_1`, `d_q`, `d_k`, `d_v`, `d_attn_out`, `d_mlp_act`, `d_splitk_out`, `d_splitk_lse`, `d_logits`, `d_sample_val`, `d_sample_idx`, `d_prompt_tokens`) consume `8,417,792` bytes (`8.03 MB`), leaving `25,136,640` bytes (`23.97 MB`) of unused capacity inside the already-allocated `GpuArena`. Furthermore, `QwenEngine::get_memory_stats()` reports `arena_->used_bytes()` (`922,054,912` bytes = `879.34 MB` in W4A16 mode) as `Total Discrete VRAM Used`.

To support hardware `__dp4a` (`W4A8`) decode while keeping both `arena_->capacity_bytes()` and `arena_->used_bytes()` (`879.34 MB`) byte-for-byte unchanged and preserving `cudaGraphExec_t` address stability, `WorkspaceBuffers` (`../../include/engine.cuh`) carves three 256-byte-aligned `Q8_1` scratch pointers (`14,336` bytes = `14 KB` total, sized for `MAX_M = 1` decode and up to `INTERMEDIATE_SIZE = 8960` elements) at fixed offsets inside the unused upper portion of `ws_.d_act_1` (`[256, 1536]` FP16 = `786,432` bytes, of which single-token decode uses at most the first `3,072` bytes and prefill chunks finish before the final-token `LM Head` step):
*   `int8_t* d_act_q8`: `8,960` bytes (aligned to `9,216` bytes at `reinterpret_cast<uint8_t*>(ws_.d_act_1) + 65536`) storing quantized `int8_t` activations $x_{q,i} \in [-127, 127]$.
*   `float* d_act_scale`: `70 * sizeof(float) = 280` bytes (aligned to `512` bytes at `reinterpret_cast<uint8_t*>(ws_.d_act_1) + 74752`) storing per-group-128 scale factors $d_g = \max_{i \in g}(|x_i|) / 127.0\text{f}$ (`12` groups for `K = 1536`, `70` groups for `K = 8960`).
*   `int32_t* d_act_sum8`: `(8960 / 8) * sizeof(int32_t) = 4,480` bytes (aligned to `4,608` bytes at `reinterpret_cast<uint8_t*>(ws_.d_act_1) + 75264`) storing precomputed 8-element scaled zero-point offsets:
    $$\text{sum8}_{k/8} = -8 \sum_{j=0}^{7} x_{q, k+j}$$

```text
+---------------------------------------------------------------------------------------+
|                   EXISTING 32 MB WORKSPACE REGION IN GPU ARENA                        |
+---------------------------------------------------+-----------------------------------+
| Existing Workspace Sub-Allocations (8.03 MB)      | Unused Workspace Slack (23.97 MB) |
| d_act_0, d_act_1 [786 KB -> carves 14 KB Q8_1     | Available inside 32 MB reservation|
|   at +64 KB: d_act_q8, d_act_scale, d_act_sum8],  | (0 extra arena_->allocate() calls;|
| d_q, d_k, d_v, d_attn_out, d_mlp_act, d_logits    |  used_bytes stays 879.34 MB)      |
+---------------------------------------------------+-----------------------------------+
```

### 2. End-to-End Decode & Prefill Execution Data Flow

```text
+=======================================================================================+
| A. DECODE PATH: ZERO-UPDATE CAPTURED CUDA GRAPH (batch_size = 1, W4A8 __dp4a)         |
+=======================================================================================+
  [DeviceControlBlock: d_token_id, d_seq_pos]
       |
       v
  (1) launch_embed_lookup_w4a16  ------------------> writes d_act_0 [1536 half]
       |
       +--- Layer l = 0 .. 27 -------------------------------------------------------+
       | (2) launch_rmsnorm_q8_1(d_act_0, norm_w)                                    |
       |       -> writes d_act_q8 [1536 int8], d_act_scale [12], d_act_sum8 [192]    |
       |                                                                             |
       | (3) launch_gemv_qkv_w4a8(QKVProjectionView, d_act_q8, d_act_scale, sum8)    |
       |       -> single 2048-row grid (4 rows/warp __dp4a) writes d_q, d_k, d_v     |
       |                                                                             |
       | (4) launch_qkv_bias_rope_kv_append(d_q, d_k, d_v, layer_k, layer_v)         |
       |       -> fast exp2f + __sincosf RoPE, appends to FP16 KV Cache              |
       |                                                                             |
       | (5) launch_attention_gqa_flash_decode(d_q, layer_k, layer_v)                |
       |       -> Stage 1: 1 warp/token uint2 loads, 0 inner-loop __syncthreads()    |
       |       -> Stage 2 epilogue: writes d_attn_out AND quantizes to d_act_q8      |
       |                                                                             |
       | (6) launch_gemv_w4a8(o_weight, d_act_q8, residual=d_act_0 -> d_act_0)       |
       |                                                                             |
       | (7) launch_rmsnorm_q8_1(d_act_0, post_norm_w)                               |
       |       -> writes d_act_q8 [1536 int8], d_act_scale [12], d_act_sum8 [192]    |
       |                                                                             |
       | (8) launch_gemv_dual_gate_up_swiglu_w4a8(gate_up_w, d_act_q8 -> d_mlp_act)  |
       |       -> 2 intermediate outputs (4 weight rows) per warp via __dp4a         |
       |                                                                             |
       | (9) launch_quantize_fp16_to_q8_1(d_mlp_act [8960 half] -> d_act_q8 [8960])  |
       |       -> 35 blocks of 32 threads (2 groups/block, < 2.5 us in L2 cache)     |
       |                                                                             |
       | (10) launch_gemv_w4a8(down_weight, d_act_q8, residual=d_act_0 -> d_act_0)   |
       +-----------------------------------------------------------------------------+
       |
       v
  (11) launch_rmsnorm_q8_1(d_act_0, final_norm_w) -> writes d_act_q8 [1536 int8]
       |
       v
  (12) launch_gemv_w4a8(embed_weight [151936, 1536], d_act_q8 -> d_logits)
       |
       v
  (13) launch_sample_and_advance(d_logits -> d_control->token_id, *h_pinned_token)

+=======================================================================================+
| B. PREFILL PATH: 2D-TILED AMPERE TENSOR CORE WMMA (BM=64, BN=64, BK=32)               |
+=======================================================================================+
  [Uncached Prompt Suffix Copied Once via cudaMemcpyAsync to ws_.d_prompt_tokens + start_pos]
       |
       +--- For each Chunk (chunk_start, chunk_len <= 256) --------------------------+
       | (0) launch_embed_lookup_w4a16(ws_.d_prompt_tokens + chunk_start -> d_act_0) |
       |       +--- Layer l = 0 .. 27 ---------------------------------------------+ |
       |       | (1) launch_rmsnorm_fp16(d_act_0 -> d_act_1 [M, 1536])             | |
       |       | (2) launch_gemm_prefill_qkv_w4a16(QKVProjectionView -> d_q,k,v)   | |
       |       |       -> 2D WMMA m16n16k16 (FP32 accum), W4->half in s_B[64][40]  | |
       |       | (3) launch_qkv_bias_rope_kv_append(M tokens)                      | |
       |       | (4) launch_attention_gqa_prefill(6 Q heads/KV head, warp/token)   | |
       |       | (5) launch_gemm_prefill_w4a16(o_weight, d_attn_out, res=d_act_0)  | |
       |       | (6) launch_rmsnorm_fp16(d_act_0 -> d_act_1 [M, 1536])             | |
       |       | (7) launch_gemm_prefill_dual_gate_up_swiglu_w4a16(-> d_mlp_act)   | |
       |       | (8) launch_gemm_prefill_w4a16(down_weight, d_mlp_act, res=d_act_0)| |
       |       +-------------------------------------------------------------------+ |
       +-----------------------------------------------------------------------------+
       |
       v
  (9)  launch_rmsnorm_q8_1(last_token_act, final_norm_w -> d_act_q8 [1536 int8])
       |
       v
  (10) launch_gemv_w4a8(embed_weight [151936, 1536], d_act_q8 -> d_logits)
       |
       v
  (11) launch_sample_and_advance(d_logits -> d_control->token_id, *h_pinned_token)
```

### 3. Shared Hardware Unpacking & Quantization Primitives (`../../include/quant_primitives.cuh`)

In `.qwenbin` v1, each `uint32_t` word in a `W4A16_G128` weight tensor packs 8 unsigned 4-bit weights ($q_0 \dots q_7 \in [0, 15]$) in interleaved nibble order:
$$\text{packed} = q_0 \mid (q_4 \ll 4) \mid (q_1 \ll 8) \mid (q_5 \ll 12) \mid (q_2 \ll 16) \mid (q_6 \ll 20) \mid (q_3 \ll 24) \mid (q_7 \ll 28)$$
with real value $w_j = (q_j - 8) \cdot s_w$.

To eliminate duplicate bit-shift and quantization code across `../../src/kernels/gemv_w4a16.cu`, `../../src/kernels/gemm_prefill.cu`, and `../../src/kernels/rmsnorm_rope.cu`, a new header `../../include/quant_primitives.cuh` provides three shared inline device primitives:

1.  **`dp4a_w4_q8_oct(uint32_t packed, int32_t x_lo, int32_t x_hi, int32_t x_sum8)` (For `W4A8` Decode):**
    *   Extracts the lower nibbles `{q0, q1, q2, q3}` (`packed & 0x0F0F0F0Fu`) and upper nibbles `{q4, q5, q6, q7}` (`(packed >> 4) & 0x0F0F0F0Fu`) as unsigned `int8x4` byte vectors in two bitwise instructions.
    *   To avoid two's-complement byte-borrow corruption (which occurs if `0x08080808` is subtracted from `w_lo` when any nibble is $< 8$), `w_lo` and `w_hi` remain unsigned in $[0, 15]$ and the precomputed zero-point offset $-8 \sum_{j=0}^{7} x_{q,j} = \text{x\_sum8}$ is passed into the accumulator seed of the first `__dp4a(w_lo, x_lo, x_sum8)` call, followed by `__dp4a(w_hi, x_hi, dot)`. This evaluates $\sum_{j=0}^{7} (q_j - 8) x_{q,j}$ in 4 integer instructions total.
2.  **`quantize_group128_fp16_to_q8_1(...)` (Shared Half-Warp `FP16 -> Q8_1` Quantizer):**
    *   Shared 16-lane half-warp helper (`8` elements per lane = `128` elements per group) that reduces `max_abs` via 4 half-warp `__shfl_xor_sync` steps, computes $d = \max(\text{max\_abs} / 127.0\text{f}, 10^{-8}\text{f})$, packs 8 quantized `int8_t` values into `(x_lo, x_hi)`, computes `x_sum8 = compute_q8_sum8(x_lo, x_hi)`, and writes `(out_q8, out_scale, out_sum8)` to either global or shared memory.
3.  **`unpack_w4_to_half2_quad(uint32_t packed, half scale, half2 out_h2[4])` (For Tensor Core Prefill & Embedding Lookup):**
    *   Unpacks `packed` into four `half2` pairs `{(q0-8)*s, (q1-8)*s}`, `{(q2-8)*s, (q3-8)*s}`, `{(q4-8)*s, (q5-8)*s}`, `{(q6-8)*s, (q7-8)*s}` for coalesced `uint4` (`8 x half`) writes into shared memory `s_B` during Tensor Core prefill and `embed_lookup_w4a16`.

### 4. Complete `Q8_1` Activation Producer Pipeline (`../../src/kernels/rmsnorm_rope.cu`, `../../src/kernels/attention_gqa.cu`)

Every linear projection in the 28-layer decode pass receives pre-quantized `Q8_1` inputs (`d_act_q8`, `d_act_scale`, `d_act_sum8`):
1.  **Fused `RMSNorm + Q8_1` (`launch_rmsnorm_q8_1`):**
    *   Launches `1` thread block of `256` threads (`8` warps) for `M = 1` decode (`HIDDEN_SIZE = 1536`), computes FP32 RMSNorm into shared memory (optionally writing `half* output_fp16` if non-null), and invokes `quantize_group128_fp16_to_q8_1` across the `12` groups (`128` elements per group) to populate `d_act_q8`, `d_act_scale`, and `d_act_sum8`.
2.  **Fused `FlashDecoding Stage 2 + Q8_1` (`launch_attention_gqa_flash_decode`):**
    *   In `attention_gqa_flash_decode_stage2_kernel`, `gridDim.x = NUM_Q_HEADS = 12`, and each 128-thread block owns one 128-element query head (`group_idx = q_h`), which aligns 1-to-1 with `QUANT_GROUP_SIZE = 128`.
    *   After combining Split-K partial outputs into the 128 FP32 head elements, when `out_q8 != nullptr`, the block reduces `max_abs` across all 128 threads, writes the FP16 output to `d_attn_out`, and simultaneously writes quantized group `q_h` to `d_act_q8`, `d_act_scale[q_h]`, and `d_act_sum8[q_h * 16 .. q_h * 16 + 15]` for `O-Proj`.
3.  **Standalone L2-Resident `quantize_fp16_to_q8_1_kernel` (`launch_quantize_fp16_to_q8_1`):**
    *   Launched before `Down-Proj` (`K = 8960`, `70` groups of `128`) with `grid = num_elements / 256` (`35` blocks for `8960`) and `block = 32` threads (`2` half-warps = `2` groups = `256` elements per block).
    *   Each half-warp (`16` threads) loads one 128-element group (`uint4` per thread) of `d_mlp_act` from the RTX 3050's 2 MB L2 cache and invokes `quantize_group128_fp16_to_q8_1` to write `d_act_q8`, `d_act_scale`, and `d_act_sum8` in `< 2.5 microseconds`.
    *   When `launch_gemv_w4a16` or `launch_gemv_dual_gate_up_swiglu_w4a16` is called standalone with raw `const half* input_x` by `../../tests/test_kernels.cu` (where global `d_act_q8` buffers are not passed), the GEMV thread block invokes `quantize_group128_fp16_to_q8_1` cooperatively into shared memory (`s_x_q8`, `s_x_scale`, `s_x_sum8`) at the start of the kernel.

### 5. 4-Row-per-Warp `W4A8` (`__dp4a`) Decode GEMV & Virtual-Row `QKVProjectionView` (`../../src/kernels/gemv_w4a16.cu`)

1.  **4-Row-per-Warp Register Tile (`gemv_w4a8_kernel`):**
    *   Uses `BLOCK_THREADS = 128` (`4` warps per block) and `ROWS_PER_WARP = 4` (`16` output rows per block, `4x` more rows per block than the baseline 1-row-per-warp kernel).
    *   At the start of the block, the 128 threads cooperatively stage `d_act_q8`, `d_act_scale`, and `d_act_sum8` into shared memory (`s_x_q8`, `s_x_scale`, `s_x_sum8`). Total shared memory is `2,352` bytes for `K = 1536` and `13,720` bytes for `K = 8960` (23% smaller than the baseline FP16 shared-memory buffer, allowing 6–8 active blocks per SM on `sm_86`).
    *   In each warp step along `K` (`256` elements = `2` groups per step across 32 lanes), each lane loads `(x_lo, x_hi, x_sum8, x_d)` from shared memory once, then evaluates 4 output rows using coalesced 128-byte warp weight loads and `dp4a_w4_q8_oct`, followed by a 32-lane warp reduction via `__shfl_down_sync`.
2.  **2-Output (4-Row) per Warp Fused SwiGLU (`gemv_dual_gate_up_swiglu_w4a8_kernel`):**
    *   Each warp computes `2` intermediate features (`4` weight rows: `gate0, up0, gate1, up1`) per shared-memory activation load via `dp4a_w4_q8_oct`, then evaluates $\text{SiLU}(\text{gate}_0) \cdot \text{up}_0$ and $\text{SiLU}(\text{gate}_1) \cdot \text{up}_1$ in lane 0.
3.  **Single-Launch Virtual-Row `QKVProjectionView` (`gemv_qkv_w4a8_kernel` & `gemv_qkv_fp16_kernel`):**
    *   Defines a descriptor struct in `../../include/kernels.cuh`:
        ```cpp
        struct QKVProjectionView {
            const void* q_weight; const half* q_scale; half* d_q;
            const void* k_weight; const half* k_scale; half* d_k;
            const void* v_weight; const half* v_scale; half* d_v;
        };
        ```
    *   Launches a single grid covering `Q_DIM + 2 * KV_DIM = 2048` rows (`128` blocks of `16` rows). Because `1536` (`96` blocks) and `256` (`16` blocks) are exact multiples of `16` (and `64` in prefill), blocks `0..95` resolve `(q_weight, q_scale, d_q)`, blocks `96..111` resolve `(k_weight, k_scale, d_k)`, and blocks `112..127` resolve `(v_weight, v_scale, d_v)` with zero intra-block warp divergence.

### 6. 2D-Tiled Ampere Tensor Core (`nvcuda::wmma`) Prefill GEMM (`../../src/kernels/gemm_prefill.cu`)

All four prefill projection kernels (`fp16`, `w4a16`, `qkv`, and `dual_gate_up_swiglu`) are unified around a 2D-tiled Ampere Tensor Core architecture using `<mma.h>` (`nvcuda::wmma`):
*   **Block & Warp Tile Geometry:**
    *   Thread block size: `128` threads (`4` warps per block).
    *   Block tile: `BM = 64` token rows $\times$ `BN = 64` output columns $\times$ `BK = 32` reduction elements per step (`grid = dim3((N + 63) / 64, (M + 63) / 64)`).
    *   Warp tile: Warp `w` (`0..3`, arranged as a $2 \times 2$ warp grid with `warp_m = w / 2`, `warp_n = w % 2`) computes a `32 x 32` output sub-tile using $2 \times 2 = 4$ `wmma::fragment<wmma::accumulator, 16, 16, 16, float>` fragments.
*   **Bank-Conflict-Free Shared Memory Union (`10.0 KB` / `10,240 B` total, well under the `48 KB` `sm_86` static limit):**
    ```cpp
    union PrefillSmem {
        struct {
            alignas(16) half s_A[64][40]; // [BM=64][BK=32 + 8 skew] = 5,120 B
            alignas(16) half s_B[64][40]; // [BN=64][BK=32 + 8 skew] = 5,120 B
        } mainloop;                       // 10,240 B (10.0 KB)
        alignas(16) float s_C[4][16][18]; // Per-warp 16x16 staging with +2 skew = 4,608 B
    };
    ```
    Padding the inner stride of `s_A` and `s_B` to `40` `half` elements (`80` bytes, an odd multiple of 16 bytes) satisfies `wmma::load_matrix_sync`'s 8-`half` alignment requirement while eliminating shared-memory bank conflicts.
*   **Mainloop (FP16 and On-the-Fly Dequantized W4A16):**
    *   In each `kt` step (`0 .. K / 32 - 1`, `k_offset = kt * 32`):
        1.  **Load `s_A` (`[64, 32]`):** The 128 threads cooperatively load `64 * 32 = 2048` `half` elements (`16` `half`s per thread) from `input_x`, zero-padding `s_A` with `0.0h` whenever `m_global >= M`.
        2.  **Load / Dequantize `s_B` (`[64, 32]`):** In `FP16` mode, threads load `uint4` vectors directly from `weight`. In `W4A16` mode, each thread loads `2` packed `uint32_t` weight words, unpacks them via `unpack_w4_to_half2_quad`, and writes `8` `half`s (`uint4`) into `s_B`.
        3.  **Tensor Core MMA (`k_sub = 0, 16`):** After `__syncthreads()`, each warp loads `wmma::matrix_a` (`row_major`, `ldm = 40`) from `s_A` and `wmma::matrix_b` (`col_major`, `ldm = 40`, representing $W^T$) from `s_B`, and executes `wmma::mma_sync` into FP32 accumulators.
*   **Epilogue (Standard Linear & Fused `Dual Gate+Up + SwiGLU`):**
    *   After the `kt` loop completes, a `__syncthreads()` barrier ensures all warps have finished reading `mainloop` (`s_A`, `s_B`) before aliasing `smem.s_C`.
    *   For standard projections, each warp stores its $2 \times 2$ `16x16` FP32 fragments through `smem.s_C[warp_id]` and writes valid rows (`m_global < M && n_global < N`) to `output_y`, adding `residual` in FP32 when provided.
    *   For `gemm_prefill_dual_gate_up_swiglu_*`, `gate_up_weight` has `2 * INTERMEDIATE_SIZE = 17,920` rows where row `2*r` is `gate[r]` and row `2*r + 1` is `up[r]`. Because each `16x16` accumulator fragment spans 16 contiguous columns (`8` adjacent `(gate, up)` pairs `2*p` and `2*p + 1` for `p = 0..7`), each warp stages its four `16x16` fragments sequentially through its own `smem.s_C[warp_id][16][18]` buffer, evaluates $\text{SiLU}(\text{gate}) \cdot \text{up}$ in FP32, and writes `8` activated `half` outputs per row per `16x16` sub-tile (`32` outputs per block row) to `output_mlp`.

### 7. Warp-per-Token Vectorized GQA Attention & Per-Stage Telemetry (`../../src/kernels/attention_gqa.cu`, `../../src/cli_bench.cpp`)

1.  **Zero-Barrier Warp-per-Token GQA Loop (`attention_gqa_flash_decode_stage1_kernel` & `attention_gqa_prefill_kernel`):**
    *   Instead of spreading 128 threads (`4` warps) across a single `HEAD_DIM = 128` vector and calling `__syncthreads()` twice per token `t`, **each 32-lane warp owns a complete `HEAD_DIM = 128` vector** by assigning `4` contiguous `half` elements (`d_base = lane_id * 4`, loaded as a 64-bit `uint2`) to each lane.
    *   All `GQA_GROUP_SIZE = 6` query heads belonging to a KV head (`kv_h in {0, 1}`) are loaded into per-lane FP32 registers `float q_reg[6][4]` (`24` registers per thread) at the start of the kernel.
    *   In the token loop, the 4 warps in a 128-thread block stride across cached tokens by 4 (`t = split_start + warp_id; t < split_end; t += 4` in FlashDecoding Stage 1; `t = warp_id; t <= cur_pos; t += 4` in Prefill GQA), reducing `Q · K` across the 32 lanes via 5 steps of `__shfl_xor_sync` with **zero inner-loop `__syncthreads()` barriers** and updating online softmax states `(m_i[h], d_i[h], o_reg[h][0..3])`.
    *   After the loop, a single cross-warp reduction in shared memory (`s_m[6][4]`, `s_d[6][4]`, `s_o[4][128]`) merges the 4 warps' states while guarding inactive warps (`if (s_d[h][w] > 0.0f)`) so short contexts (`< 4` tokens per split or `cur_pos < 3`) never evaluate `expf(-INFINITY - (-INFINITY)) = NaN`.
    *   Applying this 6-head GQA grouping to `attention_gqa_prefill_kernel` (`grid = dim3(NUM_KV_HEADS = 2, num_tokens)`) reduces prefill KV cache global memory reads by `6x`.
2.  **Fast Transcendental RoPE (`../../src/kernels/rmsnorm_rope.cu`) & Asynchronous Prefill Chunking (`../../src/engine.cu`):**
    *   In `qkv_bias_rope_kv_append_kernel`, replace `powf(ROPE_THETA, exponent)` + separate `cosf`/`sinf` calls with `exp2f(exponent * LOG2_ROPE_THETA)` (`LOG2_ROPE_THETA = 19.931568569324174f`) and hardware `__sincosf(angle, &sin_val, &cos_val)`.
    *   In `QwenEngine::prefill()` (`../../src/engine.cu:408-421`), copy the uncached prompt suffix to `ws_.d_prompt_tokens + start_pos` (within the `MAX_SEQ_LEN = 4096`-token buffer) in a single `cudaMemcpyAsync` before the chunk loop, update `execute_prefill_chunk(chunk_start, chunk_len)` to index `ws_.d_prompt_tokens + chunk_start`, remove the per-chunk `cudaStreamSynchronize(stream_)` inside the loop, and use `launch_rmsnorm_q8_1` + `launch_gemv_w4a8` for the final token's `LM Head` projection.
3.  **Per-Stage Telemetry (`--profile-stages`) & `CUDA_KERNEL_CHECK()` (`../../src/cli_bench.cpp`, `../../include/config.cuh`):**
    *   Add a `CUDA_KERNEL_CHECK()` macro invoked after every kernel launch to catch invalid launch configurations immediately during stream or graph capture.
    *   Add a `--profile-stages` CLI flag to `../../src/cli_bench.cpp` (backed by a `StageProfile` struct in `QwenEngine`) that uses `cudaEvent_t` pairs to report both prefill and decode per-operator latency breakdowns, achieved decode memory bandwidth (GB/s), and achieved prefill compute throughput (`TFLOPS = (2 * M * Total_Linear_MACs) / (prefill_sec * 1e12)`).

## Alternatives Considered

### Alternative A: Pure `W4A16` Decode with Bitwise `half2` (`__hfma2`) Unpacking Instead of `W4A8` (`__dp4a`)

*   **Pros**: Avoids dynamic `FP16 -> INT8 (Q8_1)` activation quantization and requires no `Q8_1` scratch buffers or pre-`Down-Proj` quantization kernel.
*   **Cons**: Requires 4 `__hfma2` instructions plus 4 bitwise LOP3/shift instructions per 8 weights (compared to 2 `__dp4a` instructions in `W4A8`), and requires loading twice as many activation bytes from shared memory (`3 KB` / `18 KB` in FP16 vs. `1.5 KB` / `9 KB` in INT8).
*   **Reason for Rejection**: Hardware `__dp4a` (`W4A8`) aligns directly with `.qwenbin` v1's existing split-nibble layout (`packed & 0x0F0F0F0F` and `(packed >> 4) & 0x0F0F0F0F`), cutting inner-loop ALU instruction issue count to the hardware minimum and maximizing `batch_size = 1` decode tokens/sec.

### Alternative B: Modifying `.qwenbin` v1 on Disk to Store Concatenated `QKV` Weights

*   **Pros**: Allows standard single-pointer GEMV and GEMM kernels to evaluate combined `Q`, `K`, and `V` projections without a `QKVProjectionView` descriptor.
*   **Cons**: Breaks binary backward compatibility with existing exported `.qwenbin` v1 files, forces users to re-run `../../tools/export_qwen25.py`, and invalidates per-tensor offset checks in `../../tests/test_kernels.cu`.
*   **Reason for Rejection**: Because `Q_DIM = 1536` and `KV_DIM = 256` are exact multiples of 64, passing a `QKVProjectionView` struct containing `(q_w, k_w, v_w)` achieves identical single-launch fusion with zero intra-block divergence and zero changes to `.qwenbin` v1 on disk.

### Alternative C: Raw PTX `mma.sync.aligned.m16n8k16` with Multi-Stage `cp.async` Software Pipelining for Prefill

*   **Pros**: Can extract an additional 10–15% compute utilization on large square matrices (`M >= 1024`).
*   **Cons**: Increases register pressure, approaches or exceeds the 48 KB static shared-memory ceiling on `sm_86` (especially in `dual_gate_up_swiglu`), and adds fragile inline PTX assembly across four kernel variants (`fp16`, `w4a16`, `qkv`, and `dual_gate_up_swiglu`).
*   **Reason for Rejection**: 2D-tiled `nvcuda::wmma` (`m16n16k16`, FP32 accumulator, `BM = 64, BN = 64, BK = 32`) uses `10.0 KB` (`10,240 B`) of shared memory, unifies FP16 and W4A16 prefill via C++20 templates, and drops `pp128` TTFT from `1,436.67 ms` to `~10–16 ms` (more than 2x faster than `llama.cpp`'s `33.35 ms`).

### Alternative D: Quantizing the Static KV Cache to `INT8 (Q8_0)`

*   **Pros**: Cuts the 4,096-token KV cache footprint in half (`114.69 MB -> ~59 MB`) and halves KV cache memory bandwidth at 4,096-token context lengths.
*   **Cons**: At short and medium benchmark contexts (`pp128` / `tg256`), KV cache reads represent `< 0.5%` of total per-token memory traffic (`3.7 MB` vs. `759 MB` weights), so INT8 KV dequantization instructions add ALU overhead during `tg256` while `qwen25_cuda_engine` already uses 72 MB less GPU VRAM (`+125 MB` less CPU RAM) than `llama.cpp`.
*   **Reason for Rejection**: Preserving the contiguous FP16 KV cache and upgrading `../../src/kernels/attention_gqa.cu` to warp-per-token `uint2` (`half2[2]`) register loads with zero inner-loop `__syncthreads()` accelerates both short-context and long-context attention without KV quantization loss.

## Risks, Limits & Mitigations

*   **Two's-Complement Byte-Borrow Corruption in `__dp4a` Unsigned INT4 Unpacking:**
    *   *Risk*: Subtracting `0x08080808` directly from packed `int8x4` weight bytes `w_lo` or `w_hi` causes a borrow from the adjacent byte whenever a 4-bit weight nibble is `< 8`, corrupting the upper bytes in the word.
    *   *Mitigation*: Keep `w_lo` and `w_hi` unsigned in $[0, 15]$ and fold the exact group zero-point offset $\text{x\_sum8} = -8 \sum_{j=0}^{7} x_{q,j}$ into the accumulator seed of `__dp4a(w_lo, x_lo, x_sum8)`.
*   **`sm_86` `48 KB` Static Shared-Memory Limit & WMMA Bank Conflicts:**
    *   *Risk*: Allocating separate mainloop and epilogue shared-memory arrays or unpadded `[64][32]` `half` tiles causes 4-way/8-way shared-memory bank conflicts during `wmma::load_matrix_sync`.
    *   *Mitigation*: Alias `mainloop` (`s_A[64][40]` + `s_B[64][40]` = `10,240 B`) with `s_C[4][16][18]` (`4,608 B`) inside `union PrefillSmem` (`10.0 KB` total) guarded by a post-mainloop `__syncthreads()` barrier, and pad the inner stride by `+8` `half` elements (`ldm = 40`).
*   **Out-of-Bounds Memory Access on Non-Multiple-of-64 Prompt Lengths (`M = 8` in `test_kernels.cu`):**
    *   *Risk*: A `BM = 64` Tensor Core tile reading past `M` rows on short prompts (`M = 8` in Suite 6 or tail chunks) causes illegal memory accesses or corrupts adjacent tokens.
    *   *Mitigation*: Guard global `s_A` loads with `m_global < M` (substituting `0.0h` when `m_global >= M`) and guard `output_y` / `output_mlp` epilogue writes with `m_global < M && n_global < N`.
*   **All-Zero Activation Group Division-by-Zero & Empty-Warp Online Softmax `NaN`:**
    *   *Risk*: An all-zero activation group produces `max_abs = 0.0f` (`1.0f / d = inf`), and a 4-warp GQA block running on `< 4` tokens per split (e.g., `seq_len = 8, splits = 16` in `test_kernels.cu` Suite 7) leaves Warps `1..3` with `m_i = -INFINITY, d_i = 0.0f`, causing `expf(-INFINITY - (-INFINITY)) = NaN` during cross-warp merge.
    *   *Mitigation*: Clamp `d = fmaxf(max_abs / 127.0f, 1e-8f)` in `quantize_group128_fp16_to_q8_1`, and guard every warp's contribution in the GQA cross-warp reduction with `if (s_d[h][w] > 0.0f)`.
*   **Standalone `test_kernels.cu` Callers Lacking Pre-Allocated `Q8_1` Scratch Pointers:**
    *   *Risk*: `test_kernels.cu` Suite 5 calls `launch_gemv_w4a16` and `launch_gemv_dual_gate_up_swiglu_w4a16` with raw `const half* input_x` and no `d_act_q8` workspace buffers.
    *   *Mitigation*: Both standalone `w4a16` entry points cooperatively quantize `input_x` into shared memory (`s_x_q8`, `s_x_scale`, `s_x_sum8`) at the start of each thread block using `quantize_group128_fp16_to_q8_1` and execute the same 4-row-per-warp `__dp4a` loop.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance & Latency (`sm_86` RTX 3050 Laptop GPU):**
    *   **W4A8 Decode (`tg256`, `batch_size = 1`):** `> 140.0 tokens/sec` (`< 7.14 ms` median inter-token latency), exceeding `llama.cpp` (`125.58 tokens/sec`).
    *   **W4A16 Prefill (`pp128` TTFT):** `< 30.0 ms` (`> 4,260 tokens/sec`), exceeding `llama.cpp` (`33.35 ms` / `3,838 tokens/sec`).
    *   **CUDA Graph Invariant:** Zero `cudaGraphExecUpdate` calls and exactly one `cudaGraphLaunch` call per decode token step.
*   **Memory Footprint & Backward Compatibility:**
    *   **Static `GpuArena` Size:** Strictly unchanged (`879.34 MB` for W4A16, `3,064.43 MB` for FP16); all `Q8_1` scratch pointers (`14 KB`) are carved at fixed aligned offsets inside the existing `WorkspaceBuffers` allocation so `arena_->used_bytes()` stays identical.
    *   **Model Format Compatibility:** Loads existing `.qwenbin` v1 files without modification or re-export.
*   **Rollout & Verification Safety:**
    *   All 8 kernel and end-to-end verification suites in `../../tests/test_kernels.cu` must pass against existing `models/golden_ref.bin` (`max_abs_err <= 5e-3` for FP16, `cosine_similarity >= 0.990` for W4 projections and `>= 0.985` for W4 SwiGLU).
    *   Every kernel launch wrapper must invoke `CUDA_KERNEL_CHECK()` (`cudaGetLastError()`) immediately after launch to trap invalid shared-memory or grid configurations at the point of origin.

--------------------------------------------------------------------------------

## Additional Resources

*   Brainstorm Concept Brief: [`concepts_brief.md`](../../concepts_brief.md)
*   Baseline Engine Design Document: [`qwen25_cuda_inference_engine.md`](./qwen25_cuda_inference_engine.md)
*   Empirical Benchmark Comparison vs. `llama.cpp`: [`comparison.md`](../../comparison.md)

## Future Improvements

*   **Prompt Lookup (N-Gram Speculative Decoding):** Drafting 3 to 5 continuation tokens from prompt n-gram matches and verifying them in a single `BM = 64` Tensor Core forward pass to reach `200+ effective tokens/sec` on code and structured outputs.
*   **Asynchronous Double-Buffered `cp.async` Prefill Mainloop:** Overlapping global-to-shared weight loads with `wmma::mma_sync` execution for long prompts (`M >= 1024`).

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Incremental Implementation Phases

To keep the codebase buildable and testable at every step, implementation proceeds in three incremental phases:
1.  **Phase 1 — Shared Quantization Primitives, Fast RoPE, & 4-Row-per-Warp `W4A8` (`__dp4a`) Decode + `QKVProjectionView` (Items 1–8):**
    *   Implement `include/config.cuh`, `include/tensor.cuh`, `include/quant_primitives.cuh`, `include/kernels.cuh`, `include/engine.cuh`, `src/kernels/rmsnorm_rope.cu`, `src/kernels/gemv_w4a16.cu`, and `src/kernels/gemv_fp16.cu`.
    *   *Phase 1 Checkpoint*: Build `Release` and verify `test_kernels.exe` Suites 1, 2, 3, 4, and 5 pass with `0` errors.
2.  **Phase 2 — 2D-Tiled Ampere Tensor Core (`nvcuda::wmma`) Prefill GEMM (Item 9):**
    *   Implement `src/kernels/gemm_prefill.cu` (`BM = 64, BN = 64, BK = 32` WMMA for `fp16`, `w4a16`, `qkv`, and `dual_gate_up_swiglu`).
    *   *Phase 2 Checkpoint*: Build `Release` and verify `test_kernels.exe` Suites 6a and 6b (`M = 8` prefill projections and SwiGLU) pass against `golden_ref.bin`.
3.  **Phase 3 — Warp-per-Token GQA Attention, Engine Orchestration, `StageProfile` Telemetry, & Full Benchmark Verification (Items 10–14):**
    *   Implement `src/kernels/attention_gqa.cu`, `src/engine.cu`, `src/cli_bench.cpp`, `tests/test_kernels.cu`, and update `comparison.md`.
    *   *Phase 3 Checkpoint*: Pass all suites (`1a..8b`) in `test_kernels.exe`, pass `compute-sanitizer --tool memcheck`, and verify `pp128 < 30.0 ms` and `tg256 > 140.0 tok/s` in `cli_bench.exe`.

### Target Files to Create/Modify

#### 1. `../../include/config.cuh` (Modify)

*   **Type / Scope**: Compile-Time Constants Header (`C++20 / CUDA`)
*   **Rationale**: Adds compile-time tiling constants for the 2D-tiled Ampere Tensor Core (`nvcuda::wmma`) prefill GEMM kernels, multi-row decode GEMV warps, and precomputed base-2 logarithm of `ROPE_THETA` so device kernels resolve loop bounds and shared-memory dimensions at compile time without changing `QWQwenHeader` (`4096` bytes).
*   **Proposed Changes**:
    *   Add Tensor Core prefill tile constants:
        *   `constexpr int32_t PREFILL_WMMA_BM = 64;`
        *   `constexpr int32_t PREFILL_WMMA_BN = 64;`
        *   `constexpr int32_t PREFILL_WMMA_BK = 32;`
        *   `constexpr int32_t PREFILL_SMEM_STRIDE = 40; // BK=32 + 8 half skew`
    *   Add decode multi-row warp constant: `constexpr int32_t DECODE_ROWS_PER_WARP = 4;`
    *   Add fast RoPE constant: `constexpr float LOG2_ROPE_THETA = 19.931568569324174f; // log2(1000000.0f)`
    *   Keep `QWQwenHeader`, `TokenizerBinHeader`, and `GoldenRefHeader` byte-identical.

#### 2. `../../include/tensor.cuh` (Modify)

*   **Type / Scope**: Arena Allocator & Error Diagnostics Header (`C++20 / CUDA`)
*   **Rationale**: Adds a synchronous `CUDA_KERNEL_CHECK()` macro so invalid grid, block, or shared-memory configurations are caught immediately at the kernel launch site (including during `cudaStreamBeginCapture`) with file, line, and kernel context.
*   **Proposed Changes**:
    *   Define `CUDA_KERNEL_CHECK()` macro calling `CUDA_CHECK(cudaGetLastError())` immediately after every `<<<...>>>` launch across all `.cu` files.

#### 3. `../../include/quant_primitives.cuh` (Create)

*   **Type / Scope**: Inline Device Quantization & Bit-Unpacking Header (`CUDA`)
*   **Rationale**: Centralizes `.qwenbin` v1 interleaved-nibble extraction (`q0, q4, q1, q5, q2, q6, q3, q7`), `Q8_1` group-128 quantization, and `W4A8` zero-point math in a single header shared by `../../src/kernels/gemv_w4a16.cu`, `../../src/kernels/gemm_prefill.cu`, and `../../src/kernels/rmsnorm_rope.cu`.
*   **Proposed Changes**:
    *   Declare `__device__ __forceinline__ int32_t compute_q8_sum8(int32_t x_lo, int32_t x_hi)`: returns `-8 * __dp4a(0x01010101, x_lo, __dp4a(0x01010101, x_hi, 0))`.
    *   Declare `__device__ __forceinline__ int32_t dp4a_w4_q8_oct(uint32_t packed, int32_t x_lo, int32_t x_hi, int32_t x_sum8)`: extracts `w_lo = packed & 0x0F0F0F0Fu` and `w_hi = (packed >> 4) & 0x0F0F0F0Fu` and returns `__dp4a(w_hi, x_hi, __dp4a(w_lo, x_lo, x_sum8))`.
    *   Declare `__device__ __forceinline__ void quantize_group128_fp16_to_q8_1(const float vals[8], int group_idx, int sub_lane, int8_t* out_q8, float* out_scale, int32_t* out_sum8)`: reduces `max_abs` across the 16-lane half-warp (`__shfl_xor_sync(0xFFFFFFFF, max_abs, mask)` for `mask = 8, 4, 2, 1`), computes `d = fmaxf(max_abs / 127.0f, 1e-8f)` and `inv_d = 1.0f / d`, quantizes `vals[0..7]` via `__float2int_rn(vals[j] * inv_d)` clamped to `[-127, 127]`, packs `x_lo` and `x_hi`, writes `reinterpret_cast<int2*>(out_q8 + group_idx * 128 + sub_lane * 8)[0] = make_int2(x_lo, x_hi)`, writes `out_sum8[group_idx * 16 + sub_lane] = compute_q8_sum8(x_lo, x_hi)`, and if `sub_lane == 0` writes `out_scale[group_idx] = d`.
    *   Declare `__device__ __forceinline__ void unpack_w4_to_half2_quad(uint32_t packed, half scale, half2 out_h2[4])`: extracts all 8 nibbles into four sequential `half2` pairs `{(q0-8)*s, (q1-8)*s}`, `{(q2-8)*s, (q3-8)*s}`, `{(q4-8)*s, (q5-8)*s}`, `{(q6-8)*s, (q7-8)*s}`.

#### 4. `../../include/kernels.cuh` (Modify)

*   **Type / Scope**: Kernel Launch Interface Header (`C++20 / CUDA`)
*   **Rationale**: Declares the `QKVProjectionView` struct and the new `Q8_1` quantization, `W4A8` (`__dp4a`) GEMV, and fused `QKV` launch entry points while preserving all existing `launch_*` function signatures required by `../../tests/test_kernels.cu`.
*   **Proposed Changes**:
    *   Define `struct QKVProjectionView` holding `(q_weight, q_scale, d_q, k_weight, k_scale, d_k, v_weight, v_scale, d_v)`.
    *   Add `launch_rmsnorm_q8_1(const half* input, const half* gamma, half* output_fp16_opt, int8_t* out_q8, float* out_scale, int32_t* out_sum8, cudaStream_t stream)`.
    *   Add `launch_quantize_fp16_to_q8_1(const half* input_x, int8_t* out_q8, float* out_scale, int32_t* out_sum8, int num_elements, cudaStream_t stream)`.
    *   Add `launch_gemv_w4a8(const uint32_t* qweight, const half* scales, const int8_t* input_q8, const float* input_scale, const int32_t* input_sum8, const half* residual, half* output_y, int out_features, int in_features, cudaStream_t stream)`.
    *   Add `launch_gemv_qkv_w4a8(const QKVProjectionView& view, const int8_t* input_q8, const float* input_scale, const int32_t* input_sum8, cudaStream_t stream)` and `launch_gemv_qkv_fp16(const QKVProjectionView& view, const half* input_x, cudaStream_t stream)`.
    *   Add `launch_gemv_dual_gate_up_swiglu_w4a8(const uint32_t* gate_up_qweight, const half* gate_up_scales, const int8_t* input_q8, const float* input_scale, const int32_t* input_sum8, half* output_mlp, cudaStream_t stream)`.
    *   Add `launch_gemm_prefill_qkv_fp16(const QKVProjectionView& view, const half* input_x, int M, cudaStream_t stream)` and `launch_gemm_prefill_qkv_w4a16(const QKVProjectionView& view, const half* input_x, int M, cudaStream_t stream)`.
    *   Extend `launch_attention_gqa_flash_decode` with optional defaulted parameters `int8_t* out_q8 = nullptr, float* out_scale = nullptr, int32_t* out_sum8 = nullptr` so existing callers in `../../tests/test_kernels.cu` compile without modification while `QwenEngine::execute_decode_pass()` receives fused `Q8_1` output for `O-Proj`.

#### 5. `../../include/engine.cuh` (Modify)

*   **Type / Scope**: Engine Runtime Class & Workspace Schema (`C++20 / CUDA`)
*   **Rationale**: Adds strongly-typed `Q8_1` scratch pointers to `WorkspaceBuffers` and exposes a `StageProfile` struct holding separate prefill and decode operator latency breakdowns for `--profile-stages` telemetry while keeping `ArenaStats` unchanged.
*   **Proposed Changes**:
    *   Add `int8_t* d_act_q8;`, `float* d_act_scale;`, and `int32_t* d_act_sum8;` to `WorkspaceBuffers`.
    *   Define `struct StageBreakdown { float embed_ms; float rmsnorm_ms; float qkv_ms; float rope_kv_ms; float attn_ms; float o_proj_ms; float post_norm_ms; float gate_up_swiglu_ms; float quant_down_ms; float down_proj_ms; float final_norm_ms; float lm_head_ms; float sample_ms; float total_ms; };` and `struct StageProfile { StageBreakdown prefill; StageBreakdown decode; float prefill_tflops; float decode_bw_gbs; };`, and declare `StageProfile profile_stages(const std::vector<int32_t>& prompt_tokens);` on `QwenEngine`.

#### 6. `../../src/kernels/rmsnorm_rope.cu` (Modify)

*   **Type / Scope**: Normalization, Activation Quantization & RoPE Kernels (`CUDA`)
*   **Rationale**: Implements fused `RMSNorm + Q8_1` activation quantization and standalone `FP16 -> Q8_1` group-128 quantization using `quantize_group128_fp16_to_q8_1`, vectorizes `embed_lookup_w4a16` using `unpack_w4_to_half2_quad`, and replaces `powf` calls in `qkv_bias_rope_kv_append_kernel` with `exp2f` and `__sincosf`.
*   **Proposed Changes**:
    *   Implement `rmsnorm_q8_1_kernel` (`1` block of `256` threads = `16` half-warps): computes FP32 RMSNorm for `[1536]`, and for `half_warp_id < 12` (`12` groups of `128`), loads `8` normalized values per lane (`sub_lane = threadIdx.x & 15`) and calls `quantize_group128_fp16_to_q8_1(vals, half_warp_id, sub_lane, out_q8, out_scale, out_sum8)`.
    *   Implement `quantize_fp16_to_q8_1_kernel` with deterministic launch geometry `block = 32` threads (`2` half-warps = `2` groups = `256` elements per block) and `grid = num_elements / 256` (`35` blocks for `8960`, `6` blocks for `1536`): each lane (`sub_lane = threadIdx.x & 15`, `group_idx = blockIdx.x * 2 + (threadIdx.x >> 4)`) loads `uint4` (`8 x half`) from `input_x + group_idx * 128 + sub_lane * 8`, converts to `float vals[8]`, and calls `quantize_group128_fp16_to_q8_1`.
    *   In `qkv_bias_rope_kv_append_kernel`, replace `powf(ROPE_THETA, exponent)` and `cosf`/`sinf` with `exp2f(exponent * LOG2_ROPE_THETA)` (since `exponent = -2.0f * coord_i / HEAD_DIM` is already negative) and `__sincosf(angle, &sin_val, &cos_val)`.
    *   Invoke `CUDA_KERNEL_CHECK()` after every launch.

#### 7. `../../src/kernels/gemv_w4a16.cu` (Modify)

*   **Type / Scope**: Quantized Decode Matrix-Vector Multiplication Kernels (`CUDA`)
*   **Rationale**: Replaces the 1-row-per-warp scalar FP32 W4A16 decode kernels with 4-row-per-warp (`2` outputs = `4` rows per warp for SwiGLU) `W4A8` (`__dp4a`) kernels and adds the single-launch `gemv_qkv_w4a8_kernel`.
*   **Proposed Changes**:
    *   Implement `gemv_w4a8_kernel` (`BLOCK_THREADS = 128`, `WARPS_PER_BLOCK = 4`, `ROWS_PER_WARP = 4` = `16` output rows per block): cooperatively loads `s_x_q8` (`in_features` bytes), `s_x_sum8` (`(in_features / 8) * 4` bytes), and `s_x_scale` (`(in_features / 128) * 4` bytes) into shared memory (`in_features * 49 / 32` bytes total); each warp loops over `s = 0 .. in_features / 256 - 1`, loads `(x_lo, x_hi, x_sum8, x_d)` once from shared memory (`group_idx = 2 * s + (lane_id >> 4)`, `sub_lane = lane_id & 15`), and evaluates 4 output rows using `dp4a_w4_q8_oct`.
    *   Implement `gemv_qkv_w4a8_kernel`: dispatches virtual rows `[0, 1536)` to `Q`, `[1536, 1792)` to `K`, and `[1792, 2048)` to `V` using the same 4-row-per-warp `__dp4a` core.
    *   Implement `gemv_dual_gate_up_swiglu_w4a8_kernel`: each warp computes `2` intermediate features (`r0 = 2 * warp_global`, `r1 = 2 * warp_global + 1`), evaluating 4 weight rows (`gate0, up0, gate1, up1`) per shared-memory load via `dp4a_w4_q8_oct` before applying $\text{SiLU}(g) \cdot u$ in lane 0.
    *   Update `launch_gemv_w4a16` and `launch_gemv_dual_gate_up_swiglu_w4a16` (used by standalone callers in `../../tests/test_kernels.cu` passing `const half* input_x`) to allocate the same `in_features * 49 / 32` bytes of dynamic shared memory (`s_x_q8`, `s_x_sum8`, `s_x_scale`), cooperatively quantize `input_x` in shared memory via `for (int g = threadIdx.x / 16; g < in_features / 128; g += 8)` calling `quantize_group128_fp16_to_q8_1` followed by `__syncthreads()`, and run the same 4-row-per-warp `__dp4a` accumulation path without needing external scratch buffers. (Because `in_features / 128` is even—`12` for `1536` and `70` for `8960`—both half-warps of any active 32-lane warp enter `quantize_group128_fp16_to_q8_1` together.)

#### 8. `../../src/kernels/gemv_fp16.cu` (Modify)

*   **Type / Scope**: FP16 Decode Matrix-Vector Multiplication Kernels (`CUDA`)
*   **Rationale**: Adds `launch_gemv_qkv_fp16` so FP16 decode also executes a single merged QKV launch and adds `CUDA_KERNEL_CHECK()` diagnostics.
*   **Proposed Changes**:
    *   Implement `gemv_qkv_fp16_kernel` and `launch_gemv_qkv_fp16` using `QKVProjectionView` over `2048` output rows.

#### 9. `../../src/kernels/gemm_prefill.cu` (Modify)

*   **Type / Scope**: 2D-Tiled Ampere Tensor Core (`nvcuda::wmma`) Prefill GEMM Kernels (`CUDA`)
*   **Rationale**: Replaces the `BN = 1` scalar column-per-block prefill kernels (`1,436.67 ms` for `pp128`) with unified `[BM = 64, BN = 64, BK = 32]` Ampere Tensor Core (`wmma::mma_sync` `m16n16k16`, FP32 accumulator) kernels that stream weights once per 64-token batch and achieve `< 30 ms` `pp128` TTFT.
*   **Proposed Changes**:
    *   Include `<mma.h>` and `../../include/quant_primitives.cuh`.
    *   Implement a template core or unified shared-memory mainloop (`BM = 64, BN = 64, BK = 32`, `128` threads = `4` warps in a $2 \times 2$ warp grid, each warp holding $2 \times 2$ `16x16` FP32 accumulator fragments) with bank-conflict-free `union PrefillSmem` (`s_A[64][40]`, `s_B[64][40]`, `s_C[4][16][18]`, `10,240 B` total).
    *   For `W4A16` kernels (`gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`), in step `kt` (`k_offset = kt * 32`, `group_idx = kt >> 2`), each of the 128 threads loads `2` packed `uint32_t` weight words (`row_qw[kt * 4 + word_in_32]` with `word_in_32 in 0..3`), unpacks them into `4` `half2` pairs via `unpack_w4_to_half2_quad`, and writes `uint4` into `s_B[row_n][col_k .. col_k + 7]`.
    *   Place an explicit `__syncthreads()` barrier after the `kt` loop finishes before warps store accumulator fragments into `smem.s_C[warp_id][0][0]` (`ldm = 18`).
    *   In `gemm_prefill_dual_gate_up_swiglu_*`, each warp loops over `mi = 0..1` and `ni = 0..1`, stores `acc_frag[mi][ni]` into `smem.s_C[warp_id][0][0]` (`ldm = 18`, `wmma::mem_row_major`), calls `__syncwarp()`, and has its 32 lanes (`r_local = lane_id / 8` stepping by 4 across `0..15`, `pair_p = lane_id % 8` in `0..7`) read `gate = smem.s_C[warp_id][r][2 * pair_p]` and `up = smem.s_C[warp_id][r][2 * pair_p + 1]`, compute `silu(gate) * up`, and write to `output_mlp[(m_row) * INTERMEDIATE_SIZE + (n_col_base / 2) + pair_p]`.
    *   Guard all global `input_x` loads (`0.0h` when `m_global >= M`) and `output_y` / `output_mlp` writes (`m_global < M && n_global < N`) so non-multiple-of-64 prompt lengths (such as `M = 8` in `test_kernels.cu`) execute with zero out-of-bounds reads or writes.

#### 10. `../../src/kernels/attention_gqa.cu` (Modify)

*   **Type / Scope**: GQA FlashDecoding & Causal Prefill Attention Kernels (`CUDA`)
*   **Rationale**: Eliminates inner-loop `__syncthreads()` barriers from both FlashDecoding Stage 1 and Prefill GQA Attention, shares KV cache reads across all 6 query heads in prefill, and fuses optional `Q8_1` output quantization into FlashDecoding Stage 2 for `O-Proj`.
*   **Proposed Changes**:
    *   Rewrite `attention_gqa_flash_decode_stage1_kernel` so each warp (`32` lanes) owns the `HEAD_DIM = 128` dimension (`4` contiguous `half`s = `uint2` per lane in registers `q_reg[6][4]` and `o_reg[6][4]`). The 4 warps in a 128-thread block iterate over `t = split_start + warp_id; t < split_end; t += 4`, reducing `Q · K` across the 32 lanes via `__shfl_xor_sync` with zero `__syncthreads()` inside the loop. After the loop, stage per-warp states into `__shared__ float s_m[6][4], s_d[6][4], s_o[4][128]` and guard each warp's contribution with `if (s_d[h][w] > 0.0f)` so empty warps on short contexts (`split_end - split_start < 4`) never evaluate `expf(-INFINITY - (-INFINITY)) = NaN`.
    *   Update `attention_gqa_flash_decode_stage2_kernel` (where `128` threads hold `1` scalar `float acc` each for query head `q_h`) so that when `out_q8 != nullptr`, the 128 threads reduce `max_abs` across the block via a 4-warp shared-memory max reduction, quantize `acc` to `int8_t q_val`, write `out_q8[q_h * 128 + tid] = q_val`, reduce `sum8 = -8 * q_val` across 8 adjacent lanes via `__shfl_xor_sync(0xFFFFFFFF, sum8, mask)` (`mask = 4, 2, 1`), write `out_sum8[q_h * 16 + (tid >> 3)]` (when `(tid & 7) == 0`), and write `out_scale[q_h] = d` (when `tid == 0`).
    *   Rewrite `attention_gqa_prefill_kernel` with `grid = dim3(NUM_KV_HEADS = 2, num_tokens)` and `block = 128` (`4` warps), loading all 6 query heads of KV group `kv_h` into registers `q_reg[6][4]` and sweeping `t = warp_id; t <= cur_pos; t += 4` with coalesced `uint2` K/V loads, zero inner-loop `__syncthreads()`, and the `if (s_d[h][w] > 0.0f)` guarded cross-warp shared-memory merge.

#### 11. `../../src/engine.cu` (Modify)

*   **Type / Scope**: Engine Orchestration, Arena Setup & CUDA Graph Capture (`C++20 / CUDA`)
*   **Rationale**: Carves `d_act_q8`, `d_act_scale`, and `d_act_sum8` inside the existing workspace allocation with zero change to `arena_->used_bytes()` (`879.34 MB`), wires `execute_decode_pass()`, `execute_prefill_chunk()`, and `prefill()` to the fused QKV, `W4A8` (`__dp4a`), and Tensor Core prefill kernels, removes per-chunk host synchronizations in `prefill()`, and implements `profile_stages()`.
*   **Proposed Changes**:
    *   In `QwenEngine::setup_arena()`, assign `ws_.d_act_q8 = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 65536)`, `ws_.d_act_scale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 74752)`, and `ws_.d_act_sum8 = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 75264)` (inside the `786,432`-byte `ws_.d_act_1` buffer, keeping `arena_->used_bytes()` identical at `879.34 MB`).
    *   In `QwenEngine::execute_decode_pass()`, use `launch_rmsnorm_q8_1`, `launch_gemv_qkv_w4a8` (or `launch_gemv_qkv_fp16` in FP16 mode), fused `Q8_1` epilogue in `launch_attention_gqa_flash_decode`, `launch_gemv_w4a8` for `O-Proj`, `launch_rmsnorm_q8_1`, `launch_gemv_dual_gate_up_swiglu_w4a8`, `launch_quantize_fp16_to_q8_1` + `launch_gemv_w4a8` for `Down-Proj`, and `launch_rmsnorm_q8_1` + `launch_gemv_w4a8` for `LM Head`.
    *   In `QwenEngine::execute_prefill_chunk(int chunk_start, int chunk_len)`, pass `ws_.d_prompt_tokens + chunk_start` to `launch_embed_lookup_fp16` and `launch_embed_lookup_w4a16`, and replace the three separate Q, K, V prefill GEMM calls with `launch_gemm_prefill_qkv_w4a16` (or `launch_gemm_prefill_qkv_fp16`).
    *   In `QwenEngine::prefill()`, copy the uncached prompt suffix (`prompt_tokens.data() + start_pos`, `(num_tokens - start_pos) * sizeof(int32_t)`) to `ws_.d_prompt_tokens + start_pos` in a single `cudaMemcpyAsync` before the chunk loop, remove the per-chunk `cudaStreamSynchronize(stream_)` inside the loop, and use `launch_rmsnorm_q8_1` + `launch_gemv_w4a8` for the final token's `LM Head` projection when `quant_mode_ == QuantMode::W4A16_G128`.
    *   Implement `QwenEngine::profile_stages(const std::vector<int32_t>& prompt_tokens)` using `cudaEvent_t` timers to populate both `StageProfile::prefill` and `StageProfile::decode` breakdowns, calling `reset_session()` before and after profiling so benchmark session state is clean.

#### 12. `../../src/cli_bench.cpp` (Modify)

*   **Type / Scope**: CLI Benchmark & Telemetry Harness (`C++20 / CUDA`)
*   **Rationale**: Adds `--profile-stages` operator-level breakdown and reports achieved Prefill TFLOPS (`335.4 GFLOPs` at `pp128`) and exact decode memory bandwidth.
*   **Proposed Changes**:
    *   Parse `--profile-stages` CLI flag.
    *   Compute and print **Prefill Throughput (`tokens/sec`)** and **Prefill Compute (`TFLOPS`)** alongside `TTFT (ms)`.
    *   When `--profile-stages` is passed, run `engine.profile_stages(prompt_tokens)` and print formatted tables of both prefill and decode operator latencies (ms and percentage of step time).

#### 13. `../../tests/test_kernels.cu` (Modify)

*   **Type / Scope**: Automated Kernel & End-to-End Verification Suite (`C++20 / CUDA`)
*   **Rationale**: Extends the existing 8 verification suites to validate the new `RMSNorm + Q8_1` quantizer, merged `QKVProjectionView` decode and prefill kernels, and deterministic W4A16/W4A8 greedy generation parity in Suite 8b.
*   **Proposed Changes**:
    *   Declare `const half* ref_k_raw = reinterpret_cast<const half*>(ref_k_raw_raw.data());` and `const half* ref_v_raw = reinterpret_cast<const half*>(ref_v_raw_raw.data());` alongside `ref_q_raw`.
    *   In Suite 4 and Suite 5, also test `launch_gemv_qkv_fp16` and `launch_gemv_qkv_w4a8` (after `launch_rmsnorm_q8_1`) against `ref_q_raw`, `ref_k_raw`, and `ref_v_raw` (`max_abs_err <= 5e-3` for FP16; `cosine_similarity >= 0.990` for W4A8).
    *   In Suite 6a and Suite 6b, also test `launch_gemm_prefill_qkv_fp16` and `launch_gemm_prefill_qkv_w4a16` (`M = 8`) against `ref_q_raw`, `ref_k_raw`, and `ref_v_raw`.
    *   In Suite 8b, collect 16 tokens from uncaptured stream execution (`reset_session()` -> `prefill(prompt, 0.0f)` -> 15 steps of `execute_decode_pass()` + `cudaStreamSynchronize` + `*pinned_token_ptr()`) and 16 tokens from captured `cudaGraphExec_t` replay (`reset_session()` -> `prefill(prompt, 0.0f)` -> 15 steps of `decode_step()`), and assert all 16 tokens are valid vocabulary indices (`0 <= tok < VOCAB_SIZE`) with `16 / 16` exact graph-to-stream parity.

#### 14. `../../comparison.md` (Modify)

*   **Type / Scope**: Benchmark & Architecture Documentation (`Markdown`)
*   **Rationale**: Updates empirical telemetry tables and roofline charts once `cli_bench.exe` verifies the new `pp128` (`< 30 ms`) and `tg256` (`> 140 tok/s`) measurements on the RTX 3050 Laptop GPU.
*   **Proposed Changes**:
    *   Update the empirical benchmark table and Mermaid throughput chart in Section 3 with the newly measured `TTFT (128 Tokens)`, `Decode Speed (Batch=1)`, `Inter-Token Latency (p50)`, and `Effective Bandwidth` numbers, and document the `W4A8` (`__dp4a`) + `nvcuda::wmma` Tensor Core architecture in Section 4.1.

### Verification Plan

*   **Compilation/Build**:
    *   `cmake --build build --config Release` must compile `qwen_core`, `test_kernels.exe`, `cli_bench.exe`, and `qwen_server.exe` with zero errors under MSVC C++20 and NVCC (`sm_86`).
*   **Unit & Golden Reference Tests**:
    *   `./build/Release/test_kernels.exe` must pass all FP16 and W4A16/W4A8 verification suites (`Suites 1a, 1b, 2, 3, 4, 5, 6a, 6b, 7, 8, 8b`) against `models/golden_ref.bin` and exit with code `0`.
*   **Memory & Bounds Sanitization**:
    *   `compute-sanitizer --tool memcheck ./build/Release/test_kernels.exe` must report `0 errors`.
*   **Head-to-Head Performance Benchmark vs. `llama.cpp`**:
    *   `./build/Release/cli_bench.exe --model models/qwen25_1.5b_w4a16.qwenbin --prompt-len 128 --gen-len 256 --warmup 5 --runs 10 --profile-stages` must demonstrate:
        1.  `Total Discrete VRAM Used == 879.34 MB` (unchanged arena footprint).
        2.  `Time-To-First-Token (TTFT) < 30.0 ms` (beating `llama.cpp` at `33.35 ms`).
        3.  `Decode Throughput > 140.0 tokens/sec` (`p50 ITL < 7.14 ms`, beating `llama.cpp` at `125.58 tokens/sec` / `7.96 ms`).

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
