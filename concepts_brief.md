# Pure-CUDA W4A16 Prefill GEMM Scaling & Pipeline Optimization

* **Author:** Areen-09
* **Date:** 2026-11-14

## Overview

The `qwen-engine` inference engine already beats `llama.cpp` on decode throughput (`216.79 tok/s` vs `146.73 tok/s` on Kaggle T4), but trails by ~5x on prefill (`2,936 tok/s` vs `14,337 tok/s` at `M = 128`). This concept replaces the single-buffered, fixed-tile (`64x64x32`) W4A16 prefill GEMM pipeline with a double-buffered mainloop, SM-count-aware dispatch (small-M tiles and selective Split-K within a 50–100 MB workspace budget), faster W4-to-FP16 dequantization, and vectorized epilogue writes. It also evaluates raising `PREFILL_MAX_CHUNK` from `256` to `512` and profiles `lm_head` overhead so a single build scales cleanly across both 16-SM (`sm_86`, RTX 3050 4 GB) and 40-SM+ (`sm_75`, T4) GPUs using pure CUDA/C++ with zero external libraries.

## Analysis

### 1. Why Current Prefill Fails to Scale Across Architectures
- **SM starvation at small `M` (`M <= 256`)**: With `PREFILL_WMMA_BM = 64` and `PREFILL_WMMA_BN = 64`, `o_proj` (`N = 1536`) at `M = 128` launches only `24 x 2 = 48` thread blocks. On a 16-SM RTX 3050 that is `3.0` blocks/SM, but on a 40-SM T4 it is `1.2` blocks/SM—leaving 32 SMs with a single block and no extra warps/blocks to hide barrier or memory latency.
- **Single-buffered synchronous mainloop**: `PrefillSmem` (`10,240 B`) forces two `__syncthreads()` barriers per `K = 32` step (`96` barriers for `K = 1536`, `560` barriers for `K = 8960`), with zero overlap between global loads/dequantization and Tensor Core math.
- **Dequantization and epilogue overhead**: `unpack_w4_to_half2_quad` in `include/quant_primitives.cuh` converts every 4-bit nibble through FP32 (`static_cast<float>(q) - 8.0f`), multiplies by `__half2float(scale)`, and converts back to FP16. The epilogue writes one `half` per thread per iteration with scalar `__float2half` and scalar residual loads.
- **Chunking and `lm_head` cost**: `PREFILL_MAX_CHUNK = 256` splits 512–2048 token prompts into 2–8 sequential chunks, capping `grid.y` at `4`. At `M = 128`, `lm_head` (`151936 x 1536`) runs once after the 28 layers (`~1–2 ms` out of `~43.6 ms` TTFT), so the 28-layer prefill GEMMs dominate TTFT.

### 2. Proposed Architecture & Trade-Offs
- **Double-buffered SMEM (`PrefillSmem[2]`, `BK = 32`)**: Doubling shared memory to `20,480 B` (20 KB) fits `3` blocks/SM on `sm_75` (64 KB SMEM/SM) and `4–5` blocks/SM on `sm_86` (100 KB SMEM/SM). Increasing `BK` to `64` with double buffering (`36,864 B`) is rejected because it drops `sm_75` to `1` block/SM.
- **Single-build `sm_75` + `sm_86` compatibility**: `cp.async` and `mma.sync.m16n8k16` only exist on `sm_80+` (`sm_75` supports `mma.sync.m16n8k8` and `wmma`). Register-staged double buffering (`load_a_to_reg` + `dequant_b_to_reg` for `kt + 1` while `wmma_step` executes on `stage = kt & 1`) works on both `sm_75` and `sm_86` from one codebase, with optional `#if __CUDA_ARCH__ >= 800` `cp.async` acceleration for `A`-tile loads.
- **Selective Small-M Tiles + Split-K**:
  - Apply small-M tiles (`BM = 32, BN = 64` or `32x32`) and Split-K (`split_k = 2..8`) **only** to `qkv`, `o_proj`, and `down_proj` (`N <= 2048`) when `grid_blocks < 2 * sm_count`.
  - **Do not apply Split-K to `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`**: `gate_up` has `N = 17,920` (`280` blocks in `x` alone, `14` blocks/SM at `M = 128` on T4), and SwiGLU (`silu(gate) * up`) is non-linear across `K`, so Split-K would break fusion and consume `~70 MB`.
  - **In-place residual safety**: `o_proj` and `down_proj` pass `ws_.d_act_0` as both `residual` and `output_y`. Split-K writes FP32 partials to `d_prefill_splitk` (`<= 16–32 MB`), and the reduction kernel reads `residual` and writes FP16 to `output_y` in a separate pass, avoiding data races.
- **Fast FP16x2 Dequant + Vectorized Epilogue**: Replace FP32 conversions in `unpack_w4_to_half2_quad` with FP16x2 arithmetic/bit tricks, and pack adjacent `s_C` columns into `half2`/`uint4` global stores.
- **Template Unification**: Parameterize `BM, BN, BK, STAGES, HAS_RESIDUAL, SPLIT_K` across shared mainloop helpers so `w4a16`, `qkv`, and `gate_up_swiglu` do not duplicate 9+ kernel copies.

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Double-buffered (`PrefillSmem[2]`, `BK = 32`) mainloop across all three W4A16 prefill GEMM kernels (`w4a16`, `qkv_w4a16`, `dual_gate_up_swiglu_w4a16`).
  * SM-count-aware dispatch (`cudaDevAttrMultiProcessorCount` cached at init) using small-M tiles and selective Split-K (`o_proj`, `down_proj`, `qkv`) within `<= 50–100 MB` extra workspace VRAM.
  * FP16x2 dequantization (`unpack_w4_to_half2_quad`) and vectorized `half2`/`uint4` epilogue writes.
  * Evaluate `PREFILL_MAX_CHUNK` (`256 -> 512`) and profile `lm_head` vs layer GEMMs across `M = 128, 512, 1024, 2048`.
  * Single multi-arch build (`CMAKE_CUDA_ARCHITECTURES="75;86"`) with zero decode regression (`>= 216.79 tok/s` on T4) and full numerical parity (`test_kernels` + `tools/verify_reference.py`).
* **Out-of-Scope Non-Goals:**
  * External GEMM libraries (cuBLAS, CUTLASS, or any third-party kernels).
  * Switching prefill GEMMs from W4A16 to W4A8/INT8 (`dp4a`/`imma`).
  * Modifying the decode GEMV or Flash-Decode attention kernels.

## Alternatives Weighed

* **Increase `PREFILL_WMMA_BK` from `32` to `64` with double buffering**: Rejected during design review because `2 * (64*72 + 64*72) * 2 B = 36,864 B` exceeds half of `sm_75`'s `64 KB` shared memory, capping T4 at `1` block/SM and worsening latency hiding.
* **Use `mma.sync.aligned.m16n8k16` PTX unconditionally**: Rejected because `m16n8k16` is `sm_80+` only and fails on Kaggle T4 (`sm_75`). Unified `wmma` (or `m16n8k16` guarded under `#if __CUDA_ARCH__ >= 800`) preserves single-build compatibility.
* **Apply Split-K to `dual_gate_up_swiglu`**: Rejected because `N = 17,920` already saturates 40+ SMs, and Split-K breaks SwiGLU fusion while requiring `~70 MB` of partial buffers.
* **Switch prefill to W4A8 (`__dp4a` / INT8 Tensor Cores)**: Excluded by user constraint to remain strictly W4A16.

## Key Decisions & Staging for Next Steps

1. **Phase 1 (Zero Extra VRAM)**: Optimize `unpack_w4_to_half2_quad` (FP16x2), implement register-staged `PrefillSmem[2]` (`BK = 32`) with optional `#if __CUDA_ARCH__ >= 800` `cp.async`, and vectorize epilogue stores (`half2`/`uint4`).
2. **Phase 2 (SM-Count-Aware Occupancy)**: Add `BM = 32` small-M tile and selective Split-K (`<= 16–32 MB` `d_prefill_splitk` workspace in `WorkspaceBuffers`) for `o_proj`, `down_proj`, and `qkv` when `grid_blocks < 2 * sm_count`.
3. **Phase 3 (Chunking & Multi-Arch Verification)**: Test `PREFILL_MAX_CHUNK = 512` (`+7.6 MB` VRAM), profile `lm_head` vs layer GEMMs at `M = 128..2048`, set `CMAKE_CUDA_ARCHITECTURES` to support `75;86`, and verify parity and benchmarks against `llama.cpp`.
