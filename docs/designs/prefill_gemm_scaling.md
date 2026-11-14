# Pure-CUDA W4A16 Prefill GEMM Scaling & Pipeline Optimization

*   **Author:** Areen-09
*   **Status:** Proposed
*   **Date:** 2026-11-15

--------------------------------------------------------------------------------

## Overview

This design upgrades `qwen-engine`'s W4A16 prefill GEMM pipeline so a single pure-CUDA binary scales across both low-SM GPUs (RTX 3050 Laptop, `sm_86`, 16 SMs, 4 GB VRAM) and higher-SM GPUs (Kaggle T4, `sm_75`, 40 SMs, 16 GB VRAM). The work replaces the single-buffered `64x64x32` mainloop with a register-staged double-buffered `BK = 32` pipeline (`PrefillSmem::mainloop[2]`), adds SM-count-aware `32x32` small-M tile dispatch with zero extra workspace VRAM, speeds up W4-to-FP16 nibble unpacking using FP16x2 arithmetic, and vectorizes epilogue writes.

## The Problem

On Kaggle T4 (`M = 128`, `Qwen2.5-1.5B-Instruct`), `qwen-engine` beats `llama.cpp` on decode (`216.79 tok/s` vs `146.73 tok/s`), but runs prefill at `2,936.06 tok/s` vs `llama.cpp`'s `14,336.89 tok/s` (~0.20x). Three prefill GEMM bottlenecks cause this gap:

1. **SM under-utilization at small `M` (`M <= 128`)**: Fixed `PREFILL_WMMA_BM = 64, PREFILL_WMMA_BN = 64` tiles launch only `48` blocks for `o_proj` (`N = 1536`) at `M = 128`. That gives `3.0` blocks/SM on RTX 3050 (16 SMs) but `1.2` blocks/SM on T4 (40 SMs), leaving 32 SMs with a single block and no extra waves to hide barrier or memory latency.
2. **Synchronous single-buffered mainloop**: `PrefillSmem` (`10,240 B`) forces two `__syncthreads()` barriers per `K = 32` step (`96` barriers for `K = 1536`, `560` barriers for `K = 8960`) with zero overlap between global loads/dequantization and Tensor Core math.
3. **Dequantization and epilogue overhead**: `unpack_w4_to_half2_quad` converts every 4-bit nibble through FP32 (`float` subtract, multiply, and convert back to `half`), and the epilogue writes one `half` per thread per iteration with scalar residual loads.

## Goals

*   **Beat `llama.cpp` across prefill and decode on `Qwen2.5-1.5B-Instruct`**: Exceed `14,336.89 tok/s` prefill at `M = 128` on Kaggle T4 (and scale across `M = 128, 512, 1024, 2048`) while keeping decode at or above `216.79 tok/s`.
*   **Single-build cross-architecture scaling**: Compile and run from one binary (`CMAKE_CUDA_ARCHITECTURES="75;86"`) with runtime SM-count-aware dispatch (`cudaDevAttrMultiProcessorCount`) and no `sm_80+`-only instructions on `sm_75`.
*   **Zero extra workspace VRAM**: Solve small-M SM starvation via `32x32` tiling without adding Split-K scratch buffers, preserving full headroom on the 4 GB RTX 3050.
*   **Maintain zero external dependencies and numerical parity**: Use pure CUDA/C++ only (`cuda_runtime.h`, `cuda_fp16.h`, `mma.h`, inline PTX) and pass `test_kernels` and `tools/verify_reference.py` (FP16 reference).

## Non-Goals

*   Using external GEMM libraries (cuBLAS, CUTLASS, or any third-party kernels).
*   Switching prefill GEMMs from W4A16 to W4A8/INT8 (`__dp4a` or `imma`).
*   Modifying decode GEMV (`gemv_w4a8`), Flash-Decode attention, `lm_head` dispatch, or `PREFILL_MAX_CHUNK = 256`.
*   Adding Split-K prefill GEMM reduction buffers (deferred unless post-landing profiling at `M <= 32` shows SM starvation).

## Critical User Journeys (CUJs)

### CUJ 1: Short-Prompt Prefill on Higher-SM GPU (Kaggle T4, `sm_75`, 40 SMs)
*   **Given**: `Qwen2.5-1.5B-Instruct` loaded in `qwen-engine` on T4 (`sm_75`), single multi-arch build.
*   **When**: User runs prefill with `M = 128` tokens (`examples/benchmark_llamacpp.ipynb`).
*   **Then**: SM-count-aware dispatch selects `32x32` tiles for `qkv`, `o_proj`, and `down_proj` (`192–256` blocks, `4.8–6.4` blocks/SM on T4), double-buffered `PrefillSmem::mainloop[2]` overlaps loads/dequant with Tensor Core math, and prefill tok/s exceeds `llama.cpp` (`14,336.89 tok/s`) without regressing decode (`216.79 tok/s`).

### CUJ 2: Long-Prompt Prefill Within 4 GB VRAM Ceiling (RTX 3050, `sm_86`, 16 SMs)
*   **Given**: `Qwen2.5-1.5B-Instruct` loaded on a 4 GB RTX 3050 Laptop (`sm_86`).
*   **When**: User runs prefill with `M = 512–2048` tokens (chunked at `PREFILL_MAX_CHUNK = 256`).
*   **Then**: Standard `64x64` tiles saturate the 16 SMs with zero added workspace VRAM, and prefill completes with verified numerical parity.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

The upgraded prefill pipeline keeps a single unified pure-CUDA codebase for both `sm_75` (Kaggle T4, 40 SMs, 64 KB SMEM/SM) and `sm_86` (RTX 3050, 16 SMs, 100 KB SMEM/SM) while removing the mainloop, occupancy, dequantization, and epilogue bottlenecks.

### 1. Double-Buffered Shared Memory Mainloop (`PrefillSmem::mainloop[2]`, `BK = 32`)
*   **Buffer layout**: Expand `PrefillSmem::mainloop` inside the `PrefillSmem` union from 1 stage to 2 stages (`PrefillSmem::mainloop[2]`), keeping `BK = 32` with bank-conflict skew padding (`PREFILL_SMEM_STRIDE = 40` for `s_A`/`s_B`, `18` for `s_C`):
    *   `64x64` tile: `2 * (64*40 + 64*40) * 2 B = 20,480 B` (fits `3` blocks/SM on `sm_75` 64 KB SMEM and `4–5` blocks/SM on `sm_86` 100 KB SMEM).
    *   `32x32` tile: `2 * (32*40 + 32*40) * 2 B = 10,240 B` (fits `6` blocks/SM on `sm_75` and `9` blocks/SM on `sm_86`).
*   **Universal register-staged overlap (`sm_75` + `sm_86`)**:
    *   Before the `K` loop, load and dequantize `kt = 0` into `stage 0`, followed by one `__syncthreads()`.
    *   Inside the loop for `kt = 0 .. num_kt - 2`: issue global loads for `A` (`kt + 1`) and W4-to-FP16x2 dequantization for `B` (`kt + 1`) into registers **while** `wmma_step` computes on `stage = kt & 1`.
    *   Store the prefetched registers into `stage = (kt + 1) & 1`, then execute a single `__syncthreads()` per `K`-step (cutting barriers in half: from `96` to `48` for `K = 1536`, and `560` to `280` for `K = 8960`).

```text
[Stage 0 Prefetch] -> __syncthreads()
        |
        v
For kt in 0 .. num_kt - 2:
  +-----------------------------------+   +-----------------------------------+
  | Compute: wmma_step(stage = kt&1)  |   | Prefetch (kt+1) -> Registers:     |
  | (Tensor Cores active)             |   | load_a_reg + unpack_w4_half2_quad |
  +-----------------------------------+   +-----------------------------------+
                    \                                       /
                     v                                     v
                Write prefetched registers -> stage = (kt+1)&1
                                       |
                                       v
                               __syncthreads()
        |
        v
[Final Stage wmma_step] -> [Vectorized Epilogue Store]
```

### 2. SM-Count-Aware `32x32` Small-M Tile Dispatch
*   **Device SM count caching**: Query `cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_id_)` once in `QwenEngine::setup_arena()` and pass `sm_count_` to the prefill GEMM launchers.
*   **Dispatch policy (`select_prefill_tile_mode`)**:
    1.  **`dual_gate_up_swiglu_w4a16`**: Always uses `64x64` (`N = 17,920` produces `280` blocks in `x` alone, `14` blocks/SM at `M = 128` on T4).
    2.  **`w4a16` (`o_proj`, `down_proj`) and `qkv_w4a16`**: Compute `64x64` grid block count `blocks_64 = ceil(M / 64) * ceil(N / 64)`. When `M <= 128` and `sm_count > 0` and `blocks_64 < 2 * sm_count`, dispatch the `32x32` tile; otherwise dispatch `64x64`. At `M = 128` on T4 (`sm_count = 40`), `o_proj`/`down_proj` (`blocks_64 = 48 < 80`) jump to `192` blocks (`4.8` blocks/SM) and `qkv` (`blocks_64 = 64 < 80`) jumps to `256` blocks (`6.4` blocks/SM) with zero extra VRAM.
*   **`32x32` warp & thread-to-tile mapping (128 threads, 4 warps)**:
    *   **Warp layout**: `2x2` warps (`warp_m = warp_id / 2`, `warp_n = warp_id % 2`), where each warp computes one `16x16` WMMA accumulator tile (`acc[1][1]`).
    *   **Loader mapping (`load_a_tile` & `dequant_b_row_pair`)**: 128 threads map `4` threads per row across `32` rows (`row = tid >> 2`, `col = (tid & 3) * 8`), so each thread loads/unpacks `8` FP16 elements (`1` `uint4` for `A`, `1` `uint32_t` = `8` nibbles for `B`), preserving `BK = 32` with zero out-of-bounds SMEM writes.

### 3. Fast FP16x2 Dequantization & Vectorized Epilogue
*   **FP16x2 nibble unpacking (`include/quant_primitives.cuh`)**: Because `.qwenbin v1` packs nibbles interleaved (`q0 | q4<<4 | q1<<8 | q5<<12 | ...`), `unpack_w4_to_half2_quad` de-interleaves the nibbles into `half2` pairs and performs bias subtraction and scale multiplication directly in FP16x2 arithmetic without FP32 roundtrips, cutting ALU instructions per 8-weight word inside the mainloop. Parity against the FP16 reference is verified in `tools/verify_reference.py`.
*   **Vectorized epilogue (`half2` / `uint4`)**: In `s_C[warp_id][16][18]`, adjacent columns are contiguous FP32 values in shared memory. Threads pair adjacent elements into `half2`, add vectorized residual loads when `residual != nullptr`, and issue coalesced stores gated on alignment:
    *   `uint4` (128-bit, 8 `half`s) when `N % 8 == 0` and row pointers (`output_y`, `residual`) are 16-byte aligned (true for Qwen2.5-1.5B dimensions: `HIDDEN_SIZE = 1536`, `KV_DIM = 256`, `INTERMEDIATE_SIZE = 8960`).
    *   `half2` (32-bit, 2 `half`s) fallback when `N % 2 == 0` and 4-byte aligned, with `global_row < M` boundary predication preserved for tail rows.

### 4. Helper Parameterization & Multi-Arch Build
*   **`<int BM, int BN>` helper parameterization**: Parameterize only `<int BM, int BN>` on the shared `__device__` mainloop helpers (`PrefillSmem`, `load_a_tile`, `dequant_b_row_pair`, `wmma_step`) rather than a 6-parameter kernel template, keeping `BK = 32`, `STAGES = 2`, and `dual_gate_up_swiglu`'s distinct epilogue clean.
*   **Multi-arch CMake**: Update `CMakeLists.txt` default `CMAKE_CUDA_ARCHITECTURES` from `"86"` to `"75;86"` so one build targets both Kaggle T4 and RTX 3050.

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Double-Buffered `PREFILL_WMMA_BK = 64`
*   **Pros**: Halves the number of `K`-loop iterations (`num_kt`) and aligns 2 `BK` steps per `QUANT_GROUP_SIZE = 128`.
*   **Cons**: Two stages of `s_A[64][72] + s_B[64][72]` require `36,864 B` of shared memory per block, which exceeds half of `sm_75`'s `64 KB` SMEM/SM and caps Kaggle T4 at `1` block/SM.
*   **Reason for Rejection**: Rejected during design review because dropping `sm_75` occupancy to `1` block/SM eliminates warp/block latency hiding across barriers and global memory stalls.

### Alternative B: Unconditional `mma.sync.aligned.m16n8k16` PTX & `cp.async`
*   **Pros**: Removes `wmma` fragment layout constraints and enables async global-to-SMEM copies on Ampere+.
*   **Cons**: Both `mma.sync.aligned.m16n8k16` and `cp.async` require `sm_80+` (Ampere) and fail to compile or execute on Kaggle T4 (`sm_75`, Turing). Maintaining a separate `#if __CUDA_ARCH__ >= 800` `cp.async` path for `A`-tiles (`2–4 KB`/stage) also adds dual-pipeline complexity without helping `sm_75`.
*   **Reason for Rejection**: Rejected for this design; universal register-staged double buffering + `wmma` covers both `sm_75` and `sm_86` cleanly.

### Alternative C: Selective Split-K + FP32 Reduction Kernel
*   **Pros**: Increases block count even for very small prompts (`M <= 32`).
*   **Cons**: Adds `16 MB` workspace VRAM, `gemm_prefill_splitk_reduce_kernel` (plus a `QKVProjectionView`-aware reduction for `d_q`/`d_k`/`d_v`), and non-power-of-2 `QUANT_GROUP_SIZE = 128` divisibility constraints (`split_k = 8` does not divide `1536` or `8960` into 128-element groups), while `32x32` tiling already gives `4.8–6.4` blocks/SM at `M = 128` on T4.
*   **Reason for Rejection**: Deferred; ship `32x32` small-M tiling first with zero extra VRAM, and revisit Split-K only if `M <= 32` profiling warrants it.

### Alternative D: Switch Prefill GEMMs from W4A16 to W4A8 (`__dp4a` / INT8 Tensor Cores)
*   **Pros**: Doubles theoretical Tensor Core peak on T4 (`130 TOPS` INT8 vs `65 TFLOPS` FP16) and matches `llama.cpp`'s `Q8_1` activation quantization approach.
*   **Cons**: Requires on-the-fly activation quantization per layer and changes prefill arithmetic precision away from W4A16.
*   **Reason for Rejection**: Rejected by author: prefill GEMMs must remain strictly W4A16.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Zero Added Workspace VRAM**: `32x32` small-M tiling and `PrefillSmem::mainloop[2]` (`10–20 KB` on-chip SMEM) add `0 B` of global workspace memory.
*   **Decode Latency Isolation**: Decode kernels (`gemv_w4a8`, `attention_gqa_flash_decode`) and CUDA Graph capture (`decode_graph_`) remain untouched so T4 decode throughput stays `>= 216.79 tok/s`.
*   **Zero External Dependencies**: Pure CUDA/C++ (`cuda_runtime.h`, `cuda_fp16.h`, `mma.h`, inline PTX) only.

### Future Improvements
*   Evaluate `PREFILL_MAX_CHUNK = 512` (`+7.6 MB` workspace) and `lm_head` profiling if long-prompt (`M >= 512`) bottlenecks remain after these GEMM optimizations land.
*   Evaluate `#if __CUDA_ARCH__ >= 800` `cp.async` for `A`-tiles if `sm_86` profiling shows register pressure or load stalls.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Files to Modify

#### 1. [include/config.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/config.cuh) (Modify)
*   **Type / Scope**: CUDA Compile-Time Configuration
*   **Proposed Changes**:
    *   Add `PREFILL_SMEM_STAGES = 2` (`BK = 32` and `PREFILL_MAX_CHUNK = 256` unchanged).
    *   Add `PREFILL_SMALL_BM = 32` and `PREFILL_SMALL_BN = 32`.
    *   **Rationale**: Centralizes double-buffer stage count and small-M tile dimensions.

#### 2. [include/quant_primitives.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/quant_primitives.cuh) (Modify)
*   **Type / Scope**: Device Quantization Primitives
*   **Proposed Changes**:
    *   Optimize `unpack_w4_to_half2_quad(uint32_t packed, half scale, half2 out_h2[4])` to de-interleave `.qwenbin v1` nibbles and perform subtraction and scaling in FP16x2 arithmetic without FP32 roundtrips.
    *   **Rationale**: Reduces ALU instructions per 8-weight word inside the prefill mainloop while preserving the function signature.

#### 3. [include/kernels.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/kernels.cuh) (Modify)
*   **Type / Scope**: Kernel Launch Declarations
*   **Proposed Changes**:
    *   Extend `launch_gemm_prefill_w4a16` and `launch_gemm_prefill_qkv_w4a16` with a single trailing optional parameter `int sm_count = 0`.
    *   **Rationale**: Passes cached device SM count from `QwenEngine` while keeping existing unit tests compiling unchanged.

#### 4. [src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu) (Modify)
*   **Type / Scope**: Pure-CUDA Prefill GEMM Kernels & Dispatcher
*   **Proposed Changes**:
    *   **`<int BM, int BN>` parameterized `PrefillSmem::mainloop[2]` & helpers**: Update `PrefillSmem`, `load_a_tile`, `dequant_b_row_pair`, and `wmma_step` to support both `(64, 64)` (`2` threads/row, `2x2` warps, `acc[2][2]`) and `(32, 32)` (`4` threads/row, `2x2` warps, `acc[1][1]`) with a 2-stage register-staged double-buffered mainloop across `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, and `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`.
    *   **Vectorized epilogue**: Pair adjacent FP32 elements from `s_C[warp_id][16][18]` into `half2`, add vectorized residual loads, and issue `uint4` (when `N % 8 == 0` and 16-byte aligned) or `half2` (when `N % 2 == 0` and 4-byte aligned) global stores with `global_row < M` tail predication.
    *   **SM-count-aware dispatch (`select_prefill_tile_mode`)**: Dispatch `(32, 32)` for `w4a16` and `qkv_w4a16` when `M <= 128 && sm_count > 0 && (ceil(M/64) * ceil(N/64)) < 2 * sm_count`; otherwise dispatch `(64, 64)`. Keep `dual_gate_up_swiglu` at `(64, 64)`.
    *   **Rationale**: Eliminates SM starvation on 40-SM GPUs and hides global/dequant latency behind Tensor Core math with zero extra global memory.

#### 5. [include/engine.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/engine.cuh) & [src/engine.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/engine.cu) (Modify)
*   **Type / Scope**: Engine Arena & Prefill Execution
*   **Proposed Changes**:
    *   Add `int sm_count_{0}` to `QwenEngine`; query `cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_id_)` once in `QwenEngine::setup_arena()`.
    *   Pass `sm_count_` from `execute_gqa_prefill_layer`, `execute_prefill_chunk`, and `profile_stages` to `launch_gemm_prefill_w4a16` and `launch_gemm_prefill_qkv_w4a16`.
    *   **Rationale**: Zero per-layer driver query overhead and zero extra VRAM allocation.

#### 6. [CMakeLists.txt](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/CMakeLists.txt) (Modify)
*   **Type / Scope**: Build Configuration
*   **Proposed Changes**:
    *   Update default `CMAKE_CUDA_ARCHITECTURES` from `"86"` to `"75;86"` (still overridable via `ENV{CMAKE_CUDA_ARCHITECTURES}`).
    *   **Rationale**: Produces a single fatbinary supporting both Kaggle T4 (`sm_75`) and RTX 3050 (`sm_86`) out of the box.

### Verification Plan

*   **Compilation/Build**: Build cleanly for `CMAKE_CUDA_ARCHITECTURES="75;86"` (and verify isolated `"75"` and `"86"` builds) with zero compiler warnings.
*   **Unit & Numerical Parity Tests**: Run `test_kernels` (covering `32x32` and `64x64` tiles, tail `M`, `residual == output_y`, and `unpack_w4_to_half2_quad`) and `tools/verify_reference.py` to confirm FP16 output parity against the FP16 reference.
*   **VRAM & Decode Regression Check**: Inspect `ArenaStats::workspace_bytes` on RTX 3050 (4 GB) to confirm zero added workspace VRAM, and verify T4 decode stays `>= 216.79 tok/s`.
*   **End-to-End Benchmark**: Re-run [examples/benchmark_llamacpp.ipynb](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/examples/benchmark_llamacpp.ipynb) on Kaggle T4 at `PROMPT_TOKENS = 128, 512, 1024, 2048` (and RTX 3050 at `128, 512`) and confirm `qwen-engine` beats `llama.cpp` on both prefill and decode.

--------------------------------------------------------------------------------

*Drafted with EGM*
