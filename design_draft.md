# Pure-CUDA W4A16 Prefill GEMM Scaling & Pipeline Optimization

*   **Author:** Areen-09
*   **Status:** Draft
*   **Date:** 2026-11-15

--------------------------------------------------------------------------------

## Overview

This design upgrades `qwen-engine`'s W4A16 prefill GEMM pipeline so a single pure-CUDA binary scales across both low-SM GPUs (RTX 3050 Laptop, `sm_86`, 16 SMs, 4 GB VRAM) and higher-SM GPUs (Kaggle T4, `sm_75`, 40 SMs, 16 GB VRAM). The work replaces the single-buffered `64x64x32` mainloop with a double-buffered `BK = 32` pipeline, adds SM-count-aware dispatch (small-M tiles and selective Split-K within a 50–100 MB workspace cap), speeds up W4-to-FP16 nibble unpacking, vectorizes epilogue writes, and evaluates `PREFILL_MAX_CHUNK = 512` alongside `lm_head` profiling.

## The Problem

On Kaggle T4 (`M = 128`, `Qwen2.5-1.5B-Instruct`), `qwen-engine` beats `llama.cpp` on decode (`216.79 tok/s` vs `146.73 tok/s`), but runs prefill at `2,936.06 tok/s` vs `llama.cpp`'s `14,336.89 tok/s` (~0.20x). Four codebase bottlenecks cause this gap:

1. **SM under-utilization at small `M` (`M <= 256`)**: Fixed `PREFILL_WMMA_BM = 64, PREFILL_WMMA_BN = 64` tiles launch only `48` blocks for `o_proj` (`N = 1536`) at `M = 128`. That gives `3.0` blocks/SM on RTX 3050 (16 SMs) but `1.2` blocks/SM on T4 (40 SMs), leaving 32 SMs with a single block and no extra waves to hide barrier or memory latency. `PREFILL_MAX_CHUNK = 256` also caps `grid.y` at `4` even for long prompts.
2. **Synchronous single-buffered mainloop**: `PrefillSmem` (`10,240 B`) forces two `__syncthreads()` barriers per `K = 32` step (`96` barriers for `K = 1536`, `560` barriers for `K = 8960`) with zero overlap between global loads/dequantization and Tensor Core math.
3. **Dequantization and epilogue overhead**: `unpack_w4_to_half2_quad` converts every 4-bit nibble through FP32 (`float` subtract, multiply, and convert back to `half`), and the epilogue writes one `half` per thread per iteration with scalar residual loads.
4. **Fixed `lm_head` and chunking cost**: `QwenEngine::prefill` runs `launch_rmsnorm_q8_1` + `launch_gemv_w4a8` (`151936 x 1536`) after the 28-layer prefill chunks, and `PREFILL_MAX_CHUNK = 256` splits `512–2048` token prompts into multiple sequential passes.

## Goals

*   **Beat `llama.cpp` across prefill and decode on `Qwen2.5-1.5B-Instruct`**: Exceed `14,336.89 tok/s` prefill at `M = 128` on Kaggle T4 (and scale across `M = 128, 512, 1024, 2048`) while keeping decode at or above `216.79 tok/s`.
*   **Single-build cross-architecture scaling**: Compile and run from one binary (`CMAKE_CUDA_ARCHITECTURES="75;86"`) with runtime SM-count-aware dispatch (`cudaDevAttrMultiProcessorCount`) and no `sm_80+`-only instructions on `sm_75`.
*   **Stay within the 4 GB RTX 3050 VRAM budget**: Keep all added workspace memory (Split-K reduction buffer and any chunk-size increase) at or below `50–100 MB` total.
*   **Maintain zero external dependencies and numerical parity**: Use pure CUDA/C++ only (`cuda_runtime.h`, `cuda_fp16.h`, `mma.h`, inline PTX) and pass `test_kernels` and `tools/verify_reference.py`.

## Non-Goals

*   Using external GEMM libraries (cuBLAS, CUTLASS, or any third-party kernels).
*   Switching prefill GEMMs from W4A16 to W4A8/INT8 (`__dp4a` or `imma`).
*   Modifying the decode GEMV or Flash-Decode attention kernels.

## Critical User Journeys (CUJs)

### CUJ 1: Short-Prompt Prefill on Higher-SM GPU (Kaggle T4, `sm_75`, 40 SMs)
*   **Given**: `Qwen2.5-1.5B-Instruct` loaded in `qwen-engine` on T4 (`sm_75`), single multi-arch build.
*   **When**: User runs prefill with `M = 128` tokens (`examples/benchmark_llamacpp.ipynb`).
*   **Then**: SM-count-aware dispatch selects small-M tiles / Split-K for `qkv`, `o_proj`, and `down_proj` to saturate all 40 SMs, double-buffered `PrefillSmem[2]` overlaps loads/dequant with Tensor Core math, and prefill tok/s exceeds `llama.cpp` (`14,336.89 tok/s`) without regressing decode (`216.79 tok/s`).

### CUJ 2: Long-Prompt Prefill Within 4 GB VRAM Ceiling (RTX 3050, `sm_86`, 16 SMs)
*   **Given**: `Qwen2.5-1.5B-Instruct` loaded on a 4 GB RTX 3050 Laptop (`sm_86`).
*   **When**: User runs prefill with `M = 512–2048` tokens.
*   **Then**: `WorkspaceBuffers` allocates `<= 50–100 MB` of extra VRAM (Split-K workspace + `PREFILL_MAX_CHUNK` buffers), standard `64x64` tiles saturate the 16 SMs without Split-K overhead, and prefill completes with verified numerical parity.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

The upgraded prefill pipeline keeps a single unified pure-CUDA codebase for both `sm_75` (Kaggle T4, 40 SMs, 64 KB SMEM/SM) and `sm_86` (RTX 3050, 16 SMs, 100 KB SMEM/SM) while removing the mainloop, occupancy, dequantization, and epilogue bottlenecks.

### 1. Double-Buffered Shared Memory Mainloop (`PrefillSmem[2]`, `BK = 32`)
*   **Buffer layout**: Expand `PrefillSmem::mainloop` from 1 stage (`10,240 B`) to 2 stages (`20,480 B` = `2 * (s_A[BM][40] + s_B[BN][40])`), keeping `BK = 32` so 2 stages fit `3` blocks/SM on `sm_75` (64 KB SMEM/SM) and `4–5` blocks/SM on `sm_86` (100 KB SMEM/SM).
*   **Register-staged overlap (universal `sm_75` + `sm_86` baseline)**:
    *   Before the `K` loop, load and dequantize `kt = 0` into `stage 0`, followed by one `__syncthreads()`.
    *   Inside the loop for `kt = 0 .. num_kt - 2`: issue global loads for `A` (`kt + 1`) and W4-to-FP16x2 dequantization for `B` (`kt + 1`) into registers **while** `wmma_step` computes on `stage = kt & 1`.
    *   Store the prefetched registers into `stage = (kt + 1) & 1`, then execute a single `__syncthreads()` per `K`-step (cutting barriers in half: from `96` to `48` for `K = 1536`, and `560` to `280` for `K = 8960`).
    *   Under `#if __CUDA_ARCH__ >= 800`, `A`-tile global-to-SMEM loads can optionally use `cp.async.cg.shared.global` to bypass registers for `A` while `B` continues register-staged W4 dequantization.

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
[Final Stage wmma_step] -> [Vectorized Epilogue / Split-K Store]
```

### 2. SM-Count-Aware Dispatch (Small-M Tiles & Selective Split-K)
*   **Device SM count caching**: Query `cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_id_)` once in `QwenEngine::setup_arena()` and pass `sm_count_` to the prefill GEMM launchers.
*   **Dispatch policy (`select_prefill_gemm_config`)**:
    1.  **`dual_gate_up_swiglu_w4a16`**: Always uses `BM = 64, BN = 64`, `split_k = 1`. Because `N = 17,920` produces `280` blocks in `x` alone (`14` blocks/SM at `M = 128` on T4), it is never SM-starved, and SwiGLU (`silu(gate) * up`) is non-linear across `K`.
    2.  **Small-M tile (`BM = 32, BN = 64` or `32x32`)**: Selected when `M <= 64` (or when `64x64` grid blocks `< 2 * sm_count` and `M <= 128`), doubling or quadrupling block count with zero extra VRAM.
    3.  **Selective Split-K (`split_k = 2..8`)**: Applied to `qkv`, `o_proj`, and `down_proj` (`N <= 2048`) when `grid.x * grid.y < 2 * sm_count` and `K / split_k` stays aligned to `QUANT_GROUP_SIZE = 128`.
*   **In-place residual safety & VRAM budget**: `o_proj` and `down_proj` pass `ws_.d_act_0` as both `residual` and `output_y`. When `split_k > 1`, the GEMM kernel writes FP32 partial sums to `ws_.d_prefill_splitk` (`<= 16–32 MB`, e.g., `8 * 256 * 2048 * 4 B = 16 MB`), and a dedicated reduction kernel sums the `split_k` slices, adds `residual`, and writes FP16 to `output_y` in a separate pass.

### 3. Fast FP16x2 Dequantization & Vectorized Epilogue
*   **FP16x2 nibble unpacking (`include/quant_primitives.cuh`)**: Replace the 8x `static_cast<float>(q) - 8.0f` + `__half2float` + `__float2half` sequence in `unpack_w4_to_half2_quad` with FP16x2 arithmetic/bit-manipulation (`__hsub2` / `__hmul2` using `0x6400` exponent bias or `__low2half2`/`__high2half2`), cutting ALU instructions per 8-weight word inside the mainloop.
*   **Vectorized epilogue (`half2` / `uint4`)**: In `s_C[warp_id][16][18]`, adjacent columns `(c, c + 1)` are contiguous FP32 values in shared memory. Threads pair adjacent elements into `half2` via `__float22half2_rn`, add vectorized `half2` residual loads when `residual != nullptr`, and issue coalesced 32-bit (`half2`) or 128-bit (`uint4`) global stores whenever `N` is even (all Qwen2.5-1.5B dimensions: `1536`, `256`, `8960` are multiples of 256).

### 4. Template Unification & Chunk/Build Scaling
*   **Templated mainloop**: Parameterize `template<int BM, int BN, int BK, int STAGES, bool HAS_RESIDUAL, bool SPLIT_K>` on shared tile-loading, dequantization, and `wmma_step` helpers so `w4a16`, `qkv`, and `gate_up_swiglu` share one tested mainloop without 9+ duplicated kernels.
*   **Chunking & `lm_head` profiling**: Benchmark `PREFILL_MAX_CHUNK = 512` (`+7.6 MB` activation workspace, total added VRAM `<= 40 MB`, well under the `50–100 MB` cap) and profile `lm_head` (`launch_rmsnorm_q8_1` + `launch_gemv_w4a8`) vs layer GEMMs across `M = 128, 512, 1024, 2048`.
*   **Multi-arch CMake**: Update `CMakeLists.txt` default `CMAKE_CUDA_ARCHITECTURES` from `"86"` to `"75;86"` so one build targets both Kaggle T4 and RTX 3050.

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Double-Buffered `PREFILL_WMMA_BK = 64`
*   **Pros**: Halves the number of `K`-loop iterations (`num_kt`) and aligns 2 `BK` steps per `QUANT_GROUP_SIZE = 128`.
*   **Cons**: Two stages of `s_A[64][72] + s_B[64][72]` require `36,864 B` of shared memory per block, which exceeds half of `sm_75`'s `64 KB` SMEM/SM and caps Kaggle T4 at `1` block/SM.
*   **Reason for Rejection**: Rejected during design review because dropping `sm_75` occupancy to `1` block/SM eliminates warp/block latency hiding across barriers and global memory stalls.

### Alternative B: Unconditional `mma.sync.aligned.m16n8k16` PTX & `cp.async`
*   **Pros**: Removes `wmma` fragment layout constraints and enables direct register-to-global epilogue writes without bouncing through `smem.s_C`.
*   **Cons**: Both `mma.sync.aligned.m16n8k16` and `cp.async` require `sm_80+` (Ampere) and fail to compile or execute on Kaggle T4 (`sm_75`, Turing).
*   **Reason for Rejection**: Rejected for the unified baseline because single-build `sm_75` + `sm_86` support requires `wmma` (or `m16n8k8` on `sm_75`) and register-staged prefetching, with `cp.async` restricted to `#if __CUDA_ARCH__ >= 800`.

### Alternative C: Apply Split-K to `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`
*   **Pros**: Uniform Split-K support across every prefill GEMM kernel.
*   **Cons**: `gate_up` (`N = 17,920`) already launches `280` blocks in `x` alone (`14` blocks/SM at `M = 128` on T4), so it is never SM-starved. Because SwiGLU (`silu(gate) * up`) is non-linear across `K`, Split-K would break SwiGLU fusion and require `~70 MB` (`4 * 256 * 17920 * 4 B`) of partial buffers.
*   **Reason for Rejection**: Rejected due to wasted reduction overhead, broken fusion, and excessive VRAM usage on the 4 GB RTX 3050.

### Alternative D: Switch Prefill GEMMs from W4A16 to W4A8 (`__dp4a` / INT8 Tensor Cores)
*   **Pros**: Doubles theoretical Tensor Core peak on T4 (`130 TOPS` INT8 vs `65 TFLOPS` FP16) and matches `llama.cpp`'s `Q8_1` activation quantization approach.
*   **Cons**: Requires on-the-fly activation quantization per layer and changes prefill arithmetic precision away from W4A16.
*   **Reason for Rejection**: Rejected by author: prefill GEMMs must remain strictly W4A16.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **VRAM Budget (4 GB RTX 3050 Ceiling)**: Total added workspace memory (`d_prefill_splitk` + any `PREFILL_MAX_CHUNK` increase) must stay within `50–100 MB` (target: `16–32 MB` Split-K buffer + `7.6 MB` if `PREFILL_MAX_CHUNK = 512`).
*   **Decode Latency Isolation**: Decode kernels (`gemv_w4a8`, `attention_gqa_flash_decode`) and CUDA Graph capture (`decode_graph_`) must remain untouched so T4 decode throughput stays `>= 216.79 tok/s`.
*   **Zero External Dependencies**: Pure CUDA/C++ (`cuda_runtime.h`, `cuda_fp16.h`, `mma.h`, inline PTX) only.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Files to Modify

#### 1. [include/config.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/config.cuh) (Modify)
*   **Type / Scope**: CUDA Compile-Time Configuration
*   **Proposed Changes**:
    *   Add `PREFILL_SMEM_STAGES = 2` for double-buffered `PrefillSmem[2]` (`BK = 32` unchanged).
    *   Add small-M tile constants: `PREFILL_SMALL_BM = 32`, `PREFILL_SMALL_BN = 64` (and/or `32`), and `PREFILL_MAX_SPLIT_K = 8`.
    *   Evaluate raising `PREFILL_MAX_CHUNK` from `256` to `512` (`+7.6 MB` activation workspace) after Phase 1–2 profiling confirms net speedup on `M >= 512`.
    *   **Rationale**: Centralizes tile and chunk sizing so kernel templates and `WorkspaceBuffers` stay consistent.

#### 2. [include/quant_primitives.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/quant_primitives.cuh) (Modify)
*   **Type / Scope**: Device Quantization Primitives
*   **Proposed Changes**:
    *   Optimize `unpack_w4_to_half2_quad(uint32_t packed, half scale, half2 out_h2[4])` to unpack 4-bit nibbles into `half2` pairs using FP16x2 arithmetic/bit-manipulation (`__hsub2` / `__hmul2` with `__half2half2(scale)`) instead of 8x FP32 conversions.
    *   **Rationale**: Removes ~24–32 FP32/conversion instructions per 8-weight word inside the prefill mainloop without changing function signature or numerical behavior. (Verify decode `gemv_w4a16` if it shares this helper.)

#### 3. [include/kernels.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/kernels.cuh) (Modify)
*   **Type / Scope**: Kernel Launch Declarations
*   **Proposed Changes**:
    *   Extend `launch_gemm_prefill_w4a16` and `launch_gemm_prefill_qkv_w4a16` with trailing optional parameters (`int sm_count = 0, float* splitk_workspace = nullptr, size_t splitk_workspace_bytes = 0`).
    *   **Rationale**: Passes cached device SM count and preallocated Split-K scratch memory from `QwenEngine` while preserving backward compatibility for existing unit tests that omit those arguments.

#### 4. [src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu) (Modify)
*   **Type / Scope**: Pure-CUDA Prefill GEMM Kernels & Dispatcher
*   **Proposed Changes**:
    *   **Templated `PrefillSmem` & mainloop helpers**: Parameterize `BM, BN, BK, STAGES, HAS_RESIDUAL, SPLIT_K` so `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, and `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel` share one double-buffered (`STAGES = 2`, `BK = 32`) register-staged pipeline. Guard optional `A`-tile `cp.async` under `#if __CUDA_ARCH__ >= 800`.
    *   **Vectorized epilogue**: Pair adjacent FP32 elements from `s_C[warp_id][16][18]` into `half2` (`__float22half2_rn`), add vectorized `half2` residual loads, and issue `half2`/`uint4` global stores when `N` is aligned.
    *   **SM-count-aware dispatch (`select_prefill_gemm_config`)**: Select small-M tiles (`BM = 32`) and/or `split_k = 2..8` (aligned to `QUANT_GROUP_SIZE = 128`) for `w4a16` and `qkv` when `grid_blocks < 2 * sm_count` and `splitk_workspace` is sufficient; keep `dual_gate_up_swiglu` at `64x64`, `split_k = 1`.
    *   **Split-K reduction kernel (`gemm_prefill_splitk_reduce_kernel`)**: Sum `split_k` FP32 slices from `splitk_workspace`, add `residual` (safe when `residual == output_y`), and write vectorized FP16 to `output_y`.
    *   **Rationale**: Eliminates SM starvation on 40-SM GPUs, hides global/dequant latency behind Tensor Core math, and avoids kernel duplication and in-place residual data races.

#### 5. [include/engine.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/engine.cuh) & [src/engine.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/engine.cu) (Modify)
*   **Type / Scope**: Engine Arena, Workspace, & Prefill Execution
*   **Proposed Changes**:
    *   Add `float* d_prefill_splitk{nullptr}` and `size_t prefill_splitk_bytes{0}` to `WorkspaceBuffers`, and `int sm_count_{0}` to `QwenEngine`.
    *   In `QwenEngine::setup_arena()`: query `cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device_id_)` once and allocate `d_prefill_splitk` (`<= 16–32 MB`, e.g., `8 * PREFILL_MAX_CHUNK * 2048 * sizeof(float)` = `16 MB` at `256` or `32 MB` at `512`).
    *   Pass `sm_count_` and `ws_.d_prefill_splitk` from `execute_gqa_prefill_layer`, `execute_prefill_chunk`, and `profile_stages` to `launch_gemm_prefill_w4a16` and `launch_gemm_prefill_qkv_w4a16`.
    *   **Rationale**: Ensures zero per-layer driver overhead, bounds extra VRAM well below `50–100 MB`, and keeps `decode_graph_` untouched.

#### 6. [CMakeLists.txt](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/CMakeLists.txt) (Modify)
*   **Type / Scope**: Build Configuration
*   **Proposed Changes**:
    *   Update default `CMAKE_CUDA_ARCHITECTURES` from `"86"` to `"75;86"` (still overridable via `ENV{CMAKE_CUDA_ARCHITECTURES}`).
    *   **Rationale**: Produces a single fatbinary supporting both Kaggle T4 (`sm_75`) and RTX 3050 (`sm_86`) out of the box.

### Verification Plan

*   **Compilation/Build**: Build cleanly for `CMAKE_CUDA_ARCHITECTURES="75;86"` (and verify isolated `"75"` and `"86"` builds) with zero compiler warnings or `sm_80+` PTX leaks on `sm_75`.
*   **Unit & Numerical Parity Tests**: Run `test_kernels` (including new Split-K + small-M + residual-aliasing cases) and `tools/verify_reference.py` to confirm FP16 output parity against reference.
*   **VRAM & Decode Regression Check**: Inspect `ArenaStats::workspace_bytes` on RTX 3050 (4 GB) to confirm added VRAM is `<= 50–100 MB`, and verify T4 decode stays `>= 216.79 tok/s`.
*   **End-to-End Benchmark**: Re-run [examples/benchmark_llamacpp.ipynb](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/examples/benchmark_llamacpp.ipynb) on Kaggle T4 at `PROMPT_TOKENS = 128, 512, 1024, 2048` (and RTX 3050 at `128, 512`) and confirm `qwen-engine` beats `llama.cpp` on both prefill and decode.

--------------------------------------------------------------------------------

*Drafted with EGM*
