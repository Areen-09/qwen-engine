# Handoff: Prefill GEMM Scaling & Pipeline Optimization (Pure CUDA)

## Context & Constraint
- **Hard Constraint:** **0% external libraries** — must remain pure CUDA/C++ (`cuda_runtime.h`, `cuda_fp16.h`, `mma.h` / inline PTX only; **no** cuBLAS, CUTLASS, or third-party GEMM libraries).
- **Target Hardware:** Must scale across both **RTX 3050 Laptop (`sm_86`, 16 SMs)** and higher-SM GPUs such as **Kaggle T4 (`sm_75`, 40 SMs, 65 TFLOPS FP16)**.

---

## Observed Benchmark Gap (Kaggle T4, `M = 128`, `Qwen2.5-1.5B-Instruct`)
From [examples/benchmark_llamacpp.ipynb](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/examples/benchmark_llamacpp.ipynb) (`PROMPT_TOKENS=128, GEN_TOKENS=128, RUNS=5`):

| Engine | Prefill (tok/s) | Decode (tok/s) |
| :--- | ---: | ---: |
| `qwen-engine` (W4A16 G128) | 2,936.06 | 216.79 |
| `llama.cpp` (`Q4_K_M`) | 14,336.89 | 146.73 |
| **Speedup (`qwen-engine` / `llama.cpp`)** | **0.20x** | **1.48x** |

Decode is already **1.48x faster** than `llama.cpp`. Prefill is ~5x slower (~9 TFLOPS achieved vs ~44 TFLOPS for `llama.cpp` on T4).

---

## Root Causes in Current Codebase

### 1. Grid / SM Under-Utilization at Small `M` (`M <= 256`)
In [include/config.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/config.cuh#L32-L35) and [src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu#L360-L412):
- `PREFILL_WMMA_BM = 64`, `PREFILL_WMMA_BN = 64`.
- For `M = 128`, `grid.y = 128 / 64 = 2`.
- For `o_proj` (`N = 1536`), `grid = (24, 2)` = **48 blocks total**:
  - On **RTX 3050 Laptop (16 SMs)**: `48 / 16 = 3.0` blocks/SM (enough waves to hide barrier/memory latency).
  - On **Kaggle T4 (40 SMs)**: `48 / 40 = 1.2` blocks/SM (almost every SM has only 1 block, so it stalls on every barrier).

### 2. Single-Buffered Synchronous Mainloop (No Load/Compute Overlap)
In `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, and `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel` ([src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu#L147-L154)):
- `PrefillSmem` uses a single `10,240 B` buffer (`s_A[64][40]`, `s_B[64][40]`) with `PREFILL_WMMA_BK = 32`.
- Every `K = 32` step does:
  1. `load_a_tile` + `dequant_b_row_pair` (global -> registers -> unpack W4->FP16 -> shared)
  2. `__syncthreads()`
  3. `wmma_step` (only 2 `16x16x16` steps)
  4. `__syncthreads()`
- Tensor Cores sit idle during global loads + dequant, and memory/ALU sit idle during `wmma_step`. For `K = 1536`, each block executes **96 `__syncthreads()` barriers** per GEMM.

### 3. `wmma` + Shared-Memory Epilogue Overhead
- Epilogue ([src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu#L156-L178)) stores each `16x16` accumulator through `smem.s_C[warp_id][16][18]` with `__syncwarp()` and scalar `__float2half` writes per tile.

### 4. Fixed TTFT Overhead Included in `t0 -> t1`
- In [src/engine.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/engine.cu#L554-L574), `QwenEngine::prefill` also executes `launch_rmsnorm_q8_1` + `launch_gemv_w4a8` for `lm_head` (`151936 x 1536`) after the 28-layer prefill chunk. At `M = 128`, this fixed cost is a non-trivial fraction of TTFT.

---

## Key Files to Modify
- [include/config.cuh](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/include/config.cuh) — tiling constants (`PREFILL_WMMA_BM`, `PREFILL_WMMA_BN`, `PREFILL_WMMA_BK`, `PREFILL_SMEM_STRIDE`, `PREFILL_MAX_CHUNK`)
- [src/kernels/gemm_prefill.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/kernels/gemm_prefill.cu) — `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, `gemm_prefill_dual_gate_up_swiglu_w4a16_kernel`, and launch wrappers
- [src/engine.cu](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/src/engine.cu) — `execute_gqa_prefill_layer`, `execute_prefill_chunk`, `prefill` (if dispatching by `M` or adding `split-K` workspace)
- [CMakeLists.txt](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/CMakeLists.txt) — `CMAKE_CUDA_ARCHITECTURES` (ensure any PTX/async instructions are guarded for `sm_75` vs `sm_86+`)

---

## Recommended Engineering Plan (Pure CUDA)

1. **Double-Buffered Shared Memory Mainloop (`PrefillSmem[2]`)**
   - Allocate 2 stages (`~20 KB` shared memory, still fits comfortably on both `sm_75` 64 KB and `sm_86` 100 KB SMs with >= 2–3 blocks/SM).
   - Prefetch stage `0` before the loop; inside the loop, prefetch/dequant stage `kt + 1` into registers (or next SMEM stage) while `wmma_step` / `mma.sync` executes on stage `kt`, cutting barriers per `K`-step in half and overlapping dequant/loads with Tensor Core math.
   - Optionally guard `cp.async` for `__CUDA_ARCH__ >= 800` (`sm_86`) while keeping register-staged double buffering for `sm_75` (T4).

2. **Improve SM Occupancy for Small `M` (`M <= 128`)**
   - Add a **small-`M` tile variant** (`BM = 32` or `BM = 16` with `BN = 64` or `BN = 32`, or **`split-K`** across `K = 1536` / `8960` with FP16/FP32 reduction) when `grid.x * grid.y < 2 * sm_count` (or `M <= 128`), so 40-SM GPUs like T4 get >= 2–4 blocks per SM.
   - Also check `PREFILL_WMMA_BK = 64` (aligns with `QUANT_GROUP_SIZE = 128` -> 2 `BK` steps per scale group) to amortize barrier and loop overhead.

3. **Consider `mma.sync.aligned.m16n8k16` PTX (Optional Phase 2)**
   - Replacing `nvcuda::wmma` with `mma.sync` + `ldmatrix` removes `wmma` layout constraints and allows direct register-to-global vectorized (`half2`/`uint4`) epilogue writes without bouncing through `smem.s_C`.

4. **Benchmark Across Prompt Lengths**
   - Measure `PROMPT_TOKENS` at `128`, `512`, `1024`, and `2048` in [examples/benchmark_llamacpp.ipynb](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/examples/benchmark_llamacpp.ipynb) to separate `lm_head` fixed cost from per-token prefill GEMM throughput.

---

## Verification Checklist
- Numerical parity against reference (`tools/verify_reference.py` / existing C++ & Python tests) after any kernel changes.
- Verify compilation for both `CMAKE_CUDA_ARCHITECTURES=75` (Kaggle T4) and `CMAKE_CUDA_ARCHITECTURES=86` (RTX 3050 Laptop).
- Re-run [examples/benchmark_llamacpp.ipynb](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/examples/benchmark_llamacpp.ipynb) on Kaggle T4 and confirm prefill tok/s improves without regressing decode (`216.79 tok/s`).
