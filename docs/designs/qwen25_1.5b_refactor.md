# Qwen 2.5-1.5B Single-GPU W4A16 Refactor & Optimization

*   **Author:** Areen-09
*   **Status:** Proposed
*   **Date:** 2026-11-14

--------------------------------------------------------------------------------

## Overview

`qwen-engine` currently supports three model families and two execution topologies in one codebase: Qwen 2.5-1.5B (dense GQA), Qwen 3.5-4B (hybrid DeltaNet + GQA), and Qwen 3.8-27B (hybrid DeltaNet + GQA with 2-way Tensor Parallelism), alongside both FP16 and W4A16 quantization and an HTTP server. Maintaining all of those combinations increased build and maintenance cost while leaving the target RTX 3050 GPU with unoptimized generic code paths.

This design narrows `qwen-engine` to a single-purpose, single-GPU inference engine built strictly for **Qwen 2.5-1.5B-Instruct** using **W4A16 (Group 128)** weights (with dynamic `Q8_1` 8-bit activation quantization for `batch=1` W4A8 decode and FP16 activations for prefill) on a Windows laptop **RTX 3050 (4 GB VRAM, Ampere SM 8.6)**. Removing unused models, Tensor Parallelism, FP16 runtime kernels, and the HTTP server simplifies the Python API and enables tuning the core decode and prefill kernels to beat `llama.cpp` (`Q4_K_M`) in both prefill and decode throughput (`tok/s`).

## The Problem

1. **Multi-model and multi-GPU complexity slows down iteration:** Significant portions of the C++/CUDA and Python codebase handle DeltaNet recurrence, 2-way Tensor Parallelism (`TP=2` via P2P/NCCL), FP16 runtime kernels, and an unused HTTP server (`qwen_server` / `qwen_chat`). `CMakeLists.txt` compiles six CUDA architectures (`75;80;86;89;90;90-virtual`), inflating local rebuild times.
2. **Runtime generality prevents SM 8.6-specific optimizations:** Generalizing across `head_dim=128` vs. `256`, `GQA` group sizes up to 16, and dual-rank synchronization prevents compile-time specialization and leaves unoptimized 32-bit weight loads in decode GEMV and un-tiled global memory reads in prefill attention.
3. **Unmeasured stage latency and fragmented entrypoints:** `QwenEngine::profile_stages()` currently returns zeros, so per-stage bottlenecks against `llama.cpp` are unmeasured. Meanwhile, the Python package exposes multi-GPU options, has mismatched version metadata (`0.2.0` vs. `0.1.0`), and splits setup/benchmarking across multiple notebooks.

## Goals

*   **Beat `llama.cpp` (`Q4_K_M`) on RTX 3050 (4 GB, Windows):** Achieve higher `batch=1` decode throughput (`tok/s`) and higher prompt prefill throughput (`tok/s`) than `llama.cpp` `Q4_K_M` (`llama-bench.exe -ngl 99 -fa 1 -p 128,512,1024 -n 128,256`) on the same GPU.
*   **Single-model, W4A16-only codebase:** Remove DeltaNet, `TP=2` (`communicator_p2p`, `communicator_nccl`), FP16 runtime kernels/paths, `main_server.cpp`, `web/`, and multi-model exporters. Specialize on Qwen 2.5-1.5B dimensions (`HIDDEN_SIZE=1536`, `INTERMEDIATE_SIZE=8960`, `NUM_LAYERS=28`, `NUM_Q_HEADS=12`, `NUM_KV_HEADS=2`, `HEAD_DIM=128`, `VOCAB_SIZE=151936`).
*   **Preserve `.qwenbin` v2 compatibility:** Keep existing `models/qwen25_1.5b_w4a16.qwenbin` files (216-byte `LayerDescriptor` layout; see [SPECIFICATION.md](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/docs/SPECIFICATION.md)) loadable without re-export.
*   **Measurable stage profiling:** Implement real CUDA-event timing in `profile_stages()` for both prefill and ungraphed decode so per-stage bottlenecks are directly observable before kernel tuning.
*   **Numerical parity gate:** Maintain `test_kernels` and W4A16 reference logit parity (`tools/verify_reference.py`) after cleanup and after every kernel optimization.
*   **Simpler Python API & notebook:** Clean up `qwen_engine` (single `device_id=0`, consistent `0.2.0` version, updated docs) and provide a dedicated `qwen25_1.5b.ipynb` notebook covering inference and `llama.cpp` benchmark comparison.
*   **Clean repository state:** Remove stray `.obj` files in root, allow `qwen25_1.5b.ipynb` in `.gitignore`, and keep `tools/llama_cpp/` untracked locally.

## Non-Goals

*   Supporting Qwen 3.5, Qwen 3.8, or other Qwen 2.5 sizes (`0.5B`, `3B`, `7B+`).
*   Multi-GPU / Tensor Parallelism (`TP=2`) or NCCL support.
*   FP16 runtime inference or maintaining FP16 CUDA kernels.
*   HTTP server (`main_server.cpp`) or web UI (`web/`).
*   Multi-request concurrent batching (`batch > 1` decode) unless needed after `batch=1` targets are met.

## Critical User Journeys (CUJs)

### CUJ 1: Simple Python Inference on Laptop RTX 3050
*   **Given:** A Windows machine with an RTX 3050 (4 GB VRAM), NVIDIA driver installed, and `qwen-engine` installed via `pip install .` (or editable `pip install -e . --no-build-isolation`).
*   **When:** The user loads `models/qwen25_1.5b_w4a16.qwenbin` and `models/tokenizer.bin` using `Engine.from_pretrained(...)` on `device_id=0` and calls `engine.generate(...)` or `engine.stream(...)`.
*   **Then:** The process stays under 1.1 GB total VRAM, streams valid UTF-8 ChatML output matching reference logits via CUDA Graphs, and exits cleanly with no multi-GPU or model-type configuration required.

### CUJ 2: Reproducible Benchmark vs. `llama.cpp` (`Q4_K_M`)
*   **Given:** The compiled `cli_bench` / `qwen_engine` build and local untracked `tools/llama_cpp/llama-bench.exe` with `models/qwen25_1.5b_w4a16.qwenbin` and `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`.
*   **When:** The user runs the benchmark cells in `qwen25_1.5b.ipynb` (or `cli_bench --profile-stages`) at prompt lengths `128, 512, 1024` and decode lengths `128, 256`.
*   **Then:** Both engines run on the same RTX 3050, reporting prefill `tok/s`, `batch=1` decode `tok/s`, and peak VRAM, with `qwen-engine` outperforming `llama.cpp` `Q4_K_M` on throughput.

## Technical Plan (Architecture)

### 1. Simplified Single-GPU, Single-Model Pipeline

Removing DeltaNet, `TP=2`, FP16 runtime kernels, and the HTTP server collapses the engine into a single linear pipeline specialized for Qwen 2.5-1.5B-Instruct. In decode (`batch=1`), `rmsnorm_q8_1` dynamically quantizes the FP16 residual stream into `Q8_1` (int8 values + block scales + sums) so W4A8 `__dp4a` GEMV kernels execute at integer throughput; `sample_token` writes the next token ID and updated `seq_pos` into `DeviceControlBlock` (a small device-global struct read by the static CUDA Graph so no host graph updates are needed).

```text
  models/qwen25_1.5b_w4a16.qwenbin (.qwenbin v2, mmap)
                         │
                         ▼
         QwenEngine (Single GPU, cuda:0)
         ├─ Compile-time Qwen 2.5-1.5B dims (config.cuh)
         ├─ W4A16 (G128) weights + FP16 KV cache (max_seq_len <= 4096)
         └─ Dedicated Q8_1 scratch (16 KB, separate from d_act_1)
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  Prefill (prompt_len > 1)         Decode (batch=1, CUDA Graph)
  1. embed_lookup_w4a16            1. embed_lookup_w4a16
  2. 28x GQA Layer:                2. 28x GQA Layer:
     - rmsnorm_fp16                   - rmsnorm_q8_1 (W4A8)
     - gemm_prefill_w4a16 (QKV)       - gemv_qkv_w4a8 (uint4)
     - qkv_bias_rope_kv_append        - qkv_bias_rope_kv_append
     - attention_gqa_prefill (tiled)  - attention_gqa_flash_decode (GQA=6)
     - gemm_prefill_w4a16 (O + res)   - gemv_w4a8 (O + res, uint4)
     - rmsnorm_fp16                   - rmsnorm_q8_1
     - gemm_prefill_dual_swiglu       - gemv_dual_gate_up_swiglu_w4a8
     - gemm_prefill_w4a16 (Down+res)  - gemv_w4a8 (Down + res, uint4)
  3. final rmsnorm + lm_head       3. final rmsnorm_q8_1 + lm_head
  4. sample_token (Philox/argmax)  4. sample_token -> DeviceControlBlock
```

### 2. Phase 1: Cleanup, Bugfixes, Profiling & Baseline

*   **Compile-time Specialization (`include/config.cuh`, `include/tokenizer.hpp`, `CMakeLists.txt`):**
    *   Remove `QWEN38_*` constants, `TOKEN_*_QWEN38` (and update `QwenTokenizer::is_eos()` in `include/tokenizer.hpp`), `LayerType::LAYER_DELTANET`, `QuantMode::FP16`, and legacy v1 `QWQwenHeader` constants.
    *   Preserve the 216-byte `LayerDescriptor` binary layout by replacing the 5 DeltaNet `uint64_t` fields with `uint64_t reserved_deltanet[5]` so existing `.qwenbin` v2 files load unchanged.
    *   Default `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to `"86"` (overridable via env var); remove `ENABLE_NCCL`, `communicator_*`, `deltanet_sm75.cu`, `gemv_fp16.cu`, `qwen_server`, and `qwen_chat`.
*   **Engine Cleanup, Workspace Fix & Profiling (`include/engine.cuh`, `src/engine.cu`):**
    *   Remove `WorkerOp`, `CompanionRankState`, `rank1_`, `init_rank1`, `deltanet_state()`, `conv1d_state()`, `dispatch_worker`, `wait_worker_done`, `residual_add_kernel`, `execute_deltanet_*`, FP16 dispatch, v1 loader branch, and `device_ids_.size() > 1` branches.
    *   Remove the `128 MB` minimum workspace floor (+ `64 MB` padding) in `setup_arena()` and pin `PREFILL_MAX_CHUNK = 256` (~7.8 MB workspace).
    *   Allocate a dedicated **16 KB** aligned Q8_1 scratch buffer (`9088` bytes required for `max(H=1536, I=8960)` quantized Q8_1 values + scales + sum8, padded to 16 KB) instead of aliasing `ws_.d_act_1 + 65536`.
    *   Implement `StageProfile` (`embed_ms`, `rmsnorm_ms`, `qkv_ms`, `rope_kv_ms`, `attn_ms`, `o_proj_ms`, `mlp_ms`, `lm_head_ms`, `sample_ms`) using `cudaEventRecord` in `prefill()` and an ungraphed `execute_decode_pass()` inside `profile_stages()`, then record the baseline vs. `llama.cpp` `Q4_K_M`.

### 3. Phase 2: Data-Driven SM 8.6 Kernel Optimizations

Driven by `profile_stages()` measurements against `llama.cpp` `Q4_K_M`:

1. **Vectorized W4A8 Decode GEMV (`src/kernels/gemv_w4a16.cu`):**
   *   Upgrade `gemv_w4a8_kernel`, `gemv_qkv_w4a8_kernel`, and `gemv_dual_gate_up_swiglu_w4a8_kernel` from `uint32_t` to `uint4` (128-bit) loads using an **8-lane sub-warp per row** (`8 lanes × 32 nibbles = 256 weights/step`, which evenly divides both `HIDDEN_SIZE=1536` (`6` steps) and `INTERMEDIATE_SIZE=8960` (`35` steps) and aligns 4 `uint4` loads per `128`-element quant group), targeting the 116 MB `lm_head` and MLP projections.
   *   Add an explicit 16-byte alignment check on `(offset - w_offset)` in `load_model_file()` with automatic fallback to the `uint32_t` kernel if an existing `.qwenbin` v2 tensor is misaligned.
   *   Replace per-block shared-memory copies of `input_q8`/`input_scale`/`input_sum8` with `__ldg` cached global reads on SM 8.6.
2. **Streamlined Single-Block RMSNorm+Q8_1 (`src/kernels/rmsnorm_rope.cu`):**
   *   Keep `rmsnorm_q8_1_kernel` as a single-block `<<<1, 256>>>` launch (avoiding multi-SM cooperative-launch CUDA Graph complexity) and vectorize weight/input loads to `uint4` (128-bit) to reduce SM stall cycles.
3. **Specialized GQA Attention (`src/kernels/attention_gqa.cu`):**
   *   Remove `head_dim == 256` and `MAX_GQA_GROUP = 16` branches; specialize on `HEAD_D = 128`, `GQA_GROUP_SIZE = 6`, `NUM_KV_HEADS = 2`.
   *   For decode (`attention_gqa_flash_decode_stage1_kernel` / `stage2_kernel`), keep a fixed `grid.y = FLASH_DECODE_SPLITS` (`16`) for static CUDA Graph compatibility, early-exit inactive splits on-device via `d_control->seq_pos`, and parallelize across the 6 Q heads per KV head to cut `__syncthreads()` barriers.
   *   For prefill (`attention_gqa_prefill_kernel`), replace the scalar global K/V loop with a shared-memory K/V-tiled FlashAttention kernel (`Br=16`, `Bc=32` K/V tiles in shared memory, online softmax rescaling, and causal offset `q_pos = seq_pos + m` for multi-turn/chunked prefill).
4. **SM 8.6 `cp.async` Prefill GEMM (`src/kernels/gemm_prefill.cu`):**
   *   Remove FP16 prefill kernels.
   *   Add SM 8.6 `cp.async` double-buffering to `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, and `gemm_prefill_dual_swiglu` using packed-W4 shared-memory stage buffers before register dequantization, and add a `BM=32, BN=64, BK=32` tile for `M <= 128` prompts to improve SM occupancy across 16 SMs.

### 4. VRAM Budget (RTX 3050 4 GB, Windows WDDM)

| Component | Footprint | Notes |
| :--- | :--- | :--- |
| W4A16 (G128) Model Weights | ~796 MB | Tied `lm_head` + 28 GQA layers |
| FP16 KV Cache (`max_seq_len=4096`) | 112 MB | `28 * 2 * 2 * 4096 * 128 * 2 B` |
| Activations & Q8_1 Workspace (`PREFILL_MAX_CHUNK=256`) | ~8 MB | 128 MB floor removed; dedicated 16 KB Q8_1 buffer |
| **GpuArena Subtotal** | **~916 MB** | Measured device arena |
| CUDA Context + Graph + WDDM Overhead | ~150 MB | Runtime driver & graph nodes |
| **Total Process VRAM** | **< 1.1 GB** | Leaves ~2.9 GB headroom on 4 GB GPU |

### 5. Simplified Python API & Notebook

*   **`qwen_engine` (`src/python_bindings.cpp`, `src/python/qwen_engine/`):** Remove `device_ids` and `quant_mode` parameters (single `device_id: int = 0`), remove `num_deltanet_layers` and FP16 downloader entries, bump `User-Agent` to `qwen-engine/0.2.0` in `downloader.py`, align `__version__ = "0.2.0"` across `pyproject.toml` and `__init__.py`, and validate `.qwenbin` v2 headers on load (raising `ValueError` on v1 or mismatched model dimensions).
*   **Qwen 2.5-1.5B Notebook (`qwen25_1.5b.ipynb`):** Replace `convert_models_to_qwenbin.ipynb` and `qwen_inference_and_benchmarks.ipynb` with a single `qwen25_1.5b.ipynb` covering weight export/load, ChatML streaming, VRAM stats, and head-to-head throughput benchmarks against `tools/llama_cpp/llama-bench.exe` (`Q4_K_M`).

## Alternatives Considered

### Alternative A: Keep Dynamic Config for All Qwen 2.5 Sizes (0.5B–7B+)
*   **Pros:** Supports multiple Qwen 2.5 checkpoints without recompiling; preserves existing runtime shape flexibility.
*   **Cons:** Retains runtime dimension branches, dynamic KV-cache stride calculations, and larger kernel register pressure (`MAX_GQA_GROUP`, variable `head_dim` paths), which hurt single-model SM 8.6 occupancy and slow down kernel specialization.
*   **Reason for Rejection:** Rejected by author: the engine is targeted strictly at Qwen 2.5-1.5B-Instruct, so compile-time specialization (`HIDDEN_SIZE=1536`, `INTERMEDIATE_SIZE=8960`, `HEAD_DIM=128`, `GQA_GROUP_SIZE=6`) is preferred over multi-size generality.

### Alternative B: Keep FP16 Inference Kernels as an Accuracy Baseline
*   **Pros:** Allows side-by-side FP16 vs. W4A16 perplexity and logit comparison directly inside `qwen-engine`.
*   **Cons:** Forces maintaining separate `gemv_fp16.cu`, FP16 prefill kernels, FP16 embedding lookup, and `QuantMode::FP16` dispatch paths, doubling kernel surface area and test maintenance.
*   **Reason for Rejection:** Rejected by author: accuracy parity is verified offline against PyTorch/HuggingFace reference outputs (`tools/verify_reference.py`) without shipping FP16 runtime kernels.

### Alternative C: Redesign the `.qwenbin` Binary Format for Qwen 2.5-1.5B
*   **Pros:** Allows custom tensor packing or alignment tailored purely to `uint4` SM 8.6 loads without `.qwenbin` v2 header fields.
*   **Cons:** Breaks existing `models/qwen25_1.5b_w4a16.qwenbin` files and requires re-exporting weights before benchmarking.
*   **Reason for Rejection:** Rejected by author: existing `.qwenbin` v2 files must remain loadable as-is; `LayerDescriptor` keeps its 216-byte layout via `reserved_deltanet[5]` and `load_model_file()` checks 16-byte alignment for `uint4` dispatch.

### Alternative D: Keep `tools/llama_cpp` Binaries Tracked in Git
*   **Pros:** Anyone cloning the repo gets `llama-bench.exe` and `llama-cli.exe` immediately without downloading `llama.cpp` separately.
*   **Cons:** Checks dozens of large `.dll`, `.exe`, and `.zip` binaries into git history, bloating repository size.
*   **Reason for Rejection:** Rejected by author: `tools/llama_cpp/` will be kept locally on disk for benchmarking and excluded from git tracking via `.gitignore`.

## Risks & Mitigations

*   **Misaligned `.qwenbin` v2 offsets breaking `uint4` loads:** `load_model_file()` verifies `(offset - w_offset) % 16 == 0` for every weight tensor and falls back to `uint32_t` GEMV if any tensor is unaligned.
*   **Numerical drift from removing FP16 kernels while optimizing W4A16 kernels:** Phase 1 establishes `test_kernels` + `tools/verify_reference.py` W4A16 logit parity (`atol <= 1e-2`, `rtol <= 1e-2`, cosine similarity `>= 0.999`) before any Phase 2 kernel rewrite begins, and re-runs the gate after each kernel change.
*   **CUDA Graph incompatibility during optimization:** Keeping `grid.y = FLASH_DECODE_SPLITS` fixed and keeping `rmsnorm_q8_1_kernel` single-block preserves the zero-update `cudaGraphExec_t` capture.

## Non-Functional Requirements

*   **Performance & Latency:** Beat `llama.cpp` `Q4_K_M` (`llama-bench.exe -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -fa 1 -p 128,512,1024 -n 128,256`) on Windows RTX 3050 (4 GB) for both `batch=1` decode throughput (`tok/s`) and prompt prefill throughput (`tok/s`), with CUDA-event stage timings verifiable via `profile_stages()`.
*   **Memory Footprint:** Total process VRAM (weights + FP16 KV cache at `max_seq_len=4096` + workspace + CUDA context/graph) must stay under 1.1 GB so it runs without out-of-memory errors or WDDM host-memory paging on a 4 GB laptop GPU.
*   **Backward Compatibility:** Existing `models/qwen25_1.5b_w4a16.qwenbin` (`.qwenbin` v2) and `models/tokenizer.bin` must load without re-export.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Phase 1: Cleanup, Bugfixes, Profiling & Baseline

#### Target Files to Delete / Untrack

1. **`src/kernels/deltanet_sm75.cu`, `include/deltanet.cuh`** (Delete)
   * **Type / Scope**: CUDA Kernel & Header
   * **Proposed Changes**: Remove Gated DeltaNet kernels and header declarations; only used by Qwen 3.5/3.8.
2. **`src/communicator_p2p.cu`, `src/communicator_nccl.cu`, `include/communicator.cuh`** (Delete)
   * **Type / Scope**: Multi-GPU Communication
   * **Proposed Changes**: Remove 2-way Tensor Parallelism P2P and NCCL communicators; engine runs strictly on a single GPU.
3. **`src/kernels/gemv_fp16.cu`** (Delete)
   * **Type / Scope**: CUDA Kernel
   * **Proposed Changes**: Remove FP16 decode GEMV kernels; engine supports W4A16 (G128) only.
4. **`src/main_server.cpp`, `web/`** (Delete)
   * **Type / Scope**: HTTP Server & Web Frontend
   * **Proposed Changes**: Remove unused local HTTP/SSE server and web UI assets.
5. **`tools/export_streaming_safetensors.py`, `tools/test_server.py`** (Delete)
   * **Type / Scope**: Multi-Model Exporter & Server Test
   * **Proposed Changes**: Remove streaming safetensors exporter for larger multi-file models and HTTP server test script.
6. **`tests/test_tensor_parallel.py`, `tests/test_dynamic_engine.py`** (Delete)
   * **Type / Scope**: Test Suite
   * **Proposed Changes**: Remove TP=2 and multi-model dynamic engine tests that cover deleted subsystems.
7. **`convert_models_to_qwenbin.ipynb`, `qwen_inference_and_benchmarks.ipynb`** (Delete)
   * **Type / Scope**: Notebooks
   * **Proposed Changes**: Replace multi-model notebooks with a single dedicated `qwen25_1.5b.ipynb`.
8. **`attention_gqa.obj`, `deltanet_sm75.obj`, `engine.obj`, `gemv_w4a16.obj`, `models/qwen25_1.5b_fp16.qwenbin`** (Delete / Clean)
   * **Type / Scope**: Stray Build Artifacts & Unused FP16 Weights
   * **Proposed Changes**: Remove stray `.obj` files from the repo root and remove FP16 weights no longer used at runtime.
9. **`tools/llama_cpp/`** (Untrack via `.gitignore`, keep locally)
   * **Type / Scope**: Local Benchmark Binaries
   * **Proposed Changes**: Run `git rm -r --cached tools/llama_cpp` and keep `tools/llama_cpp/` in `.gitignore` so local `llama.cpp` binaries remain on disk without bloating git.

#### Target Files to Create / Modify (Phase 1)

10. **`.gitignore`** (Modify)
    * **Type / Scope**: Repo Hygiene
    * **Proposed Changes**: Keep `*.obj`, `build/`, `dist/`, `tools/llama_cpp/`, `*.qwenbin`, and `*.gguf` ignored, and add `!qwen25_1.5b.ipynb` (or remove `*.ipynb` ignore) so `qwen25_1.5b.ipynb` is tracked.
11. **`CMakeLists.txt`, `build.bat`, `.github/workflows/wheels.yml`** (Modify)
    * **Type / Scope**: Build & CI Configuration
    * **Proposed Changes**: Default `CMAKE_CUDA_ARCHITECTURES` to `"86"` (overridable via env var); remove `ENABLE_NCCL`, `communicator_*`, `deltanet_sm75.cu`, `gemv_fp16.cu`, `qwen_server`, and `qwen_chat` targets from `CMakeLists.txt`, `build.bat`, and CI workflow.
12. **`include/config.cuh` & `include/tokenizer.hpp`** (Modify)
    * **Type / Scope**: Model Configuration & Tokenizer
    * **Proposed Changes**: Remove `QWEN38_*`, `TOKEN_*_QWEN38`, `LayerType::LAYER_DELTANET`, `QuantMode::FP16`, and legacy `QWQwenHeader` v1 constants. Update `QwenTokenizer::is_eos()` in `include/tokenizer.hpp` to remove `TOKEN_ENDOFTEXT_QWEN38`. Replace the 5 DeltaNet fields in `LayerDescriptor` with `uint64_t reserved_deltanet[5]` to preserve `sizeof(LayerDescriptor) == 216`. Pin `PREFILL_MAX_CHUNK = 256` and Qwen 2.5-1.5B constants (`HIDDEN_SIZE=1536`, `INTERMEDIATE_SIZE=8960`, `NUM_LAYERS=28`, `NUM_Q_HEADS=12`, `NUM_KV_HEADS=2`, `HEAD_DIM=128`, `VOCAB_SIZE=151936`).
13. **`include/engine.cuh` & `src/engine.cu`** (Modify)
    * **Type / Scope**: Core Inference Engine
    * **Proposed Changes**: Remove `WorkerOp`, `CompanionRankState`, `rank1_`, `init_rank1`, `deltanet_state()`, `conv1d_state()`, `execute_deltanet_*`, `residual_add_kernel`, FP16 dispatch, v1 loader branch, and `device_ids_.size() > 1` branches. Remove the `128 MB` workspace floor (+ `64 MB` padding) in `setup_arena()`. Allocate a dedicated `16 KB` aligned Q8_1 scratch buffer (`d_act_q8`, `d_act_scale`, `d_act_sum8`) instead of aliasing `d_act_1 + 65536`. Implement `StageProfile` CUDA-event timing in `QwenEngine::profile_stages()` for `prefill()` and an ungraphed `execute_decode_pass()`. Validate `.qwenbin` v2 header dimensions and `QuantMode::W4A16_G128` on load, throwing `std::runtime_error` on v1 or mismatched models.
14. **`include/kernels.cuh`, `include/tensor.cuh`, `src/kernels/sampling.cu`** (Modify)
    * **Type / Scope**: Kernel Declarations & Sampling Cleanup
    * **Proposed Changes**: Remove FP16, DeltaNet, `PinnedCandidate`, and `launch_sample_candidate` (`TP=2`) declarations/implementations; remove unused `head_dim`/`num_kv_heads` runtime parameters where specialized for Qwen 2.5-1.5B.
15. **`src/cli_bench.cpp`** (Modify)
    * **Type / Scope**: C++ Benchmark CLI
    * **Proposed Changes**: Remove `QuantMode::FP16` and multi-GPU CLI branches; print `StageProfile` fields (`embed_ms`, `rmsnorm_ms`, `qkv_ms`, `rope_kv_ms`, `attn_ms`, `o_proj_ms`, `mlp_ms`, `lm_head_ms`, `sample_ms`) under `--profile-stages`.
16. **`src/python_bindings.cpp`, `src/python/qwen_engine/__init__.py`, `src/python/qwen_engine/engine.py`, `src/python/qwen_engine/downloader.py`, `src/python/qwen_engine/_core.pyi`, `pyproject.toml`** (Modify)
    * **Type / Scope**: Python Package & Bindings
    * **Proposed Changes**: Remove `QuantMode::FP16` and multi-GPU `device_ids` constructor overload from `python_bindings.cpp`; expose single `device_id: int = 0` and `max_seq_len` in `Engine` and `_core.pyi` (fixing `W4A16_G128`); remove `num_deltanet_layers` and FP16 downloader entries; update `User-Agent` to `qwen-engine/0.2.0`, align `__version__ = "0.2.0"`, and update `pyproject.toml` description.
17. **`tools/verify_reference.py`, `tools/export_qwen25.py`, `tools/convert_gguf_to_qwenbin.py`, `tests/test_kernels.cu`, `tests/test_python_bindings.py`, `tests/test_gguf_converter.py`** (Modify)
    * **Type / Scope**: Verification & Unit Tests
    * **Proposed Changes**: Update `tools/verify_reference.py` to test W4A16 logits against HuggingFace/PyTorch reference outputs (`atol <= 1e-2`, `rtol <= 1e-2`, cosine similarity `>= 0.999`). Update `test_kernels.cu` and `test_python_bindings.py` to remove FP16/DeltaNet/TP=2 cases and assert Q8_1 buffer non-overlap (`d_act_q8` outside `[d_act_1, d_act_1 + size)`) and non-zero `profile_stages()` output.

### Phase 2: Data-Driven SM 8.6 Kernel Optimizations & Docs

18. **`src/kernels/gemv_w4a16.cu`** (Modify)
    * **Type / Scope**: CUDA Decode GEMV Kernel
    * **Proposed Changes**: Upgrade `gemv_w4a8_kernel`, `gemv_qkv_w4a8_kernel`, and `gemv_dual_gate_up_swiglu_w4a8_kernel` to 128-bit `uint4` loads using an 8-lane sub-warp per row (`256` weights/step, dividing `1536` and `8960`), with 16-byte alignment check in `load_model_file()` (falling back to `uint32_t` if misaligned), and replace per-block shared-memory activation reloads with `__ldg` cached reads.
19. **`src/kernels/rmsnorm_rope.cu`** (Modify)
    * **Type / Scope**: CUDA Normalization & RoPE Kernel
    * **Proposed Changes**: Remove `embed_lookup_fp16_kernel`. Vectorize `rmsnorm_q8_1_kernel` loads/stores to `uint4` within its single-block `<<<1, 256>>>` launch to preserve static CUDA Graph capture.
20. **`src/kernels/attention_gqa.cu`** (Modify)
    * **Type / Scope**: CUDA GQA Attention Kernels
    * **Proposed Changes**: Remove `head_dim == 256` and `MAX_GQA_GROUP = 16` branches; specialize on `HEAD_D = 128`, `GQA_GROUP_SIZE = 6`, `NUM_KV_HEADS = 2`. In `attention_gqa_flash_decode_stage1_kernel`/`stage2_kernel`, keep fixed `grid.y = FLASH_DECODE_SPLITS` (`16`), early-exit inactive splits via `d_control->seq_pos`, and parallelize across the 6 Q heads per KV head. Replace `attention_gqa_prefill_kernel` with a shared-memory K/V-tiled FlashAttention kernel (`Br=16`, `Bc=32`, online softmax, causal offset `q_pos = seq_pos + m`).
21. **`src/kernels/gemm_prefill.cu`** (Modify)
    * **Type / Scope**: CUDA Prefill GEMM Kernel
    * **Proposed Changes**: Remove FP16 prefill kernels. Add SM 8.6 `cp.async` double-buffering (staging packed W4 + scales in shared memory before register dequantization) to `gemm_prefill_w4a16_kernel`, `gemm_prefill_qkv_w4a16_kernel`, and `gemm_prefill_dual_swiglu`, plus a `BM=32, BN=64, BK=32` tile for `M <= 128` prompts.
22. **`qwen25_1.5b.ipynb`** (Create)
    * **Type / Scope**: User Notebook
    * **Proposed Changes**: Single self-contained notebook for Qwen 2.5-1.5B: export/load `qwen25_1.5b_w4a16.qwenbin`, interactive ChatML streaming, VRAM check, and reproducible `batch=1` decode & prefill `tok/s` benchmark against `tools/llama_cpp/llama-bench.exe` (`Q4_K_M`).
23. **`README.md`, `docs/MODELS.md`, `docs/SPECIFICATION.md`** (Modify)
    * **Type / Scope**: Documentation
    * **Proposed Changes**: Update docs to reflect the focused Qwen 2.5-1.5B single-GPU W4A16 engine, updated quickstart, `reserved_deltanet[5]` note in `SPECIFICATION.md`, and RTX 3050 benchmark results.

### Verification Plan

*   **Compilation / Build:** Build `qwen_core`, `cli_bench`, `test_kernels`, and the Python wheel (`pip install -e . --no-build-isolation`) targeting SM 8.6 with zero compiler warnings or errors.
*   **Unit & Kernel Parity Tests:** Run `test_kernels`, `pytest tests/`, and `python tools/verify_reference.py` after Phase 1 and after each Phase 2 kernel change to verify `atol <= 1e-2`, `rtol <= 1e-2`, and cosine similarity `>= 0.999` against reference logits.
*   **Stage Profiling:** Run `cli_bench --profile-stages` to confirm non-zero CUDA-event stage timings across all 28 layers and `lm_head`.
*   **Head-to-Head Benchmark vs. `llama.cpp` (`Q4_K_M`):** Run `qwen25_1.5b.ipynb` and `tools/llama_cpp/llama-bench.exe -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -fa 1 -p 128,512,1024 -n 128,256`, verifying `qwen-engine` beats `llama.cpp` in both prefill `tok/s` and `batch=1` decode `tok/s` while keeping total process VRAM under 1.1 GB.

--------------------------------------------------------------------------------

*Drafted with EGM*
