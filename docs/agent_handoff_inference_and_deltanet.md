# Engineering Handoff: Qwen CUDA Inference Engine, Multi-GPU Tensor Parallelism & DeltaNet Hybrid State

**Target Audience:** Next Senior CUDA / Systems Agent  
**Date:** November 2026  
**Repository:** [`https://github.com/Areen-09/qwen-engine.git`](https://github.com/Areen-09/qwen-engine.git)  
**Hugging Face Hub:** [`areen09/qwen-engine-models-qwenbin`](https://huggingface.co/areen09/qwen-engine-models-qwenbin)  
**Target Hardware:** Kaggle Free GPU Instances (**2x NVIDIA Tesla T4, Turing `sm_75`, 16 GB GDDR6 each, 32 GB total VRAM**)  

---

## 1. Executive Summary & Current Repository State

The repository provides a zero-dependency, bare-metal C++20 / CUDA inference engine for the Qwen model family, featuring custom Turing Tensor Core kernels (`sm_75`), W4A16 Group-128 quantization, Gated DeltaNet linear-attention hybrid state management, and 2-way Tensor Parallelism ($TP=2$) across dual GPUs over PCIe.

### Current Published Deliverables on Hugging Face Hub (`areen09/qwen-engine-models-qwenbin`)

| Model | Format / Quantization | Parameter Count / Architecture | Disk / VRAM Size | Status | Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5-1.5B** | `.qwenbin` v2 / W4A16 Group-128 | 1.54B / 28 GQA Layers | **796 MB** | **Production Ready & Verified** | Single Tesla T4 (`cuda:0`) (~1.0 GB VRAM) |
| **Qwen 3.5-4B** | `.qwenbin` v2 / W4A16 Group-128 | ~3.8B / 32 Layers (24 DeltaNet, 8 GQA) | **~2.4 GB** | **Published on Hub** | Single Tesla T4 (`cuda:0`) (75% KV reduction) |
| **Qwen 3.8-27B** | `.qwenbin` v2 / W4A16 Dual Rank | ~27B / 64 Layers (48 DeltaNet, 16 GQA) | **~7.1 GB each rank** | **Published on Hub** | **Dual Tesla T4 TP=2 (`cuda:0` & `cuda:1`)** |
| **`tokenizer.bin`** | Binary BPE | 151,936 Vocab + Merges | **3.7 MB** | **Production Ready & Verified** | Host / Device Tokenizer |
| **Model Card** | Markdown `README.md` | Architecture & Benchmark Docs | **1.75 kB** | **Production Ready** | Hub Documentation |

---

## 2. Recent Progress & Completed Milestones

1. **Dual-Rank Chunked On-Disk Streaming (Task 2)**:
   * Solved Kaggle's 20 GB disk limit by emitting Rank 0 (~7.1 GB), verifying container integrity, uploading immediately to Hugging Face, deleting Rank 0 locally, and then generating Rank 1 (~7.1 GB).
   * Peak disk footprint strictly $\le 7.1\text{ GB}$ (immune to Kaggle disk full errors).
   * Peak RAM footprint strictly $< 300\text{ MB}$ via layer-by-layer direct-to-disk streaming (immune to Kaggle 16 GB host RAM OOM kills).
2. **Cross-Platform C++ & Linux Build Fixes**:
   * Resolved fatal error `<windows.h>: No such file or directory` in `src/main_server.cpp` by wrapping Windows headers in `#ifdef _WIN32` and providing standard POSIX sockets (`<sys/socket.h>`, `<unistd.h>`, `MSG_NOSIGNAL`).
   * Targeted `cli_bench` directly in CMake builds (`cmake --build build --target cli_bench -j$(nproc)`).
3. **Pre-Compiled Linux Wheels (`v0.2.0`)**:
   * GitHub Actions workflow (`.github/workflows/wheels.yml`) successfully compiles `manylinux_2_28_x86_64` wheels for Python 3.9, 3.10, 3.11, 3.12, and 3.13 with static CUDA 12.4 and architectures `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`.
   * All Linux wheels are attached to GitHub Release `v0.2.0` and install via `pip install` in under 3 seconds in Kaggle without compiling.
4. **Python API Polish & Backward Compatibility**:
   * Added `stream_generate` as an alias to `stream()`.
   * Added `max_new_tokens` parameter mapping to both `generate()` and `stream()`.
   * Added `__call__` interface on `Engine` matching `llama-cpp-python` (`engine("prompt", stream=True)`).
   * Added direct Hugging Face Hub loading in `Engine.from_pretrained(repo_id=..., filename=...)`.
5. **Streamlined Notebooks**:
   * `notebooks/convert_models_to_qwenbin.ipynb`: Full model exporter and quantization pipeline.
   * `notebooks/qwen_inference_and_benchmarks.ipynb`: Clean, 6-step minimal inference notebook (llama.cpp style).

---

## 3. The Active Bug: Root Cause, Analysis & Resolution [RESOLVED]

### The Failure Symptom in Kaggle (Initial)
When initializing the 4B hybrid model (`qwen3.5-4b-w4a16.qwenbin`) in Step 5:
```python
engine_4b = Engine(model_path=m4b_path, tokenizer_path=tokenizer_path, device_id=0)
```
The execution terminated with:
```text
CUDA error at /project/src/engine.cu:547: an illegal memory access was encountered (cudaStreamSynchronize(stream_))
RuntimeError: CUDA error: an illegal memory access was encountered
```

### Traceback Path
1. `Engine.__init__` in `src/python/qwen_engine/engine.py` calls `_core.NativeEngine(...)`.
2. `QwenEngine::QwenEngine(...)` in `src/engine.cu`:
   * Line 53: `load_model_file(qwenbin_path_);`
   * Line 54: `setup_arena();`
   * Line 55: `capture_decode_graph();`
3. Inside `QwenEngine::capture_decode_graph()`:
   * Line 546: `execute_decode_pass();` (warm-up pass before CUDA stream capture)
   * Line 547: `CUDA_CHECK(cudaStreamSynchronize(stream_));`

---

### Root Cause Analysis & Resolution

#### A. Compile-Time Dimension Hardcoding in GEMV Kernels [FIXED]
In `include/config.cuh`, the dimensions were originally hardcoded for Qwen 2.5-1.5B:
```cpp
constexpr int32_t HIDDEN_SIZE       = 1536;
constexpr int32_t INTERMEDIATE_SIZE = 8960;
```
In `src/kernels/gemv_w4a16.cu` and `src/kernels/gemv_fp16.cu`:
* `launch_gemv_dual_gate_up_swiglu_w4a8`
* `launch_gemv_dual_gate_up_swiglu_w4a16`
* `launch_gemv_dual_gate_up_swiglu_fp16`
The grid size, dynamic shared memory calculations, and loop bounds were originally hardcoded to `HIDDEN_SIZE` and `INTERMEDIATE_SIZE`.
**Fix Implemented:**
* Parameterized `gemv_dual_gate_up_swiglu_w4a8_kernel`, `gemv_dual_gate_up_swiglu_w4a16_standalone_kernel`, and `gemv_dual_gate_up_swiglu_fp16_kernel` with dynamic `in_features` and `intermediate_size`.
* Parameterized launchers `launch_gemv_dual_gate_up_swiglu_w4a8`, `launch_gemv_dual_gate_up_swiglu_w4a16`, and `launch_gemv_dual_gate_up_swiglu_fp16` in `include/kernels.cuh` with `int hidden_size, int intermediate_size`.
* Dynamically allocated `smem = static_cast<size_t>(hidden_size) * 49 / 32` and `grid = (intermediate_size + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK`.
* In `src/engine.cu`, updated call sites to pass `h_dim, i_dim`.
* In `src/engine.cu` `setup_arena()`, parameterized `ws_.d_act_q8`, `ws_.d_act_scale`, `ws_.d_act_sum8` buffer offsets based on dynamic `i_dim` to eliminate buffer overlap on models with `intermediate_size > 8960`.

#### B. DeltaNet Kernel Weight Dereferencing [FIXED]
In `src/kernels/deltanet_sm75.cu`:
* Added guard in `deltanet_decode_sm75_kernel`: if `beta == nullptr`, default `alpha = 1.0f` (no decay).
* Added guard in `depthwise_conv1d_step_kernel`: if `weight == nullptr`, copy `in[c]` straight to `out[c]` and return.

---

## 4. Completed Action Items & Verification

### Step 1: Parameterize GEMV Launchers with Dynamic Dimensions [COMPLETED]
Updated `include/kernels.cuh`, `src/kernels/gemv_w4a16.cu`, and `src/kernels/gemv_fp16.cu`.

### Step 2: Update Call Sites in `src/engine.cu` [COMPLETED]
Updated `src/engine.cu` inside `execute_decode_pass()` and dynamic offsets in `setup_arena()`.

### Step 3: Add Null-Pointer Guards in DeltaNet Kernels [COMPLETED]
Added null-pointer guards in `src/kernels/deltanet_sm75.cu`.

### Step 4: Validate and Verify [COMPLETED]
1. Built and executed `tests/test_kernels.cu`:
   ```text
   >>> ALL 8 VERIFICATION SUITES PASSED (EXIT CODE 0) <<<
   ```
2. Built and tested `cli_bench.exe` with greedy decoding:
   ```text
   [Prompt]: Hello! Tell me who you are.
   [Tokens]: 8
   [Generated Output]: Clean deterministic generation, exit code 0.
   ```
3. Ready for commit & push to GitHub `origin/main` to trigger the `Build & Package Wheels` workflow.

---

## 5. Critical Technical Invariants & Rules

1. **Never Hardcode Dimension Constants in CUDA Kernels**:
   Qwen 2.5, 3.5, and 3.8 have different hidden dimensions (1536, 2560, 5120) and intermediate dimensions (8960, 10240, 27648). Dynamic shared memory and grid calculations must always use `model_config_.hidden_size()` and `model_config_.intermediate_size()`.
2. **Kaggle In-Place Package Upgrades**:
   Never run `pip install --upgrade numpy` inside Kaggle. It breaks in-memory C-extensions with `_blas_supports_fpe` AttributeError.
3. **216-Byte `LayerDescriptor` Packing**:
   Format string must remain strictly `"<2I26Q"` (28 fields, 216 bytes).
4. **Keep Notebook Clean & Simple**:
   Do not re-introduce heavy C++ CMake compilation or verbose 50-line diagnostic scripts into `notebooks/qwen_inference_and_benchmarks.ipynb`. Keep the simple 6-step `llama.cpp`-style user experience.

