# Implementation Readiness Review Report: Stage 9 Audit

**Target:** `docs/designs/multi_model_test_suite_and_docs_expansion.md` (Lines 244–369)

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| `egm-implementation-readiness` | completed |
| `egm-codebase-details` | completed |
| `egm-style` | completed |

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Complete Absence of Phasing & Incremental Checkpoints - Severity: Blocker
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Lines 244–369 (`## Detailed Implementation Plan`)
* **Discrepancy Description**: The Detailed Implementation Plan does not partition the work into logical, incremental phases. It presents an unordered, flat catalog of files to create/modify followed directly by the verification plan.
* **Rationale & Recommendation**: The Implementation Readiness Standard explicitly mandates: *"The plan must partition the work into logical, incremental phases. Each phase must leave the codebase in a stable, buildable, and testable state."* Reorganize the implementation plan into distinct phases (e.g., Phase 1: CUDA C++ Synthetic Kernel Test Harness in `tests/test_kernels.cu`; Phase 2: Python Synthetic Topology & Rollback Tests in `tests/test_dynamic_engine.py`; Phase 3: Core Documentation Authoring for `README.md`, `docs/MODELS.md`, and `docs/SPECIFICATION.md`). Each phase must include its own build/test checkpoint.

---

### 2. Non-Existent Kernel Launcher `launch_rmsnorm` & Compile-Time `HIDDEN_SIZE` Hardcoding - Severity: Blocker
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Line 254 (`run_synthetic_rmsnorm_suite(int hidden_size)`)
* **Discrepancy Description**: The plan states that `run_synthetic_rmsnorm_suite(int hidden_size)` "launches `launch_rmsnorm`". However:
  1. In `include/kernels.cuh`, the launcher is named `launch_rmsnorm_fp16`, not `launch_rmsnorm`.
  2. `launch_rmsnorm_fp16` has signature `(const half* input, const half* gamma, half* output, int num_tokens, cudaStream_t stream)` and does not take `hidden_size`.
  3. Inside `src/kernels/rmsnorm_rope.cu` (lines 131–135), `rmsnorm_fp16_kernel` hardcodes compile-time constant `HIDDEN_SIZE = 1536` (`constexpr int NUM_H2 = HIDDEN_SIZE / 2;`).
* **Rationale & Recommendation**: Calling `run_synthetic_rmsnorm_suite(int hidden_size)` with dynamic dimensions will fail to compile or will ignore `hidden_size`. Furthermore, `src/kernels/rmsnorm_rope.cu` is not listed in Target Files to Modify. Either:
  - Add `src/kernels/rmsnorm_rope.cu` and `include/kernels.cuh` to the target modification scope to dynamically parameterize `rmsnorm_fp16_kernel` with `hidden_size`, OR
  - Explicitly restrict the unit test to the baseline dimension `HIDDEN_SIZE` (1536) and update the launcher reference to `launch_rmsnorm_fp16`.

---

### 3. RoPE Verification Missing Kernel Signature, Bias Handling, and Tolerance Contracts - Severity: Blocker
* **Reviewer Lens:** `egm-codebase-details` / `egm-implementation-readiness`
* **Affected Scope:** Line 255 (`run_synthetic_rope_suite(int seq_len, int head_dim)`)
* **Discrepancy Description**: The plan specifies `run_synthetic_rope_suite(int seq_len, int head_dim)` to verify RoPE against CPU calculations, but leaves the entire CUDA invocation mechanism undefined:
  1. There is no standalone `launch_rope` launcher in the codebase. Rotary embedding is fused in `launch_qkv_bias_rope_kv_append` (`include/kernels.cuh` lines 202–214).
  2. `launch_qkv_bias_rope_kv_append` requires Q, K, and V bias buffers, KV cache output buffers, and a `DeviceControlBlock* d_control` holding sequence position.
  3. `HEAD_DIM` is hardcoded to 128 in `include/config.cuh` and `rmsnorm_rope.cu`.
  4. The plan does not define how dummy bias/cache pointers are initialized, what `DeviceControlBlock` fields are populated, or what numerical error tolerance applies.
* **Rationale & Recommendation**: Specify the exact launcher called (`launch_qkv_bias_rope_kv_append`), how inputs are wired (e.g. zero-initialized biases, preallocated dummy KV cache, `d_control->seq_pos = 0`), fix `head_dim` to 128 (matching kernel constraints), and establish explicit absolute and relative tolerance bounds.

---

### 4. Single-Step DeltaNet Decode Launcher Mismatch with Sequence Recurrence Sweep - Severity: Blocker
* **Reviewer Lens:** `egm-codebase-details` / `egm-implementation-readiness`
* **Affected Scope:** Lines 262–266 (`run_synthetic_deltanet_suite(int seq_len, int state_dim, int head_dim)`)
* **Discrepancy Description**: 
  1. `launch_deltanet_decode_sm75` (`include/deltanet.cuh`) has signature `(const half* q, const half* k, const half* v, const half* beta, float* s_state, half* out, int batch_size, int num_heads, int head_dim, cudaStream_t stream)`. It executes a *single autoregressive decode step* per invocation, not a multi-step sequence kernel.
  2. The function signature in line 262 accepts `int seq_len`, but line 265 states it "invokes `launch_deltanet_decode_sm75` on GPU across sequence lengths $T \in \{1, 4, 16, 64\}$".
  3. The plan omits `batch_size` and `num_heads`, which are required by the launcher.
  4. `head_dim` and `state_dim` in `src/kernels/deltanet_sm75.cu` are hardcoded to 128x128 in shared memory tiling and thread indexing (`smem_subtile[2][32][64]`).
* **Rationale & Recommendation**: Clarify the signature to `run_synthetic_deltanet_suite()` (or parameterize by `int num_heads`), explicitly specify the CPU sequential simulation loop and the corresponding GPU sequential step loop ($t = 0 \dots T-1$ invoking `launch_deltanet_decode_sm75` with `batch_size = 1`, `head_dim = 128`), and document state initialization (`cudaMemset(s_state, 0, ...)`).

---

### 5. Synthetic GGUF Generation Dimension Mismatch for 4B and 27B Scales - Severity: Blocker
* **Reviewer Lens:** `egm-codebase-details` / `egm-implementation-readiness`
* **Affected Scope:** Lines 277–285 (`test_qwen35_4b_topology_parity` and `test_qwen38_27b_topology_parity`)
* **Discrepancy Description**: 
  1. Lines 278 and 284 direct tests to use `test_gguf_converter.create_synthetic_gguf` with $H=2560$ and $H=5120$. However, `create_synthetic_gguf` in `tests/test_gguf_converter.py` hardcodes:
     - `qwen2.attention.head_count = 2`
     - `qwen2.attention.head_count_kv = 2`
     - `qwen2.attention.key_length = 128`
     This produces $N_{heads} \times D_{head} = 2 \times 128 = 256 \ne H$. In `src/engine.cu`, Conv1d buffer sizing and DeltaNet dispatch expect $N_{heads} \times D_{head} == H$.
  2. A 4-layer mock model with $H=5120, I=27648$ generates tensor weights exceeding ~3.4 GB written to disk during pytest execution.
  3. `tests/test_gguf_converter.py` is omitted from the Target Files to Modify list.
* **Rationale & Recommendation**: 
  - Add `tests/test_gguf_converter.py` to the Target Files list and parameterize `create_synthetic_gguf` with `head_count` and `head_dim` such that $N_{heads} \times D_{head} = H$ (e.g. for 4B, `head_count=20, head_dim=128`; for 27B, `head_count=40, head_dim=128`).
  - Alternatively, keep synthetic topology test dimensions scaled down ($H=256, I=512$) to avoid multi-gigabyte disk writes in pytest tempdirs, while verifying topology ratio (3:1) and selective KV cache arithmetic.

---

### 6. Zero-Decision Violation in `test_kernels.cu` `main()` Call Sites - Severity: Major
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Lines 268–272 (`Update main(int argc, char** argv)`)
* **Discrepancy Description**: The plan states that `main()` must "execute all synthetic suites unconditionally by default." However, `run_synthetic_rmsnorm_suite`, `run_synthetic_rope_suite`, `run_synthetic_deltanet_suite`, and `run_synthetic_conv1d_suite` all take arguments in their proposed signatures. The plan specifies zero concrete values passed at the call sites in `main()`.
* **Rationale & Recommendation**: Explicitly specify the parameter values passed to each suite function in `main()` (e.g. `run_synthetic_rmsnorm_suite(1536)`, `run_synthetic_rope_suite(8, 128)`, `run_synthetic_conv1d_suite(2560, 4)`), or define the functions as parameterless runners that internally iterate over the target dimension sets.

---

### 7. Ambiguous Kernel Coverage and Assertions in SwiGLU Sweeps - Severity: Major
* **Reviewer Lens:** `egm-codebase-details` / `egm-style`
* **Affected Scope:** Lines 256–261 (`run_synthetic_gemv_swiglu_suite()`)
* **Discrepancy Description**: The plan states that `run_synthetic_gemv_swiglu_suite()` "verifies dynamic shared memory sizing... and output bounds without crashing or memory corruption."
  1. The codebase provides three distinct launchers: `launch_gemv_dual_gate_up_swiglu_fp16`, `launch_gemv_dual_gate_up_swiglu_w4a16`, and `launch_gemv_dual_gate_up_swiglu_w4a8`. The plan does not specify which kernel variants are executed.
  2. "Verifies output bounds" is vague: does it assert numerical parity against a CPU reference, or simply check that output values are finite (no NaNs/Infs)?
* **Rationale & Recommendation**: Explicitly list the target kernel variants (FP16 and W4A16), specify input generation procedures (deterministic PRNG), and define exact assertions (e.g. CPU reference parity within $\le 5 \cdot 10^{-3}$, and assertion of zero NaN/Inf outputs).

---

### 8. Incorrect Test Binary Output Path in Verification Plan - Severity: Major
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Lines 353–356 (`./build/tests/Release/test_kernels.exe`)
* **Discrepancy Description**: In `CMakeLists.txt` (line 151), `test_kernels` is added directly at the top-level project (`add_executable(test_kernels tests/test_kernels.cu)`). CMake/MSVC generates the binary in `build/Release/test_kernels.exe` (or `build/test_kernels` on Linux), NOT in `build/tests/Release/test_kernels.exe`.
* **Rationale & Recommendation**: Update the test invocation path in the verification plan to `./build/Release/test_kernels.exe` (Windows) / `./build/test_kernels` (Linux).

---

### 9. Missing Python C-Extension Build Prerequisite in Verification Plan - Severity: Major
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Lines 357–360 (`pytest tests/test_dynamic_engine.py ...`)
* **Discrepancy Description**: The Verification Plan runs `pytest` immediately without instructing the build or installation of the nanobind `_core` C-extension target. On a clean checkout or following C++ changes, running `pytest` will fail with `ModuleNotFoundError: No module named 'qwen_engine._core'`.
* **Rationale & Recommendation**: Insert a compilation/install step prior to pytest execution:
  ```bash
  pip install -e . --no-build-isolation
  # or: cmake --build build --config Release --target _core -j4
  ```

---

### 10. Underspecified Determinism Test Setup in `test_hybrid_rollback_determinism` - Severity: Major
* **Reviewer Lens:** `egm-implementation-readiness` / `egm-style`
* **Affected Scope:** Lines 286–292 (`test_hybrid_rollback_determinism()`)
* **Discrepancy Description**: 
  1. The plan specifies "Invokes `engine.rollback_to(current_pos - 3)`" without specifying whether `engine` is `qwen_engine.Engine` (high-level wrapper requiring `tokenizer.bin`) or `_core.NativeEngine` (low-level, token-ID based). Synthetic models in tempdirs have no tokenizer unless explicitly generated.
  2. The prompt token sequence and sampling parameters are omitted (temperature must be explicitly set to `0.0` for greedy determinism).
* **Rationale & Recommendation**: Explicitly specify that the test operates on `_core.NativeEngine`, define the test prompt (e.g. `[1, 10, 100, 1000]`), set temperature to `0.0f`, and document the exact mock model generation parameters.

---

### 11. Underspecified Content Specifications for Documentation Artifacts - Severity: Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Lines 293–337 (`README.md`, `docs/MODELS.md`, `docs/SPECIFICATION.md`)
* **Discrepancy Description**: Sections for `docs/MODELS.md` and `docs/SPECIFICATION.md` provide high-level topic lists rather than complete drafts or concrete schemas (e.g., exact 28-field ordered layout for `LayerDescriptor`, full LaTeX memory formulas).
* **Rationale & Recommendation**: Provide explicit table templates and exact field name lists directly in the plan so the implementer does not have to cross-reference header files or invent table formatting.

---

### 12. Multi-Rank Loading Contract Omitted for `Engine.from_pretrained` - Severity: Nit
* **Reviewer Lens:** `egm-codebase-details` / `egm-style`
* **Affected Scope:** Lines 302–303 (`README.md` Quickstart)
* **Discrepancy Description**: Line 302 suggests `Engine.from_pretrained(repo_id="...", filename="...")` alongside `device_ids=[0, 1]`. For Qwen 3.8-27B, the model is split across two separate files (`qwen3.8-27b-w4a16-rank0.qwenbin` and `qwen3.8-27b-w4a16-rank1.qwenbin`). `Engine.from_pretrained` does not currently accept multiple filenames or auto-resolve rank prefixes.
* **Rationale & Recommendation**: Document the exact dual-GPU invocation contract in the Quickstart guide (e.g. passing the rank0 file path or downloading both rank files).

---

## Conclusion & Gate Recommendation

**Standard Evaluation:** 🔴 **FAIL (Standard Not Met)**  
The Detailed Implementation Plan violates the **Zero-Decision Rule** and the **Phasing & Checkpoints** mandate of the Implementation Readiness Standard. An implementer unfamiliar with the engine internals would encounter broken builds, unresolved kernel launcher names, and missing test execution parameters.

Addressing the 5 Blocker findings and 5 Major findings detailed above will bring the document to complete implementation readiness.

