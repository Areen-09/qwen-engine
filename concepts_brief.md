# Multi-Model Test Suite and Documentation Expansion

* **Author:** Areen-09
* **Date:** 2026-11-09
* **Status:** Approved Feature Concept (Ready for egm-draft)
* **Target Architectures:** Qwen 2.5-1.5B (Dense GQA), Qwen 3.5-4B (Hybrid Single-GPU), Qwen 3.8-27B (Hybrid Dual-GPU TP=2)
* **Target Hardware:** NVIDIA Tesla T4 (`sm_75`, Single and Dual-GPU)

---

## 1. Overview & Motivation

The repository originated as a bare-metal CUDA C++20 inference engine for dense `Qwen2.5-1.5B-Instruct`. It now supports hybrid linear-attention models:
- **Qwen 3.5-4B:** Single-GPU hybrid alternating Gated DeltaNet (`sm_75` 16 KB SMEM cooperative recurrence) and GQA in a 3:1 stacking ratio.
- **Qwen 3.8-27B:** Dual-GPU Tensor Parallel ($TP=2$) hybrid architecture over PCIe (direct BAR1 P2P or staging fallback on Windows, NCCL on Linux).

However, test coverage and repository documentation remain anchored to the initial 1.5B dense baseline:
1. `tests/test_kernels.cu` hardcodes 1.5B dimensions (hidden size 1536, intermediate size 8960) and aborts if `models/golden_ref.bin` is missing. It executes zero tests on Gated DeltaNet recurrence or depthwise Conv1d steps.
2. The testing suite lacks parameterized sweeps to verify matrix-vector operations across hidden dimensions 2560 (4B) and 5120 (27B).
3. Documentation is fragmented across handoff notes, lacking a consolidated user guide for multi-GPU deployment on accessible hardware (such as Kaggle 2x T4 instances) and lacking a formal `.qwenbin` v2 technical specification for automated agents.

This initiative establishes:
1. A two-tiered test suite: standalone parameterized mathematical validation on synthetic tensors (zero file dependencies) alongside an opt-in golden reference parity path.
2. Full mathematical and state-rollback validation for DeltaNet recurrence and depthwise Conv1d.
3. A decoupled dual-audience documentation architecture: user quickstart and hardware support matrix (`README.md`, `docs/MODELS.md`) and technical binary specification (`docs/SPECIFICATION.md`).

---

## 2. Concrete Architectural Boundaries & Model Matrix

Support is strictly confined to three concrete model architectures:

| Specification | Qwen 2.5-1.5B | Qwen 3.5-4B | Qwen 3.8-27B |
| :--- | :--- | :--- | :--- |
| **Model Type** | Dense Baseline | Hybrid Linear-Attention | Hybrid Linear-Attention |
| **Layer Arrangement** | 28 GQA (100% Attention) | 32 Layers: 24 DeltaNet + 8 GQA (3:1 ratio) | 64 Layers: 48 DeltaNet + 16 GQA (3:1 ratio) |
| **Hidden Dimension ($H$)** | 1,536 | 2,560 | 5,120 |
| **Intermediate Dimension ($I$)** | 8,960 | 10,240 | 27,648 |
| **Attention Heads (Q / KV)** | 12 / 2 (head dim 128) | 20 / 4 (head dim 128) | 40 / 8 (20 / 4 per rank in TP=2) |
| **Quantization Format** | FP16 / W4A16 Group 128 | FP16 / W4A16 Group 128 | W4A16 Group 128 (Mandatory on dual T4) |
| **Hardware Target** | 1x GPU (>= 2 GB VRAM) | 1x Tesla T4 (16 GB VRAM) | 2x Tesla T4 (TP=2, ~8.75 GB / GPU) |

*Explicit Non-Goals:* Multimodal models (Qwen 2.5-VL), arbitrary layer interleavings, or $N$-way Tensor Parallelism beyond dual-GPU ($TP=2$).

---

## 3. Testing Architecture & Failure Mode Prevention

### 3.1 Eliminating File-Dependency Blockers
- **Current Defect:** `test_kernels.cu` fails immediately at startup when `models/golden_ref.bin` is absent.
- **Remedy:** Restructure the executable into two execution modes:
  1. **Default Mode (Standalone):** Executes synthetic mathematical unit tests with deterministically initialized tensors (seed-based pseudo-random data). Validates embedding lookup, RMSNorm, RoPE, GQA FlashDecoding, W4A16 dequantization GEMV, DeltaNet decode recurrence, and depthwise Conv1d steps. Requires zero external files.
  2. **Golden Reference Mode (`--golden <path>`):** Opt-in verification against pre-generated PyTorch reference activations when available.

### 3.2 DeltaNet & Conv1d Mathematical Parity Verification
- **Recurrent State Formula:**
  $$\alpha = \frac{1}{1 + \exp(\beta)}, \quad S_t = \alpha S_{t-1} + v_t k_t^T, \quad o_t = S_t q_t$$
  Where $S_t \in \mathbb{R}^{128 \times 128}$ FP32, inputs $q, k, v \in \mathbb{R}^{128}$ FP16, output $o_t \in \mathbb{R}^{128}$ FP16.
- **Test Implementation:**
  - Host CPU computes reference recurrence in FP64/FP32 across $T \in \{1, 4, 16, 64\}$ sequence steps.
  - CUDA kernel `launch_deltanet_decode_sm75` executes on device.
  - Parity asserted: Max absolute error $\le 10^{-3}$, Cosine similarity $\ge 0.9999$.
- **Depthwise Conv1d Step:**
  - Host CPU simulates rolling 3-element history buffer with fused SiLU activation.
  - Device kernel `launch_depthwise_conv1d_step` executed; asserts output parity within $\le 5 \cdot 10^{-3}$.

### 3.3 Dynamic Dimension Sweeps
Unit tests in C++ and Python (`tests/test_dynamic_engine.py`) sweep parameterized dimensions:
- $H \in \{256, 1536, 2560, 5120\}$
- $I \in \{512, 8960, 10240, 27648\}$
This validates that kernel grid/block launch bounds, shared memory tiling, and memory indexing correctly handle 4B and 27B dimensions without memory corruption.

### 3.4 Hybrid Rollback Determinism
Extend `tests/test_dynamic_engine.py` with multi-step autoregressive generation followed by session rollback:
- Generate 5 tokens on a 3:1 hybrid DeltaNet/GQA model.
- Rollback sequence position by 3 tokens (`rollback_to(current_pos - 3)`).
- Re-generate 3 tokens; assert identical token IDs to verify recurrent state and KV cache rollback determinism.

---

## 4. Documentation Architecture

Documentation is decoupled into three clear tiers:

### 4.1 Tier 1: Repository Overview & Quickstart (`README.md`)
- Supported models table with parameter sizes, layer configurations, and VRAM requirements.
- Quickstart guide for Python package (`qwen-engine`).
- 2x Tesla T4 Kaggle setup instructions (`Engine(model_path, device_ids=[0, 1])`).

### 4.2 Tier 2: Model Architecture & Deployment Guide (`docs/MODELS.md`)
- Detailed layer-by-layer architectural comparison across 1.5B, 4B, and 27B.
- Memory consumption breakdown (Weights, Selective KV Cache, DeltaNet FP32 State, Dynamic Workspace).
- Quantization guide: FP16 baseline vs W4A16 group-128 warp-interleaved packing.
- Multi-GPU execution on dual-T4 platforms (PCIe P2P, WDDM staging, Linux NCCL).

### 4.3 Tier 3: Technical Engineering Specification (`docs/SPECIFICATION.md`)
- **Container Format:** Full specification of `.qwenbin` v2 container (64-byte aligned header, magic `0x5157454E`).
- **LayerDescriptor Struct:** Binary layout of the 216-byte `LayerDescriptor` (`<2I26Q`), including tensor offsets for GQA, DeltaNet, SwiGLU, and quantization scales.
- **Memory & Invariant Layouts:** Selective KV cache allocation for hybrid models (omitting KV cache on DeltaNet layers, reducing memory footprint by up to 75%).
- **Multi-GPU Communication Protocol:** Dual-rank TP=2 tensor slicing and all-reduce synchronization barriers.

---

## 5. Goals vs. Non-Goals

* **In-Scope Goals:**
  * Refactor `tests/test_kernels.cu` into standalone mathematical unit tests covering dynamic dimensions (1536, 2560, 5120) and Gated DeltaNet recurrence without file dependencies.
  * Implement analytical CPU-vs-CUDA mathematical parity checks for DeltaNet state updates and conv1d steps on synthetic tensors.
  * Expand Python test coverage (`tests/test_dynamic_engine.py`) to systematically validate layer configurations, memory stats, and rollback behavior across 1.5B, 4B, and 27B-equivalent synthetic topologies.
  * Author centralized `docs/MODELS.md` support matrix and `docs/SPECIFICATION.md` covering `.qwenbin` v2 and hybrid memory layout.
  * Update root `README.md` with multi-model capabilities and dual-T4 execution.

* **Out-of-Scope Non-Goals:**
  * Multi-gigabyte model weight downloads or golden reference regeneration in CI pipelines.
  * Multimodal support (Qwen 2.5-VL, vision encoders, MRoPE).
  * Arbitrary N-way Tensor Parallelism beyond dual-GPU ($TP=2$) on Turing/Ampere architectures.

---

## 6. Alternatives Weighed

* *Option A: Monolithic Documentation in `README.md`:* Rejected. Mixes 216-byte binary struct specifications with quickstart tutorials, adding cognitive overhead for end users.
* *Option B: Crash-Only Synthetic Testing:* Rejected. Merely checking that kernels do not trigger CUDA runtime errors fails to detect numerical drift or state corruption in recurrent linear attention.
* *Option C: Requiring Golden References for All Models:* Rejected. Infeasible due to Kaggle 20 GB disk limits and absence of free GPU runners in CI.

---

## 7. Key Decisions & Staging for Next Steps

1. Decouple `test_kernels.cu` into standalone synthetic math suites (default) and opt-in `--golden` suites.
2. Establish a 3-tier documentation layout (`README.md` for users, `docs/MODELS.md` for deployment/benchmarking, and `docs/SPECIFICATION.md` for agents/internals).
3. Ready for formal design specification via `/egm-draft`.
