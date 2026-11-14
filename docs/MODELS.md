# Qwen Engine: Model Architecture & Deployment Guide

This document specifies the architectural characteristics, memory footprint calculations, and hardware deployment configuration for **Qwen 2.5-1.5B-Instruct** in `qwen-engine`.

---

## 1. Target Model Specification

| Property | Value |
| :--- | :--- |
| **Model** | Qwen 2.5-1.5B-Instruct (`1.54B` parameters) |
| **Hidden Size (`H`)** | `1536` |
| **Intermediate Size (`I`)** | `8960` |
| **Layers (`L`)** | `28` (Dense GQA) |
| **Attention Heads (`N_Q` / `N_KV`)** | `12` Query heads / `2` Key-Value heads (`6:1` GQA) |
| **Head Dimension (`D_head`)** | `128` |
| **Vocabulary Size (`V`)** | `151936` (tied word embeddings) |
| **Max Context Length (`S_max`)** | `4096` |
| **Quantization** | W4A16 Group-128 (`QuantMode::W4A16_G128 = 1`) |
| **Target Hardware** | Single GPU (`cuda:0`) — RTX 3050 Laptop GPU (4 GB VRAM, SM 8.6) |

---

## 2. Architecture & Execution Pipeline

- **Grouped-Query Attention (GQA):** 12 Query heads share 2 Key/Value heads (`GQA_GROUP_SIZE = 6`, `HEAD_DIM = 128`) with NeoX-style Rotary Position Embeddings (`rope_theta = 1000000.0`).
- **Prefill Attention:** Shared-memory K/V-tiled causal FlashAttention (`Br=16, Bc=32`) with online softmax.
- **Decode Attention:** 2D Split-K FlashDecoding (`grid = (2, 16)`) with single-barrier cross-warp merge and fused Q8_1 epilogue quantization for `o_proj`.
- **Linear Projections:**
  - **Prefill:** `64x64x32` WMMA Tensor Core GEMM (`PrefillSmem`, 10 KB shared memory) with on-the-fly W4->FP16 dequantization.
  - **Decode:** 8-lane sub-warp `uint4` (`__ldg`) W4A8 `__dp4a` GEMV with on-the-fly Q8_1 activation quantization.
- **Zero-Update CUDA Graph:** The entire 28-layer decode step + LM head + greedy/top-k sampling is captured into a single `cudaGraphExec_t` driven by `DeviceControlBlock`.

---

## 3. GPU Memory Footprint (`GpuArena`)

All device memory is allocated upfront inside a single contiguous `GpuArena` (`879.35 MB` discrete VRAM used on RTX 3050, `21.47%` of `4095.50 MB`):

$$\text{Total VRAM} = \text{Weights} + \text{KV Cache} + \text{Workspace}$$

### 3.1 Weight Memory (`759.31 MB`)
In W4A16 Group-128 quantization, each weight consumes 4 bits ($0.5\text{ B}$) plus 1 FP16 scale per 128 weights ($\frac{2}{128} = 0.015625\text{ B}$), totaling $\approx 0.515625\text{ B/weight}$.

### 3.2 Static KV Cache (`112.00 MB`)
Allocated in FP16 across all 28 GQA layers up to `max_seq_len = 4096`:
$$\text{KV Cache Bytes} = 28 \times 2 \times 4096 \times 128 \times 2 \times 2 = 117,440,512\text{ bytes} = 112.00\text{ MB}$$

### 3.3 Exact Workspace Region (`8.04 MB`)
Holds `PREFILL_MAX_CHUNK = 512` activation buffers (`d_act_0`, `d_act_1`, `d_q`, `d_k`, `d_v`, `d_attn_out`, `d_mlp_act`), Split-K FlashDecoding partial buffers, logits (`VOCAB_SIZE * 2 B`), sampling scratch, and a dedicated `16 KB` Q8_1 scratch region (`d_act_q8`, `d_act_scale`, `d_act_sum8`).
