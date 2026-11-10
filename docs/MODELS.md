# Qwen Engine: Model Architecture & Deployment Guide

This document specifies the architectural characteristics, memory footprint calculations, and hardware deployment strategies across the supported models in the `qwen-engine` ecosystem.

---

## 1. Supported Models Matrix

| Model | Parameters | Hidden Size ($H$) | Intermediate Size ($I$) | Total Layers | Topology & Ratio | Quantization | Disk Size | VRAM Footprint | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5-1.5B** | 1.54B | 1536 | 8960 | 28 | Pure Dense GQA (28 GQA) | W4A16 (G128) | **796 MB** | **~943 MB** | Single GPU (`cuda:0`) / RTX 3050 (4 GB) / Tesla T4 |
| **Qwen 3.5-4B** | ~3.8B | 2560 | 10240 | 32 | Hybrid Linear (24 DeltaNet : 8 GQA, 3:1) | W4A16 (G128) | **~2.4 GB** | **~2.8 GB** | Single GPU (`cuda:0`) / Tesla T4 (16 GB) |
| **Qwen 3.8-27B** | ~27B | 5120 | 27648 | 64 | Hybrid Linear (48 DeltaNet : 16 GQA, 3:1) | W4A16 Dual-Rank ($TP=2$) | **~7.1 GB / rank** | **~7.8 GB / GPU** | Dual GPU (`cuda:0` & `cuda:1`) / 2x Tesla T4 (32 GB total) |

---

## 2. Architectural Comparison

### 2.1 Pure Dense Transformer (`Qwen 2.5-1.5B`)
- **Attention:** 12 Query heads, 2 Key/Value heads (6:1 GQA ratio), 128 head dimension.
- **Context Handling:** Standard causal grouped-query attention evaluated with 2D Split-K FlashDecoding across all 28 layers.
- **KV Cache Allocation:** Allocated for all 28 layers across the full context window (`MAX_SEQ_LEN = 4096`).

### 2.2 Hybrid Linear Attention (`Qwen 3.5-4B`)
- **Interleaving Pattern:** Strict 3:1 stacking ratio:
  $$\text{Layer } l = \begin{cases} \text{GQA (Grouped-Query Attention)}, & \text{if } (l + 1) \equiv 0 \pmod 4 \\ \text{Gated DeltaNet (Linear Attention)}, & \text{otherwise} \end{cases}$$
- **DeltaNet Layers (75% of model):**
  - Depthwise Conv1d step with sliding history window $K=4$ ($history\_len = 3$).
  - FP32 recurrent state matrix update:
    $$\alpha_t = \frac{1}{1 + \exp(\beta_t)}, \quad S_t = \alpha_t S_{t-1} + v_t k_t^T, \quad o_t = S_t q_t$$
  - Sub-tiled cooperative sm_75 kernel (16 KB shared memory, 4 warps).
  - Zero KV cache allocation on disk or in device memory.
- **GQA Layers (25% of model):** Standard FlashDecoding with rotary position embeddings (NeoX RoPE).

### 2.3 Dual-GPU Tensor Parallelism (`Qwen 3.8-27B`, $TP=2$)
- **Interleaving Pattern:** 64 total layers with 3:1 ratio (48 Gated DeltaNet layers, 16 GQA layers).
- **Partitioning Strategy:**
  - Attention Q, K, V projections and MLP Gate, Up projections: Column-parallel split across rank 0 and rank 1.
  - Attention Out projection and MLP Down projection: Row-parallel split across rank 0 and rank 1.
- **Communication:** PCIe P2P direct transfer or host pinned staging between Rank 0 and Rank 1 after row-parallel projections.

---

## 3. GPU Memory Footprint Formulas

The total device memory allocation in `qwen-engine` is composed of five distinct memory regions managed inside a unified contiguous `GpuArena`:

$$\text{Total VRAM} = \text{Weights} + \text{KV Cache} + \text{DeltaNet State} + \text{Conv1d Cache} + \text{Workspace}$$

### 3.1 Weight Memory (W4A16 Group-128)
For a model with $L$ layers, hidden dimension $H$, and intermediate dimension $I$:
$$\text{Weight Bytes} = \text{Bytes}_{\text{Embed}} + \sum_{l=0}^{L-1} \left( \text{Bytes}_{\text{Attn/DeltaNet}}^{(l)} + \text{Bytes}_{\text{MLP}}^{(l)} + \text{Bytes}_{\text{Norm}}^{(l)} \right) + \text{Bytes}_{\text{Head}}$$
- In W4A16 Group-128 quantization, weight elements consume 4 bits ($0.5\text{ bytes}$), plus 1 FP16 scale per 128 weights ($\frac{2}{128} = 0.015625\text{ bytes}$), totaling $\approx 0.515625\text{ bytes/weight}$ (a 74.2% reduction relative to FP16).

### 3.2 Selective KV Cache Memory (75% Footprint Reduction)
Unlike dense transformers which allocate KV caches across all layers, hybrid models in `qwen-engine` allocate KV cache strictly on GQA layers ($N_{\text{GQA}}$):
$$\text{KV Cache Bytes} = N_{\text{GQA}} \times N_{\text{KV\_Heads}} \times \text{MaxSeqLen} \times D_{\text{Head}} \times 2 \times 2$$
where the trailing factor $2 \times 2$ accounts for 2 bytes per FP16 element and 2 distinct projection buffers (Key and Value).
- **Qwen 2.5-1.5B (28 GQA layers, 4096 context):**
  $$28 \times 2 \times 4096 \times 128 \times 2 \times 2 = 117,440,512\text{ bytes} \approx 112\text{ MB}$$
- **Qwen 3.5-4B (8 GQA layers, 4096 context):**
  $$8 \times 2 \times 4096 \times 128 \times 2 \times 2 = 33,554,432\text{ bytes} \approx 32\text{ MB}$$
  *(Without selective allocation, a dense 32-layer 4B model would require 128 MB — saving 96 MB).*
- **Qwen 3.8-27B (16 GQA layers, 4096 context, $TP=2$ with 4 KV heads partitioned to 2 per rank):**
  $$16 \times 2 \times 4096 \times 128 \times 2 \times 2 = 67,108,864\text{ bytes} \approx 64\text{ MB per GPU}$$

### 3.3 Gated DeltaNet Recurrent State Memory
The recurrent state matrix $S$ is stored in FP32 precision to prevent numerical drift over long sequence generations:
$$\text{DeltaNet State Bytes} = N_{\text{DeltaNet}} \times N_{\text{Attn\_Heads}} \times 128 \times 128 \times \text{sizeof}(\text{float})$$
- **Qwen 3.5-4B (24 DeltaNet layers, 16 attention heads):**
  $$24 \times 16 \times 128 \times 128 \times 4 = 25,165,824\text{ bytes} \approx 24\text{ MB}$$
- **Qwen 3.8-27B (48 DeltaNet layers, 32 attention heads partitioned to 16 per rank):**
  $$48 \times 16 \times 128 \times 128 \times 4 = 50,331,648\text{ bytes} \approx 48\text{ MB per GPU}$$

### 3.4 Depthwise Conv1d History Memory
$$\text{Conv1d State Bytes} = N_{\text{DeltaNet}} \times (N_{\text{Attn\_Heads}} \times D_{\text{Head}}) \times (K - 1) \times \text{sizeof}(\text{half})$$
- For Qwen 3.5-4B ($K=4$, history length 3):
  $$24 \times (16 \times 128) \times 3 \times 2 = 294,912\text{ bytes} \approx 288\text{ KB}$$

### 3.5 Dynamic Scratch Arena
$$\text{Workspace Bytes} = \max\left(128\text{ MB}, \quad \text{PREFILL\_MAX\_CHUNK} \times \max(H, I) \times 4 + \text{VocabSize} \times 2 + 64\text{ MB}\right)$$
- Accommodates prefill activation tensors, Split-K FlashDecoding partial accumulators, sampling scratch spaces, and W4A8 dynamic quantization buffers (`d_act_q8`, `d_act_scale`, `d_act_sum8`).

---

## 4. Multi-GPU Deployment Guide ($TP=2$)

### 4.1 Kaggle 2x Tesla T4 Configuration
Kaggle GPU instances provide 2x NVIDIA Tesla T4 GPUs (16 GB GDDR6 each, PCIe 3.0 x16 interconnect). `qwen-engine` leverages 2-way Tensor Parallelism to distribute Qwen 3.8-27B across both devices. A complete, runnable notebook is provided at [`notebooks/qwen38_27b_dual_t4_kaggle.ipynb`](../notebooks/qwen38_27b_dual_t4_kaggle.ipynb).

```python
from qwen_engine import Engine, SamplingParams

# Initialize 27B model across GPU 0 and GPU 1
engine = Engine.from_pretrained(
    "areen09/qwen-engine-models-qwenbin",
    filename="qwen3.8-27b-w4a16",
    device_ids=[0, 1]
)

prompt = "<|im_start|>user\nWrite a concise summary of general relativity.<|im_end|>\n<|im_start|>assistant\n"
for token in engine.stream(prompt, sampling_params=SamplingParams(temperature=0.7, max_tokens=256)):
    print(token, end="", flush=True)
print()
```

### 4.2 Inter-GPU Communication Backends
- **Linux (NCCL):** If compiled with `-DENABLE_NCCL=ON`, `CommunicatorNccl` manages `ncclAllReduce` calls directly on PCIe/NVLink streams with sub-microsecond synchronization overhead.
- **Windows / Direct P2P (BAR1):** If peer access is supported (`cudaDeviceCanAccessPeer`), `CommunicatorP2P` registers cross-device memory pointers and synchronizes via CUDA events (`cudaEventRecord` and `cudaStreamWaitEvent`), bypassing the host CPU.
- **Windows / WDDM Host Staging Fallback:** On Windows systems where the WDDM driver disables direct GPU-to-GPU peer aperture mapping, `qwen-engine` automatically falls back to host pinned-memory double buffering without crashing.

