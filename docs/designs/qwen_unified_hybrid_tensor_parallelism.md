# Unified Qwen Engine: Cross-Platform Hybrid Architecture Expansion

*   **Author:** Areen-09
*   **Status:** Implemented
*   **Date:** 2026-11-08

---

## Overview

The `qwen_engine` currently provides single-GPU inference for standard transformer architectures. Open-weights models in the Qwen family span three distinct architectural configurations: pure Grouped Query Attention (Qwen 2.5), hybrid linear-attention with Gated DeltaNet (Qwen 3.5), and hybrid large-parameter models (Qwen 3.8).

This design establishes a unified C++ inference runtime capable of executing all three model generations on both Windows and Linux hosts. The engine avoids code bifurcation by standardizing on a self-describing model container format (`.qwenbin` v2) with tag-dispatched layer execution, dynamic memory partitioning, an offline model transpilation bridge, and an abstract 2-way tensor parallelism interface supporting direct peer memory on Windows (in TCC driver mode) and NCCL on Linux.

Target hardware encompasses single and dual NVIDIA Tesla T4 16GB GPUs (`sm_75` Turing architecture over PCIe Gen3 x16).

---

## Problem Statement

Deploying modern open-source language models across different operational environments encounters three major structural bottlenecks:

1. **Architectural Divergence Across Generations**:
   Qwen 2.5 relies on standard Transformer layers with Grouped Query Attention (GQA). Qwen 3.5 and Qwen 3.8 introduce hybrid architectures combining standard attention with recurrent Gated DeltaNet linear attention in a 3:1 stacking ratio. Supporting these models by creating distinct engine binaries (`qwen25_engine`, `qwen35_engine`, `qwen38_engine`) triples maintenance overhead, duplicates core memory management, CUDA Graph dispatch, KV cache rollback logic, and tokenizers, and fragments downstream client integration.

2. **Cross-Platform Multi-GPU Execution Barriers**:
   Scaling inference to models exceeding 16 GB (such as Qwen 3.8-27B) requires tensor parallelism. Linux inference runtimes rely on NVIDIA NCCL, which lacks official native support on Windows. Developers and edge deployments on Windows are forced to use single-GPU setups or unsupported custom ports, preventing unified deployment workflows across development and production environments.

3. **Format Incompatibility and Hardware Saturation on Turing (`sm_75`)**:
   Community models are distributed primarily in GGUF format with BF16 weights or GGML quantization schemes incompatible with custom high-throughput GPU kernels. Furthermore, Tesla T4 GPUs lack native BF16 arithmetic hardware, possess strict 64 KB per-SM shared memory limits, restrict threads to 255 32-bit registers, and cannot hold unquantized 27B parameter models within their combined 32 GB VRAM budget.

A unified engine runtime with offline transpilation, platform-abstracted communications, and physically viable shared-memory sub-tiling solves these issues without maintaining diverging codebases.

---

## Goals & Non-Goals

### Goals
* **Unified Engine Runtime**: Execute Qwen 2.5, Qwen 3.5, and Qwen 3.8 through one unified C++ binary and Python API using dynamic metadata and tag-dispatched layer loops.
* **Offline Transpilation Pipeline**: Provide an offline Python transpiler (`tools/convert_gguf_to_qwenbin.py`) and generalized exporter (`tools/export_qwen25.py`) converting standard model weights into validated `.qwenbin` v2 binaries, including FP16 casting and warp-interleaved W4A16 group-128 quantization packing.
* **Cross-Platform 2-Way Tensor Parallelism**: Implement an abstract communicator interface (`ICommunicator`) supporting 2-way tensor parallelism:
  * Linux: Backed by NVIDIA NCCL All-Reduce.
  * Windows: Backed by direct CUDA P2P peer memory access in TCC driver mode, with automatic fallback to host-pinned staging buffers when running under WDDM.
* **Hardware-Grounded `sm_75` Optimization**: Provide custom fused Gated DeltaNet kernels tailored to Turing (`sm_75`):
  * Use cooperative 2D sub-tiling with double-buffered shared memory ($\le 16\text{ KB}$ per thread block) to operate strictly within Turing's 255-register per-thread hardware ceiling.
  * Store recurrent state matrices in global memory in FP32 (`float`) to guarantee numerical stability across 2048+ decode tokens.
* **Preserve Core Runtime Capabilities**: Retain fine-grained `prefill()`, `decode_step()`, and `rollback_to()` interfaces, enabling real-time token streaming via `PyEngineStream` and $O(1)$ transactional KV-cache rollback.
* **Target Validation Deployments**:
  * Qwen 2.5-1.5B on 1x Tesla T4 (functional regression parity).
  * Qwen 3.5-4B on 1x Tesla T4 (hybrid DeltaNet execution).
  * Qwen 3.8-27B on 2x Tesla T4 (2-way tensor parallelism and W4A16 group-128 quantization).

### Non-Goals
* **Independent Engine Binaries**: Creating separate binaries or dedicated class hierarchies per model generation.
* **Runtime GGUF Parsing in C++**: Dynamic parsing, runtime dequantization, or executing GGUF files directly inside the C++ runtime. All ingestion occurs offline.
* **Multi-Node Distributed Inference**: Scaling across multiple physical host machines over InfiniBand or Ethernet. Scope is intra-node multi-GPU (2 devices).
* **Pipeline Parallelism and CPU Offloading**: Layer-wise pipeline parallelism or dynamic weight swapping between host RAM and GPU VRAM.
* **Non-Qwen Architectures**: Generalizing kernel abstractions to Llama, Mistral, or Gemma architectures.
* **Multimodal / Vision Inputs**: Processing non-text modalities. Scope is restricted to text token generation backbones.

---

## Critical User Journeys (CUJs)

### CUJ 1: Converting and Serving a Pure Attention Baseline (Qwen 2.5)
* **Given**: An unquantized model of Qwen 2.5-1.5B and a single Tesla T4 GPU running Windows or Linux.
* **When**: The user converts the model to `.qwenbin` v2 and initializes `Engine("qwen2.5-1.5b.qwenbin")`.
* **Then**: The engine identifies all layers as `LAYER_GQA`, allocates 0 bytes for DeltaNet state, completes forward inference, and produces token sequences matching reference outputs with logits within $1e-3$ floating-point tolerance.

### CUJ 2: Converting and Serving a Hybrid Linear-Attention Model (Qwen 3.5)
* **Given**: A Qwen 3.5-4B model with a 3:1 hybrid ratio of Gated DeltaNet to standard attention on a single Tesla T4 GPU.
* **When**: The user converts the model to `.qwenbin` v2 and executes autoregressive text generation.
* **Then**: The engine dynamically allocates KV cache buffers only for full-attention layers, allocates FP32 recurrent state memory for DeltaNet layers, executes shared-memory sub-tiled state updates, and delivers inference within the 16 GB VRAM budget.

### CUJ 3: Dual-GPU 2-Way Tensor Parallelism on Windows (Qwen 3.8-27B)
* **Given**: A Windows host with 2x NVIDIA Tesla T4 16GB GPUs in TCC mode and a W4A16-quantized `.qwenbin` v2 artifact of Qwen 3.8-27B.
* **When**: The user initializes `Engine("qwen3.8-27b-w4a16.qwenbin", device_ids=[0, 1])`.
* **Then**: The engine establishes peer memory access across devices 0 and 1 via `WindowsP2PCommunicator`, shards linear weights across devices, captures the decode pass into a zero-update CUDA Graph, and performs synchronized autoregressive token generation across both devices within a 14 GB per-GPU VRAM ceiling.

---

## Proposed Architecture

### 1. End-to-End System Flow

The system separates offline ingestion from the unified runtime execution path:

```text
[ Model Weights: Hugging Face / GGUF (BF16 / FP16) ]
                         |
                         v
          [ Offline Model Transpiler / Exporter ]
            - Downcasts BF16 to FP16 with dynamic range clamping
            - Packs W4A16 Group-128 quantization in warp-interleaved layout
            - Emits .qwenbin v2 container (64-bit Header + Descriptors + Weights)
                         |
                         v
                [ .qwenbin v2 File ]
                         |
                         v
             [ PlatformFileMap (mmap) ]
                         |
            +------------+------------+
            |                         |
            v                         v
   [ ModelConfig & Descriptors ] [ Dynamic GpuArena ]
     - num_layers, dimensions     - Weights Area
     - LAYER_GQA vs DELTANET      - KV Cache (GQA layers only)
     - Complete tensor offsets    - Recurrent State (FP32, DeltaNet only)
            |                     - Conv1d History State Buffer
            |                     - Dynamic Workspace (~128 MB prefill chunk)
            +------------+------------+
                         |
                         v
           [ qwen::QwenEngine Forward Pipeline ]
                         |
      +------------------+------------------+
      |                                     |
[ LAYER_GQA Layer ]              [ LAYER_DELTANET Layer ]
  - RMSNorm (rmsnorm_rope.cu)      - RMSNorm
  - QKV GEMM / GEMV                - Depthwise Conv1d (conv1d state cache)
  - Q, K, V Bias Addition          - In-Projection & Gating Projections
  - RoPE rotary embedding          - Fused Gated DeltaNet (deltanet_sm75.cu)
  - GQA Attention (WMMA / Flash)     * Cooperative 2D SMEM sub-tiling
  - KV Cache Write                   * FP32 global state accumulation
  - Out Projection                 - Out Projection
      |                                     |
      +------------------+------------------+
                         |
                         v
      [ 2-Way TP All-Reduce 1 (if device_ids.size() > 1): ICommunicator ]
        * Linux: NcclCommunicator (Row-parallel partial sum all-reduce)
        * Windows TCC: WindowsP2PCommunicator (Direct BAR1 P2P)
        * Windows WDDM: Host-Pinned Staging Fallback (CUDA Graph Bypassed)
                         |
                         v
        [ Intermediate Residual Connection: x_mid = x + attn_out ]
                         |
                         v
                 [ RMSNorm 2 (Pre-MLP) ]
                         |
                         v
                   [ SwiGLU MLP ]
                     - Gate & Up Projections (GEMM / GEMV)
                     - SiLU activation & elementwise product
                     - Down Projection (GEMM / GEMV)
                         |
                         v
      [ 2-Way TP All-Reduce 2 (if device_ids.size() > 1): ICommunicator ]
                         |
                         v
          [ Final Layer Residual Connection: x_out = x_mid + mlp_out ]
                         |
                         v
            [ Final RMSNorm & LM Head (vocab_size projection) ]
                         |
                         v
         [ Greedy / Top-P Sampler & PyEngineStream Token Yield ]
```

---

### 2. Binary Container Specification (`.qwenbin` v2)

The engine migrates from compile-time constants in `include/config.cuh` to a dynamic header format with 64-bit file offsets, preventing integer overflow on model files exceeding 4 GB.

#### File Header Layout
```cpp
struct alignas(64) QwenBinHeaderV2 {
    uint32_t magic;                 // 0x5157454E ("QWEN")
    uint32_t version;               // 2
    uint32_t hidden_size;           // Hidden dimension (e.g. 1536, 2560, 5120)
    uint32_t intermediate_size;     // MLP intermediate dimension
    uint32_t num_layers;            // Total layer count
    uint32_t num_attention_heads;   // Attention query heads
    uint32_t num_kv_heads;          // Key/Value heads
    uint32_t head_dim;              // Head dimension (typically 128)
    uint32_t vocab_size;            // Vocabulary size (e.g. 151936)
    uint32_t max_seq_len;           // Context window ceiling
    uint32_t quant_type;            // 0 = FP16, 1 = W4A16_G128
    uint32_t tie_word_embeddings;   // 1 if lm_head shares embed_tokens, else 0
    float    rms_norm_eps;          // Normalization epsilon (e.g. 1e-6)
    float    rope_theta;            // RoPE base frequency (e.g. 1000000.0)
    uint32_t reserved_flags;        // Alignment padding
    
    // 64-bit File Offsets
    uint64_t layer_table_offset;    // Byte offset to LayerDescriptor array
    uint64_t embed_tokens_offset;   // Byte offset to input embedding table
    uint64_t final_norm_offset;     // Byte offset to final RMSNorm weights
    uint64_t lm_head_offset;        // Byte offset to output unembedding weights
    uint64_t weights_offset;        // Byte offset to aligned layer weight payload
    uint64_t reserved[6];           // Future expansion headroom
};
```

#### Layer Descriptors
Following the file header, an array of `LayerDescriptor` elements of length `num_layers` defines dynamic execution dispatch and tensor locations:
```cpp
enum LayerType : uint32_t {
    LAYER_GQA = 0,       // Standard Grouped Query Attention
    LAYER_DELTANET = 1   // Gated DeltaNet Linear Attention
};

struct alignas(8) LayerDescriptor {
    LayerType type;                 // Execution tag
    uint32_t  reserved;             // Explicit struct padding ensuring 8-byte alignment
    
    // Common Normalization Offsets
    uint64_t norm1_weight_offset;   // Pre-attention / Pre-DeltaNet RMSNorm
    uint64_t norm2_weight_offset;   // Pre-MLP RMSNorm
    
    // GQA Specific Offsets (Valid when type == LAYER_GQA)
    uint64_t qkv_weight_offset;     // Fused or strided Q, K, V projection weights
    uint64_t q_bias_offset;         // Q attention bias
    uint64_t k_bias_offset;         // K attention bias
    uint64_t v_bias_offset;         // V attention bias
    uint64_t attn_out_weight_offset;// Attention output projection
    
    // DeltaNet Specific Offsets (Valid when type == LAYER_DELTANET)
    uint64_t conv1d_weight_offset;  // Depthwise 1D convolution weights
    uint64_t conv1d_bias_offset;    // Depthwise 1D convolution bias
    uint64_t in_proj_weight_offset; // Input projection (Q, K, V, Z gates)
    uint64_t decay_weight_offset;   // Decay projection weights (alpha / beta gate)
    uint64_t delta_out_weight_offset;// DeltaNet output projection
    
    // MLP Offsets
    uint64_t mlp_gate_offset;       // SwiGLU gate projection
    uint64_t mlp_up_offset;         // SwiGLU up projection
    uint64_t mlp_down_offset;       // SwiGLU down projection
    
    // Quantization Offsets (Valid when quant_type == W4A16_G128)
    uint64_t quant_scales_offset;   // FP16 scaling factors
    uint64_t quant_zeros_offset;    // Zero points (or 0 for symmetric quantization)
};

#### ModelConfig Class Interface
```cpp
class ModelConfig {
public:
    explicit ModelConfig(const QwenBinHeaderV2& header, const LayerDescriptor* layers)
        : header_(header), layers_(layers, layers + header.num_layers) {}

    // Model Dimensions
    int32_t hidden_size() const { return header_.hidden_size; }
    int32_t intermediate_size() const { return header_.intermediate_size; }
    int32_t num_layers() const { return header_.num_layers; }
    int32_t num_attention_heads() const { return header_.num_attention_heads; }
    int32_t num_kv_heads() const { return header_.num_kv_heads; }
    int32_t head_dim() const { return header_.head_dim; }
    int32_t vocab_size() const { return header_.vocab_size; }
    int32_t max_seq_len() const { return header_.max_seq_len; }
    uint32_t quant_type() const { return header_.quant_type; }
    bool tie_word_embeddings() const { return header_.tie_word_embeddings != 0; }
    float rms_norm_eps() const { return header_.rms_norm_eps; }
    float rope_theta() const { return header_.rope_theta; }

    // Layer Descriptors & Offsets
    const LayerDescriptor& layer(size_t idx) const { return layers_.at(idx); }
    const QwenBinHeaderV2& header() const { return header_; }

private:
    QwenBinHeaderV2 header_;
    std::vector<LayerDescriptor> layers_;
};
```
```

---

### 3. Memory Subsystem & Dynamic Arena Allocation

The `GpuArena` allocates contiguous VRAM on each active GPU dynamically at initialization based on `ModelConfig`:

1. **Selective KV Cache Allocation & Compact Indexing Table**:
   * KV cache memory is allocated exclusively for layers tagged `LAYER_GQA`:
     $$\text{Bytes}_{KV} = 2 \times N_{GQA} \times \text{max\_seq\_len} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{sizeof(half)}$$
   * For pure GQA models (Qwen 2.5), $N_{GQA} = num\_layers$.
   * For hybrid models (Qwen 3.5 / 3.8), $N_{GQA} = \frac{1}{4} num\_layers$. Linear attention layers require zero KV cache pages.
   * **Compact Layer Indexing**: To prevent out-of-bounds memory accesses when indexing KV cache in hybrid models, `setup_arena()` generates a dense lookup table `std::vector<int> gqa_layer_idx(num_layers, -1)`. If `layers[l].type == LAYER_GQA`, `gqa_layer_idx[l] = gqa_count++`. Kernels index KV cache using `d_k_cache_ + (gqa_layer_idx[l] * max_seq_len * num_kv_heads * head_dim)`.
2. **DeltaNet State Tensor Memory (FP32)**:
   * Recurrent associative state matrices are allocated exclusively for layers tagged `LAYER_DELTANET` using 32-bit floats to guarantee numerical stability:
     $$\text{Bytes}_{state} = N_{DELTANET} \times \text{num\_attention\_heads} \times 128 \times 128 \times \text{sizeof(float)}$$
   * Conv1d rolling history state buffer:
     $$\text{Bytes}_{conv1d} = N_{DELTANET} \times \text{num\_attention\_heads} \times \text{head\_dim} \times (\text{kernel\_size} - 1) \times \text{sizeof(half)}$$
3. **Workspace Scratchpad Allocation & Prefill Execution Path**:
   * Sized dynamically to accommodate prompt prefill sequence chunks ($L_{chunk} = 256$), intermediate MLP activations, vocabulary projection logits, and GEMM scratch:
     $$\text{Bytes}_{workspace} = L_{chunk} \times \max(hidden\_size, intermediate\_size) \times 2 \times 2 + vocab\_size \times 2 + \text{GEMM\_SCRATCH\_BYTES}$$
   * Sizing allocates $\sim 128\text{ MB}$ per GPU, preventing out-of-bounds writes during prefill execution.
   * **DeltaNet Prefill Execution**: Inside `execute_prefill_chunk()`, while GQA layers process full 256-token prompt tiles via flash attention, DeltaNet layers execute a sequential GPU stream loop over token positions $t \in [0, chunk\_len)$. At each step $t$, the engine launches `launch_depthwise_conv1d_step` followed by `launch_deltanet_decode_sm75` on the persistent recurrent state matrix, updating $S_t$ and output activations with zero host synchronizations.
4. **Session Leasing & Rollback Semantics**:
   * On pure GQA models, `rollback_to(target_seq_pos)` operates in $O(1)$ by updating `current_seq_pos_ = target_seq_pos`.
   * On hybrid DeltaNet models, because recurrent associative states $S_t$ accumulate in-place, `rollback_to(target_seq_pos)` is defined as resetting state matrices to zero and re-evaluating prompt prefill up to `target_seq_pos`, ensuring deterministic mathematical continuity.

---

### 4. Gated DeltaNet Implementation on Turing (`sm_75`)

DeltaNet implements linear recurrent attention with input-dependent decay gating:
$$\alpha_t = \exp(-\text{softplus}(\beta_t))$$
$$S_t = \alpha_t S_{t-1} + v_t \otimes k_t^T$$
$$o_t = S_t q_t$$

#### Cooperative 2D Shared-Memory Sub-Tiling
* **Hardware Limit**: Turing GPUs (`sm_75`) enforce a strict hardware ceiling of **255 32-bit registers per thread**. Loading the full $128 \times 128$ state matrix in registers across 32 threads requires 512 registers/thread in FP32, causing catastrophic register spilling into local VRAM.
* **Kernel Architecture (`deltanet_sm75.cu`)**:
  * Instead of per-warp register streaming, the decode kernel partitions the $128 \times 128$ associative state matrix into $32 \times 64$ sub-tiles.
  * A thread block of 128 threads (4 warps) cooperatively streams a sub-tile into double-buffered `__shared__` memory:
    $$\text{SMEM per block} = 32 \times 64 \times 4\text{ bytes} \times 2\text{ (double buffer)} = 16\text{ KB}$$
  * $16\text{ KB}$ consumes only 25% of Turing's 64 KB configurable SMEM, allowing an active occupancy of 3 to 4 thread blocks per SM.
  * Outer product updates and decay gating accumulate in FP32 registers across the sub-tile without exceeding 64 registers per thread.
  * The updated state is written back to global memory in FP32 (`float`), and output projection vectors $o_t$ are evaluated via warp reduction.

---

### 5. Cross-Platform Tensor Parallelism Architecture (`ICommunicator`)

The multi-GPU tensor parallel architecture partitions linear weights across 2 devices ($TP = 2$):

```text
+---------------------------------------------------------------------------------+
|                               ICommunicator Base                                |
|  virtual void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type,   |
|                              cudaStream_t stream) = 0;                          |
|  virtual void sync(cudaStream_t stream) = 0;                                    |
|  virtual int rank() const = 0;                                                  |
|  virtual int world_size() const = 0;                                            |
+---------------------------------------------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
                 v                                           v
+---------------------------------+         +---------------------------------+
|        NcclCommunicator         |         |     WindowsP2PCommunicator      |
|  (Linux: SPMD ncclAllReduce)    |         |  (Windows: TCC P2P / Staging)   |
+---------------------------------+         +---------------------------------+
```

#### Multi-Device Threading & Concurrency Model
* `qwen::QwenEngine` instantiates two communicator handles (one per rank) and executes multi-GPU operations using a dedicated host worker thread pool (one host thread per GPU).
* Each thread manages its device context (`cudaSetDevice(rank)`), memory arena, and execution stream independently, preventing deadlocks during peer synchronization.

#### 2-Way Tensor Parallel Weight Sharding Policy
Weights are sliced across the 2 GPUs upon initialization in `setup_arena()` to adhere strictly to the 14.0 GB per-GPU VRAM ceiling:
1. **Column-Parallel Projections (Split across Output Channels)**:
   * **GQA Attention**: Q, K, V projections are sharded along head dimensions. Device 0 owns heads $0 \dots (H/2 - 1)$; Device 1 owns heads $H/2 \dots (H - 1)$.
   * **DeltaNet Projections**: In-projection ($W_{in}$) and decay/beta projection ($W_{decay}$) are sharded along output heads.
   * **SwiGLU MLP**: Gate projection ($W_{gate}$) and Up projection ($W_{up}$) are sliced along $intermediate\_size / 2$.
2. **Row-Parallel Projections (Split across Input Channels)**:
   * **Attention Out-Projection ($W_{out}$)**: Sliced along input channels. Each GPU computes a partial output vector; `ICommunicator::all_reduce_sum()` accumulates the partial sums before intermediate residual addition.
   * **DeltaNet Out-Projection ($W_{out}$)**: Sliced along input head dimensions, followed by `all_reduce_sum()`.
   * **SwiGLU Down-Projection ($W_{down}$)**: Sliced along intermediate channels ($intermediate\_size / 2$), followed by `all_reduce_sum()`.
3. **Quantization Block Alignment (W4A16 Group-128)**:
   * Sharding boundaries along intermediate and hidden dimensions are verified to be integer multiples of 128 elements (64 bytes of packed `uint32` values). FP16 scale factor blocks are sliced identically, ensuring group index alignment.
4. **Token Embeddings & LM Head**:
   * When `tie_word_embeddings == 1`, `embed_tokens` is replicated across both GPUs to avoid communication during initial token embedding lookup.
   * In un-tied 27B configurations, `lm_head` is column-sharded across $vocab\_size / 2$, and ranks perform an All-Gather or local Top-P reduction.

#### Windows Driver Modes and CUDA Graph Capture
* **TCC Mode (Direct BAR1 P2P)**:
  * Devices access peer buffers directly via PCIe BAR1 mapping (`cudaDeviceEnablePeerAccess`).
  * Cross-GPU synchronization uses asynchronous stream events (`cudaEventRecord` / `cudaStreamWaitEvent`), which are fully capture-compatible with CUDA Graphs.
  * Zero-update CUDA Graph decode is active under TCC mode.
* **WDDM Mode (Host-Pinned Staging Fallback)**:
  * When direct P2P access is denied by Windows WDDM drivers, All-Reduce falls back to double-buffered host-pinned staging buffers.
  * Because host CPU event polling cannot be captured within a CUDA Graph (`cudaErrorStreamCaptureUnsupported`), **CUDA Graph capture is automatically bypassed when running under WDDM staging**, falling back to standard eager stream kernel launches.

---

## Public Interfaces and Component Boundary Map

### C++ Engine API (`include/engine.cuh`)
```cpp
namespace qwen {

enum class DataType : uint32_t {
    FP16 = 0,
    FP32 = 1,
    INT32 = 2
};

struct EngineConfig {
    std::string model_path;
    std::string tokenizer_path;
    std::vector<int> device_ids{0};
    size_t max_seq_len{2048};
    bool force_wddm_staging{false};
};

class QwenEngine {
public:
    explicit QwenEngine(const EngineConfig& config);
    explicit QwenEngine(const std::string& model_path, int device_id = 0);
    ~QwenEngine();

    // Session Leasing
    bool try_acquire_session();
    void release_session();

    // Fine-Grained Inference Lifecycle
    void prefill(const std::vector<int32_t>& prompt_tokens);
    int32_t decode_step();
    void rollback_to(size_t target_seq_pos);
    void reset_session();

    // Metadata Accessors
    const ModelConfig& config() const;
    int current_seq_len() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen
```

### Python Nanobind Interface (`src/python_bindings.cpp`)
```cpp
NB_MODULE(_core, m) {
    m.def("cuda_device_count", []() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    });

    nb::class_<qwen::QwenEngine>(m, "NativeEngine")
        .def(nb::init<const std::string&, int>(),
             nb::arg("model_path"),
             nb::arg("device_id") = 0)
        .def("try_acquire_session", &qwen::QwenEngine::try_acquire_session)
        .def("release_session", &qwen::QwenEngine::release_session)
        .def("prefill", &qwen::QwenEngine::prefill, nb::arg("prompt_tokens"))
        .def("decode_step", &qwen::QwenEngine::decode_step)
        .def("rollback_to", &qwen::QwenEngine::rollback_to, nb::arg("target_seq_pos"))
        .def("reset_session", &qwen::QwenEngine::reset_session);

    nb::class_<PyEngineStream>(m, "EngineStream")
        .def("__iter__", [](PyEngineStream& s) -> PyEngineStream& { return s; })
        .def("__next__", &PyEngineStream::next, nb::call_guard<nb::gil_scoped_release>());
}
```

---

## Alternatives Considered

### Alternative A: Forking into Multiple Engine Binaries (`qwen25_engine`, `qwen35_engine`, `qwen38_engine`)
* **Pros**: Allows each binary to hardcode compile-time dimensions and kernel launch bounds.
* **Cons**: Triples long-term maintenance overhead, duplicates file mapping, memory arenas, CUDA Graph capture, and KV cache rollback logic, and fragments the Python client API.
* **Reason for Rejection**: Excessive code duplication and maintenance overhead. Unified dynamic metadata cleanly supports all three architectures in a single engine.

### Alternative B: Native In-Engine GGUF Parsing and Execution via GGML Kernels
* **Pros**: Users can run `.gguf` files directly from community hubs without an offline transpilation step.
* **Cons**: Embeds heavy file parsing, tensor dictionary tracking, and type unpacking into the C++ runtime. Prevents custom Turing `sm_75` kernel optimizations (such as 2D sub-tiled DeltaNet updates and native warp-interleaved W4A16 group-128 quantization).
* **Reason for Rejection**: Substantial runtime complexity and kernel throughput degradation. An offline Python transpiler isolates file format parsing from GPU execution.

### Alternative C: Porting NVIDIA NCCL to Windows
* **Pros**: Identical All-Reduce code paths and algorithms across both Windows and Linux.
* **Cons**: NVIDIA NCCL provides no official Windows support. Community ports rely on emulated POSIX translation layers, unmaintained patches, and are prone to driver-level deadlocks.
* **Reason for Rejection**: High stability risk and lack of vendor support. Resolved by creating `ICommunicator` with a native `WindowsP2PCommunicator` backend.

### Alternative D: Full-Tile SMEM Buffering vs. Per-Warp Register Streaming
* **Pros**: Full-tile buffering is conceptually simple; register streaming avoids shared memory entirely.
* **Cons**: Full $128 \times 128$ FP32 tile in SMEM consumes all 64 KB, reducing SM occupancy to 1 block. Full-tile per-warp register streaming requires 512 registers/thread, exceeding Turing's 255-register limit and causing catastrophic register spilling.
* **Reason for Rejection**: Both extremes violate Turing physical hardware limits. Resolved by cooperative 2D sub-tiling ($32 \times 64$) using only $16\text{ KB}$ SMEM and $\le 64$ registers per thread.

### Alternative E: Pipeline Parallelism (PP) Across 2 GPUs
* **Pros**: Eliminates All-Reduce synchronization after attention and MLP blocks; communication occurs only at layer boundaries.
* **Cons**: During single-stream autoregressive token generation (batch size = 1), pipeline bubbles force one GPU to remain idle while the other executes its layer subset, cutting hardware utilization to 50%.
* **Reason for Rejection**: 50% device idle penalty during autoregressive decode. 2-way Tensor Parallelism keeps both devices active during every layer.

---

## Risks & Mitigations

### 1. PCIe Gen3 Bandwidth Saturation during Tensor Parallel All-Reduce
* **Risk**: Dual Tesla T4 GPUs communicate across PCIe Gen3 x16 (~15.75 GB/s bidirectional). In 2-way Tensor Parallelism, two All-Reduce operations occur per layer. For large models (Qwen 3.8-27B with 64 layers), All-Reduce transfers can saturate the PCIe root complex.
* **Mitigation**: Communication is restricted to hidden state activations ($hidden\_size \times 2\text{ bytes}$), which equals only $10\text{ KB}$ per All-Reduce for $hidden\_size = 5120$. Pure data transfer time per layer is $< 1.5\ \mu\text{s}$, well within PCIe Gen3 transmission capacity.

### 2. Decode Throughput Collapse under Windows WDDM Staging Fallback
* **Risk**: Under WDDM, All-Reduce falls back to host-pinned staging buffers. 128 host-synchronized roundtrips per token at 50–100 $\mu\text{s}$ per roundtrip adds ~12.8 ms of driver overhead per token, capping throughput below 78 tok/s.
* **Mitigation**: TCC driver mode is established as the mandatory operational prerequisite for latency-sensitive multi-GPU deployments. The engine emits a prominent operational warning if WDDM is detected, automatically bypasses CUDA Graph capture, and documents that WDDM multi-GPU execution is restricted to offline batch processing.

### 3. Numerical Dynamic Range Overflow during BF16 to FP16 Downcasting
* **Risk**: Pretrained Qwen weights use BF16 (exponent range $\approx 10^{-38}$ to $10^{38}$). Downcasting to FP16 (maximum value $65,504$) on models with SwiGLU activations and recurrent state accumulations risks silent `Inf` / `NaN` generation.
* **Mitigation**: The offline converter inspects activation distributions, applies dynamic range clamping, accumulates RMSNorm and SwiGLU gating in FP32, and keeps DeltaNet recurrent state matrices strictly in FP32 (`float`).

### 4. VRAM Exhaustion on Dual 16 GB GPUs at Maximum Context Length
* **Risk**: Qwen 3.8-27B in W4A16 consumes ~7 GB of weight memory per GPU. At extended context sequences ($max\_seq\_len = 2048$), KV cache and recurrent states could exhaust remaining VRAM.
* **Mitigation**: Because Qwen 3.8 uses a 3:1 hybrid stacking ratio, KV cache is only allocated for the 25% full-attention layers. Dynamic arena initialization calculates exact peak VRAM requirements upfront and rejects initialization if allocation exceeds 14.5 GB, preserving 1.5 GB of headroom for driver overhead.

### 5. PCIe Access Control Services (ACS) Blocking Direct Peer Access
* **Risk**: Certain motherboard chipsets and virtualization hypervisors enforce PCIe ACS, blocking direct peer-to-peer memory transactions between GPUs.
* **Mitigation**: `WindowsP2PCommunicator` explicitly probes `cudaDeviceCanAccessPeer()` during initialization. If blocked by ACS, it safely falls back to host-pinned staging buffers and outputs actionable troubleshooting guidance to disable ACS in BIOS.

---

## Non-Functional Requirements

### Security & Privacy
* **Local Memory Isolation**: Weights and activations reside exclusively in host RAM and dedicated GPU VRAM. No external network endpoints, telemetry, or remote dependencies are invoked.
* **Memory Safety in Ingestion**: The offline converter validates magic bytes, tensor count bounds, and alignment before writing `.qwenbin` v2. The C++ loader validates header bounds, version numbers, and file size offsets before mapping memory to prevent out-of-bounds reads and memory corruption.

### Performance & Latency
* **Turing SM Occupancy**: 2D sub-tiled DeltaNet updates limit shared memory consumption to $16\text{ KB}$ per block, sustaining theoretical warp occupancy above 50% (at least 32 active warps per SM) on Tesla T4.
* **Communication Latency on PCIe Gen3**:
  * Linux: NCCL All-Reduce ring communication pipelined across PCIe root complex.
  * Windows TCC: BAR1 direct peer writes bypass host RAM, keeping All-Reduce data transfer throughput bounded by PCIe Gen3 theoretical bandwidth (~15.75 GB/s bidirectional).
  * Windows WDDM: Host-pinned staging buffer fallback incurs two host-device PCIe copies per All-Reduce; configure GPUs in TCC mode for latency-sensitive deployments.

### Rollout & Compatibility
* **Backward Compatibility**: Existing Qwen 2.5 `.qwenbin` v1 files trigger an informative error guiding users to re-export via `tools/export_qwen25.py` or `tools/convert_gguf_to_qwenbin.py`.
* **Zero Runtime Dependencies**: The package retains zero external math runtime dependencies (`dependencies = []` in `pyproject.toml`, no PyTorch). Device queries use native CUDA runtime calls (`cudaGetDeviceCount`).

---

## External References

* [Qwen 2.5 Technical Report](https://arxiv.org/abs/2409.12191)
* [Qwen 3.5 / Gated DeltaNet Architecture Disclosures](https://arxiv.org/abs/2412.06464)
* [NVIDIA Turing Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
* [NVIDIA GPUDirect P2P Communication Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)

---

## Future Improvements

* **FlashAttention-3 Integration on Newer Architectures**: Porting to Ada/Hopper architectures (`sm_89`/`sm_90`) utilizing hardware FP8 and TMA instructions.
* **4-Way and 8-Way Tensor Parallelism**: Extending `ICommunicator` to support scale-up multi-GPU topologies across PCIe switches or NVLink bridges.

---

## Detailed Implementation Plan

The implementation proceeds in four sequential phases, structured to eliminate testability deadlocks:

```text
+-------------------------------------------------------------------------+
| Phase 1: Model Transpilation & Quantization Tooling                     |
|  - Generalize tools/export_qwen25.py to emit .qwenbin v2 container       |
|  - Implement tools/convert_gguf_to_qwenbin.py with GGUF ingestion       |
|  - Pack W4A16 Group-128 quantization in warp-interleaved layout         |
|  - Emit 64-bit QwenBinHeaderV2 with full tensor offset tables           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| Phase 2: Dynamic Engine & Unified Layer Abstraction                     |
|  - Refactor include/config.cuh to dynamic ModelConfig                   |
|  - Implement .qwenbin v2 container parser in src/engine.cu              |
|  - Update src/engine.cu dynamic layer loop for LAYER_GQA                |
|  - Update downstream targets (cli_bench.cpp, main_server.cpp)           |
|  - Verify greedy token and logit parity (1e-3) on Qwen 2.5-1.5B         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| Phase 3: Fused Gated DeltaNet sm_75 Kernels                             |
|  - Create include/deltanet.cuh and src/kernels/deltanet_sm75.cu         |
|  - Implement cooperative 2D SMEM sub-tiled decode kernel (16 KB SMEM)   |
|  - Store recurrent states in FP32; allocate Conv1d state cache          |
|  - Validate single-GPU Qwen 3.5-4B hybrid execution                     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
| Phase 4: Cross-Platform 2-Way Tensor Parallelism                        |
|  - Implement include/communicator.cuh (ICommunicator & DataType)        |
|  - Implement src/communicator_nccl.cu (Linux SPMD backend)              |
|  - Implement src/communicator_win_p2p.cu (TCC direct P2P / WDDM staging)|
|  - Establish worker thread pool in QwenEngine for multi-device execution|
|  - Validate 2-way TP Qwen 3.8-27B on 2x Tesla T4 16GB GPUs             |
+-------------------------------------------------------------------------+
```

---

### Target Files to Create/Modify

#### 1. `tools/export_qwen25.py` (Modify)
* **Scope**: Python CLI Exporter
* **Changes**: Update header packing to emit `.qwenbin` v2 (`QwenBinHeaderV2`) with 64-bit offsets for `embed_tokens`, `final_norm`, `lm_head`, and attention QKV biases (`q_bias`, `k_bias`, `v_bias`). Generalize to export arbitrary Qwen 2.5 configurations from Hugging Face.
* **Rationale**: Produces validated `.qwenbin` v2 containers in Phase 1 before engine modifications.

#### 2. `tools/convert_gguf_to_qwenbin.py` (Create)
* **Scope**: Python CLI Transpiler
* **Changes**: Ingest GGUF models via `gguf` reader. Cast BF16 tensors to FP16 with dynamic range clamping. Apply warp-interleaved W4A16 group-128 quantization packing matching existing CUDA kernels. Detect layer types (`LAYER_GQA` vs `LAYER_DELTANET`) and emit `QwenBinHeaderV2`, `LayerDescriptor` table, and aligned weight blocks.
* **Rationale**: Enables ingestion of community GGUF models across Qwen 2.5, 3.5, and 3.8.

#### 3. `include/config.cuh` (Modify)
* **Scope**: C++ / CUDA Header
* **Changes**: Define `struct alignas(64) QwenBinHeaderV2`, `enum LayerType`, `struct alignas(8) LayerDescriptor`, and `class ModelConfig`. Provide dynamic accessors for dimensions, head counts, and layer tensor offsets. Retain backward-compatible `constexpr int32_t` dimension defaults for Qwen 2.5-1.5B (`NUM_LAYERS = 28`, `HIDDEN_SIZE = 1536`, etc.) to prevent breaking legacy build configurations.
* **Rationale**: Replaces compile-time constants with dynamic container configuration.

#### 4. `include/kernels.cuh`, `src/kernels/gemv_fp16.cu`, `src/kernels/gemv_w4a16.cu`, `src/kernels/attention_gqa.cu` (Modify)
* **Scope**: CUDA C++ Kernel Signatures & Launchers
* **Changes**: Parameterize kernel launch wrappers with dynamic dimensions (`int hidden_size`, `int intermediate_size`, `int num_q_heads`, `int num_kv_heads`, `int gqa_group_size`) instead of referencing fixed compile-time constants from `config.cuh`. Compute grid and thread block dimensions dynamically based on runtime `ModelConfig`.
* **Rationale**: Enables existing WMMA and GEMV kernels to execute across variable model dimensions (from 0.5B up to 27B) without shared-memory or grid bounds violations.

#### 5. `include/engine.cuh` (Modify)
* **Scope**: C++ / CUDA Header
* **Changes**: Update `qwen::QwenEngine` to accept `EngineConfig` (including `device_ids`). Preserve fine-grained lifecycle methods (`prefill`, `decode_step`, `rollback_to`, `try_acquire_session`, `reset_session`). Declare layer execution dispatchers (`execute_gqa_layer`, `execute_deltanet_layer`).
* **Rationale**: Retains the established streaming and session management interfaces while supporting multi-GPU and hybrid layer dispatch.

#### 6. `src/engine.cu` (Modify)
* **Scope**: CUDA C++ Source
* **Changes**: Implement container deserialization and header verification in `load_model_file`. Implement dynamic layer loop dispatching based on `LayerDescriptor::type`. Construct `gqa_layer_idx` mapping table for compact KV-cache indexing. Expand `setup_arena()` workspace buffer to $\sim 128	ext{ MB}$ to support prefill chunk sequences. Implement multi-device worker thread dispatch, tensor parallel weight sharding, and conditional CUDA Graph capture (active under single-GPU and Windows TCC mode; bypassed under WDDM staging).
* **Rationale**: Core forward execution engine orchestrating dynamic layers and tensor parallelism.

#### 7. `include/tokenizer.hpp` & `src/tokenizer.cpp` (Modify)
* **Scope**: C++ Tokenizer Module
* **Changes**: Add buffer-based initialization constructor `QwenTokenizer(const void* buffer, size_t size)` alongside the existing filesystem constructor.
* **Rationale**: Enables initializing tokenizer vocabularies from memory-mapped `.qwenbin` containers or standalone `tokenizer.bin` buffers.

#### 8. `include/deltanet.cuh` (Create)
* **Scope**: CUDA Header (Linear Attention Interface)
* **Changes**: Declare host launcher signatures:
  ```cpp
  void launch_deltanet_decode_sm75(
      const half* q, const half* k, const half* v, const half* beta,
      float* s_state, half* out, int batch_size, int num_heads, int head_dim,
      cudaStream_t stream);
  void launch_depthwise_conv1d_step(
      const half* in, const half* weight, const half* bias, half* conv_state,
      half* out, int num_channels, int kernel_size, cudaStream_t stream);
  ```
* **Rationale**: Public interface for Gated DeltaNet linear attention and Conv1d kernels.

#### 9. `src/kernels/deltanet_sm75.cu` (Create)
* **Scope**: CUDA C++ Source
* **Changes**: Implement the cooperative 2D sub-tiled DeltaNet decode kernel. Partitions $128 	imes 128$ state updates into $32 	imes 64$ sub-tiles cooperatively loaded into $16	ext{ KB}$ double-buffered shared memory. Accumulates updates in FP32 registers ($\le 64$ registers/thread) and writes state back to global VRAM in FP32 (`float`). Implement depthwise Conv1d sliding-window step kernel.
* **Rationale**: Executes DeltaNet on Tesla T4 while respecting the 255-register per-thread ceiling and preserving SM occupancy.

#### 10. `include/communicator.cuh` (Create)
* **Scope**: C++ Header (Communication Interface)
* **Changes**: Declare `enum class DataType : uint32_t { FP16, FP32, INT32 };` and abstract base class `ICommunicator` with pure virtual methods:
  ```cpp
  virtual void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type, cudaStream_t stream) = 0;
  virtual void sync(cudaStream_t stream) = 0;
  virtual int rank() const = 0;
  virtual int world_size() const = 0;
  ```
  Declare factory function `create_communicator(int rank, int world_size, const std::vector<int>& device_ids, bool force_staging)`.
* **Rationale**: Hardware-abstracted multi-GPU communication interface.

#### 11. `src/communicator_nccl.cu` (Create)
* **Scope**: CUDA C++ Source (Linux NCCL Backend)
* **Changes**: Implement `NcclCommunicator` wrapping `ncclCommInitRank` and `ncclAllReduce` using `ncclSum`. Guard entire file with `#ifndef _WIN32`.
* **Rationale**: Provides NCCL-backed All-Reduce operations across PCIe and NVLink on Linux platforms.

#### 12. `src/communicator_win_p2p.cu` (Create)
* **Scope**: CUDA C++ Source (Windows P2P Backend)
* **Changes**: Implement `WindowsP2PCommunicator`. Check `cudaDeviceCanAccessPeer`. In TCC mode, enable peer access and execute direct BAR1 writes with cross-stream CUDA event synchronization (`cudaEventRecord`/`cudaStreamWaitEvent`). In WDDM mode, allocate double-buffered pinned host buffers (`cudaHostAlloc`) and execute staging copies. Guard entire file with `#ifdef _WIN32`.
* **Rationale**: Native multi-GPU tensor parallelism on Windows without external NCCL libraries.

#### 13. `src/python_bindings.cpp` (Modify)
* **Scope**: Nanobind C++ Module
* **Changes**: Update `NativeEngine` binding constructor to accept `device_ids` list. Expose `cuda_device_count()` helper querying CUDA runtime. Maintain `PyEngineStream` binding for token streaming with GIL release.
* **Rationale**: Python bindings exposing multi-GPU configuration while maintaining streaming semantics.

#### 14. `src/python/qwen_engine/engine.py` (Modify)
* **Scope**: Python Wrapper Module
* **Changes**: Update `Engine` wrapper to validate `device_ids` using native `_core.cuda_device_count()` (zero PyTorch dependency). Preserve `stream()` and `stream_async()` generators.
* **Rationale**: Python API client interface with zero external runtime dependencies.

#### 15. `src/cli_bench.cpp`, `src/main_server.cpp`, `tests/test_kernels.cu` (Modify)
* **Scope**: C++ Executables & Native Tests
* **Changes**:
  * `src/cli_bench.cpp`: Add CLI flag `--devices <id,id,...>`, split comma-separated device IDs into `std::vector<int>`, and forward to `EngineConfig.device_ids`. Update engine initialization to use dynamic `EngineConfig`.
  * `src/main_server.cpp`: Query `engine.config().max_seq_len()` for sequence bounds checks instead of relying on global `MAX_SEQ_LEN`.
  * `tests/test_kernels.cu`: Update test kernel invocations to pass dynamic dimension parameters.
* **Rationale**: Prevents downstream compilation and runtime breakage during CMake build and enables multi-GPU benchmarking.

#### 16. `CMakeLists.txt` (Modify)
* **Scope**: CMake Build Configuration
* **Changes**: Add custom NCCL discovery block (`find_path(NCCL_INCLUDE_DIR nccl.h)`, `find_library(NCCL_LIBRARY nccl)`) on Linux. Conditionally compile `communicator_nccl.cu` on Linux or `communicator_win_p2p.cu` on Windows. Add `target_link_libraries(qwen_core PUBLIC ${NCCL_LIBRARY})` and `target_include_directories(qwen_core PUBLIC ${NCCL_INCLUDE_DIR})` to ensure downstream executable targets link NCCL symbols on Linux. Add `deltanet_sm75.cu` to target sources.
* **Rationale**: Configures cross-platform compilation and clean downstream linkage across MSVC and GCC/Clang.

#### 17. `tests/test_dynamic_engine.py` (Create)
* **Scope**: Python Test Suite
* **Changes**: Test single-GPU loading and token generation for Qwen 2.5-1.5B and Qwen 3.5-4B using lightweight synthetic mock `.qwenbin` v2 fixtures (calibrated to 4 layers: 3 DeltaNet, 1 GQA to verify the 3:1 stacking ratio and 25% KV cache allocation) for CI testing, and full model tests for hardware benches.
* **Rationale**: Automated regression and functional verification for dynamic model execution.

#### 18. `tests/test_tensor_parallel.py` (Create)
* **Scope**: Multi-GPU Test Suite
* **Changes**: Verify 2-way All-Reduce numerical correctness on dual GPUs against CPU reference. Decorate multi-GPU tests with `@pytest.mark.skipif(_core.cuda_device_count() < 2, reason="Requires >= 2 CUDA devices")` to allow seamless test execution on single-GPU developer workstations.
* **Rationale**: Verifies functional correctness and numerical precision of `ICommunicator` backends with safe CI gating.

#### 19. `tests/test_gguf_converter.py` (Create)
* **Scope**: Python Unit Test Suite
* **Changes**: Unit tests verifying GGUF file parsing, BF16-to-FP16 downcasting with dynamic range clamping, W4A16 group-128 quantization packing, 64-bit header offset alignment, and LayerDescriptor table serialization.
* **Rationale**: Verifies model transpilation integrity and container compliance before engine loading.

---

## Verification Plan

### 1. Compilation & Build Verification
* **Windows Build**:
  * Command: `cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"` followed by `cmake --build build --config Release`.
  * Success Criteria: Zero compiler errors and zero warnings (`/WX` enabled); produces `_core.pyd` and `cli_bench.exe`.
* **Linux Build**:
  * Command: `cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" && cmake --build build -j$(nproc)`.
  * Success Criteria: Clean build linking NCCL; Python module built and importable.

### 2. Unit Testing & Offline Converter Verification
* **Converter Test**:
  * Command: `pytest tests/test_gguf_converter.py -v`.
  * Success Criteria: All tests pass for GGUF parsing, FP16 casting, W4A16 scale packing, and 64-bit header byte offset alignment.

### 3. Single-GPU Regression & Hybrid Layer Verification
* **Qwen 2.5-1.5B Parity**:
  * Command: `pytest tests/test_dynamic_engine.py -k "qwen25" -v`.
  * Success Criteria: Greedy token ID sequence match and logit parity within FP16 tolerance ($1e-3$) against baseline `.qwenbin` v1. DeltaNet state allocation verified as exactly 0 bytes.
* **Qwen 3.5-4B Hybrid Execution**:
  * Command: `pytest tests/test_dynamic_engine.py -k "qwen35" -v`.
  * Success Criteria: Zero residual GPU allocations (`cudaMemGetInfo` delta == 0 bytes) between iteration 1 and iteration 100. KV cache consumption verified at 25% of equivalent pure-attention layer models.

### 4. Multi-GPU Tensor Parallelism Verification
* **Windows 2x Tesla T4 Test**:
  * Command: `pytest tests/test_tensor_parallel.py -v`.
  * Success Criteria: Direct BAR1 P2P access confirmed under TCC mode; double-buffered host staging buffers confirmed under WDDM mode. Logits across 2 GPUs match single-device reference within $1e-3$ relative error. Single-device environments cleanly skip multi-device tests without failure.
* **Qwen 3.8-27B Scale Validation**:
  * Command: `cli_bench.exe --model qwen3.8-27b-w4a16.qwenbin --devices 0,1 --prompt-len 32 --gen-len 128`.
  * Success Criteria: Peak VRAM allocation remains $< 14.0	ext{ GB}$ per GPU. 128-token generation completes without `cudaErrorMemoryAllocation` at sustained throughput $\ge 25	ext{ tokens/sec}$.
