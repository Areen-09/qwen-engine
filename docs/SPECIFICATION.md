# Qwen Engine: Technical Engineering Specification (`.qwenbin` v2)

* **Version:** 2.0  
* **Byte Alignment:** 64-byte boundary  
* **Endianness:** Little-Endian (`<`)  

---

## 1. Container Overview

The `.qwenbin` v2 binary container is a high-performance, single-file model serialization format designed for bare-metal memory-mapped (`mmap`) runtime ingestion in CUDA C++20 and Python without external framework dependencies.

```text
+----------------------------------------------------------------+
| QwenBinHeaderV2 (192 bytes, 64-byte aligned)                   |
| - Magic: 0x5157454E ("QWEN")                                   |
| - Architecture dimensions, quantization mode, layer counts     |
+----------------------------------------------------------------+
| Layer Table: LayerDescriptor[N_LAYERS]                         |
| - 216 bytes per descriptor (format: "<2I26Q")                  |
| - Execution dispatch type (0 = LAYER_GQA, 1 = LAYER_DELTANET)  |
| - 26x 64-bit tensor payload byte offsets                       |
+----------------------------------------------------------------+
| Padding to 256-byte alignment                                  |
+----------------------------------------------------------------+
| Global Weights: Embeddings, Final RMSNorm, LM Head             |
+----------------------------------------------------------------+
| Layer Weights: Attention / DeltaNet / SwiGLU MLP / Quant Scales|
+----------------------------------------------------------------+
```

---

## 2. Global Container Header (`QwenBinHeaderV2`)

Located at byte offset `0` of the file, aligned to 64 bytes.

| Offset | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `uint32` | Magic identification word: `0x5157454E` (ASCII `"QWEN"`). |
| `0x04` | `version` | `uint32` | Container format version: `2`. |
| `0x08` | `hidden_size` | `uint32` | Residual stream dimension ($H$, e.g. 1536, 2560, 5120). |
| `0x0C` | `intermediate_size` | `uint32` | SwiGLU MLP inner hidden dimension ($I$, e.g. 8960, 10240, 27648). |
| `0x10` | `num_layers` | `uint32` | Total transformer decoder layer count ($L$, e.g. 28, 32, 64). |
| `0x14` | `num_attention_heads` | `uint32` | Attention Query heads ($N_Q$, e.g. 12, 16, 32). |
| `0x18` | `num_kv_heads` | `uint32` | Attention Key/Value heads ($N_{KV}$, e.g. 2, 2, 4). |
| `0x1C` | `head_dim` | `uint32` | Dimension per attention head ($D_{\text{head}} = 128$). |
| `0x20` | `vocab_size` | `uint32` | Vocabulary length ($V = 151936$). |
| `0x24` | `max_seq_len` | `uint32` | Maximum context window ceiling ($S_{\text{max}} = 4096$). |
| `0x28` | `quant_type` | `uint32` | Quantization algorithm: `0 = FP16`, `1 = W4A16_G128`. |
| `0x2C` | `tie_word_embeddings` | `uint32` | `1` if `lm_head` is tied to `embed_tokens`, otherwise `0`. |
| `0x30` | `rms_norm_eps` | `float32` | RMSNorm epsilon stabilizer (e.g. `1e-6`). |
| `0x34` | `rope_theta` | `float32` | Base angular frequency for RoPE (e.g. `1000000.0`). |
| `0x38` | `reserved_flags` | `uint32` | Reserved bitfield flags (set to `0`). |
| `0x3C` | `padding0` | `uint32` | Explicit 4-byte padding ensuring 8-byte offset alignment. |
| `0x40` | `layer_table_offset` | `uint64` | Byte offset to `LayerDescriptor` array. |
| `0x48` | `embed_tokens_offset` | `uint64` | Byte offset to input token embedding matrix. |
| `0x50` | `final_norm_offset` | `uint64` | Byte offset to final pre-logits RMSNorm weights. |
| `0x58` | `lm_head_offset` | `uint64` | Byte offset to unembedding matrix (or `0` if tied). |
| `0x60` | `weights_offset` | `uint64` | Byte offset to contiguous aligned layer weights payload. |
| `0x68` | `reserved[0]` | `uint64` | Offset to embedding quantization scales (in W4A16 mode). |
| `0x70` | `reserved[1..5]` | `5 x uint64` | Reserved for future format expansions. |

---

## 3. 216-Byte `LayerDescriptor` Packing (`"<2I26Q"`)

Each layer is described by an immutable 216-byte struct. The struct contains exactly 28 packed fields (2 `uint32` words followed by 26 `uint64` words):

```python
# Python struct format string
LAYER_DESCRIPTOR_FORMAT = "<2I26Q"
assert struct.calcsize(LAYER_DESCRIPTOR_FORMAT) == 216
```

### Complete Field Order

| Index | Field Name | Type | Size | Valid Under | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `type` | `uint32` | 4 B | All | `0 = LAYER_GQA`, `1 = LAYER_DELTANET`. |
| `1` | `reserved` | `uint32` | 4 B | All | Struct alignment padding to 8-byte boundary. |
| `2` | `norm1_weight_offset` | `uint64` | 8 B | All | Pre-attention or pre-DeltaNet RMSNorm weight. |
| `3` | `norm2_weight_offset` | `uint64` | 8 B | All | Pre-MLP RMSNorm weight. |
| `4` | `qkv_weight_offset` | `uint64` | 8 B | GQA | Fused QKV (or Q projection) weight offset. |
| `5` | `q_bias_offset` | `uint64` | 8 B | GQA | Query attention bias vector offset. |
| `6` | `k_bias_offset` | `uint64` | 8 B | GQA | Key attention bias vector offset. |
| `7` | `v_bias_offset` | `uint64` | 8 B | GQA | Value attention bias vector offset. |
| `8` | `attn_out_weight_offset`| `uint64` | 8 B | GQA | Attention output projection ($O$) weight offset. |
| `9` | `conv1d_weight_offset` | `uint64` | 8 B | DeltaNet | Depthwise Conv1d weight offset ($C \times K$). |
| `10` | `conv1d_bias_offset` | `uint64` | 8 B | DeltaNet | Depthwise Conv1d bias vector offset ($C$). |
| `11` | `in_proj_weight_offset`| `uint64` | 8 B | DeltaNet | DeltaNet input projection matrix offset. |
| `12` | `decay_weight_offset` | `uint64` | 8 B | DeltaNet | Beta decay gate projection weight offset. |
| `13` | `delta_out_weight_offset`|`uint64` | 8 B | DeltaNet | DeltaNet output projection matrix offset. |
| `14` | `mlp_gate_offset` | `uint64` | 8 B | All | SwiGLU Gate projection weight offset. |
| `15` | `mlp_up_offset` | `uint64` | 8 B | All | SwiGLU Up projection weight offset. |
| `16` | `mlp_down_offset` | `uint64` | 8 B | All | SwiGLU Down projection weight offset. |
| `17` | `quant_scales_offset` | `uint64` | 8 B | W4A16 | Shared quantization scales array. |
| `18` | `quant_zeros_offset` | `uint64` | 8 B | W4A16 | Quantization zero-points (or `0` for symmetric). |
| `19` | `k_weight_offset` | `uint64` | 8 B | GQA | Key projection weight offset (unfused mode). |
| `20` | `v_weight_offset` | `uint64` | 8 B | GQA | Value projection weight offset (unfused mode). |
| `21` | `q_scale_offset` | `uint64` | 8 B | W4A16 GQA | Query projection scale factors. |
| `22` | `k_scale_offset` | `uint64` | 8 B | W4A16 GQA | Key projection scale factors. |
| `23` | `v_scale_offset` | `uint64` | 8 B | W4A16 GQA | Value projection scale factors. |
| `24` | `attn_out_scale_offset`| `uint64` | 8 B | W4A16 GQA | Attention output projection scale factors. |
| `25` | `mlp_gate_scale_offset`| `uint64` | 8 B | W4A16 MLP | SwiGLU Gate projection scale factors. |
| `26` | `mlp_up_scale_offset` | `uint64` | 8 B | W4A16 MLP | SwiGLU Up projection scale factors. |
| `27` | `mlp_down_scale_offset`| `uint64` | 8 B | W4A16 MLP | SwiGLU Down projection scale factors. |

---

## 4. Dynamic Memory Arena & Shared Memory Sizing

### 4.1 Activation Quantization Offsets in Workspace (`ws_.d_act_1`)
To prevent activation buffer overlap when scaling across dynamic intermediate dimensions ($I$), the Q8_1 activation staging buffers are calculated with variable offsets inside `ws_.d_act_1`:

```cpp
size_t q8_offset    = 65536; // 64 KB safety headroom
size_t q8_bytes     = ((static_cast<size_t>(i_dim) + 255) / 256) * 256;
size_t scale_offset = q8_offset + q8_bytes;
size_t num_groups   = (static_cast<size_t>(i_dim) + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE;
size_t scale_bytes  = ((num_groups * sizeof(float) + 255) / 256) * 256;
size_t sum8_offset  = scale_offset + scale_bytes;

ws_.d_act_q8    = reinterpret_cast<int8_t*>(ws_.d_act_1 + q8_offset);
ws_.d_act_scale = reinterpret_cast<float*>(ws_.d_act_1 + scale_offset);
ws_.d_act_sum8  = reinterpret_cast<int32_t*>(ws_.d_act_1 + sum8_offset);
```

### 4.2 Shared Memory Footprint Derivation
For W4A16 GEMV kernels executing with on-the-fly Q8_1 activation quantization, each thread block allocates dynamic shared memory across three components:
1. **INT8 Quantized Vector (`s_x_q8`):** $H \times 1\text{ byte}$
2. **INT32 Precomputed Block Sums (`s_x_sum8`):** $(H / 8) \times 4\text{ bytes} = H / 2\text{ bytes}$
3. **FP32 Group Scales (`s_x_scale`):** $(H / 128) \times 4\text{ bytes} = H / 32\text{ bytes}$

$$\text{Total SMEM} = H \times \left(1 + \frac{1}{2} + \frac{1}{32}\right) = H \times \frac{49}{32}\text{ bytes}$$

| Dimension ($H$) | Scale Family | Shared Memory Required | Turing Hardware Limit |
| :--- | :--- | :--- | :--- |
| `256` | Synthetic Small | $392\text{ bytes}$ | 64 KB |
| `1536` | Qwen 2.5-1.5B | $2,352\text{ bytes}$ | 64 KB |
| `2560` | Qwen 3.5-4B | $3,920\text{ bytes}$ | 64 KB |
| `5120` | Qwen 3.8-27B | $7,840\text{ bytes}$ | 64 KB |

---

## 5. Dual-Rank Tensor Parallelism ($TP=2$) Partitioning

In 2-way Tensor Parallelism, weights are partitioned across Rank 0 (`cuda:0`) and Rank 1 (`cuda:1`):

### 5.1 Column-Parallel Invariant
$$\text{Output Dimension } N \rightarrow \frac{N}{2}$$
- **Q Projection:** $W_Q \in \mathbb{R}^{H \times \frac{Q\_DIM}{2}}$
- **K Projection:** $W_K \in \mathbb{R}^{H \times \frac{KV\_DIM}{2}}$
- **V Projection:** $W_V \in \mathbb{R}^{H \times \frac{KV\_DIM}{2}}$
- **SwiGLU Gate/Up:** $W_{\text{gate}} \in \mathbb{R}^{H \times \frac{I}{2}}, \quad W_{\text{up}} \in \mathbb{R}^{H \times \frac{I}{2}}$

### 5.2 Row-Parallel Invariant
$$\text{Input Dimension } K \rightarrow \frac{K}{2}$$
- **O Projection:** $W_O \in \mathbb{R}^{\frac{Q\_DIM}{2} \times H}$
- **Down Projection:** $W_{\text{down}} \in \mathbb{R}^{\frac{I}{2} \times H}$

### 5.3 Communication Reduction Invariant
After each row-parallel GEMV execution:
$$Y = Y^{(0)} + Y^{(1)}$$
An All-Reduce Sum barrier across PCIe synchronizes both ranks into `ws_.d_act_0`, restoring the global residual stream.

