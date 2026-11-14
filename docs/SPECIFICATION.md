# Qwen Engine: Technical Engineering Specification (`.qwenbin`)

* **Supported Formats:** `.qwenbin` v1 (`QWQwenHeader`) and `.qwenbin` v2 (`QwenBinHeaderV2`)
* **Byte Alignment:** 64-byte header alignment (v2), 4096-byte header alignment (v1)
* **Endianness:** Little-Endian (`<`)

---

## 1. Container Overview

The `.qwenbin` binary container is a single-file model format designed for memory-mapped (`mmap`) runtime ingestion in C++20/CUDA without external framework dependencies. `QwenEngine` validates that the loaded file uses `W4A16_G128` quantization and matches Qwen 2.5-1.5B-Instruct dimensions (`H=1536, I=8960, L=28, N_Q=12, N_KV=2, D_head=128, V=151936`).

```text
+----------------------------------------------------------------+
| QwenBinHeaderV2 (192 bytes, 64-byte aligned)                   |
| - Magic: 0x5157454E ("QWEN"), version = 2                      |
| - Architecture dimensions, W4A16_G128 mode, layer counts       |
+----------------------------------------------------------------+
| Layer Table: LayerDescriptor[28]                               |
| - 216 bytes per descriptor (format: "<2I26Q")                  |
| - type = 0 (LAYER_GQA)                                         |
| - 26x 64-bit tensor payload byte offsets                       |
+----------------------------------------------------------------+
| Padding to 256-byte alignment                                  |
+----------------------------------------------------------------+
| Global Weights: Embeddings (+ scales), Final RMSNorm, LM Head  |
+----------------------------------------------------------------+
| Layer Weights: GQA Attention / SwiGLU MLP / Group-128 Scales   |
+----------------------------------------------------------------+
```

---

## 2. Global Container Header (`QwenBinHeaderV2`)

Located at byte offset `0` of `.qwenbin` v2 files, aligned to 64 bytes:

| Offset | Field Name | Type | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `uint32` | Magic identification word: `0x5157454E` (ASCII `"QWEN"`). |
| `0x04` | `version` | `uint32` | Container format version: `2`. |
| `0x08` | `hidden_size` | `uint32` | Residual stream dimension (`1536`). |
| `0x0C` | `intermediate_size` | `uint32` | SwiGLU MLP inner hidden dimension (`8960`). |
| `0x10` | `num_layers` | `uint32` | Total transformer decoder layer count (`28`). |
| `0x14` | `num_attention_heads` | `uint32` | Attention Query heads (`12`). |
| `0x18` | `num_kv_heads` | `uint32` | Attention Key/Value heads (`2`). |
| `0x1C` | `head_dim` | `uint32` | Dimension per attention head (`128`). |
| `0x20` | `vocab_size` | `uint32` | Vocabulary length (`151936`). |
| `0x24` | `max_seq_len` | `uint32` | Maximum context window ceiling (`4096`). |
| `0x28` | `quant_type` | `uint32` | Quantization algorithm: `1 = W4A16_G128`. |
| `0x2C` | `tie_word_embeddings` | `uint32` | `1` if `lm_head` is tied to `embed_tokens`, otherwise `0`. |
| `0x30` | `rms_norm_eps` | `float32` | RMSNorm epsilon stabilizer (`1e-6`). |
| `0x34` | `rope_theta` | `float32` | Base angular frequency for RoPE (`1000000.0`). |
| `0x38` | `reserved_flags` | `uint32` | Reserved bitfield flags (`0`). |
| `0x3C` | `padding0` | `uint32` | Explicit 4-byte padding ensuring 8-byte offset alignment. |
| `0x40` | `layer_table_offset` | `uint64` | Byte offset to `LayerDescriptor` array. |
| `0x48` | `embed_tokens_offset` | `uint64` | Byte offset to input token embedding matrix. |
| `0x50` | `final_norm_offset` | `uint64` | Byte offset to final pre-logits RMSNorm weights. |
| `0x58` | `lm_head_offset` | `uint64` | Byte offset to unembedding matrix (or `0` if tied). |
| `0x60` | `weights_offset` | `uint64` | Byte offset to contiguous aligned layer weights payload. |
| `0x68` | `reserved[0]` | `uint64` | Offset to embedding quantization scales. |
| `0x70` | `reserved[1..5]` | `5 x uint64` | Reserved (`reserved[1]` holds `lm_head` scales if untied). |

---

## 3. 216-Byte `LayerDescriptor` Packing (`"<2I26Q"`)

Each layer is described by an immutable 216-byte struct (`2` `uint32` words + `26` `uint64` words):

| Index | Field Name | Type | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0` | `type` | `uint32` | 4 B | `0 = LAYER_GQA`. |
| `1` | `reserved` | `uint32` | 4 B | Struct alignment padding to 8-byte boundary. |
| `2` | `norm1_weight_offset` | `uint64` | 8 B | Pre-attention RMSNorm weight. |
| `3` | `norm2_weight_offset` | `uint64` | 8 B | Pre-MLP RMSNorm weight. |
| `4` | `qkv_weight_offset` | `uint64` | 8 B | Query projection weight offset. |
| `5` | `q_bias_offset` | `uint64` | 8 B | Query attention bias vector offset. |
| `6` | `k_bias_offset` | `uint64` | 8 B | Key attention bias vector offset. |
| `7` | `v_bias_offset` | `uint64` | 8 B | Value attention bias vector offset. |
| `8` | `attn_out_weight_offset` | `uint64` | 8 B | Attention output projection (`o_proj`) weight offset. |
| `9..13` | `reserved_deltanet[5]` | `5 x uint64` | 40 B | Reserved zeroed padding preserving 216-byte binary layout. |
| `14` | `mlp_gate_offset` | `uint64` | 8 B | Interleaved SwiGLU Gate+Up projection weight offset. |
| `15` | `mlp_up_offset` | `uint64` | 8 B | Same as `mlp_gate_offset` in interleaved layout. |
| `16` | `mlp_down_offset` | `uint64` | 8 B | SwiGLU Down projection weight offset. |
| `17` | `quant_scales_offset` | `uint64` | 8 B | Shared quantization scales array. |
| `18` | `quant_zeros_offset` | `uint64` | 8 B | Zero points (`0` for symmetric quantization). |
| `19` | `k_weight_offset` | `uint64` | 8 B | Key projection weight offset. |
| `20` | `v_weight_offset` | `uint64` | 8 B | Value projection weight offset. |
| `21` | `q_scale_offset` | `uint64` | 8 B | Query projection scale factors. |
| `22` | `k_scale_offset` | `uint64` | 8 B | Key projection scale factors. |
| `23` | `v_scale_offset` | `uint64` | 8 B | Value projection scale factors. |
| `24` | `attn_out_scale_offset` | `uint64` | 8 B | Attention output projection scale factors. |
| `25` | `mlp_gate_scale_offset` | `uint64` | 8 B | Interleaved Gate+Up projection scale factors. |
| `26` | `mlp_up_scale_offset` | `uint64` | 8 B | Same as `mlp_gate_scale_offset`. |
| `27` | `mlp_down_scale_offset` | `uint64` | 8 B | SwiGLU Down projection scale factors. |

---

## 4. Workspace & Shared Memory Sizing

### 4.1 Dedicated Q8_1 Scratch Buffer
`ws_.d_act_q8`, `ws_.d_act_scale`, and `ws_.d_act_sum8` are allocated in a dedicated `16 KB` region inside `WorkspaceBuffers` (completely separate from `ws_.d_act_1` to prevent aliasing across prefill/decode buffers):
- `d_act_q8`: `9216 B` (covers `I = 8960` `int8_t` activations)
- `d_act_scale`: `512 B` (covers `70` `float` group scales)
- `d_act_sum8`: `4608 B` (covers `1120` `int32_t` block sums)

### 4.2 GEMV Shared Memory Footprint
Each W4A8 GEMV block stages Q8_1 activations in shared memory:
$$\text{Total SMEM} = K \times \left(1 + \frac{1}{2} + \frac{1}{32}\right) = K \times \frac{49}{32}\text{ bytes}$$

| Dimension (`K`) | Layer Projections | Shared Memory Required |
| :--- | :--- | :--- |
| `1536` | QKV, O-Proj, Gate+Up, LM Head | `2,352 bytes` |
| `8960` | MLP Down-Proj | `13,720 bytes` |
