#!/usr/bin/env python3
"""
Offline Model Transpiler: Convert GGUF to .qwenbin v2 container format.
Supports Qwen 2.5 (Pure GQA), Qwen 3.5 & Qwen 3.8 (Hybrid Gated DeltaNet).
Performs BF16 to FP16 downcasting with dynamic range clamping,
warp-interleaved W4A16 group-128 quantization, and 64-bit container serialization.
"""

import argparse
import struct
import os
import sys
import numpy as np

MAGIC_QWEN = 0x5157454E  # "QWEN"
MAGIC_QTOK = 0x51544F4B  # "QTOK"
QUANT_GROUP_SIZE = 128

LAYER_GQA = 0
LAYER_DELTANET = 1

# GGML Types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 16

# GGUF Metadata Value Types
GGUF_METADATA_VALUE_TYPE_UINT8   = 0
GGUF_METADATA_VALUE_TYPE_INT8    = 1
GGUF_METADATA_VALUE_TYPE_UINT16  = 2
GGUF_METADATA_VALUE_TYPE_INT16   = 3
GGUF_METADATA_VALUE_TYPE_UINT32  = 4
GGUF_METADATA_VALUE_TYPE_INT32   = 5
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_BOOL    = 7
GGUF_METADATA_VALUE_TYPE_STRING  = 8
GGUF_METADATA_VALUE_TYPE_ARRAY   = 9
GGUF_METADATA_VALUE_TYPE_UINT64  = 10
GGUF_METADATA_VALUE_TYPE_INT64   = 11
GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12

def bf16_to_fp16_clamped(bf16_arr: np.ndarray) -> np.ndarray:
    """
    Converts a uint16 array containing BF16 bits to float16 with dynamic range clamping.
    BF16 has 1 sign bit, 8 exponent bits, 7 mantissa bits.
    Max FP16 representable value is 65504.0.
    """
    u16 = bf16_arr.astype(np.uint16)
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    clamped_f32 = np.clip(f32, -65504.0, 65504.0)
    # Handle NaNs or Infs
    clamped_f32 = np.nan_to_num(clamped_f32, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return clamped_f32.astype(np.float16)

def f32_to_fp16_clamped(f32_arr: np.ndarray) -> np.ndarray:
    clamped = np.clip(f32_arr, -65504.0, 65504.0)
    clamped = np.nan_to_num(clamped, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return clamped.astype(np.float16)

def quantize_and_pack_w4a16(weight_matrix: np.ndarray):
    """
    Quantizes an [N, K] matrix into W4A16 group-128 format with warp-interleaved layout.
    Returns (packed_qweight [N, K/8] uint32, scales [N, K/128] float16)
    """
    N, K = weight_matrix.shape
    assert K % QUANT_GROUP_SIZE == 0
    assert K % 256 == 0, f"K must be multiple of 256 for warp interleaving, got {K}"
    num_groups_per_row = K // QUANT_GROUP_SIZE
    num_warp_steps = K // 256

    w_grouped = weight_matrix.reshape(N, num_groups_per_row, QUANT_GROUP_SIZE).astype(np.float32)

    max_abs = np.max(np.abs(w_grouped), axis=-1, keepdims=True)
    scales = (max_abs / 7.0).astype(np.float16)
    scales_f32 = np.where(scales == 0, np.float16(1e-5), scales).astype(np.float32)

    q = np.clip(np.round(w_grouped / scales_f32) + 8.0, 0, 15).astype(np.uint32)

    q_sub = q.reshape(N, num_groups_per_row, 16, 8)
    q0 = q_sub[..., 0]
    q1 = q_sub[..., 1]
    q2 = q_sub[..., 2]
    q3 = q_sub[..., 3]
    q4 = q_sub[..., 4]
    q5 = q_sub[..., 5]
    q6 = q_sub[..., 6]
    q7 = q_sub[..., 7]

    w32 = (q0) | (q4 << 4) | (q1 << 8) | (q5 << 12) | (q2 << 16) | (q6 << 20) | (q3 << 24) | (q7 << 28)
    w32_paired = w32.reshape(N, num_warp_steps, 2, 16)
    packed = np.ascontiguousarray(w32_paired.reshape(N, num_warp_steps * 32))

    return packed, scales.reshape(N, num_groups_per_row)

class GGUFReader:
    """Self-contained zero-dependency GGUF reader supporting v2 and v3."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, "rb")
        self.metadata = {}
        self.tensor_info = {}
        self.tensor_data_offset = 0
        self._parse_header()

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _read_str(self) -> str:
        length = struct.unpack("<Q", self.file.read(8))[0]
        data = self.file.read(length)
        return data.decode("utf-8", errors="replace")

    def _read_val(self, val_type):
        if val_type == GGUF_METADATA_VALUE_TYPE_UINT8:
            return struct.unpack("<B", self.file.read(1))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_INT8:
            return struct.unpack("<b", self.file.read(1))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_UINT16:
            return struct.unpack("<H", self.file.read(2))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_INT16:
            return struct.unpack("<h", self.file.read(2))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_UINT32:
            return struct.unpack("<I", self.file.read(4))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_INT32:
            return struct.unpack("<i", self.file.read(4))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return struct.unpack("<f", self.file.read(4))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_BOOL:
            return struct.unpack("<?", self.file.read(1))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_STRING:
            return self._read_str()
        elif val_type == GGUF_METADATA_VALUE_TYPE_UINT64:
            return struct.unpack("<Q", self.file.read(8))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_INT64:
            return struct.unpack("<q", self.file.read(8))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return struct.unpack("<d", self.file.read(8))[0]
        elif val_type == GGUF_METADATA_VALUE_TYPE_ARRAY:
            sub_type = struct.unpack("<I", self.file.read(4))[0]
            arr_len = struct.unpack("<Q", self.file.read(8))[0]
            return [self._read_val(sub_type) for _ in range(arr_len)]
        else:
            raise ValueError(f"Unknown GGUF metadata value type: {val_type}")

    def _parse_header(self):
        magic = self.file.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic}")
        version = struct.unpack("<I", self.file.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        tensor_count = struct.unpack("<Q", self.file.read(8))[0]
        kv_count = struct.unpack("<Q", self.file.read(8))[0]

        for _ in range(kv_count):
            key = self._read_str()
            val_type = struct.unpack("<I", self.file.read(4))[0]
            val = self._read_val(val_type)
            self.metadata[key] = val

        alignment = self.metadata.get("general.alignment", 32)

        for _ in range(tensor_count):
            name = self._read_str()
            n_dims = struct.unpack("<I", self.file.read(4))[0]
            dims = [struct.unpack("<Q", self.file.read(8))[0] for _ in range(n_dims)]
            type_id = struct.unpack("<I", self.file.read(4))[0]
            offset = struct.unpack("<Q", self.file.read(8))[0]
            self.tensor_info[name] = {
                "dims": dims,
                "type": type_id,
                "offset": offset,
            }

        curr_pos = self.file.tell()
        padding = (alignment - (curr_pos % alignment)) % alignment
        self.tensor_data_offset = curr_pos + padding

    def read_tensor(self, name: str) -> np.ndarray:
        if name not in self.tensor_info:
            raise KeyError(f"Tensor {name} not found in GGUF file")
        info = self.tensor_info[name]
        type_id = info["type"]
        dims = info["dims"] # In GGUF, dims are [cols, rows, ...]
        abs_offset = self.tensor_data_offset + info["offset"]
        self.file.seek(abs_offset)

        num_elements = 1
        for d in dims:
            num_elements *= d

        if type_id == GGML_TYPE_F32:
            raw = self.file.read(num_elements * 4)
            arr = np.frombuffer(raw, dtype=np.float32)
        elif type_id == GGML_TYPE_F16:
            raw = self.file.read(num_elements * 2)
            arr = np.frombuffer(raw, dtype=np.float16)
        elif type_id == GGML_TYPE_BF16:
            raw = self.file.read(num_elements * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = bf16_to_fp16_clamped(u16)
        else:
            # Fallback reading raw bytes if needed
            raw = self.file.read(num_elements * 2)
            arr = np.frombuffer(raw, dtype=np.float16)

        # In GGUF, 2D matrix dims are [cols, rows], so numpy shape is [rows, cols]
        if len(dims) == 2:
            shape = (dims[1], dims[0])
            arr = arr.reshape(shape)
        elif len(dims) == 1:
            arr = arr.reshape((dims[0],))
        return arr

def export_tokenizer_from_gguf(reader: GGUFReader, output_path: str):
    tokens = reader.metadata.get("tokenizer.ggml.tokens", None)
    if not tokens:
        print("[Converter] Warning: GGUF does not contain tokenizer tokens. Generating fallback...")
        return

    merges = reader.metadata.get("tokenizer.ggml.merges", [])
    vocab_size = len(tokens)
    print(f"[Converter] Extracting tokenizer with {vocab_size} tokens to {output_path}...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        header = struct.pack("<IIII", MAGIC_QTOK, vocab_size, 0, 151645)
        f.write(header)

        for i, tok_str in enumerate(tokens):
            if isinstance(tok_str, str):
                p_bytes = tok_str.encode("utf-8", errors="replace")
            else:
                p_bytes = bytes(tok_str)
            f.write(struct.pack("<H", len(p_bytes)))
            f.write(p_bytes)
            f.write(struct.pack("<i", i))

def convert_gguf_to_qwenbin(gguf_path: str, output_path: str, mode: str = "w4a16", tokenizer_out: str = None):
    print(f"[Converter] Parsing GGUF container: {gguf_path}...")
    is_w4a16 = (mode == "w4a16")
    quant_mode_val = 1 if is_w4a16 else 0

    with GGUFReader(gguf_path) as reader:
        # Detect model architecture parameters from GGUF metadata
        meta = reader.metadata
        arch = meta.get("general.architecture", "qwen2")
        hidden_size = meta.get(f"{arch}.embedding_length", 1536)
        intermediate_size = meta.get(f"{arch}.feed_forward_length", 8960)
        num_layers = meta.get(f"{arch}.block_count", 28)
        num_q_heads = meta.get(f"{arch}.attention.head_count", 12)
        num_kv_heads = meta.get(f"{arch}.attention.head_count_kv", 2)
        head_dim = meta.get(f"{arch}.attention.key_length", hidden_size // num_q_heads)
        vocab_size = meta.get(f"{arch}.vocab_size", 151936)
        max_seq_len = meta.get(f"{arch}.context_length", 4096)
        rms_norm_eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
        rope_theta = float(meta.get(f"{arch}.rope.freq_base", 1000000.0))

        # Check embed_tokens and lm_head
        embed_name = "token_embd.weight"
        if embed_name not in reader.tensor_info:
            embed_name = "model.embed_tokens.weight"
        embed_w = reader.read_tensor(embed_name)
        if embed_w.dtype != np.float16:
            embed_w = f32_to_fp16_clamped(embed_w)
        vocab_size = embed_w.shape[0]
        hidden_size = embed_w.shape[1]

        lm_head_name = "output.weight"
        if lm_head_name not in reader.tensor_info:
            lm_head_name = "lm_head.weight"
        tie_word_embeddings = 1
        if lm_head_name in reader.tensor_info and lm_head_name != embed_name:
            lm_head_w = reader.read_tensor(lm_head_name)
            if not np.array_equal(embed_w, lm_head_w):
                tie_word_embeddings = 0
        else:
            lm_head_w = embed_w

        final_norm_name = "output_norm.weight"
        if final_norm_name not in reader.tensor_info:
            final_norm_name = "model.norm.weight"
        final_norm_w = reader.read_tensor(final_norm_name)
        if final_norm_w.dtype != np.float16:
            final_norm_w = f32_to_fp16_clamped(final_norm_w)

        # Detect layer types
        layer_types = []
        for l in range(num_layers):
            # Check if layer has conv1d or linear_attn
            deltanet_keys = [
                f"blk.{l}.conv1d.weight",
                f"blk.{l}.linear_attn.in_proj.weight",
                f"blk.{l}.linear_attn.decay.weight",
                f"blk.{l}.deltanet.weight"
            ]
            is_deltanet = any(k in reader.tensor_info for k in deltanet_keys)
            layer_types.append(LAYER_DELTANET if is_deltanet else LAYER_GQA)

        num_deltanet = sum(1 for t in layer_types if t == LAYER_DELTANET)
        num_gqa = num_layers - num_deltanet
        print(f"[Converter] Model configuration: {num_layers} layers ({num_gqa} GQA, {num_deltanet} DeltaNet)")
        print(f"  - Hidden Size: {hidden_size}, Intermediate: {intermediate_size}")
        print(f"  - Heads: Q={num_q_heads}, KV={num_kv_heads}, Head Dim={head_dim}")

        if tokenizer_out:
            export_tokenizer_from_gguf(reader, tokenizer_out)

        # Prepare output .qwenbin v2 container
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            layer_table_offset = 4096
            layer_table_size = num_layers * 216
            weights_offset = (layer_table_offset + layer_table_size + 255) & ~255
            f.seek(weights_offset)

            def write_tensor(arr_bytes):
                offset = f.tell()
                f.write(arr_bytes)
                rem = f.tell() % 256
                if rem != 0:
                    f.write(b"\x00" * (256 - rem))
                return offset

            # 1. Embed tokens
            if is_w4a16:
                q_w, s_w = quantize_and_pack_w4a16(embed_w)
                embed_w_offset = write_tensor(q_w.tobytes())
                embed_s_offset = write_tensor(s_w.tobytes())
            else:
                embed_w_offset = write_tensor(embed_w.tobytes())
                embed_s_offset = 0

            # LM head
            if tie_word_embeddings == 0:
                if is_w4a16:
                    q_lm, s_lm = quantize_and_pack_w4a16(lm_head_w)
                    lm_head_offset = write_tensor(q_lm.tobytes())
                else:
                    lm_head_offset = write_tensor(lm_head_w.tobytes())
            else:
                lm_head_offset = embed_w_offset

            # Final norm
            final_norm_offset = write_tensor(final_norm_w.tobytes())

            # 2. Layers
            layer_descriptors = []

            for l in range(num_layers):
                l_type = layer_types[l]

                # Input norm
                norm1_name = f"blk.{l}.attn_norm.weight"
                if norm1_name not in reader.tensor_info:
                    norm1_name = f"model.layers.{l}.input_layernorm.weight"
                norm1_w = reader.read_tensor(norm1_name)
                norm1_off = write_tensor(f32_to_fp16_clamped(norm1_w).tobytes())

                # Post attention norm
                norm2_name = f"blk.{l}.ffn_norm.weight"
                if norm2_name not in reader.tensor_info:
                    norm2_name = f"model.layers.{l}.post_attention_layernorm.weight"
                norm2_w = reader.read_tensor(norm2_name)
                norm2_off = write_tensor(f32_to_fp16_clamped(norm2_w).tobytes())

                q_off = q_scale_off = q_b_off = 0
                k_off = k_scale_off = k_b_off = 0
                v_off = v_scale_off = v_b_off = 0
                o_off = o_scale_off = 0

                conv1d_w_off = conv1d_b_off = in_proj_off = decay_off = delta_out_off = 0

                if l_type == LAYER_GQA:
                    # GQA tensors
                    q_name = f"blk.{l}.attn_q.weight" if f"blk.{l}.attn_q.weight" in reader.tensor_info else f"model.layers.{l}.self_attn.q_proj.weight"
                    k_name = f"blk.{l}.attn_k.weight" if f"blk.{l}.attn_k.weight" in reader.tensor_info else f"model.layers.{l}.self_attn.k_proj.weight"
                    v_name = f"blk.{l}.attn_v.weight" if f"blk.{l}.attn_v.weight" in reader.tensor_info else f"model.layers.{l}.self_attn.v_proj.weight"
                    o_name = f"blk.{l}.attn_output.weight" if f"blk.{l}.attn_output.weight" in reader.tensor_info else f"model.layers.{l}.self_attn.o_proj.weight"

                    q_w = f32_to_fp16_clamped(reader.read_tensor(q_name))
                    k_w = f32_to_fp16_clamped(reader.read_tensor(k_name))
                    v_w = f32_to_fp16_clamped(reader.read_tensor(v_name))
                    o_w = f32_to_fp16_clamped(reader.read_tensor(o_name))

                    # Biases
                    q_b_name = f"blk.{l}.attn_q.bias" if f"blk.{l}.attn_q.bias" in reader.tensor_info else f"model.layers.{l}.self_attn.q_proj.bias"
                    k_b_name = f"blk.{l}.attn_k.bias" if f"blk.{l}.attn_k.bias" in reader.tensor_info else f"model.layers.{l}.self_attn.k_proj.bias"
                    v_b_name = f"blk.{l}.attn_v.bias" if f"blk.{l}.attn_v.bias" in reader.tensor_info else f"model.layers.{l}.self_attn.v_proj.bias"

                    q_b = f32_to_fp16_clamped(reader.read_tensor(q_b_name)) if q_b_name in reader.tensor_info else np.zeros((q_w.shape[0],), dtype=np.float16)
                    k_b = f32_to_fp16_clamped(reader.read_tensor(k_b_name)) if k_b_name in reader.tensor_info else np.zeros((k_w.shape[0],), dtype=np.float16)
                    v_b = f32_to_fp16_clamped(reader.read_tensor(v_b_name)) if v_b_name in reader.tensor_info else np.zeros((v_w.shape[0],), dtype=np.float16)

                    if is_w4a16:
                        packed_q, scale_q = quantize_and_pack_w4a16(q_w)
                        packed_k, scale_k = quantize_and_pack_w4a16(k_w)
                        packed_v, scale_v = quantize_and_pack_w4a16(v_w)
                        packed_o, scale_o = quantize_and_pack_w4a16(o_w)
                        q_off = write_tensor(packed_q.tobytes())
                        q_scale_off = write_tensor(scale_q.tobytes())
                        k_off = write_tensor(packed_k.tobytes())
                        k_scale_off = write_tensor(scale_k.tobytes())
                        v_off = write_tensor(packed_v.tobytes())
                        v_scale_off = write_tensor(scale_v.tobytes())
                        o_off = write_tensor(packed_o.tobytes())
                        o_scale_off = write_tensor(scale_o.tobytes())
                    else:
                        q_off = write_tensor(q_w.tobytes())
                        k_off = write_tensor(k_w.tobytes())
                        v_off = write_tensor(v_w.tobytes())
                        o_off = write_tensor(o_w.tobytes())

                    q_b_off = write_tensor(q_b.tobytes())
                    k_b_off = write_tensor(k_b.tobytes())
                    v_b_off = write_tensor(v_b.tobytes())

                else:
                    # DeltaNet tensors
                    c1d_w_name = f"blk.{l}.conv1d.weight"
                    c1d_b_name = f"blk.{l}.conv1d.bias"
                    in_proj_name = f"blk.{l}.linear_attn.in_proj.weight"
                    decay_name = f"blk.{l}.linear_attn.decay.weight"
                    delta_out_name = f"blk.{l}.linear_attn.out_proj.weight"

                    c1d_w = f32_to_fp16_clamped(reader.read_tensor(c1d_w_name)) if c1d_w_name in reader.tensor_info else np.zeros((num_q_heads * head_dim, 4), dtype=np.float16)
                    c1d_b = f32_to_fp16_clamped(reader.read_tensor(c1d_b_name)) if c1d_b_name in reader.tensor_info else np.zeros((num_q_heads * head_dim,), dtype=np.float16)
                    in_proj_w = f32_to_fp16_clamped(reader.read_tensor(in_proj_name)) if in_proj_name in reader.tensor_info else np.zeros((num_q_heads * head_dim * 2, hidden_size), dtype=np.float16)
                    decay_w = f32_to_fp16_clamped(reader.read_tensor(decay_name)) if decay_name in reader.tensor_info else np.zeros((num_q_heads,), dtype=np.float16)
                    delta_out_w = f32_to_fp16_clamped(reader.read_tensor(delta_out_name)) if delta_out_name in reader.tensor_info else np.zeros((hidden_size, num_q_heads * head_dim), dtype=np.float16)

                    conv1d_w_off = write_tensor(c1d_w.tobytes())
                    conv1d_b_off = write_tensor(c1d_b.tobytes())
                    in_proj_off = write_tensor(in_proj_w.tobytes())
                    decay_off = write_tensor(decay_w.tobytes())
                    delta_out_off = write_tensor(delta_out_w.tobytes())

                # MLP SwiGLU tensors
                gate_name = f"blk.{l}.ffn_gate.weight" if f"blk.{l}.ffn_gate.weight" in reader.tensor_info else f"model.layers.{l}.mlp.gate_proj.weight"
                up_name   = f"blk.{l}.ffn_up.weight" if f"blk.{l}.ffn_up.weight" in reader.tensor_info else f"model.layers.{l}.mlp.up_proj.weight"
                down_name = f"blk.{l}.ffn_down.weight" if f"blk.{l}.ffn_down.weight" in reader.tensor_info else f"model.layers.{l}.mlp.down_proj.weight"

                gate_w = f32_to_fp16_clamped(reader.read_tensor(gate_name))
                up_w   = f32_to_fp16_clamped(reader.read_tensor(up_name))
                down_w = f32_to_fp16_clamped(reader.read_tensor(down_name))

                gate_up_w = np.empty((2 * intermediate_size, hidden_size), dtype=np.float16)
                gate_up_w[0::2, :] = gate_w
                gate_up_w[1::2, :] = up_w

                gu_off = gu_scale_off = 0
                dn_off = dn_scale_off = 0
                if is_w4a16:
                    packed_gu, scale_gu = quantize_and_pack_w4a16(gate_up_w)
                    packed_dn, scale_dn = quantize_and_pack_w4a16(down_w)
                    gu_off = write_tensor(packed_gu.tobytes())
                    gu_scale_off = write_tensor(scale_gu.tobytes())
                    dn_off = write_tensor(packed_dn.tobytes())
                    dn_scale_off = write_tensor(scale_dn.tobytes())
                else:
                    gu_off = write_tensor(gate_up_w.tobytes())
                    dn_off = write_tensor(down_w.tobytes())

                # Build LayerDescriptor
                desc = struct.pack(
                    "<2I26Q",
                    l_type, 0,
                    norm1_off, norm2_off,
                    q_off, q_b_off, k_b_off, v_b_off, o_off,
                    conv1d_w_off, conv1d_b_off, in_proj_off, decay_off, delta_out_off,
                    gu_off, gu_off, dn_off,
                    q_scale_off, 0,
                    k_off, v_off,
                    q_scale_off, k_scale_off, v_scale_off, o_scale_off,
                    gu_scale_off, gu_scale_off, dn_scale_off
                )
                layer_descriptors.append(desc)

            total_file_size = f.tell()

            # Write LayerDescriptor table
            f.seek(layer_table_offset)
            for desc in layer_descriptors:
                f.write(desc)

            # Write QwenBinHeaderV2
            f.seek(0)
            header_buf = bytearray(4096)
            struct.pack_into(
                "<12I2f2I11Q",
                header_buf,
                0,
                MAGIC_QWEN,
                2, # version 2
                hidden_size,
                intermediate_size,
                num_layers,
                num_q_heads,
                num_kv_heads,
                head_dim,
                vocab_size,
                max_seq_len,
                quant_mode_val,
                tie_word_embeddings,
                rms_norm_eps,
                rope_theta,
                0, # reserved_flags
                0, # padding0
                layer_table_offset,
                embed_w_offset,
                final_norm_offset,
                lm_head_offset,
                weights_offset,
                embed_s_offset, # reserved[0]
                0, 0, 0, 0, 0   # reserved[1..5]
            )
            f.write(header_buf)

    print(f"[Converter] Done! Exported {output_path} ({total_file_size / (1024*1024):.2f} MB).")

def main():
    parser = argparse.ArgumentParser(description="Convert GGUF models to .qwenbin v2 container")
    parser.add_argument("--gguf", required=True, help="Input .gguf file path")
    parser.add_argument("--out", required=True, help="Output .qwenbin container path")
    parser.add_argument("--mode", choices=["fp16", "w4a16"], default="w4a16", help="Precision mode")
    parser.add_argument("--tokenizer-out", default=None, help="Optional output tokenizer.bin path")
    args = parser.parse_args()

    convert_gguf_to_qwenbin(args.gguf, args.out, mode=args.mode, tokenizer_out=args.tokenizer_out)

if __name__ == "__main__":
    main()

