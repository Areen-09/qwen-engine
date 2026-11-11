"""
qwen_engine.exporter: Shared quantization and serialization primitives for .qwenbin v2.
"""

import os
import struct
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

MAGIC_QWEN = 0x5157454E  # "QWEN"
MAGIC_QTOK = 0x51544F4B  # "QTOK"
QUANT_GROUP_SIZE = 128

LAYER_GQA = 0
LAYER_DELTANET = 1

TOKEN_ENDOFTEXT_QWEN25 = 151645
TOKEN_ENDOFTEXT_QWEN38 = 248044


def bf16_to_fp16_clamped(bf16_arr: np.ndarray) -> np.ndarray:
    """
    Converts a uint16 array of BF16 bit patterns to float16 with dynamic range clamping
    to [-65504.0, 65504.0] to prevent sm_75 half-precision overflow.
    """
    u16 = bf16_arr.astype(np.uint16)
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    clamped_f32 = np.clip(f32, -65504.0, 65504.0)
    clamped_f32 = np.nan_to_num(clamped_f32, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return clamped_f32.astype(np.float16)


def f32_to_fp16_clamped(f32_arr: np.ndarray) -> np.ndarray:
    """Clamps float32 array safely within float16 bounds [-65504.0, 65504.0]."""
    clamped = np.clip(f32_arr, -65504.0, 65504.0)
    clamped = np.nan_to_num(clamped, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return clamped.astype(np.float16)


def quantize_and_pack_w4a16(weight_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes an [N, K] matrix into W4A16 group-128 format with warp-interleaved layout
    optimized for Turing (sm_75) Tensor Core WMMA and fast GEMV kernels.

    Returns:
        packed_qweight: [N, K // 8] uint32
        scales: [N, K // 128] float16
    """
    N, K = weight_matrix.shape
    assert K % QUANT_GROUP_SIZE == 0, f"K ({K}) must be divisible by {QUANT_GROUP_SIZE}"
    assert K % 256 == 0, f"K ({K}) must be divisible by 256 for warp interleaving"
    num_groups_per_row = K // QUANT_GROUP_SIZE
    num_warp_steps = K // 256

    # Instantaneous fast-path for placeholder/synthetic zero arrays
    if not np.any(weight_matrix):
        packed = np.full((N, num_warp_steps * 32), 0x88888888, dtype=np.uint32)
        scales = np.full((N, num_groups_per_row), 1e-5, dtype=np.float16)
        return packed, scales

    w_grouped = weight_matrix.reshape(N, num_groups_per_row, QUANT_GROUP_SIZE).astype(np.float32)

    max_abs = np.max(np.abs(w_grouped), axis=-1, keepdims=True)
    scales = (max_abs / 7.0).astype(np.float16)
    scales_f32 = np.where(scales == 0, np.float16(1e-5), scales).astype(np.float32)

    q = np.clip(np.round(w_grouped / scales_f32) + 8.0, 0, 15).astype(np.uint32)

    # Vectorized nibble packing into 32-bit words
    q_sub = q.reshape(N, num_groups_per_row, 16, 8)
    q0, q1, q2, q3 = q_sub[..., 0], q_sub[..., 1], q_sub[..., 2], q_sub[..., 3]
    q4, q5, q6, q7 = q_sub[..., 4], q_sub[..., 5], q_sub[..., 6], q_sub[..., 7]

    w32 = (q0) | (q4 << 4) | (q1 << 8) | (q5 << 12) | (q2 << 16) | (q6 << 20) | (q3 << 24) | (q7 << 28)
    w32_paired = w32.reshape(N, num_warp_steps, 2, 16)
    packed = np.ascontiguousarray(w32_paired.reshape(N, num_warp_steps * 32))

    return packed, scales.reshape(N, num_groups_per_row)


def pack_layer_descriptor(
    l_type: int,
    norm1_off: int,
    norm2_off: int,
    q_off: int = 0,
    q_b_off: int = 0,
    k_b_off: int = 0,
    v_b_off: int = 0,
    o_off: int = 0,
    conv1d_w_off: int = 0,
    conv1d_b_off: int = 0,
    in_proj_off: int = 0,
    decay_off: int = 0,
    delta_out_off: int = 0,
    gu_off: int = 0,
    dn_off: int = 0,
    q_scale_off: int = 0,
    k_off: int = 0,
    v_off: int = 0,
    k_scale_off: int = 0,
    v_scale_off: int = 0,
    o_scale_off: int = 0,
    gu_scale_off: int = 0,
    dn_scale_off: int = 0,
) -> bytes:
    """Emits strictly 216-byte LayerDescriptor using format '<2I26Q'."""
    return struct.pack(
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


def pack_header_v2(
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int,
    quant_mode: int,
    tie_word_embeddings: int,
    rms_norm_eps: float,
    rope_theta: float,
    layer_table_offset: int,
    embed_w_offset: int,
    final_norm_offset: int,
    lm_head_offset: int,
    weights_offset: int,
    embed_s_offset: int = 0,
    lm_head_s_offset: int = 0,
) -> bytearray:
    """Emits 4096-byte page-aligned QwenBinHeaderV2 with 64-bit offsets."""
    buf = bytearray(4096)
    struct.pack_into(
        "<12I2f2I11Q",
        buf, 0,
        MAGIC_QWEN, 2,
        hidden_size, intermediate_size, num_layers, num_q_heads, num_kv_heads, head_dim, vocab_size, max_seq_len,
        quant_mode, tie_word_embeddings, rms_norm_eps, rope_theta, 0, 0,
        layer_table_offset, embed_w_offset, final_norm_offset, lm_head_offset, weights_offset,
        embed_s_offset, lm_head_s_offset, 0, 0, 0, 0
    )
    return buf


def export_tokenizer(tokenizer: Any, output_path: str, eos_id: int = TOKEN_ENDOFTEXT_QWEN25) -> None:
    """Exports tokenizer vocab, merges, and special tokens to binary tokenizer.bin."""
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_size = len(sorted_vocab)

    merges: List[Tuple[int, int, int]] = []
    merges_list = getattr(tokenizer, "_merges", None) or getattr(tokenizer, "bpe_ranks", None)
    if isinstance(merges_list, dict):
        merges_list = list(merges_list.keys())
    if merges_list:
        for pair in merges_list:
            left_token, right_token = pair
            if left_token in vocab and right_token in vocab:
                merged_str = left_token + right_token
                if merged_str in vocab:
                    merges.append((vocab[left_token], vocab[right_token], vocab[merged_str]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        header = struct.pack("<IIII", MAGIC_QTOK, vocab_size, len(merges), eos_id)
        f.write(header)

        for piece_str, token_id in sorted_vocab:
            decoded_text = tokenizer.convert_tokens_to_string([piece_str])
            piece_bytes = decoded_text.encode("utf-8", errors="replace")
            f.write(struct.pack("<H", len(piece_bytes)))
            f.write(piece_bytes)
            f.write(struct.pack("<i", token_id))

        for left_id, right_id, merged_id in merges:
            f.write(struct.pack("<iii", left_id, right_id, merged_id))


def verify_qwenbin_v2(filepath: str) -> Dict[str, Any]:
    """Parses and validates a .qwenbin v2 container according to engine specs."""
    assert os.path.exists(filepath), f"File {filepath} not found"
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        header_buf = f.read(4096)
        assert len(header_buf) == 4096, "File too small for 4096-byte header"

        fields = struct.unpack_from("<12I2f2I11Q", header_buf, 0)
        magic = fields[0]
        version = fields[1]
        hidden_size = fields[2]
        intermediate_size = fields[3]
        num_layers = fields[4]
        num_q_heads = fields[5]
        num_kv_heads = fields[6]
        head_dim = fields[7]
        vocab_size = fields[8]
        max_seq_len = fields[9]
        quant_type = fields[10]
        tie_word_embeddings = fields[11]
        rms_norm_eps = fields[12]
        rope_theta = fields[13]

        layer_table_offset = fields[16]
        embed_tokens_offset = fields[17]
        final_norm_offset = fields[18]
        lm_head_offset = fields[19]
        weights_offset = fields[20]
        embed_scale_offset = fields[21]

        # Validations
        assert magic == MAGIC_QWEN, f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_QWEN)}"
        assert version == 2, f"Unsupported version: {version}, expected 2"
        assert layer_table_offset == 4096, f"Invalid layer table offset: {layer_table_offset}"
        assert weights_offset % 256 == 0, f"Weights offset {weights_offset} is not 256-byte aligned"

        f.seek(layer_table_offset)
        layer_types = []
        for l in range(num_layers):
            desc_raw = f.read(216)
            assert len(desc_raw) == 216, f"Truncated layer descriptor at layer {l}"
            desc_fields = struct.unpack("<2I26Q", desc_raw)
            layer_types.append(desc_fields[0])

    num_gqa = sum(1 for t in layer_types if t == LAYER_GQA)
    num_deltanet = sum(1 for t in layer_types if t == LAYER_DELTANET)

    return {
        "num_layers": num_layers,
        "num_gqa": num_gqa,
        "num_deltanet": num_deltanet,
        "quant_type": quant_type,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "tie_word_embeddings": tie_word_embeddings,
        "file_size_mb": file_size / (1024 * 1024),
    }

