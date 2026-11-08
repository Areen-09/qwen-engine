#!/usr/bin/env python3
"""
Offline Model Exporter & Group-128 Quantizer for Qwen2.5
Generates .qwenbin v2 container files and binary tokenizer.bin
Supports arbitrary model sizes and configurations from HuggingFace.
"""

import argparse
import struct
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

MAGIC_QWEN = 0x5157454E  # "QWEN"
MAGIC_QTOK = 0x51544F4B  # "QTOK"
QUANT_GROUP_SIZE = 128

LAYER_GQA = 0
LAYER_DELTANET = 1

class LayerTensorIdx:
    INPUT_LAYERNORM          = 0
    Q_WEIGHT                 = 1
    Q_SCALE                  = 2
    Q_BIAS                   = 3
    K_WEIGHT                 = 4
    K_SCALE                  = 5
    K_BIAS                   = 6
    V_WEIGHT                 = 7
    V_SCALE                  = 8
    V_BIAS                   = 9
    O_WEIGHT                 = 10
    O_SCALE                  = 11
    POST_ATTENTION_LAYERNORM = 12
    GATE_UP_WEIGHT           = 13
    GATE_UP_SCALE            = 14
    DOWN_WEIGHT              = 15
    DOWN_SCALE               = 16
    COUNT                    = 17

def quantize_and_pack_w4a16(weight_matrix: np.ndarray):
    """
    Quantizes an [N, K] matrix into W4A16 group-128 format with warp-interleaved layout.
    Returns (packed_qweight [N, K/8] uint32, scales [N, K/128] float16)
    """
    N, K = weight_matrix.shape
    assert K % QUANT_GROUP_SIZE == 0
    assert K % 256 == 0
    num_groups_per_row = K // QUANT_GROUP_SIZE
    num_warp_steps = K // 256

    w_grouped = weight_matrix.reshape(N, num_groups_per_row, QUANT_GROUP_SIZE).astype(np.float32)

    max_abs = np.max(np.abs(w_grouped), axis=-1, keepdims=True)
    scales = (max_abs / 7.0).astype(np.float16)
    scales_f32 = np.where(scales == 0, np.float16(1e-5), scales).astype(np.float32)

    q = np.clip(np.round(w_grouped / scales_f32) + 8.0, 0, 15).astype(np.uint32)

    # Fast vectorized nibble packing
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

    # Interleave group 2*s (lanes 0..15) and group 2*s + 1 (lanes 16..31)
    w32_paired = w32.reshape(N, num_warp_steps, 2, 16)
    packed = np.ascontiguousarray(w32_paired.reshape(N, num_warp_steps * 32))

    return packed, scales.reshape(N, num_groups_per_row)

def export_tokenizer(tokenizer, output_path):
    print(f"[Exporter] Serializing binary tokenizer to {output_path}...")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_size = len(sorted_vocab)

    merges = []
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
        header = struct.pack("<IIII", MAGIC_QTOK, vocab_size, len(merges), 151645)
        f.write(header)

        for piece_str, token_id in sorted_vocab:
            decoded_text = tokenizer.convert_tokens_to_string([piece_str])
            piece_bytes = decoded_text.encode("utf-8", errors="replace")
            f.write(struct.pack("<H", len(piece_bytes)))
            f.write(piece_bytes)
            f.write(struct.pack("<i", token_id))

        for left_id, right_id, merged_id in merges:
            f.write(struct.pack("<iii", left_id, right_id, merged_id))

    print(f"[Exporter] Serialized {vocab_size} tokens and {len(merges)} merges.")

def export_dummy_tokenizer(output_path, vocab_size=151646):
    print(f"[Exporter] Generating offline synthetic tokenizer at {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        header = struct.pack("<IIII", MAGIC_QTOK, vocab_size, 0, 151645)
        f.write(header)

        for i in range(vocab_size):
            if i == 151643:
                p = b"<|endoftext|>"
            elif i == 151644:
                p = b"<|im_start|>"
            elif i == 151645:
                p = b"<|im_end|>"
            elif i < 256:
                p = bytes([i])
            else:
                p = f"tok_{i}".encode("utf-8")
            f.write(struct.pack("<H", len(p)))
            f.write(p)
            f.write(struct.pack("<i", i))
    print(f"[Exporter] Offline tokenizer created with {vocab_size} tokens.")

def export_model(model_id, mode, output_path, tokenizer_out, dummy=False, version=2):
    is_w4a16 = (mode == "w4a16")
    quant_mode_val = 1 if is_w4a16 else 0

    if dummy:
        print("[Exporter] Generating deterministic model in memory (seed=42)...")
        torch.manual_seed(42)
        vocab_size = 151936
        hidden_size = 1536
        intermediate_size = 8960
        num_layers = 28
        num_q_heads = 12
        num_kv_heads = 2
        head_dim = 128
        max_seq_len = 4096
        rms_norm_eps = 1e-6
        rope_theta = 1000000.0
        tie_word_embeddings = 1

        config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=max_seq_len,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=True
        )
        model = Qwen2ForCausalLM(config).half()
        export_dummy_tokenizer(tokenizer_out, vocab_size=vocab_size)
        state_dict = model.state_dict()
    else:
        print(f"[Exporter] Loading {model_id} from HuggingFace or local checkpoint...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        export_tokenizer(tokenizer, tokenizer_out)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True
            )
        state_dict = model.state_dict()
        cfg = model.config
        vocab_size = getattr(cfg, "vocab_size", 151936)
        hidden_size = getattr(cfg, "hidden_size", 1536)
        intermediate_size = getattr(cfg, "intermediate_size", 8960)
        num_layers = getattr(cfg, "num_hidden_layers", 28)
        num_q_heads = getattr(cfg, "num_attention_heads", 12)
        num_kv_heads = getattr(cfg, "num_key_value_heads", 2)
        head_dim = getattr(cfg, "head_dim", hidden_size // num_q_heads)
        max_seq_len = getattr(cfg, "max_position_embeddings", 4096)
        rms_norm_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
        tie_word_embeddings = 1 if getattr(cfg, "tie_word_embeddings", True) else 0

    embed_w = state_dict["model.embed_tokens.weight"].numpy()
    lm_head_w = state_dict["lm_head.weight"].numpy() if "lm_head.weight" in state_dict else embed_w
    if not np.array_equal(embed_w, lm_head_w):
        print("[Exporter] Notice: lm_head is distinct from embed_tokens.")
        tie_word_embeddings = 0

    print(f"[Exporter] Preparing .qwenbin v{version} in {mode.upper()} mode...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        # Determine payload offset
        if version == 2:
            # 4096-byte header + layer descriptors
            layer_table_offset = 4096
            layer_table_size = num_layers * 216
            weights_offset = (layer_table_offset + layer_table_size + 255) & ~255
            f.seek(weights_offset)
        else:
            weights_offset = 4096
            f.seek(4096)

        def write_tensor(arr_bytes):
            offset = f.tell()
            f.write(arr_bytes)
            remainder = f.tell() % 256
            if remainder != 0:
                f.write(b"\x00" * (256 - remainder))
            return offset

        # 1. Embed tokens
        print("  - Exporting embed_tokens...")
        if is_w4a16:
            q_w, s_w = quantize_and_pack_w4a16(embed_w)
            embed_w_offset = write_tensor(q_w.tobytes())
            embed_s_offset = write_tensor(s_w.tobytes())
        else:
            embed_w_offset = write_tensor(embed_w.astype(np.float16).tobytes())
            embed_s_offset = 0

        # LM Head (if untied)
        if tie_word_embeddings == 0:
            print("  - Exporting untied lm_head...")
            if is_w4a16:
                q_lm, s_lm = quantize_and_pack_w4a16(lm_head_w)
                lm_head_offset = write_tensor(q_lm.tobytes())
            else:
                lm_head_offset = write_tensor(lm_head_w.astype(np.float16).tobytes())
        else:
            lm_head_offset = embed_w_offset

        # 2. Final norm
        final_norm_w = state_dict["model.norm.weight"].numpy().astype(np.float16)
        final_norm_offset = write_tensor(final_norm_w.tobytes())

        # 3. Layers
        layer_offsets = [[0] * LayerTensorIdx.COUNT for _ in range(num_layers)]
        layer_descriptors = []

        for l in range(num_layers):
            prefix = f"model.layers.{l}."
            if l % 7 == 0:
                print(f"  - Exporting Layer {l} .. {min(l + 6, num_layers - 1)}...")

            in_norm = state_dict[prefix + "input_layernorm.weight"].numpy().astype(np.float16)
            norm1_offset = write_tensor(in_norm.tobytes())
            layer_offsets[l][LayerTensorIdx.INPUT_LAYERNORM] = norm1_offset

            q_w = state_dict[prefix + "self_attn.q_proj.weight"].numpy()
            q_b = state_dict[prefix + "self_attn.q_proj.bias"].numpy().astype(np.float16) if (prefix + "self_attn.q_proj.bias") in state_dict else np.zeros((q_w.shape[0],), dtype=np.float16)
            k_w = state_dict[prefix + "self_attn.k_proj.weight"].numpy()
            k_b = state_dict[prefix + "self_attn.k_proj.bias"].numpy().astype(np.float16) if (prefix + "self_attn.k_proj.bias") in state_dict else np.zeros((k_w.shape[0],), dtype=np.float16)
            v_w = state_dict[prefix + "self_attn.v_proj.weight"].numpy()
            v_b = state_dict[prefix + "self_attn.v_proj.bias"].numpy().astype(np.float16) if (prefix + "self_attn.v_proj.bias") in state_dict else np.zeros((v_w.shape[0],), dtype=np.float16)

            q_scale_off = 0
            k_scale_off = 0
            v_scale_off = 0
            if is_w4a16:
                packed_q, scale_q = quantize_and_pack_w4a16(q_w)
                packed_k, scale_k = quantize_and_pack_w4a16(k_w)
                packed_v, scale_v = quantize_and_pack_w4a16(v_w)
                q_off = write_tensor(packed_q.tobytes())
                q_scale_off = write_tensor(scale_q.tobytes())
                k_off = write_tensor(packed_k.tobytes())
                k_scale_off = write_tensor(scale_k.tobytes())
                v_off = write_tensor(packed_v.tobytes())
                v_scale_off = write_tensor(scale_v.tobytes())

                layer_offsets[l][LayerTensorIdx.Q_WEIGHT] = q_off
                layer_offsets[l][LayerTensorIdx.Q_SCALE]  = q_scale_off
                layer_offsets[l][LayerTensorIdx.K_WEIGHT] = k_off
                layer_offsets[l][LayerTensorIdx.K_SCALE]  = k_scale_off
                layer_offsets[l][LayerTensorIdx.V_WEIGHT] = v_off
                layer_offsets[l][LayerTensorIdx.V_SCALE]  = v_scale_off
            else:
                q_off = write_tensor(q_w.astype(np.float16).tobytes())
                k_off = write_tensor(k_w.astype(np.float16).tobytes())
                v_off = write_tensor(v_w.astype(np.float16).tobytes())
                layer_offsets[l][LayerTensorIdx.Q_WEIGHT] = q_off
                layer_offsets[l][LayerTensorIdx.K_WEIGHT] = k_off
                layer_offsets[l][LayerTensorIdx.V_WEIGHT] = v_off

            q_b_off = write_tensor(q_b.tobytes())
            k_b_off = write_tensor(k_b.tobytes())
            v_b_off = write_tensor(v_b.tobytes())
            layer_offsets[l][LayerTensorIdx.Q_BIAS] = q_b_off
            layer_offsets[l][LayerTensorIdx.K_BIAS] = k_b_off
            layer_offsets[l][LayerTensorIdx.V_BIAS] = v_b_off

            o_w = state_dict[prefix + "self_attn.o_proj.weight"].numpy()
            o_scale_off = 0
            if is_w4a16:
                packed_o, scale_o = quantize_and_pack_w4a16(o_w)
                o_off = write_tensor(packed_o.tobytes())
                o_scale_off = write_tensor(scale_o.tobytes())
                layer_offsets[l][LayerTensorIdx.O_WEIGHT] = o_off
                layer_offsets[l][LayerTensorIdx.O_SCALE]  = o_scale_off
            else:
                o_off = write_tensor(o_w.astype(np.float16).tobytes())
                layer_offsets[l][LayerTensorIdx.O_WEIGHT] = o_off

            post_norm = state_dict[prefix + "post_attention_layernorm.weight"].numpy().astype(np.float16)
            norm2_off = write_tensor(post_norm.tobytes())
            layer_offsets[l][LayerTensorIdx.POST_ATTENTION_LAYERNORM] = norm2_off

            gate_w = state_dict[prefix + "mlp.gate_proj.weight"].numpy()
            up_w   = state_dict[prefix + "mlp.up_proj.weight"].numpy()
            gate_up_w = np.empty((2 * intermediate_size, hidden_size), dtype=np.float32)
            gate_up_w[0::2, :] = gate_w
            gate_up_w[1::2, :] = up_w

            gate_up_scale_off = 0
            if is_w4a16:
                packed_gu, scale_gu = quantize_and_pack_w4a16(gate_up_w)
                gu_off = write_tensor(packed_gu.tobytes())
                gate_up_scale_off = write_tensor(scale_gu.tobytes())
                layer_offsets[l][LayerTensorIdx.GATE_UP_WEIGHT] = gu_off
                layer_offsets[l][LayerTensorIdx.GATE_UP_SCALE]  = gate_up_scale_off
            else:
                gu_off = write_tensor(gate_up_w.astype(np.float16).tobytes())
                layer_offsets[l][LayerTensorIdx.GATE_UP_WEIGHT] = gu_off

            down_w = state_dict[prefix + "mlp.down_proj.weight"].numpy()
            down_scale_off = 0
            if is_w4a16:
                packed_dn, scale_dn = quantize_and_pack_w4a16(down_w)
                dn_off = write_tensor(packed_dn.tobytes())
                down_scale_off = write_tensor(scale_dn.tobytes())
                layer_offsets[l][LayerTensorIdx.DOWN_WEIGHT] = dn_off
                layer_offsets[l][LayerTensorIdx.DOWN_SCALE]  = down_scale_off
            else:
                dn_off = write_tensor(down_w.astype(np.float16).tobytes())
                layer_offsets[l][LayerTensorIdx.DOWN_WEIGHT] = dn_off

            if version == 2:
                # Build LayerDescriptor (216 bytes)
                desc = struct.pack(
                    "<2I26Q",
                    LAYER_GQA, 0,
                    norm1_off, norm2_off,
                    q_off, q_b_off, k_b_off, v_b_off, o_off,
                    0, 0, 0, 0, 0, # DeltaNet fields (conv1d_w, conv1d_b, in_proj, decay, out)
                    gu_off, gu_off, dn_off, # mlp_gate, mlp_up, mlp_down
                    q_scale_off, 0, # quant_scales_offset, quant_zeros_offset
                    k_off, v_off, # k_weight, v_weight
                    q_scale_off, k_scale_off, v_scale_off, o_scale_off,
                    gate_up_scale_off, gate_up_scale_off, down_scale_off
                )
                layer_descriptors.append(desc)

        total_file_size = f.tell()

        if version == 2:
            # Write LayerDescriptor table
            f.seek(layer_table_offset)
            for desc in layer_descriptors:
                f.write(desc)

            # Write QwenBinHeaderV2 (4096 bytes)
            f.seek(0)
            header_buf = bytearray(4096)
            struct.pack_into(
                "<12I2f2I11Q",
                header_buf,
                0,
                MAGIC_QWEN,
                2, # version
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
                0, # padding0 (alignment to 64 bytes)
                layer_table_offset,
                embed_w_offset,
                final_norm_offset,
                lm_head_offset,
                weights_offset,
                embed_s_offset, # reserved[0]
                0, 0, 0, 0, 0   # reserved[1..5]
            )
            f.write(header_buf)
        else:
            # Write legacy QWQwenHeader at byte 0
            f.seek(0)
            header_buf = bytearray(4096)
            struct.pack_into(
                "<12I3Q",
                header_buf,
                0,
                MAGIC_QWEN,
                1,
                quant_mode_val,
                vocab_size,
                hidden_size,
                intermediate_size,
                num_layers,
                num_q_heads,
                num_kv_heads,
                head_dim,
                QUANT_GROUP_SIZE,
                0,
                embed_w_offset,
                embed_s_offset,
                final_norm_offset
            )
            offset_pos = struct.calcsize("<12I3Q")
            for l in range(num_layers):
                for t in range(LayerTensorIdx.COUNT):
                    struct.pack_into("<Q", header_buf, offset_pos, layer_offsets[l][t])
                    offset_pos += 8
            f.write(header_buf)

    file_mb = total_file_size / (1024.0 * 1024.0)
    print(f"[Exporter] Successfully exported {output_path} ({file_mb:.2f} MB, version {version}).")

def main():
    parser = argparse.ArgumentParser(description="Export Qwen2.5 to .qwenbin")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", choices=["fp16", "w4a16"], default="w4a16")
    parser.add_argument("--out", default="models/qwen25_1.5b_w4a16.qwenbin")
    parser.add_argument("--tokenizer-out", default="models/tokenizer.bin")
    parser.add_argument("--dummy", action="store_true", help="Generate initialized model offline without downloading weights")
    parser.add_argument("--format-version", type=int, choices=[1, 2], default=2, help="Container format version (1 or 2)")
    args = parser.parse_args()

    export_model(args.model_id, args.mode, args.out, args.tokenizer_out, dummy=args.dummy, version=args.format_version)

if __name__ == "__main__":
    main()
