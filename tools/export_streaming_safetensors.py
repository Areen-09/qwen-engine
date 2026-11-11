#!/usr/bin/env python3
"""
tools/export_streaming_safetensors.py: Out-of-core shard-by-shard SafeTensor streaming exporter
for Qwen 3.8-27B dual Tesla T4 TP=2 deployment strictly within Kaggle's 20.0 GiB disk ceiling.
"""

import os
import sys
import gc
import json
import struct
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "python"))

import numpy as np

from qwen_engine.exporter import (
    MAGIC_QWEN,
    QUANT_GROUP_SIZE,
    LAYER_GQA,
    LAYER_DELTANET,
    TOKEN_ENDOFTEXT_QWEN38,
    bf16_to_fp16_clamped,
    f32_to_fp16_clamped,
    quantize_and_pack_w4a16,
    pack_layer_descriptor,
    pack_header_v2,
    export_tokenizer,
    verify_qwenbin_v2,
)


def align256(off: int) -> int:
    rem = off % 256
    return off + (256 - rem) if rem != 0 else off


def check_disk_guard(target_dir: str, required_bytes: int, safety_margin_mb: int = 500) -> None:
    """Enforces dynamic watermark invariant: free_disk >= required_bytes + safety_margin."""
    os.makedirs(target_dir, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    margin_bytes = safety_margin_mb * 1024 * 1024
    if free < required_bytes + margin_bytes:
        free_mb = free / (1024 * 1024)
        req_mb = (required_bytes + margin_bytes) / (1024 * 1024)
        raise RuntimeError(
            f"[Disk Watermark Guard] Insufficient disk space in {target_dir}. "
            f"Available: {free_mb:.2f} MB, Required: {req_mb:.2f} MB."
        )


def compute_container_layout(
    num_layers: int = 64,
    hidden_size: int = 5120,
    intermediate_size: int = 17408,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    vocab_size: int = 248320,
    device_shards: int = 2,
) -> Dict[str, Any]:
    """Calculates deterministic byte offsets for every tensor in a rank's .qwenbin file."""
    rank_q_heads = num_q_heads // device_shards
    rank_kv_heads = num_kv_heads // device_shards
    rank_intermediate = intermediate_size // device_shards
    rank_vocab = vocab_size // device_shards

    layer_table_offset = 4096
    layer_table_size = num_layers * 216
    weights_offset = align256(layer_table_offset + layer_table_size)

    cur_off = weights_offset

    # 1. embed_tokens: full [vocab_size, hidden_size] W4A16 group-128
    embed_w_off = cur_off
    cur_off = align256(cur_off + vocab_size * (hidden_size // 8))

    embed_s_off = cur_off
    cur_off = align256(cur_off + vocab_size * (hidden_size // 128) * 2)

    # 2. lm_head: column-sharded [rank_vocab, hidden_size] W4A16 group-128
    lm_head_off = cur_off
    cur_off = align256(cur_off + rank_vocab * (hidden_size // 8))

    lm_head_s_off = cur_off
    cur_off = align256(cur_off + rank_vocab * (hidden_size // 128) * 2)

    # 3. final_norm: [hidden_size] FP16
    final_norm_off = cur_off
    cur_off = align256(cur_off + hidden_size * 2)

    # 4. Layers
    layer_offsets = []
    layer_types = [LAYER_GQA if (l % 4 == 3) else LAYER_DELTANET for l in range(num_layers)]

    for l in range(num_layers):
        l_type = layer_types[l]
        offs = {"type": l_type}

        offs["norm1"] = cur_off
        cur_off = align256(cur_off + hidden_size * 2)

        offs["norm2"] = cur_off
        cur_off = align256(cur_off + hidden_size * 2)

        if l_type == LAYER_GQA:
            q_rows = rank_q_heads * head_dim
            offs["q"] = cur_off
            cur_off = align256(cur_off + q_rows * (hidden_size // 8))
            offs["q_scale"] = cur_off
            cur_off = align256(cur_off + q_rows * (hidden_size // 128) * 2)

            k_rows = rank_kv_heads * head_dim
            offs["k"] = cur_off
            cur_off = align256(cur_off + k_rows * (hidden_size // 8))
            offs["k_scale"] = cur_off
            cur_off = align256(cur_off + k_rows * (hidden_size // 128) * 2)

            offs["v"] = cur_off
            cur_off = align256(cur_off + k_rows * (hidden_size // 8))
            offs["v_scale"] = cur_off
            cur_off = align256(cur_off + k_rows * (hidden_size // 128) * 2)

            o_k = rank_q_heads * head_dim
            offs["o"] = cur_off
            cur_off = align256(cur_off + hidden_size * (o_k // 8))
            offs["o_scale"] = cur_off
            cur_off = align256(cur_off + hidden_size * (o_k // 128) * 2)

            offs["q_b"] = cur_off
            cur_off = align256(cur_off + q_rows * 2)
            offs["k_b"] = cur_off
            cur_off = align256(cur_off + k_rows * 2)
            offs["v_b"] = cur_off
            cur_off = align256(cur_off + k_rows * 2)

            # Unused in GQA
            offs["conv1d_w"] = offs["conv1d_b"] = offs["in_proj"] = offs["decay"] = offs["delta_out"] = 0
        else:
            delta_channels = rank_q_heads * head_dim
            offs["conv1d_w"] = cur_off
            cur_off = align256(cur_off + 4 * delta_channels * 2)

            offs["conv1d_b"] = cur_off
            cur_off = align256(cur_off + delta_channels * 2)

            offs["in_proj"] = cur_off
            cur_off = align256(cur_off + delta_channels * hidden_size * 2)

            offs["decay"] = cur_off
            cur_off = align256(cur_off + rank_q_heads * 2)

            offs["delta_out"] = cur_off
            cur_off = align256(cur_off + hidden_size * delta_channels * 2)

            # Unused in DeltaNet
            offs["q"] = offs["q_scale"] = offs["q_b"] = 0
            offs["k"] = offs["k_scale"] = offs["k_b"] = 0
            offs["v"] = offs["v_scale"] = offs["v_b"] = 0
            offs["o"] = offs["o_scale"] = 0

        # MLP SwiGLU: gate + up concatenated
        offs["gu"] = cur_off
        cur_off = align256(cur_off + 2 * rank_intermediate * (hidden_size // 8))
        offs["gu_scale"] = cur_off
        cur_off = align256(cur_off + 2 * rank_intermediate * (hidden_size // 128) * 2)

        offs["dn"] = cur_off
        cur_off = align256(cur_off + hidden_size * (rank_intermediate // 8))
        offs["dn_scale"] = cur_off
        cur_off = align256(cur_off + hidden_size * (rank_intermediate // 128) * 2)

        layer_offsets.append(offs)

    return {
        "layer_table_offset": layer_table_offset,
        "weights_offset": weights_offset,
        "embed_w_off": embed_w_off,
        "embed_s_off": embed_s_off,
        "lm_head_off": lm_head_off,
        "lm_head_s_off": lm_head_s_off,
        "final_norm_off": final_norm_off,
        "layer_offsets": layer_offsets,
        "total_file_size": cur_off,
    }


def export_streaming_safetensors(
    model_id: str,
    output_dir: str,
    device_shards: int = 2,
    synthetic_prototype: bool = False,
    test_toy: bool = False,
    hf_token: str = None,
) -> Tuple[str, str]:
    """
    Streams and converts Qwen 3.8-27B checkpoint into symmetric TP=2 dual-rank containers.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if test_toy:
        num_layers = 4
        hidden_size = 512
        intermediate_size = 1024
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 128
        rotary_dim = 64
        vocab_size = 1024
        max_seq_len = 128
        rms_norm_eps = 1e-6
        rope_theta = 1000000.0
    else:
        num_layers = 64
        hidden_size = 5120
        intermediate_size = 17408
        num_q_heads = 32
        num_kv_heads = 4
        head_dim = 256
        rotary_dim = 64
        vocab_size = 248320
        max_seq_len = 4096
        rms_norm_eps = 1e-6
        rope_theta = 1000000.0

    rank_q_heads = num_q_heads // device_shards
    rank_kv_heads = num_kv_heads // device_shards
    rank_intermediate = intermediate_size // device_shards
    rank_vocab = vocab_size // device_shards

    print("=" * 80)
    print(f"Qwen 3.8-27B SafeTensor Streaming Exporter (TP={device_shards})")
    print(f"  Target: {model_id}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Architecture: {num_layers} layers (48 DeltaNet, 16 GQA in 3:1 pattern)")
    print(f"  Dimensions: H={hidden_size}, I={intermediate_size} (Rank I={rank_intermediate}), "
          f"Q_heads={num_q_heads} (Rank Q={rank_q_heads}), KV_heads={num_kv_heads} (Rank KV={rank_kv_heads}), "
          f"head_dim={head_dim}, rotary_dim={rotary_dim}, vocab={vocab_size} (Rank vocab={rank_vocab})")
    print("=" * 80)

    rank0_filename = "qwen3.8-27b-tp2-rank0-w4a16.qwenbin"
    rank1_filename = "qwen3.8-27b-tp2-rank1-w4a16.qwenbin"
    rank0_path = str(out_path / rank0_filename)
    rank1_path = str(out_path / rank1_filename)

    layout = compute_container_layout(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        device_shards=device_shards,
    )
    total_file_size = layout["total_file_size"]
    print(f"Pre-calculated Container Size per GPU: {total_file_size / (1024**3):.3f} GB ({total_file_size / (1024**2):.2f} MB)")

    if synthetic_prototype or test_toy:
        print("\n[Synthetic Prototype Mode Active]")
        print("Generating bit-exact dual-rank containers with pre-calculated layout table...")

        if not test_toy:
            check_disk_guard(str(out_path), total_file_size * 2, safety_margin_mb=500)

        for rank in range(device_shards):
            rank_path = rank0_path if rank == 0 else rank1_path
            print(f"[Rank {rank} / {device_shards - 1}] Pre-allocating container: {rank_path}...")
            with open(rank_path, "wb") as f:
                f.seek(total_file_size - 1)
                f.write(b"\x00")

                # Fill with fast zero arrays
                f.seek(layout["embed_w_off"])
                pq_emb, ps_emb = quantize_and_pack_w4a16(np.zeros((vocab_size, hidden_size), dtype=np.float32))
                f.write(pq_emb.tobytes())
                f.seek(layout["embed_s_off"])
                f.write(ps_emb.tobytes())

                f.seek(layout["lm_head_off"])
                pq_lm, ps_lm = quantize_and_pack_w4a16(np.zeros((rank_vocab, hidden_size), dtype=np.float32))
                f.write(pq_lm.tobytes())
                f.seek(layout["lm_head_s_off"])
                f.write(ps_lm.tobytes())

                f.seek(layout["final_norm_off"])
                f.write(np.ones((hidden_size,), dtype=np.float16).tobytes())

                # Write layer descriptors
                f.seek(layout["layer_table_offset"])
                for l in range(num_layers):
                    lo = layout["layer_offsets"][l]
                    desc = pack_layer_descriptor(
                        l_type=lo["type"],
                        norm1_off=lo["norm1"],
                        norm2_off=lo["norm2"],
                        q_off=lo["q"],
                        q_b_off=lo["q_b"],
                        k_b_off=lo["k_b"],
                        v_b_off=lo["v_b"],
                        o_off=lo["o"],
                        conv1d_w_off=lo["conv1d_w"],
                        conv1d_b_off=lo["conv1d_b"],
                        in_proj_off=lo["in_proj"],
                        decay_off=lo["decay"],
                        delta_out_off=lo["delta_out"],
                        gu_off=lo["gu"],
                        dn_off=lo["dn"],
                        q_scale_off=lo["q_scale"],
                        k_off=lo["k"],
                        v_off=lo["v"],
                        k_scale_off=lo["k_scale"],
                        v_scale_off=lo["v_scale"],
                        o_scale_off=lo["o_scale"],
                        gu_scale_off=lo["gu_scale"],
                        dn_scale_off=lo["dn_scale"],
                    )
                    f.write(desc)

                # Write Header
                f.seek(0)
                header = pack_header_v2(
                    hidden_size=hidden_size,
                    intermediate_size=rank_intermediate,
                    num_layers=num_layers,
                    num_q_heads=rank_q_heads,
                    num_kv_heads=rank_kv_heads,
                    head_dim=head_dim,
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    quant_mode=1,
                    tie_word_embeddings=0,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    layer_table_offset=layout["layer_table_offset"],
                    embed_w_offset=layout["embed_w_off"],
                    final_norm_offset=layout["final_norm_off"],
                    lm_head_offset=layout["lm_head_off"],
                    weights_offset=layout["weights_offset"],
                    embed_s_offset=layout["embed_s_off"],
                    lm_head_s_offset=layout["lm_head_s_off"],
                )
                f.write(header)

            stats = verify_qwenbin_v2(rank_path)
            print(f"  Container Size: {stats['file_size_mb']:.2f} MB")

        # Tokenizer
        tok_path = str(out_path / "tokenizer.bin")
        print(f"\n[Tokenizer] Emitting tokenizer.bin with eos_id={TOKEN_ENDOFTEXT_QWEN38}...")
        with open(tok_path, "wb") as f:
            header = struct.pack("<IIII", 0x51544F4B, vocab_size, 0, TOKEN_ENDOFTEXT_QWEN38)
            f.write(header)
            for tid in range(vocab_size):
                tok_str = f"<token_{tid}>".encode("utf-8")
                f.write(struct.pack("<H", len(tok_str)))
                f.write(tok_str)
                f.write(struct.pack("<i", tid))
        print(f"  Tokenizer Size: {os.path.getsize(tok_path) / (1024 * 1024):.2f} MB")
        return rank0_path, rank1_path

    # =========================================================================
    # LIVE SAFETENSOR SHARD-BY-SHARD OUT-OF-CORE STREAMING CONVERTER
    # =========================================================================
    from huggingface_hub import hf_hub_download, HfApi
    from transformers import AutoTokenizer
    import safetensors.torch

    print(f"\n[Live Exporter] Accessing Hugging Face Hub for {model_id}...")
    api = HfApi(token=hf_token)

    # 1. Export Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    tok_path = str(out_path / "tokenizer.bin")
    export_tokenizer(tok, tok_path, eos_id=TOKEN_ENDOFTEXT_QWEN38)

    # 2. Pre-allocate rank0 and rank1 sparse files
    print(f"\n[Pre-allocation] Initializing destination containers ({total_file_size / (1024**3):.2f} GB each)...")
    check_disk_guard(str(out_path), total_file_size * 2 + 5000 * 1024 * 1024, safety_margin_mb=500)

    for r_path in (rank0_path, rank1_path):
        with open(r_path, "wb") as f:
            f.seek(total_file_size - 1)
            f.write(b"\x00")

    # Read SafeTensor index
    index_file = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", token=hf_token)
    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    shard_filenames = sorted(list(set(weight_map.values())))
    print(f"Identified {len(shard_filenames)} remote safetensor shards.")

    f0 = open(rank0_path, "r+b")
    f1 = open(rank1_path, "r+b")

    try:
        staging_dir = out_path / "staging_shards"
        staging_dir.mkdir(parents=True, exist_ok=True)

        for shard_idx, shard_fname in enumerate(shard_filenames):
            print(f"\n{'='*75}")
            print(f"[{shard_idx + 1}/{len(shard_filenames)}] Streaming Remote Shard: {shard_fname}")
            print(f"{'='*75}")

            # Dynamic disk guard check before downloading shard
            check_disk_guard(str(staging_dir), 4500 * 1024 * 1024, safety_margin_mb=500)

            downloaded = hf_hub_download(
                repo_id=model_id,
                filename=shard_fname,
                local_dir=str(staging_dir),
                token=hf_token
            )

            state_dict = safetensors.torch.load_file(downloaded)

            for key, tensor in state_dict.items():
                arr = tensor.float().numpy()

                if key == "model.embed_tokens.weight":
                    print(f"  - Quantizing and packing embed_tokens.weight ({arr.shape})...")
                    chunk_rows = 16384
                    for r_start in range(0, vocab_size, chunk_rows):
                        r_end = min(r_start + chunk_rows, vocab_size)
                        pq, ps = quantize_and_pack_w4a16(arr[r_start:r_end, :])

                        w_byte_off = layout["embed_w_off"] + r_start * (hidden_size // 8)
                        s_byte_off = layout["embed_s_off"] + r_start * (hidden_size // 128) * 2

                        f0.seek(w_byte_off); f0.write(pq.tobytes())
                        f0.seek(s_byte_off); f0.write(ps.tobytes())
                        f1.seek(w_byte_off); f1.write(pq.tobytes())
                        f1.seek(s_byte_off); f1.write(ps.tobytes())
                        del pq, ps

                elif key == "lm_head.weight":
                    print(f"  - Slicing and quantizing column-sharded lm_head.weight ({arr.shape})...")
                    chunk_rows = 16384
                    # Rank 0: rows [0, rank_vocab)
                    for r_start in range(0, rank_vocab, chunk_rows):
                        r_end = min(r_start + chunk_rows, rank_vocab)
                        pq, ps = quantize_and_pack_w4a16(arr[r_start:r_end, :])
                        f0.seek(layout["lm_head_off"] + r_start * (hidden_size // 8))
                        f0.write(pq.tobytes())
                        f0.seek(layout["lm_head_s_off"] + r_start * (hidden_size // 128) * 2)
                        f0.write(ps.tobytes())
                        del pq, ps

                    # Rank 1: rows [rank_vocab, vocab_size)
                    for r_start in range(0, rank_vocab, chunk_rows):
                        r_end = min(r_start + chunk_rows, rank_vocab)
                        pq, ps = quantize_and_pack_w4a16(arr[rank_vocab + r_start : rank_vocab + r_end, :])
                        f1.seek(layout["lm_head_off"] + r_start * (hidden_size // 8))
                        f1.write(pq.tobytes())
                        f1.seek(layout["lm_head_s_off"] + r_start * (hidden_size // 128) * 2)
                        f1.write(ps.tobytes())
                        del pq, ps

                elif key == "model.norm.weight":
                    print(f"  - Storing final norm weight ({arr.shape})...")
                    norm_fp16 = f32_to_fp16_clamped(arr)
                    f0.seek(layout["final_norm_off"]); f0.write(norm_fp16.tobytes())
                    f1.seek(layout["final_norm_off"]); f1.write(norm_fp16.tobytes())

                elif key.startswith("model.layers."):
                    parts = key.split(".")
                    l = int(parts[2])
                    sub = ".".join(parts[3:])
                    offs = layout["layer_offsets"][l]

                    if sub == "input_layernorm.weight":
                        norm1_fp16 = f32_to_fp16_clamped(arr)
                        f0.seek(offs["norm1"]); f0.write(norm1_fp16.tobytes())
                        f1.seek(offs["norm1"]); f1.write(norm1_fp16.tobytes())

                    elif sub == "post_attention_layernorm.weight":
                        norm2_fp16 = f32_to_fp16_clamped(arr)
                        f0.seek(offs["norm2"]); f0.write(norm2_fp16.tobytes())
                        f1.seek(offs["norm2"]); f1.write(norm2_fp16.tobytes())

                    elif sub == "self_attn.q_proj.weight":
                        # Rank 0: [0, 4096), Rank 1: [4096, 8192)
                        pq0, ps0 = quantize_and_pack_w4a16(arr[0 : rank_q_heads * head_dim, :])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[rank_q_heads * head_dim : 2 * rank_q_heads * head_dim, :])
                        f0.seek(offs["q"]); f0.write(pq0.tobytes())
                        f0.seek(offs["q_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["q"]); f1.write(pq1.tobytes())
                        f1.seek(offs["q_scale"]); f1.write(ps1.tobytes())

                    elif sub == "self_attn.k_proj.weight":
                        pq0, ps0 = quantize_and_pack_w4a16(arr[0 : rank_kv_heads * head_dim, :])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[rank_kv_heads * head_dim : 2 * rank_kv_heads * head_dim, :])
                        f0.seek(offs["k"]); f0.write(pq0.tobytes())
                        f0.seek(offs["k_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["k"]); f1.write(pq1.tobytes())
                        f1.seek(offs["k_scale"]); f1.write(ps1.tobytes())

                    elif sub == "self_attn.v_proj.weight":
                        pq0, ps0 = quantize_and_pack_w4a16(arr[0 : rank_kv_heads * head_dim, :])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[rank_kv_heads * head_dim : 2 * rank_kv_heads * head_dim, :])
                        f0.seek(offs["v"]); f0.write(pq0.tobytes())
                        f0.seek(offs["v_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["v"]); f1.write(pq1.tobytes())
                        f1.seek(offs["v_scale"]); f1.write(ps1.tobytes())

                    elif sub == "self_attn.o_proj.weight":
                        # Row-partitioned: Rank 0 cols [0, 4096), Rank 1 cols [4096, 8192)
                        pq0, ps0 = quantize_and_pack_w4a16(arr[:, 0 : rank_q_heads * head_dim])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[:, rank_q_heads * head_dim : 2 * rank_q_heads * head_dim])
                        f0.seek(offs["o"]); f0.write(pq0.tobytes())
                        f0.seek(offs["o_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["o"]); f1.write(pq1.tobytes())
                        f1.seek(offs["o_scale"]); f1.write(ps1.tobytes())

                    elif sub == "self_attn.q_proj.bias":
                        b0 = f32_to_fp16_clamped(arr[0 : rank_q_heads * head_dim])
                        b1 = f32_to_fp16_clamped(arr[rank_q_heads * head_dim : 2 * rank_q_heads * head_dim])
                        f0.seek(offs["q_b"]); f0.write(b0.tobytes())
                        f1.seek(offs["q_b"]); f1.write(b1.tobytes())

                    elif sub == "self_attn.k_proj.bias":
                        b0 = f32_to_fp16_clamped(arr[0 : rank_kv_heads * head_dim])
                        b1 = f32_to_fp16_clamped(arr[rank_kv_heads * head_dim : 2 * rank_kv_heads * head_dim])
                        f0.seek(offs["k_b"]); f0.write(b0.tobytes())
                        f1.seek(offs["k_b"]); f1.write(b1.tobytes())

                    elif sub == "self_attn.v_proj.bias":
                        b0 = f32_to_fp16_clamped(arr[0 : rank_kv_heads * head_dim])
                        b1 = f32_to_fp16_clamped(arr[rank_kv_heads * head_dim : 2 * rank_kv_heads * head_dim])
                        f0.seek(offs["v_b"]); f0.write(b0.tobytes())
                        f1.seek(offs["v_b"]); f1.write(b1.tobytes())

                    elif sub == "conv1d.weight":
                        c_w = arr.reshape(4, -1)
                        w0 = f32_to_fp16_clamped(c_w[:, 0 : rank_q_heads * head_dim])
                        w1 = f32_to_fp16_clamped(c_w[:, rank_q_heads * head_dim : 2 * rank_q_heads * head_dim])
                        f0.seek(offs["conv1d_w"]); f0.write(w0.tobytes())
                        f1.seek(offs["conv1d_w"]); f1.write(w1.tobytes())

                    elif sub == "conv1d.bias":
                        b0 = f32_to_fp16_clamped(arr[0 : rank_q_heads * head_dim])
                        b1 = f32_to_fp16_clamped(arr[rank_q_heads * head_dim : 2 * rank_q_heads * head_dim])
                        f0.seek(offs["conv1d_b"]); f0.write(b0.tobytes())
                        f1.seek(offs["conv1d_b"]); f1.write(b1.tobytes())

                    elif sub == "linear_attn.in_proj.weight":
                        w0 = f32_to_fp16_clamped(arr[0 : rank_q_heads * head_dim, :])
                        w1 = f32_to_fp16_clamped(arr[rank_q_heads * head_dim : 2 * rank_q_heads * head_dim, :])
                        f0.seek(offs["in_proj"]); f0.write(w0.tobytes())
                        f1.seek(offs["in_proj"]); f1.write(w1.tobytes())

                    elif sub == "linear_attn.decay.weight":
                        d0 = f32_to_fp16_clamped(arr[0 : rank_q_heads])
                        d1 = f32_to_fp16_clamped(arr[rank_q_heads : 2 * rank_q_heads])
                        f0.seek(offs["decay"]); f0.write(d0.tobytes())
                        f1.seek(offs["decay"]); f1.write(d1.tobytes())

                    elif sub == "linear_attn.out_proj.weight":
                        w0 = f32_to_fp16_clamped(arr[:, 0 : rank_q_heads * head_dim])
                        w1 = f32_to_fp16_clamped(arr[:, rank_q_heads * head_dim : 2 * rank_q_heads * head_dim])
                        f0.seek(offs["delta_out"]); f0.write(w0.tobytes())
                        f1.seek(offs["delta_out"]); f1.write(w1.tobytes())

                    elif sub == "mlp.gate_proj.weight":
                        pq0, ps0 = quantize_and_pack_w4a16(arr[0 : rank_intermediate, :])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[rank_intermediate : 2 * rank_intermediate, :])
                        f0.seek(offs["gu"]); f0.write(pq0.tobytes())
                        f0.seek(offs["gu_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["gu"]); f1.write(pq1.tobytes())
                        f1.seek(offs["gu_scale"]); f1.write(ps1.tobytes())

                    elif sub == "mlp.up_proj.weight":
                        pq0, ps0 = quantize_and_pack_w4a16(arr[0 : rank_intermediate, :])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[rank_intermediate : 2 * rank_intermediate, :])
                        gu_w_up_off = offs["gu"] + rank_intermediate * (hidden_size // 8)
                        gu_s_up_off = offs["gu_scale"] + rank_intermediate * (hidden_size // 128) * 2
                        f0.seek(gu_w_up_off); f0.write(pq0.tobytes())
                        f0.seek(gu_s_up_off); f0.write(ps0.tobytes())
                        f1.seek(gu_w_up_off); f1.write(pq1.tobytes())
                        f1.seek(gu_s_up_off); f1.write(ps1.tobytes())

                    elif sub == "mlp.down_proj.weight":
                        pq0, ps0 = quantize_and_pack_w4a16(arr[:, 0 : rank_intermediate])
                        pq1, ps1 = quantize_and_pack_w4a16(arr[:, rank_intermediate : 2 * rank_intermediate])
                        f0.seek(offs["dn"]); f0.write(pq0.tobytes())
                        f0.seek(offs["dn_scale"]); f0.write(ps0.tobytes())
                        f1.seek(offs["dn"]); f1.write(pq1.tobytes())
                        f1.seek(offs["dn_scale"]); f1.write(ps1.tobytes())

            # Delete downloaded shard immediately to preserve disk space
            del state_dict
            gc.collect()
            if os.path.exists(downloaded):
                os.remove(downloaded)
                print(f"  ✓ Cleaned local staging shard {shard_fname} to reclaim disk space.")

        # Write layer descriptors and headers for both rank files
        print("\n[Writing Metadata] Emitting 4096-byte headers and layer descriptor tables...")
        for rank, f_handle in enumerate([f0, f1]):
            f_handle.seek(layout["layer_table_offset"])
            for l in range(num_layers):
                lo = layout["layer_offsets"][l]
                desc = pack_layer_descriptor(
                    l_type=lo["type"],
                    norm1_off=lo["norm1"],
                    norm2_off=lo["norm2"],
                    q_off=lo["q"],
                    q_b_off=lo["q_b"],
                    k_b_off=lo["k_b"],
                    v_b_off=lo["v_b"],
                    o_off=lo["o"],
                    conv1d_w_off=lo["conv1d_w"],
                    conv1d_b_off=lo["conv1d_b"],
                    in_proj_off=lo["in_proj"],
                    decay_off=lo["decay"],
                    delta_out_off=lo["delta_out"],
                    gu_off=lo["gu"],
                    dn_off=lo["dn"],
                    q_scale_off=lo["q_scale"],
                    k_off=lo["k"],
                    v_off=lo["v"],
                    k_scale_off=lo["k_scale"],
                    v_scale_off=lo["v_scale"],
                    o_scale_off=lo["o_scale"],
                    gu_scale_off=lo["gu_scale"],
                    dn_scale_off=lo["dn_scale"],
                )
                f_handle.write(desc)

            f_handle.seek(0)
            header = pack_header_v2(
                hidden_size=hidden_size,
                intermediate_size=rank_intermediate,
                num_layers=num_layers,
                num_q_heads=rank_q_heads,
                num_kv_heads=rank_kv_heads,
                head_dim=head_dim,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                quant_mode=1,
                tie_word_embeddings=0,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                layer_table_offset=layout["layer_table_offset"],
                embed_w_offset=layout["embed_w_off"],
                final_norm_offset=layout["final_norm_off"],
                lm_head_offset=layout["lm_head_off"],
                weights_offset=layout["weights_offset"],
                embed_s_offset=layout["embed_s_off"],
                lm_head_s_offset=layout["lm_head_s_off"],
            )
            f_handle.write(header)
    finally:
        f0.close()
        f1.close()

    print(f"\n[Integrity Verification] Verifying generated containers...")
    verify_qwenbin_v2(rank0_path)
    verify_qwenbin_v2(rank1_path)
    print("\n✓ Live SafeTensor Streaming Conversion Completed Successfully!")

    return rank0_path, rank1_path


def main():
    parser = argparse.ArgumentParser(description="Out-of-Core Shard Streaming SafeTensor Exporter for Qwen 3.8-27B")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3.8-27B", help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--device-shards", type=int, default=2, help="Number of tensor parallel shards (default 2)")
    parser.add_argument("--synthetic-prototype", action="store_true", help="Generate architectural prototype container")
    parser.add_argument("--test-toy", action="store_true", help="Generate lightweight toy prototype for local testing (<1 MB)")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face user access token")

    args = parser.parse_args()
    export_streaming_safetensors(
        model_id=args.model_id,
        output_dir=args.output_dir,
        device_shards=args.device_shards,
        synthetic_prototype=args.synthetic_prototype,
        test_toy=args.test_toy,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
