import os
import struct
import tempfile
import numpy as np
import pytest
from tools.convert_gguf_to_qwenbin import (
    bf16_to_fp16_clamped,
    f32_to_fp16_clamped,
    quantize_and_pack_w4a16,
    GGUFReader,
    convert_gguf_to_qwenbin,
    MAGIC_QWEN,
    LAYER_GQA,
    LAYER_DELTANET,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGUF_METADATA_VALUE_TYPE_UINT32,
    GGUF_METADATA_VALUE_TYPE_FLOAT32,
    GGUF_METADATA_VALUE_TYPE_STRING,
)

def test_bf16_to_fp16_clamping():
    # Construct BF16 representations:
    # 1.0f in float32: 0x3F800000 -> BF16 is 0x3F80
    # 70000.0f in float32: exceeds 65504 -> should clamp to 65504.0 in FP16
    val_1_0 = np.array([0x3F80], dtype=np.uint16)
    fp16_res = bf16_to_fp16_clamped(val_1_0)
    assert np.isclose(fp16_res[0], 1.0, atol=1e-3)

    # Float32 70000.0f = 0x4788B000 -> BF16 = 0x4789
    val_huge = np.array([0x4789], dtype=np.uint16)
    fp16_huge = bf16_to_fp16_clamped(val_huge)
    assert fp16_huge[0] == np.float16(65504.0)

    # Negative large number -> -65504.0
    val_neg_huge = np.array([0xC789], dtype=np.uint16)
    fp16_neg_huge = bf16_to_fp16_clamped(val_neg_huge)
    assert fp16_neg_huge[0] == np.float16(-65504.0)

def test_w4a16_quantize_and_pack():
    np.random.seed(42)
    N = 64
    K = 512
    weights = np.random.randn(N, K).astype(np.float32)

    packed_q, scales = quantize_and_pack_w4a16(weights)
    assert packed_q.shape == (N, K // 8)
    assert scales.shape == (N, K // 128)
    assert packed_q.dtype == np.uint32
    assert scales.dtype == np.float16

def create_synthetic_gguf(path, num_layers=4, hidden_size=256, intermediate_size=512):
    """Generates a valid minimal synthetic GGUF v3 file."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3)) # version 3
        
        # Metadata KV pairs
        metadata_pairs = [
            ("general.architecture", GGUF_METADATA_VALUE_TYPE_STRING, "qwen2"),
            ("qwen2.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, hidden_size),
            ("qwen2.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, intermediate_size),
            ("qwen2.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, num_layers),
            ("qwen2.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, 2),
            ("qwen2.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, 2),
            ("qwen2.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, 128),
            ("qwen2.vocab_size", GGUF_METADATA_VALUE_TYPE_UINT32, 1024),
            ("qwen2.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, 512),
            ("qwen2.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, 1e-6),
            ("qwen2.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, 1000000.0),
        ]
        
        # Define tensors
        tensors = []
        tensors.append(("token_embd.weight", [hidden_size, 1024], GGML_TYPE_F16))
        tensors.append(("output_norm.weight", [hidden_size], GGML_TYPE_F16))

        for l in range(num_layers):
            tensors.append((f"blk.{l}.attn_norm.weight", [hidden_size], GGML_TYPE_F16))
            tensors.append((f"blk.{l}.ffn_norm.weight", [hidden_size], GGML_TYPE_F16))
            tensors.append((f"blk.{l}.ffn_gate.weight", [hidden_size, intermediate_size], GGML_TYPE_F16))
            tensors.append((f"blk.{l}.ffn_up.weight", [hidden_size, intermediate_size], GGML_TYPE_F16))
            tensors.append((f"blk.{l}.ffn_down.weight", [intermediate_size, hidden_size], GGML_TYPE_F16))
            
            # Hybrid 3:1 stacking: layer 0,1,2 DeltaNet, layer 3 GQA
            if l < 3:
                tensors.append((f"blk.{l}.conv1d.weight", [4, hidden_size], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.conv1d.bias", [hidden_size], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.linear_attn.in_proj.weight", [hidden_size, hidden_size * 2], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.linear_attn.decay.weight", [2], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.linear_attn.out_proj.weight", [hidden_size, hidden_size], GGML_TYPE_F16))
            else:
                tensors.append((f"blk.{l}.attn_q.weight", [hidden_size, hidden_size], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.attn_k.weight", [hidden_size, hidden_size], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.attn_v.weight", [hidden_size, hidden_size], GGML_TYPE_F16))
                tensors.append((f"blk.{l}.attn_output.weight", [hidden_size, hidden_size], GGML_TYPE_F16))

        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata_pairs)))

        def write_str(s):
            b = s.encode("utf-8")
            f.write(struct.pack("<Q", len(b)))
            f.write(b)

        for k, vtype, val in metadata_pairs:
            write_str(k)
            f.write(struct.pack("<I", vtype))
            if vtype == GGUF_METADATA_VALUE_TYPE_UINT32:
                f.write(struct.pack("<I", val))
            elif vtype == GGUF_METADATA_VALUE_TYPE_FLOAT32:
                f.write(struct.pack("<f", val))
            elif vtype == GGUF_METADATA_VALUE_TYPE_STRING:
                write_str(val)

        # Calculate offsets
        tensor_data_list = []
        curr_offset = 0
        for name, dims, type_id in tensors:
            num_elem = 1
            for d in dims:
                num_elem *= d
            raw = np.zeros(num_elem, dtype=np.float16).tobytes()
            tensor_data_list.append(raw)

        for i, (name, dims, type_id) in enumerate(tensors):
            write_str(name)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", type_id))
            f.write(struct.pack("<Q", curr_offset))
            curr_offset += len(tensor_data_list[i])

        # Padding to 32 bytes alignment
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\x00" * pad)

        # Write tensor data
        for data in tensor_data_list:
            f.write(data)

def test_convert_gguf_to_qwenbin_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_file = os.path.join(tmpdir, "model.gguf")
        qwenbin_file = os.path.join(tmpdir, "model.qwenbin")
        create_synthetic_gguf(gguf_file, num_layers=4, hidden_size=256, intermediate_size=512)

        convert_gguf_to_qwenbin(gguf_file, qwenbin_file, mode="fp16")

        # Verify .qwenbin v2 container header
        with open(qwenbin_file, "rb") as f:
            header_bytes = f.read(4096)
            unpacked = struct.unpack_from("<12I2f2I11Q", header_bytes, 0)
            magic = unpacked[0]
            version = unpacked[1]
            hidden_size = unpacked[2]
            intermediate_size = unpacked[3]
            num_layers = unpacked[4]
            layer_table_offset = unpacked[16]
            embed_offset = unpacked[17]
            final_norm_offset = unpacked[18]
            weights_offset = unpacked[20]

            assert magic == MAGIC_QWEN
            assert version == 2
            assert hidden_size == 256
            assert intermediate_size == 512
            assert num_layers == 4
            assert layer_table_offset == 4096
            assert embed_offset >= weights_offset
            assert final_norm_offset >= weights_offset

            # Read layer descriptors
            f.seek(layer_table_offset)
            descriptors = []
            for _ in range(num_layers):
                desc_bytes = f.read(216)
                desc = struct.unpack("<2I26Q", desc_bytes)
                descriptors.append(desc)

            assert len(descriptors) == 4
            # Layers 0, 1, 2 should be DeltaNet, Layer 3 should be GQA
            assert descriptors[0][0] == LAYER_DELTANET
            assert descriptors[1][0] == LAYER_DELTANET
            assert descriptors[2][0] == LAYER_DELTANET
            assert descriptors[3][0] == LAYER_GQA

