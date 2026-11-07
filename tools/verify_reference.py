#!/usr/bin/env python3
"""
PyTorch Reference Exporter for Qwen2.5-1.5B
Dumps intermediate activations and greedy tokens into models/golden_ref.bin
"""

import argparse
import struct
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

MAGIC_GOLD = 0x474F4C44  # "GOLD"
PROMPT_TOKENS = [151644, 8948, 198, 2610, 525, 264, 10950, 17847]  # 8 tokens

def main():
    parser = argparse.ArgumentParser(description="Export golden reference tensors for kernel unit tests")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="models/golden_ref.bin")
    parser.add_argument("--dummy", action="store_true", help="Use offline initialized model with seed 42")
    args = parser.parse_args()

    if args.dummy:
        print("[Reference] Using offline initialized model (seed=42)...")
        torch.manual_seed(42)
        config = Qwen2Config(
            vocab_size=151936,
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True
        )
        model = Qwen2ForCausalLM(config).half()
    else:
        try:
            print(f"[Reference] Loading {args.model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[Reference] HuggingFace load failed ({e}). Falling back to offline model...")
            torch.manual_seed(42)
            config = Qwen2Config(
                vocab_size=151936,
                hidden_size=1536,
                intermediate_size=8960,
                num_hidden_layers=28,
                num_attention_heads=12,
                num_key_value_heads=2,
                max_position_embeddings=4096,
                rope_theta=1000000.0,
                rms_norm_eps=1e-6,
                tie_word_embeddings=True
            )
            model = Qwen2ForCausalLM(config).half()

    model.eval()
    device = torch.device("cpu")
    input_ids = torch.tensor([PROMPT_TOKENS], dtype=torch.long, device=device)

    tensors = {}

    # Hook embedding
    embed_tokens = model.model.embed_tokens(input_ids)
    tensors[0] = embed_tokens.detach().cpu().numpy().astype(np.float16)

    # Layer 0 inspection
    layer0 = model.model.layers[0]

    # 1. Input layernorm
    norm_hidden = layer0.input_layernorm(embed_tokens)
    tensors[1] = norm_hidden.detach().cpu().numpy().astype(np.float16)

    # 2-4. Raw Q, K, V projections (before bias)
    q_weight = layer0.self_attn.q_proj.weight
    k_weight = layer0.self_attn.k_proj.weight
    v_weight = layer0.self_attn.v_proj.weight

    q_raw = torch.matmul(norm_hidden, q_weight.t())
    k_raw = torch.matmul(norm_hidden, k_weight.t())
    v_raw = torch.matmul(norm_hidden, v_weight.t())
    tensors[2] = q_raw.detach().cpu().numpy().astype(np.float16)
    tensors[3] = k_raw.detach().cpu().numpy().astype(np.float16)
    tensors[4] = v_raw.detach().cpu().numpy().astype(np.float16)

    # 5-7. Biased + RoPE Q, K, V
    q_biased = q_raw + layer0.self_attn.q_proj.bias
    k_biased = k_raw + layer0.self_attn.k_proj.bias
    v_biased = v_raw + layer0.self_attn.v_proj.bias
    tensors[7] = v_biased.detach().cpu().numpy().astype(np.float16)

    position_ids = torch.arange(8, dtype=torch.long, device=device).unsqueeze(0)
    rotary_emb = model.model.rotary_emb
    cos, sin = rotary_emb(v_biased, position_ids)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(x, cos, sin):
        return (x * cos) + (rotate_half(x) * sin)

    q_reshaped = q_biased.view(1, 8, 12, 128).transpose(1, 2)
    k_reshaped = k_biased.view(1, 8, 2, 128).transpose(1, 2)

    q_rot = apply_rope(q_reshaped, cos, sin)
    k_rot = apply_rope(k_reshaped, cos, sin)

    tensors[5] = q_rot.transpose(1, 2).reshape(1, 8, 1536).detach().cpu().numpy().astype(np.float16)
    tensors[6] = k_rot.transpose(1, 2).reshape(1, 8, 256).detach().cpu().numpy().astype(np.float16)

    # 8. Attention output
    # Layer 0 full attention forward pass
    attn_outputs = layer0.self_attn(
        norm_hidden,
        position_embeddings=(cos, sin),
        attention_mask=None
    )
    attn_out = attn_outputs[0]
    tensors[8] = attn_out.detach().cpu().numpy().astype(np.float16)

    hidden_post_attn = embed_tokens + attn_out

    # 9. Post attention layernorm
    post_norm = layer0.post_attention_layernorm(hidden_post_attn)
    tensors[9] = post_norm.detach().cpu().numpy().astype(np.float16)

    # 10. SwiGLU output
    gate = layer0.mlp.gate_proj(post_norm)
    up   = layer0.mlp.up_proj(post_norm)
    swiglu = layer0.mlp.act_fn(gate) * up
    tensors[10] = swiglu.detach().cpu().numpy().astype(np.float16)

    # 11. MLP output
    down = layer0.mlp.down_proj(swiglu)
    layer0_out = hidden_post_attn + down
    tensors[11] = layer0_out.detach().cpu().numpy().astype(np.float16)

    # Full forward pass to get final logits on the 8 prompt tokens
    with torch.no_grad():
        full_out = model(input_ids)
        final_logits = full_out.logits[0, -1, :].cpu().numpy().astype(np.float32)
        tensors[12] = final_logits

    # Generate 32 greedy continuation tokens
    print("[Reference] Generating 32 greedy continuation tokens...")
    curr_ids = input_ids.clone()
    greedy_tokens = []
    with torch.no_grad():
        for _ in range(32):
            out = model(curr_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1).item()
            greedy_tokens.append(next_token)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token]], device=device)], dim=-1)

    print(f"[Reference] 32 Greedy tokens: {greedy_tokens}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.seek(512)
        tensor_offsets = [0] * 13
        for idx in range(13):
            arr = tensors[idx]
            offset = f.tell()
            f.write(arr.tobytes())
            remainder = f.tell() % 256
            if remainder != 0:
                f.write(b"\x00" * (256 - remainder))
            tensor_offsets[idx] = offset

        f.seek(0)
        header_buf = bytearray(512)
        struct.pack_into(
            "<II8i32i13Q",
            header_buf,
            0,
            MAGIC_GOLD,
            len(PROMPT_TOKENS),
            *PROMPT_TOKENS,
            *greedy_tokens,
            *tensor_offsets
        )
        f.write(header_buf)

    print(f"[Reference] Golden reference saved to {args.out}.")

if __name__ == "__main__":
    main()
