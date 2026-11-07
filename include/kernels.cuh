#pragma once

#include "config.cuh"
#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qwen {

// 1. Embedding lookup
void launch_embed_lookup_fp16(
    const half* embed_table,
    const int32_t* d_token_ids,
    half* out_hidden,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
);

void launch_embed_lookup_w4a16(
    const uint32_t* embed_qweight,
    const half* embed_scales,
    const int32_t* d_token_ids,
    half* out_hidden,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
);

// QKV Projection View descriptor for single-launch fused QKV projections
struct QKVProjectionView {
    const void* q_weight; const half* q_scale; half* d_q;
    const void* k_weight; const half* k_scale; half* d_k;
    const void* v_weight; const half* v_scale; half* d_v;
};

// 2. RMSNorm & Quantization
void launch_rmsnorm_fp16(
    const half* input,
    const half* gamma,
    half* output,
    int num_tokens,
    cudaStream_t stream
);

void launch_rmsnorm_q8_1(
    const half* input,
    const half* gamma,
    half* output_fp16_opt,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8,
    cudaStream_t stream
);

void launch_quantize_fp16_to_q8_1(
    const half* input_x,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8,
    int num_elements,
    cudaStream_t stream
);

// 3. Vectorized FP16 GEMV (decode)
void launch_gemv_fp16(
    const half* weight,
    const half* input_x,
    const half* residual,
    half* output_y,
    int out_features,
    int in_features,
    cudaStream_t stream
);

void launch_gemv_qkv_fp16(
    const QKVProjectionView& view,
    const half* input_x,
    cudaStream_t stream
);

void launch_gemv_dual_gate_up_swiglu_fp16(
    const half* gate_up_weight,
    const half* input_x,
    half* output_mlp,
    cudaStream_t stream
);

// 4. Fused W4A16 & W4A8 GEMV (decode)
void launch_gemv_w4a16(
    const uint32_t* qweight,
    const half* scales,
    const half* input_x,
    const half* residual,
    half* output_y,
    int out_features,
    int in_features,
    cudaStream_t stream
);

void launch_gemv_w4a8(
    const uint32_t* qweight,
    const half* scales,
    const int8_t* input_q8,
    const float* input_scale,
    const int32_t* input_sum8,
    const half* residual,
    half* output_y,
    int out_features,
    int in_features,
    cudaStream_t stream
);

void launch_gemv_qkv_w4a8(
    const QKVProjectionView& view,
    const int8_t* input_q8,
    const float* input_scale,
    const int32_t* input_sum8,
    cudaStream_t stream
);

void launch_gemv_dual_gate_up_swiglu_w4a16(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const half* input_x,
    half* output_mlp,
    cudaStream_t stream
);

void launch_gemv_dual_gate_up_swiglu_w4a8(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const int8_t* input_q8,
    const float* input_scale,
    const int32_t* input_sum8,
    half* output_mlp,
    cudaStream_t stream
);

// 5. Tiled Prefill GEMM (M <= 256)
void launch_gemm_prefill_fp16(
    const half* weight,
    const half* input_x,
    const half* residual,
    half* output_y,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

void launch_gemm_prefill_qkv_fp16(
    const QKVProjectionView& view,
    const half* input_x,
    int M,
    cudaStream_t stream
);

void launch_gemm_prefill_dual_gate_up_swiglu_fp16(
    const half* gate_up_weight,
    const half* input_x,
    half* output_mlp,
    int M,
    cudaStream_t stream
);

void launch_gemm_prefill_w4a16(
    const uint32_t* qweight,
    const half* scales,
    const half* input_x,
    const half* residual,
    half* output_y,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

void launch_gemm_prefill_qkv_w4a16(
    const QKVProjectionView& view,
    const half* input_x,
    int M,
    cudaStream_t stream
);

void launch_gemm_prefill_dual_gate_up_swiglu_w4a16(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const half* input_x,
    half* output_mlp,
    int M,
    cudaStream_t stream
);

// 6. QKV Bias + NeoX Split-Half RoPE + KV Cache write
void launch_qkv_bias_rope_kv_append(
    half* q,
    half* k,
    half* v,
    const half* q_bias,
    const half* k_bias,
    const half* v_bias,
    half* layer_k_cache,
    half* layer_v_cache,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
);

// 7. GQA FlashDecoding & Causal Prefill Attention
void launch_attention_gqa_flash_decode(
    const half* q,
    const half* layer_k_cache,
    const half* layer_v_cache,
    float* splitk_partial_out,
    float* splitk_partial_lse,
    half* attn_out,
    const DeviceControlBlock* d_control,
    cudaStream_t stream,
    int8_t* out_q8 = nullptr,
    float* out_scale = nullptr,
    int32_t* out_sum8 = nullptr
);

void launch_attention_gqa_prefill(
    const half* q,
    const half* layer_k_cache,
    const half* layer_v_cache,
    half* attn_out,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
);

// 8. Device Argmax / Gumbel-Max Sampling & Sequence Advance
void launch_sample_and_advance(
    const half* logits,
    float* sample_scratch_val,
    int32_t* sample_scratch_idx,
    DeviceControlBlock* d_control,
    int32_t* d_mapped_pinned_token,
    cudaStream_t stream
);

} // namespace qwen
