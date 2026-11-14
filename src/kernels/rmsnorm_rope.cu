#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "quant_primitives.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// Embedding Lookup Kernel (W4A16)
// ----------------------------------------------------------------------------

__global__ void embed_lookup_w4a16_kernel(
    const uint32_t* __restrict__ embed_qweight,
    const half* __restrict__ embed_scales,
    const int32_t* __restrict__ d_token_ids,
    half* __restrict__ out_hidden,
    const DeviceControlBlock* __restrict__ d_control,
    int num_tokens,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int32_t token_id = 0;
    if (d_token_ids != nullptr) {
        token_id = d_token_ids[token_idx];
    } else if (d_control != nullptr) {
        token_id = d_control->token_id;
    }
    if (token_id < 0 || token_id >= VOCAB_SIZE) {
        token_id = 0;
    }

    int num_u4 = hidden_size / 8;
    int num_groups = hidden_size / QUANT_GROUP_SIZE;
    half* dst = out_hidden + static_cast<size_t>(token_idx) * hidden_size;

    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        int group_idx = i / (QUANT_GROUP_SIZE / 8);
        half scale = embed_scales[static_cast<size_t>(token_id) * num_groups + group_idx];
        uint32_t packed = embed_qweight[static_cast<size_t>(token_id) * num_u4 + i];

        half2 h2[4];
        unpack_w4_to_half2_quad(packed, scale, h2);

        reinterpret_cast<uint4*>(dst + i * 8)[0] = *reinterpret_cast<const uint4*>(h2);
    }
}

void launch_embed_lookup_w4a16(
    const uint32_t* embed_qweight,
    const half* embed_scales,
    const int32_t* d_token_ids,
    half* out_hidden,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream,
    int hidden_size
) {
    int block_size = 256;
    int grid_size = num_tokens;
    embed_lookup_w4a16_kernel<<<grid_size, block_size, 0, stream>>>(
        embed_qweight, embed_scales, d_token_ids, out_hidden, d_control, num_tokens, hidden_size
    );
    CUDA_KERNEL_CHECK();
}

// ----------------------------------------------------------------------------
// Fused RMSNorm Kernel (FP16 output)
// ----------------------------------------------------------------------------

__global__ void rmsnorm_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    half* __restrict__ output,
    int num_tokens,
    int hidden_size
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const half2* in_h2 = reinterpret_cast<const half2*>(input + static_cast<size_t>(token_idx) * hidden_size);
    const half2* gamma_h2 = reinterpret_cast<const half2*>(gamma);
    half2* out_h2 = reinterpret_cast<half2*>(output + static_cast<size_t>(token_idx) * hidden_size);

    int num_h2 = hidden_size / 2;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < num_h2; i += blockDim.x) {
        half2 val = in_h2[i];
        float2 f2 = __half22float2(val);
        sum_sq += f2.x * f2.x + f2.y * f2.y;
    }

    // Warp reduction (32 lanes)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // Reduce warp sums in warp 0
    float total_sum = 0.0f;
    if (threadIdx.x < 8) {
        total_sum = s_warp_sums[threadIdx.x];
    }
    #pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        total_sum += __shfl_down_sync(0xFF, total_sum, offset);
    }

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total_sum / static_cast<float>(hidden_size) + RMS_NORM_EPS);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    half2 inv_rms_h2 = __float2half2_rn(inv_rms);

    for (int i = threadIdx.x; i < num_h2; i += blockDim.x) {
        half2 g = gamma_h2[i];
        half2 val = in_h2[i];
        half2 norm_val = __hmul2(val, inv_rms_h2);
        out_h2[i] = __hmul2(norm_val, g);
    }
}

void launch_rmsnorm_fp16(
    const half* input,
    const half* gamma,
    half* output,
    int num_tokens,
    cudaStream_t stream,
    int hidden_size
) {
    rmsnorm_fp16_kernel<<<num_tokens, 256, 0, stream>>>(input, gamma, output, num_tokens, hidden_size);
    CUDA_KERNEL_CHECK();
}

// ----------------------------------------------------------------------------
// Fused RMSNorm + Q8_1 Activation Quantization Kernel
// ----------------------------------------------------------------------------

__global__ void rmsnorm_q8_1_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    half* __restrict__ output_fp16_opt,
    int8_t* __restrict__ out_q8,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_sum8,
    int hidden_size
) {
    extern __shared__ __align__(16) half s_norm[];

    const uint4* in_u4 = reinterpret_cast<const uint4*>(input);
    const uint4* gamma_u4 = reinterpret_cast<const uint4*>(gamma);
    uint4* s_out_u4 = reinterpret_cast<uint4*>(s_norm);
    uint4* out_fp16_u4 = reinterpret_cast<uint4*>(output_fp16_opt);

    int num_u4 = hidden_size / 8;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        uint4 v_u4 = __ldg(&in_u4[i]);
        const half2* vh2 = reinterpret_cast<const half2*>(&v_u4);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f2 = __half22float2(vh2[k]);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    float total_sum = 0.0f;
    if (threadIdx.x < 8) {
        total_sum = s_warp_sums[threadIdx.x];
    }
    #pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        total_sum += __shfl_down_sync(0xFF, total_sum, offset);
    }

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total_sum / static_cast<float>(hidden_size) + RMS_NORM_EPS);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    half2 inv_rms_h2 = __float2half2_rn(inv_rms);

    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        uint4 v_u4 = __ldg(&in_u4[i]);
        uint4 g_u4 = __ldg(&gamma_u4[i]);
        uint4 r_u4;
        const half2* vh2 = reinterpret_cast<const half2*>(&v_u4);
        const half2* gh2 = reinterpret_cast<const half2*>(&g_u4);
        half2* rh2 = reinterpret_cast<half2*>(&r_u4);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            rh2[k] = __hmul2(__hmul2(vh2[k], inv_rms_h2), gh2[k]);
        }
        s_out_u4[i] = r_u4;
        if (out_fp16_u4 != nullptr) {
            out_fp16_u4[i] = r_u4;
        }
    }
    __syncthreads();

    // Quantize into Q8_1 across groups of 128 elements
    int num_groups = hidden_size / QUANT_GROUP_SIZE;
    int half_warp_id = threadIdx.x >> 4;
    int sub_lane = threadIdx.x & 15;

    for (int group_idx = half_warp_id; group_idx < num_groups; group_idx += (blockDim.x >> 4)) {
        uint4 g_u4 = *reinterpret_cast<const uint4*>(s_norm + group_idx * QUANT_GROUP_SIZE + sub_lane * 8);
        const half* g_src = reinterpret_cast<const half*>(&g_u4);
        float vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            vals[j] = __half2float(g_src[j]);
        }
        quantize_group128_fp16_to_q8_1(vals, group_idx, sub_lane, out_q8, out_scale, out_sum8);
    }
}

void launch_rmsnorm_q8_1(
    const half* input,
    const half* gamma,
    half* output_fp16_opt,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8,
    cudaStream_t stream,
    int hidden_size
) {
    size_t smem = static_cast<size_t>(hidden_size) * sizeof(half);
    rmsnorm_q8_1_kernel<<<1, 256, smem, stream>>>(
        input, gamma, output_fp16_opt, out_q8, out_scale, out_sum8, hidden_size
    );
    CUDA_KERNEL_CHECK();
}

// ----------------------------------------------------------------------------
// Standalone FP16 -> Q8_1 Group-128 Quantization Kernel
// ----------------------------------------------------------------------------

__global__ void quantize_fp16_to_q8_1_kernel(
    const half* __restrict__ input_x,
    int8_t* __restrict__ out_q8,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_sum8,
    int num_elements
) {
    // block = 32 threads (2 half-warps = 2 groups = 256 elements per block)
    // grid = num_elements / 256
    int group_idx = blockIdx.x * 2 + (threadIdx.x >> 4);
    int sub_lane = threadIdx.x & 15;

    int base_elem = group_idx * QUANT_GROUP_SIZE + sub_lane * 8;
    if (base_elem + 7 >= num_elements) return;

    const uint4* in_u4 = reinterpret_cast<const uint4*>(input_x + base_elem);
    uint4 raw_u4 = *in_u4;
    const half* in_h = reinterpret_cast<const half*>(&raw_u4);

    float vals[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        vals[j] = __half2float(in_h[j]);
    }

    quantize_group128_fp16_to_q8_1(vals, group_idx, sub_lane, out_q8, out_scale, out_sum8);
}

void launch_quantize_fp16_to_q8_1(
    const half* input_x,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8,
    int num_elements,
    cudaStream_t stream
) {
    int blocks = num_elements / 256;
    quantize_fp16_to_q8_1_kernel<<<blocks, 32, 0, stream>>>(
        input_x, out_q8, out_scale, out_sum8, num_elements
    );
    CUDA_KERNEL_CHECK();
}

// ----------------------------------------------------------------------------
// Fused QKV Bias + NeoX Split-Half RoPE + KV Cache Append
// ----------------------------------------------------------------------------

__global__ void qkv_bias_rope_kv_append_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    half* __restrict__ v,
    const half* __restrict__ q_bias,
    const half* __restrict__ k_bias,
    const half* __restrict__ v_bias,
    half* __restrict__ layer_k_cache,
    half* __restrict__ layer_v_cache,
    const DeviceControlBlock* __restrict__ d_control,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int rotary_dim,
    float rope_theta
) {
    int m = blockIdx.x;
    if (m >= num_tokens) return;

    int pos = 0;
    if (d_control != nullptr) {
        pos = d_control->seq_pos + m;
        if (pos >= max_seq_len) {
            pos = max_seq_len - 1;
        }
    }

    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    half* q_m = q + static_cast<size_t>(m) * q_dim;
    half* k_m = k + static_cast<size_t>(m) * kv_dim;
    half* v_m = v + static_cast<size_t>(m) * kv_dim;

    // 1. Vectorized bias addition using half2 (if bias provided)
    const half2* qb_h2 = reinterpret_cast<const half2*>(q_bias);
    const half2* kb_h2 = reinterpret_cast<const half2*>(k_bias);
    const half2* vb_h2 = reinterpret_cast<const half2*>(v_bias);
    half2* qm_h2 = reinterpret_cast<half2*>(q_m);
    half2* km_h2 = reinterpret_cast<half2*>(k_m);
    half2* vm_h2 = reinterpret_cast<half2*>(v_m);

    for (int i = threadIdx.x; i < q_dim / 2; i += blockDim.x) {
        if (qb_h2) qm_h2[i] = __hadd2(qm_h2[i], qb_h2[i]);
    }
    for (int i = threadIdx.x; i < kv_dim / 2; i += blockDim.x) {
        if (kb_h2) km_h2[i] = __hadd2(km_h2[i], kb_h2[i]);
        if (vb_h2) vm_h2[i] = __hadd2(vm_h2[i], vb_h2[i]);
    }

    // Shared memory staging for RoPE angles (half_rotary <= 64)
    int eff_rotary = (rotary_dim > 0 && rotary_dim <= head_dim) ? rotary_dim : head_dim;
    int half_rotary = eff_rotary / 2;
    __shared__ float s_sin[64];
    __shared__ float s_cos[64];

    if (threadIdx.x < half_rotary) {
        int coord_i = threadIdx.x;
        float exponent = -2.0f * static_cast<float>(coord_i) / static_cast<float>(eff_rotary);
        float log2_theta = log2f(rope_theta);
        float theta = exp2f(exponent * log2_theta);
        float angle = static_cast<float>(pos) * theta;
        __sincosf(angle, &s_sin[coord_i], &s_cos[coord_i]);
    }
    __syncthreads();

    // 2. Partial RoPE using precomputed shared angles on [0, half_rotary)
    int total_q_pairs = num_q_heads * half_rotary;
    int total_k_pairs = num_kv_heads * half_rotary;
    int total_pairs   = total_q_pairs + total_k_pairs;

    for (int pair_idx = threadIdx.x; pair_idx < total_pairs; pair_idx += blockDim.x) {
        bool is_q = (pair_idx < total_q_pairs);
        int head_idx = is_q ? (pair_idx / half_rotary) : ((pair_idx - total_q_pairs) / half_rotary);
        int coord_i  = pair_idx % half_rotary;

        float sin_val = s_sin[coord_i];
        float cos_val = s_cos[coord_i];

        half* head_ptr = is_q ? (q_m + head_idx * head_dim) : (k_m + head_idx * head_dim);
        float x0 = __half2float(head_ptr[coord_i]);
        float x1 = __half2float(head_ptr[coord_i + half_rotary]);

        float rot0 = x0 * cos_val - x1 * sin_val;
        float rot1 = x0 * sin_val + x1 * cos_val;

        head_ptr[coord_i]               = __float2half(rot0);
        head_ptr[coord_i + half_rotary] = __float2half(rot1);
    }
    __syncthreads();

    // 3. Write rotated k and biased v to KV Cache using half2
    for (int i = threadIdx.x; i < kv_dim / 2; i += blockDim.x) {
        int idx = i * 2;
        int kv_head = idx / head_dim;
        int d = idx % head_dim;
        size_t cache_offset = (static_cast<size_t>(kv_head) * max_seq_len + pos) * head_dim + d;

        *reinterpret_cast<half2*>(layer_k_cache + cache_offset) = km_h2[i];
        *reinterpret_cast<half2*>(layer_v_cache + cache_offset) = vm_h2[i];
    }
}

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
    cudaStream_t stream,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int rotary_dim,
    float rope_theta
) {
    if (rotary_dim <= 0) rotary_dim = head_dim;
    qkv_bias_rope_kv_append_kernel<<<num_tokens, 256, 0, stream>>>(
        q, k, v, q_bias, k_bias, v_bias, layer_k_cache, layer_v_cache, d_control, num_tokens,
        num_q_heads, num_kv_heads, head_dim, max_seq_len, rotary_dim, rope_theta
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
