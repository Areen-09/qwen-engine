#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "quant_primitives.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// Embedding Lookup Kernels
// ----------------------------------------------------------------------------

__global__ void embed_lookup_fp16_kernel(
    const half* __restrict__ embed_table,
    const int32_t* __restrict__ d_token_ids,
    half* __restrict__ out_hidden,
    const DeviceControlBlock* __restrict__ d_control,
    int num_tokens
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

    const half* src = embed_table + static_cast<size_t>(token_id) * HIDDEN_SIZE;
    half* dst = out_hidden + static_cast<size_t>(token_idx) * HIDDEN_SIZE;

    // Vectorized 128-bit loads (8 halfs per uint4)
    const uint4* src_u4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_u4 = reinterpret_cast<uint4*>(dst);
    int num_u4 = HIDDEN_SIZE / 8; // 1536 / 8 = 192

    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        dst_u4[i] = src_u4[i];
    }
}

__global__ void embed_lookup_w4a16_kernel(
    const uint32_t* __restrict__ embed_qweight,
    const half* __restrict__ embed_scales,
    const int32_t* __restrict__ d_token_ids,
    half* __restrict__ out_hidden,
    const DeviceControlBlock* __restrict__ d_control,
    int num_tokens
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

    int tid = threadIdx.x;
    if (tid >= 192) return;

    int group_idx = tid / 16;   // 0 .. 11
    int lane_idx  = tid % 16;   // 0 .. 15

    half scale = embed_scales[static_cast<size_t>(token_id) * (HIDDEN_SIZE / QUANT_GROUP_SIZE) + group_idx];
    uint32_t packed = embed_qweight[static_cast<size_t>(token_id) * 192 + tid];

    half2 h2[4];
    unpack_w4_to_half2_quad(packed, scale, h2);

    half* dst = out_hidden + static_cast<size_t>(token_idx) * HIDDEN_SIZE + group_idx * QUANT_GROUP_SIZE + lane_idx * 8;
    reinterpret_cast<uint4*>(dst)[0] = *reinterpret_cast<const uint4*>(h2);
}

void launch_embed_lookup_fp16(
    const half* embed_table,
    const int32_t* d_token_ids,
    half* out_hidden,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = num_tokens;
    embed_lookup_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        embed_table, d_token_ids, out_hidden, d_control, num_tokens
    );
    CUDA_KERNEL_CHECK();
}

void launch_embed_lookup_w4a16(
    const uint32_t* embed_qweight,
    const half* embed_scales,
    const int32_t* d_token_ids,
    half* out_hidden,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream
) {
    int block_size = 192;
    int grid_size = num_tokens;
    embed_lookup_w4a16_kernel<<<grid_size, block_size, 0, stream>>>(
        embed_qweight, embed_scales, d_token_ids, out_hidden, d_control, num_tokens
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
    int num_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const half2* in_h2 = reinterpret_cast<const half2*>(input + static_cast<size_t>(token_idx) * HIDDEN_SIZE);
    const half2* gamma_h2 = reinterpret_cast<const half2*>(gamma);
    half2* out_h2 = reinterpret_cast<half2*>(output + static_cast<size_t>(token_idx) * HIDDEN_SIZE);

    constexpr int NUM_H2 = HIDDEN_SIZE / 2; // 768
    constexpr int ELEMENTS_PER_THREAD = NUM_H2 / 256; // 3 half2 = 6 elements

    half2 cached[ELEMENTS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * 256;
        half2 val = in_h2[idx];
        cached[i] = val;
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
        s_inv_rms = rsqrtf(total_sum / static_cast<float>(HIDDEN_SIZE) + RMS_NORM_EPS);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    half2 inv_rms_h2 = __float2half2_rn(inv_rms);

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * 256;
        half2 g = gamma_h2[idx];
        half2 norm_val = __hmul2(cached[i], inv_rms_h2);
        out_h2[idx] = __hmul2(norm_val, g);
    }
}

void launch_rmsnorm_fp16(
    const half* input,
    const half* gamma,
    half* output,
    int num_tokens,
    cudaStream_t stream
) {
    rmsnorm_fp16_kernel<<<num_tokens, 256, 0, stream>>>(input, gamma, output, num_tokens);
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
    int32_t* __restrict__ out_sum8
) {
    __shared__ half s_norm[HIDDEN_SIZE]; // 1536 * 2 = 3072 B

    const half2* in_h2 = reinterpret_cast<const half2*>(input);
    const half2* gamma_h2 = reinterpret_cast<const half2*>(gamma);

    constexpr int NUM_H2 = HIDDEN_SIZE / 2; // 768
    constexpr int ELEMENTS_PER_THREAD = NUM_H2 / 256; // 3 half2 = 6 elements

    half2 cached[ELEMENTS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * 256;
        half2 val = in_h2[idx];
        cached[i] = val;
        float2 f2 = __half22float2(val);
        sum_sq += f2.x * f2.x + f2.y * f2.y;
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
        s_inv_rms = rsqrtf(total_sum / static_cast<float>(HIDDEN_SIZE) + RMS_NORM_EPS);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    half2 inv_rms_h2 = __float2half2_rn(inv_rms);
    half2* s_out_h2 = reinterpret_cast<half2*>(s_norm);

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * 256;
        half2 g = gamma_h2[idx];
        half2 norm_val = __hmul2(cached[i], inv_rms_h2);
        half2 res = __hmul2(norm_val, g);
        s_out_h2[idx] = res;
        if (output_fp16_opt != nullptr) {
            reinterpret_cast<half2*>(output_fp16_opt)[idx] = res;
        }
    }
    __syncthreads();

    // Quantize into Q8_1 across 12 groups of 128 elements
    // 256 threads = 16 half-warps; groups 0 .. 11 are handled by half-warps 0 .. 11
    int half_warp_id = threadIdx.x >> 4;
    int sub_lane = threadIdx.x & 15;

    if (half_warp_id < 12) {
        int group_idx = half_warp_id;
        const half* g_src = s_norm + group_idx * QUANT_GROUP_SIZE + sub_lane * 8;
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
    cudaStream_t stream
) {
    rmsnorm_q8_1_kernel<<<1, 256, 0, stream>>>(
        input, gamma, output_fp16_opt, out_q8, out_scale, out_sum8
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
    int num_tokens
) {
    int m = blockIdx.x;
    if (m >= num_tokens) return;

    int pos = 0;
    if (d_control != nullptr) {
        pos = d_control->seq_pos + m;
        if (pos >= MAX_SEQ_LEN) {
            pos = MAX_SEQ_LEN - 1;
        }
    }

    half* q_m = q + static_cast<size_t>(m) * Q_DIM;
    half* k_m = k + static_cast<size_t>(m) * KV_DIM;
    half* v_m = v + static_cast<size_t>(m) * KV_DIM;

    // 1. Vectorized bias addition using half2
    const half2* qb_h2 = reinterpret_cast<const half2*>(q_bias);
    const half2* kb_h2 = reinterpret_cast<const half2*>(k_bias);
    const half2* vb_h2 = reinterpret_cast<const half2*>(v_bias);
    half2* qm_h2 = reinterpret_cast<half2*>(q_m);
    half2* km_h2 = reinterpret_cast<half2*>(k_m);
    half2* vm_h2 = reinterpret_cast<half2*>(v_m);

    for (int i = threadIdx.x; i < Q_DIM / 2; i += blockDim.x) {
        qm_h2[i] = __hadd2(qm_h2[i], qb_h2[i]);
    }
    for (int i = threadIdx.x; i < KV_DIM / 2; i += blockDim.x) {
        km_h2[i] = __hadd2(km_h2[i], kb_h2[i]);
        vm_h2[i] = __hadd2(vm_h2[i], vb_h2[i]);
    }

    // Shared memory staging for 64 RoPE angles (computed once per block by first 64 threads)
    __shared__ float s_sin[64];
    __shared__ float s_cos[64];

    if (threadIdx.x < 64) {
        int coord_i = threadIdx.x;
        float exponent = -2.0f * static_cast<float>(coord_i) / static_cast<float>(HEAD_DIM);
        float theta = exp2f(exponent * LOG2_ROPE_THETA);
        float angle = static_cast<float>(pos) * theta;
        __sincosf(angle, &s_sin[coord_i], &s_cos[coord_i]);
    }
    __syncthreads();

    // 2. NeoX Split-Half RoPE using precomputed shared angles
    constexpr int HALF_HEAD = HEAD_DIM / 2; // 64
    constexpr int TOTAL_Q_PAIRS = NUM_Q_HEADS * HALF_HEAD; // 12 * 64 = 768
    constexpr int TOTAL_K_PAIRS = NUM_KV_HEADS * HALF_HEAD; // 2 * 64 = 128
    constexpr int TOTAL_PAIRS   = TOTAL_Q_PAIRS + TOTAL_K_PAIRS; // 896

    for (int pair_idx = threadIdx.x; pair_idx < TOTAL_PAIRS; pair_idx += blockDim.x) {
        bool is_q = (pair_idx < TOTAL_Q_PAIRS);
        int head_idx = is_q ? (pair_idx / HALF_HEAD) : ((pair_idx - TOTAL_Q_PAIRS) / HALF_HEAD);
        int coord_i  = pair_idx % HALF_HEAD; // 0 .. 63

        float sin_val = s_sin[coord_i];
        float cos_val = s_cos[coord_i];

        half* head_ptr = is_q ? (q_m + head_idx * HEAD_DIM) : (k_m + head_idx * HEAD_DIM);
        float x0 = __half2float(head_ptr[coord_i]);
        float x1 = __half2float(head_ptr[coord_i + HALF_HEAD]);

        float rot0 = x0 * cos_val - x1 * sin_val;
        float rot1 = x0 * sin_val + x1 * cos_val;

        head_ptr[coord_i]             = __float2half(rot0);
        head_ptr[coord_i + HALF_HEAD] = __float2half(rot1);
    }
    __syncthreads();

    // 3. Write rotated k and biased v to KV Cache using half2
    for (int i = threadIdx.x; i < KV_DIM / 2; i += blockDim.x) {
        int idx = i * 2;
        int kv_head = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        size_t cache_offset = (static_cast<size_t>(kv_head) * MAX_SEQ_LEN + pos) * HEAD_DIM + d;

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
    cudaStream_t stream
) {
    qkv_bias_rope_kv_append_kernel<<<num_tokens, 256, 0, stream>>>(
        q, k, v, q_bias, k_bias, v_bias, layer_k_cache, layer_v_cache, d_control, num_tokens
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
