#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// GQA Split-K FlashDecoding Stage 1: Zero-Barrier Warp-per-Token
// Grid: (num_kv_heads, FLASH_DECODE_SPLITS = 16)
// Threads: 128 (4 warps: warp 0 .. 3)
// Each 32-lane warp owns HEAD_DIM = 128 (4 elements per lane as uint2)
// ----------------------------------------------------------------------------

constexpr int MAX_GQA_GROUP = 16;

template <int HEAD_D>
__global__ void attention_gqa_flash_decode_stage1_kernel(
    const half* __restrict__ q,
    const half* __restrict__ layer_k_cache,
    const half* __restrict__ layer_v_cache,
    float* __restrict__ splitk_partial_out,
    float* __restrict__ splitk_partial_lse,
    const DeviceControlBlock* __restrict__ d_control,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len
) {
    int kv_head   = blockIdx.x;
    int split_idx = blockIdx.y;

    int gqa_group_size = num_q_heads / num_kv_heads;

    int seq_pos = 0;
    if (d_control != nullptr) {
        seq_pos = d_control->seq_pos;
    }
    int seq_len = min(seq_pos + 1, max_seq_len);

    int tokens_per_split = (seq_len + FLASH_DECODE_SPLITS - 1) / FLASH_DECODE_SPLITS;
    int split_start = split_idx * tokens_per_split;
    int split_end   = min(split_start + tokens_per_split, seq_len);

    constexpr int ELEMS_PER_LANE = (HEAD_D == 256) ? 8 : 4;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int d_base  = lane_id * ELEMS_PER_LANE;

    int q_head_base = kv_head * gqa_group_size;

    // Guard for empty split
    if (split_start >= split_end) {
        if (tid < gqa_group_size) {
            int q_h = q_head_base + tid;
            size_t lse_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * 2;
            splitk_partial_lse[lse_idx + 0] = -INFINITY;
            splitk_partial_lse[lse_idx + 1] = 0.0f;
        }
        return;
    }

    // Load Query vectors belonging to this KV head into per-lane registers
    float q_reg[MAX_GQA_GROUP][ELEMS_PER_LANE];
    for (int h = 0; h < gqa_group_size; ++h) {
        const half* q_h_ptr = q + static_cast<size_t>(q_head_base + h) * head_dim + d_base;
        if constexpr (HEAD_D == 256) {
            uint4 raw_u4 = *reinterpret_cast<const uint4*>(q_h_ptr);
            const half* raw_h = reinterpret_cast<const half*>(&raw_u4);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                q_reg[h][p] = __half2float(raw_h[p]);
            }
        } else {
            uint2 raw_u2 = *reinterpret_cast<const uint2*>(q_h_ptr);
            const half* raw_h = reinterpret_cast<const half*>(&raw_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                q_reg[h][p] = __half2float(raw_h[p]);
            }
        }
    }

    // Per-warp online softmax accumulators
    float m_i[MAX_GQA_GROUP];
    float d_i[MAX_GQA_GROUP];
    float o_reg[MAX_GQA_GROUP][ELEMS_PER_LANE];

    for (int h = 0; h < gqa_group_size; ++h) {
        m_i[h] = -INFINITY;
        d_i[h] = 0.0f;
        #pragma unroll
        for (int p = 0; p < ELEMS_PER_LANE; ++p) {
            o_reg[h][p] = 0.0f;
        }
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_D));

    // Inner loop: 4 warps stride by 4 with ZERO __syncthreads() barriers
    for (int t = split_start + warp_id; t < split_end; t += 4) {
        size_t cache_base = (static_cast<size_t>(kv_head) * max_seq_len + t) * head_dim + d_base;

        float k_val[ELEMS_PER_LANE], v_val[ELEMS_PER_LANE];
        if constexpr (HEAD_D == 256) {
            uint4 k_u4 = *reinterpret_cast<const uint4*>(layer_k_cache + cache_base);
            uint4 v_u4 = *reinterpret_cast<const uint4*>(layer_v_cache + cache_base);
            const half* k_h = reinterpret_cast<const half*>(&k_u4);
            const half* v_h = reinterpret_cast<const half*>(&v_u4);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                k_val[p] = __half2float(k_h[p]);
                v_val[p] = __half2float(v_h[p]);
            }
        } else {
            uint2 k_u2 = *reinterpret_cast<const uint2*>(layer_k_cache + cache_base);
            uint2 v_u2 = *reinterpret_cast<const uint2*>(layer_v_cache + cache_base);
            const half* k_h = reinterpret_cast<const half*>(&k_u2);
            const half* v_h = reinterpret_cast<const half*>(&v_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                k_val[p] = __half2float(k_h[p]);
                v_val[p] = __half2float(v_h[p]);
            }
        }

        for (int h = 0; h < gqa_group_size; ++h) {
            float prod = 0.0f;
            #pragma unroll
            for (int p = 0; p < ELEMS_PER_LANE; ++p) {
                prod += q_reg[h][p] * k_val[p];
            }

            // Warp reduction: broadcast dot sum to all 32 lanes
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                prod += __shfl_xor_sync(0xFFFFFFFF, prod, mask);
            }

            float score = prod * scale;
            float m_curr = fmaxf(m_i[h], score);
            float alpha = expf(m_i[h] - m_curr);
            float beta  = expf(score - m_curr);

            d_i[h] = d_i[h] * alpha + beta;
            #pragma unroll
            for (int p = 0; p < ELEMS_PER_LANE; ++p) {
                o_reg[h][p] = o_reg[h][p] * alpha + beta * v_val[p];
            }
            m_i[h] = m_curr;
        }
    }

    // Shared memory cross-warp reduction
    __shared__ float s_m[MAX_GQA_GROUP][4];
    __shared__ float s_d[MAX_GQA_GROUP][4];
    __shared__ float s_o[4][HEAD_D];

    if (lane_id < gqa_group_size) {
        s_m[lane_id][warp_id] = m_i[lane_id];
        s_d[lane_id][warp_id] = d_i[lane_id];
    }
    __syncthreads();

    // Merge each head sequentially
    for (int h = 0; h < gqa_group_size; ++h) {
        // Store this warp's o_reg into shared memory
        #pragma unroll
        for (int p = 0; p < ELEMS_PER_LANE; ++p) {
            s_o[warp_id][d_base + p] = o_reg[h][p];
        }
        __syncthreads();

        // Warp 0 merges the 4 warps' states
        if (warp_id == 0) {
            float m_max = -INFINITY;
            for (int w = 0; w < 4; ++w) {
                if (s_d[h][w] > 0.0f) {
                    m_max = fmaxf(m_max, s_m[h][w]);
                }
            }

            float d_total = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (s_d[h][w] > 0.0f) {
                    d_total += s_d[h][w] * expf(s_m[h][w] - m_max);
                }
            }

            // Stride across elements in warp 0
            for (int elem = lane_id; elem < head_dim; elem += 32) {
                float merged_o = 0.0f;
                if (d_total > 0.0f) {
                    for (int w = 0; w < 4; ++w) {
                        if (s_d[h][w] > 0.0f) {
                            float factor = expf(s_m[h][w] - m_max);
                            merged_o += factor * s_o[w][elem];
                        }
                    }
                }

                int q_h = q_head_base + h;
                size_t out_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * head_dim + elem;
                splitk_partial_out[out_idx] = merged_o;
            }

            if (lane_id == 0) {
                int q_h = q_head_base + h;
                size_t lse_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * 2;
                splitk_partial_lse[lse_idx + 0] = m_max;
                splitk_partial_lse[lse_idx + 1] = d_total;
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// GQA Split-K FlashDecoding Stage 2: Combine Splits + Optional Q8_1 Epilogue
// Grid: (num_q_heads, 1)
// Threads: 256 (8 warps)
// ----------------------------------------------------------------------------

__global__ void attention_gqa_flash_decode_stage2_kernel(
    const float* __restrict__ splitk_partial_out,
    const float* __restrict__ splitk_partial_lse,
    half* __restrict__ attn_out,
    int8_t* __restrict__ out_q8,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_sum8,
    int head_dim
) {
    int q_h = blockIdx.x;
    int tid = threadIdx.x; // 0 .. 255 (8 warps)

    __shared__ float s_split_m[FLASH_DECODE_SPLITS];
    __shared__ float s_split_d[FLASH_DECODE_SPLITS];
    __shared__ float s_m_global;
    __shared__ float s_d_global;

    if (tid < FLASH_DECODE_SPLITS) {
        size_t lse_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + tid) * 2;
        s_split_m[tid] = splitk_partial_lse[lse_idx + 0];
        s_split_d[tid] = splitk_partial_lse[lse_idx + 1];
    }
    __syncthreads();

    if (tid == 0) {
        float m_max = -INFINITY;
        for (int s = 0; s < FLASH_DECODE_SPLITS; ++s) {
            if (s_split_d[s] > 0.0f) {
                m_max = fmaxf(m_max, s_split_m[s]);
            }
        }
        s_m_global = m_max;

        float d_sum = 0.0f;
        for (int s = 0; s < FLASH_DECODE_SPLITS; ++s) {
            if (s_split_d[s] > 0.0f) {
                d_sum += s_split_d[s] * expf(s_split_m[s] - m_max);
            }
        }
        s_d_global = d_sum;
    }
    __syncthreads();

    float m_global = s_m_global;
    float d_global = s_d_global;

    float acc = 0.0f;
    if (tid < head_dim) {
        if (d_global > 0.0f) {
            float inv_d = 1.0f / d_global;
            for (int s = 0; s < FLASH_DECODE_SPLITS; ++s) {
                if (s_split_d[s] > 0.0f) {
                    float weight = expf(s_split_m[s] - m_global) * inv_d;
                    size_t out_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + s) * head_dim + tid;
                    acc += weight * splitk_partial_out[out_idx];
                }
            }
        }
        attn_out[static_cast<size_t>(q_h) * head_dim + tid] = __float2half(acc);
    }

    // Fused Q8_1 epilogue quantization for O-Proj
    if (out_q8 != nullptr) {
        __shared__ float s_warp_max[8];
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        float abs_val = (tid < head_dim) ? fabsf(acc) : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            abs_val = fmaxf(abs_val, __shfl_xor_sync(0xFFFFFFFF, abs_val, mask));
        }
        if (lane_id == 0) {
            s_warp_max[warp_id] = abs_val;
        }
        __syncthreads();

        __shared__ float s_block_d;
        if (tid == 0) {
            int num_warps = (head_dim + 31) / 32;
            float b_max = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                b_max = fmaxf(b_max, s_warp_max[w]);
            }
            s_block_d = fmaxf(b_max / 127.0f, 1e-8f);
            if (out_scale != nullptr) {
                out_scale[q_h] = s_block_d;
            }
        }
        __syncthreads();

        if (tid < head_dim) {
            float d = s_block_d;
            int v = __float2int_rn(acc / d);
            v = max(-127, min(127, v));
            int8_t q_val = static_cast<int8_t>(v);
            out_q8[static_cast<size_t>(q_h) * head_dim + tid] = q_val;

            // Reduce sum8 = -8 * sum_{j=0}^7 q_val across 8 adjacent lanes
            int32_t sum8 = -8 * static_cast<int32_t>(q_val);
            sum8 += __shfl_xor_sync(0xFFFFFFFF, sum8, 4);
            sum8 += __shfl_xor_sync(0xFFFFFFFF, sum8, 2);
            sum8 += __shfl_xor_sync(0xFFFFFFFF, sum8, 1);

            if ((tid & 7) == 0 && out_sum8 != nullptr) {
                out_sum8[q_h * (head_dim / 8) + (tid >> 3)] = sum8;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Causal Prefill Attention Kernel (Zero inner-loop barriers, 6 Q heads shared)
// Grid: (NUM_KV_HEADS, num_tokens)
// Threads: 128 (4 warps)
// ----------------------------------------------------------------------------

template <int HEAD_D>
__global__ void attention_gqa_prefill_kernel(
    const half* __restrict__ q,
    const half* __restrict__ layer_k_cache,
    const half* __restrict__ layer_v_cache,
    half* __restrict__ attn_out,
    const DeviceControlBlock* __restrict__ d_control,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int hidden_size
) {
    int kv_h = blockIdx.x;
    int m    = blockIdx.y;
    if (m >= num_tokens) return;

    constexpr int ELEMS_PER_LANE = (HEAD_D == 256) ? 8 : 4;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int d_base  = lane_id * ELEMS_PER_LANE;

    int gqa_group_size = num_q_heads / num_kv_heads;
    int q_dim = num_q_heads * head_dim;
    int q_head_base = kv_h * gqa_group_size;

    int base_pos = 0;
    if (d_control != nullptr) {
        base_pos = d_control->seq_pos;
    }
    int cur_pos = min(base_pos + m, max_seq_len - 1);
    int seq_len = cur_pos + 1;

    // Load Query vectors into registers
    float q_reg[MAX_GQA_GROUP][ELEMS_PER_LANE];
    for (int h = 0; h < gqa_group_size; ++h) {
        size_t q_idx = static_cast<size_t>(m) * q_dim + (q_head_base + h) * head_dim + d_base;
        if constexpr (HEAD_D == 256) {
            uint4 raw_u4 = *reinterpret_cast<const uint4*>(q + q_idx);
            const half* raw_h = reinterpret_cast<const half*>(&raw_u4);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                q_reg[h][p] = __half2float(raw_h[p]);
            }
        } else {
            uint2 raw_u2 = *reinterpret_cast<const uint2*>(q + q_idx);
            const half* raw_h = reinterpret_cast<const half*>(&raw_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                q_reg[h][p] = __half2float(raw_h[p]);
            }
        }
    }

    float m_i[MAX_GQA_GROUP];
    float d_i[MAX_GQA_GROUP];
    float o_reg[MAX_GQA_GROUP][ELEMS_PER_LANE];

    for (int h = 0; h < gqa_group_size; ++h) {
        m_i[h] = -INFINITY;
        d_i[h] = 0.0f;
        #pragma unroll
        for (int p = 0; p < ELEMS_PER_LANE; ++p) {
            o_reg[h][p] = 0.0f;
        }
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_D));

    // Sweeping cached tokens: 4 warps stride by 4 with ZERO __syncthreads() barriers
    for (int t = warp_id; t < seq_len; t += 4) {
        size_t cache_base = (static_cast<size_t>(kv_h) * max_seq_len + t) * head_dim + d_base;

        float k_val[ELEMS_PER_LANE], v_val[ELEMS_PER_LANE];
        if constexpr (HEAD_D == 256) {
            uint4 k_u4 = *reinterpret_cast<const uint4*>(layer_k_cache + cache_base);
            uint4 v_u4 = *reinterpret_cast<const uint4*>(layer_v_cache + cache_base);
            const half* k_h = reinterpret_cast<const half*>(&k_u4);
            const half* v_h = reinterpret_cast<const half*>(&v_u4);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                k_val[p] = __half2float(k_h[p]);
                v_val[p] = __half2float(v_h[p]);
            }
        } else {
            uint2 k_u2 = *reinterpret_cast<const uint2*>(layer_k_cache + cache_base);
            uint2 v_u2 = *reinterpret_cast<const uint2*>(layer_v_cache + cache_base);
            const half* k_h = reinterpret_cast<const half*>(&k_u2);
            const half* v_h = reinterpret_cast<const half*>(&v_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                k_val[p] = __half2float(k_h[p]);
                v_val[p] = __half2float(v_h[p]);
            }
        }

        for (int h = 0; h < gqa_group_size; ++h) {
            float prod = 0.0f;
            #pragma unroll
            for (int p = 0; p < ELEMS_PER_LANE; ++p) {
                prod += q_reg[h][p] * k_val[p];
            }

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                prod += __shfl_xor_sync(0xFFFFFFFF, prod, mask);
            }

            float score = prod * scale;
            float m_curr = fmaxf(m_i[h], score);
            float alpha = expf(m_i[h] - m_curr);
            float beta  = expf(score - m_curr);

            d_i[h] = d_i[h] * alpha + beta;
            #pragma unroll
            for (int p = 0; p < ELEMS_PER_LANE; ++p) {
                o_reg[h][p] = o_reg[h][p] * alpha + beta * v_val[p];
            }
            m_i[h] = m_curr;
        }
    }

    // Shared memory cross-warp reduction
    __shared__ float s_m[MAX_GQA_GROUP][4];
    __shared__ float s_d[MAX_GQA_GROUP][4];
    __shared__ float s_o[4][HEAD_D];

    if (lane_id < gqa_group_size) {
        s_m[lane_id][warp_id] = m_i[lane_id];
        s_d[lane_id][warp_id] = d_i[lane_id];
    }
    __syncthreads();

    for (int h = 0; h < gqa_group_size; ++h) {
        #pragma unroll
        for (int p = 0; p < ELEMS_PER_LANE; ++p) {
            s_o[warp_id][d_base + p] = o_reg[h][p];
        }
        __syncthreads();

        if (warp_id == 0) {
            float m_max = -INFINITY;
            for (int w = 0; w < 4; ++w) {
                if (s_d[h][w] > 0.0f) {
                    m_max = fmaxf(m_max, s_m[h][w]);
                }
            }

            float d_total = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (s_d[h][w] > 0.0f) {
                    d_total += s_d[h][w] * expf(s_m[h][w] - m_max);
                }
            }

            for (int elem = lane_id; elem < head_dim; elem += 32) {
                float merged_o = 0.0f;
                if (d_total > 0.0f) {
                    for (int w = 0; w < 4; ++w) {
                        if (s_d[h][w] > 0.0f) {
                            float factor = expf(s_m[h][w] - m_max);
                            merged_o += factor * s_o[w][elem];
                        }
                    }
                }

                float final_val = (d_total > 0.0f) ? (merged_o / d_total) : 0.0f;
                size_t out_idx = static_cast<size_t>(m) * hidden_size + (q_head_base + h) * head_dim + elem;
                attn_out[out_idx] = __float2half(final_val);
            }
        }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// Host Launch Functions
// ----------------------------------------------------------------------------

void launch_attention_gqa_flash_decode(
    const half* q,
    const half* layer_k_cache,
    const half* layer_v_cache,
    float* splitk_partial_out,
    float* splitk_partial_lse,
    half* attn_out,
    const DeviceControlBlock* d_control,
    cudaStream_t stream,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8
) {
    dim3 grid_stage1(num_kv_heads, FLASH_DECODE_SPLITS);
    if (head_dim == 256) {
        attention_gqa_flash_decode_stage1_kernel<256><<<grid_stage1, 128, 0, stream>>>(
            q, layer_k_cache, layer_v_cache, splitk_partial_out, splitk_partial_lse, d_control,
            num_q_heads, num_kv_heads, head_dim, max_seq_len
        );
    } else {
        attention_gqa_flash_decode_stage1_kernel<128><<<grid_stage1, 128, 0, stream>>>(
            q, layer_k_cache, layer_v_cache, splitk_partial_out, splitk_partial_lse, d_control,
            num_q_heads, num_kv_heads, head_dim, max_seq_len
        );
    }
    CUDA_KERNEL_CHECK();

    dim3 grid_stage2(num_q_heads, 1);
    attention_gqa_flash_decode_stage2_kernel<<<grid_stage2, 256, 0, stream>>>(
        splitk_partial_out, splitk_partial_lse, attn_out, out_q8, out_scale, out_sum8, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void launch_attention_gqa_prefill(
    const half* q,
    const half* layer_k_cache,
    const half* layer_v_cache,
    half* attn_out,
    const DeviceControlBlock* d_control,
    int num_tokens,
    cudaStream_t stream,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int hidden_size
) {
    dim3 grid(num_kv_heads, num_tokens);
    if (head_dim == 256) {
        attention_gqa_prefill_kernel<256><<<grid, 128, 0, stream>>>(
            q, layer_k_cache, layer_v_cache, attn_out, d_control, num_tokens,
            num_q_heads, num_kv_heads, head_dim, max_seq_len, hidden_size
        );
    } else {
        attention_gqa_prefill_kernel<128><<<grid, 128, 0, stream>>>(
            q, layer_k_cache, layer_v_cache, attn_out, d_control, num_tokens,
            num_q_heads, num_kv_heads, head_dim, max_seq_len, hidden_size
        );
    }
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
