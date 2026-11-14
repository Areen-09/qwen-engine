#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// GQA Split-K FlashDecoding Stage 1: Zero-Barrier Warp-per-Token (SM 8.6)
// Specialized for Qwen 2.5-1.5B (HEAD_D = 128, GQA_GROUP_SIZE = 6)
// Grid: (num_kv_heads, FLASH_DECODE_SPLITS = 16)
// Threads: 128 (4 warps)
// ----------------------------------------------------------------------------

constexpr int QWEN_GQA_GROUP = 6;
constexpr int QWEN_HEAD_D    = 128;

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

    constexpr int ELEMS_PER_LANE = 4; // 32 lanes * 4 = 128
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int d_base  = lane_id * ELEMS_PER_LANE;

    int q_head_base = kv_head * gqa_group_size;

    // Early-exit inactive splits on-device (preserving fixed grid.y for CUDA Graph)
    if (split_start >= split_end) {
        if (tid < gqa_group_size) {
            int q_h = q_head_base + tid;
            size_t lse_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * 2;
            splitk_partial_lse[lse_idx + 0] = -INFINITY;
            splitk_partial_lse[lse_idx + 1] = 0.0f;
        }
        return;
    }

    // Load Query vectors into registers (6 heads * 4 floats = 24 floats)
    float q_reg[QWEN_GQA_GROUP][ELEMS_PER_LANE];
    #pragma unroll
    for (int h = 0; h < QWEN_GQA_GROUP; ++h) {
        if (h < gqa_group_size) {
            const half* q_h_ptr = q + static_cast<size_t>(q_head_base + h) * head_dim + d_base;
            uint2 raw_u2 = __ldg(reinterpret_cast<const uint2*>(q_h_ptr));
            const half* raw_h = reinterpret_cast<const half*>(&raw_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                q_reg[h][p] = __half2float(raw_h[p]);
            }
        }
    }

    float m_i[QWEN_GQA_GROUP];
    float d_i[QWEN_GQA_GROUP];
    float o_reg[QWEN_GQA_GROUP][ELEMS_PER_LANE];

    #pragma unroll
    for (int h = 0; h < QWEN_GQA_GROUP; ++h) {
        m_i[h] = -INFINITY;
        d_i[h] = 0.0f;
        #pragma unroll
        for (int p = 0; p < ELEMS_PER_LANE; ++p) {
            o_reg[h][p] = 0.0f;
        }
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(QWEN_HEAD_D));

    for (int t = split_start + warp_id; t < split_end; t += 4) {
        size_t cache_base = (static_cast<size_t>(kv_head) * max_seq_len + t) * head_dim + d_base;
        uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(layer_k_cache + cache_base));
        uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(layer_v_cache + cache_base));
        const half* k_h = reinterpret_cast<const half*>(&k_u2);
        const half* v_h = reinterpret_cast<const half*>(&v_u2);

        float k_val[ELEMS_PER_LANE], v_val[ELEMS_PER_LANE];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            k_val[p] = __half2float(k_h[p]);
            v_val[p] = __half2float(v_h[p]);
        }

        #pragma unroll
        for (int h = 0; h < QWEN_GQA_GROUP; ++h) {
            if (h < gqa_group_size) {
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
    }

    // Single-barrier parallel cross-warp reduction across all 6 Q heads
    __shared__ float s_m[QWEN_GQA_GROUP][4];
    __shared__ float s_d[QWEN_GQA_GROUP][4];
    __shared__ float s_o[4][QWEN_GQA_GROUP][QWEN_HEAD_D];

    if (lane_id < gqa_group_size) {
        s_m[lane_id][warp_id] = m_i[lane_id];
        s_d[lane_id][warp_id] = d_i[lane_id];
    }
    #pragma unroll
    for (int h = 0; h < QWEN_GQA_GROUP; ++h) {
        if (h < gqa_group_size) {
            #pragma unroll
            for (int p = 0; p < ELEMS_PER_LANE; ++p) {
                s_o[warp_id][h][d_base + p] = o_reg[h][p];
            }
        }
    }
    __syncthreads();

    // All 128 threads merge (h, elem) pairs in parallel
    int total_elems = gqa_group_size * head_dim;
    for (int idx = tid; idx < total_elems; idx += blockDim.x) {
        int h = idx / head_dim;
        int elem = idx % head_dim;

        float m_max = -INFINITY;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_d[h][w] > 0.0f) {
                m_max = fmaxf(m_max, s_m[h][w]);
            }
        }

        float d_total = 0.0f;
        float merged_o = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_d[h][w] > 0.0f) {
                float factor = expf(s_m[h][w] - m_max);
                d_total += s_d[h][w] * factor;
                merged_o += factor * s_o[w][h][elem];
            }
        }

        int q_h = q_head_base + h;
        size_t out_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * head_dim + elem;
        splitk_partial_out[out_idx] = (d_total > 0.0f) ? merged_o : 0.0f;

        if (elem == 0) {
            size_t lse_idx = (static_cast<size_t>(q_h) * FLASH_DECODE_SPLITS + split_idx) * 2;
            splitk_partial_lse[lse_idx + 0] = m_max;
            splitk_partial_lse[lse_idx + 1] = d_total;
        }
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
// Shared-Memory K/V-Tiled Causal FlashAttention Prefill Kernel (Br=16, Bc=32)
// Grid: (num_q_heads, ceil(num_tokens / 16))
// Threads: 128 (4 warps, each warp owns 4 query rows)
// ----------------------------------------------------------------------------

constexpr int PREFILL_BR = 16;
constexpr int PREFILL_BC = 32;

__global__ void attention_gqa_prefill_flash_kernel(
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
    int q_h = blockIdx.x;
    int q_tile = blockIdx.y;
    int m_start = q_tile * PREFILL_BR;
    if (m_start >= num_tokens) return;

    int gqa_group_size = num_q_heads / num_kv_heads;
    int kv_h = q_h / gqa_group_size;
    int q_dim = num_q_heads * head_dim;

    int base_pos = 0;
    if (d_control != nullptr) {
        base_pos = d_control->seq_pos;
    }

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int d_base  = lane_id * 4; // 32 lanes * 4 = 128

    // Each warp owns 4 query rows: m = m_start + warp_id * 4 + r
    int q_pos[4];
    float q_reg[4][4];
    float m_i[4];
    float d_i[4];
    float o_reg[4][4];

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int m = m_start + warp_id * 4 + r;
        q_pos[r] = (m < num_tokens) ? min(base_pos + m, max_seq_len - 1) : -1;
        m_i[r] = -INFINITY;
        d_i[r] = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            o_reg[r][p] = 0.0f;
            q_reg[r][p] = 0.0f;
        }
        if (m < num_tokens) {
            size_t q_idx = static_cast<size_t>(m) * q_dim + static_cast<size_t>(q_h) * head_dim + d_base;
            uint2 raw_u2 = __ldg(reinterpret_cast<const uint2*>(q + q_idx));
            const half* raw_h = reinterpret_cast<const half*>(&raw_u2);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                q_reg[r][p] = __half2float(raw_h[p]);
            }
        }
    }

    int last_m = min(m_start + PREFILL_BR - 1, num_tokens - 1);
    int max_kv_len = min(base_pos + last_m, max_seq_len - 1) + 1;

    __shared__ __align__(16) half s_K[PREFILL_BC][QWEN_HEAD_D];
    __shared__ __align__(16) half s_V[PREFILL_BC][QWEN_HEAD_D];

    const float scale = 1.0f / sqrtf(static_cast<float>(QWEN_HEAD_D));
    const size_t kv_head_offset = static_cast<size_t>(kv_h) * max_seq_len * head_dim;

    for (int t_base = 0; t_base < max_kv_len; t_base += PREFILL_BC) {
        int tile_len = min(PREFILL_BC, max_kv_len - t_base);

        // Cooperatively load K and V tiles into shared memory (uint4 = 8 halfs, 16 threads per row)
        int total_u4 = tile_len * (QWEN_HEAD_D / 8); // <= 32 * 16 = 512
        for (int idx = tid; idx < total_u4; idx += blockDim.x) {
            int k_row = idx >> 4;
            int k_col_u4 = idx & 15;
            size_t src_idx = kv_head_offset + static_cast<size_t>(t_base + k_row) * head_dim + k_col_u4 * 8;
            *reinterpret_cast<uint4*>(&s_K[k_row][k_col_u4 * 8]) = __ldg(reinterpret_cast<const uint4*>(layer_k_cache + src_idx));
            *reinterpret_cast<uint4*>(&s_V[k_row][k_col_u4 * 8]) = __ldg(reinterpret_cast<const uint4*>(layer_v_cache + src_idx));
        }
        __syncthreads();

        for (int k_idx = 0; k_idx < tile_len; ++k_idx) {
            int kv_pos = t_base + k_idx;
            uint2 k_u2 = *reinterpret_cast<const uint2*>(&s_K[k_idx][d_base]);
            uint2 v_u2 = *reinterpret_cast<const uint2*>(&s_V[k_idx][d_base]);
            const half* k_h = reinterpret_cast<const half*>(&k_u2);
            const half* v_h = reinterpret_cast<const half*>(&v_u2);

            float k_val[4], v_val[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                k_val[p] = __half2float(k_h[p]);
                v_val[p] = __half2float(v_h[p]);
            }

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (q_pos[r] >= kv_pos) {
                    float prod = q_reg[r][0] * k_val[0] + q_reg[r][1] * k_val[1]
                               + q_reg[r][2] * k_val[2] + q_reg[r][3] * k_val[3];
                    #pragma unroll
                    for (int mask = 16; mask > 0; mask >>= 1) {
                        prod += __shfl_xor_sync(0xFFFFFFFF, prod, mask);
                    }

                    float score = prod * scale;
                    float m_curr = fmaxf(m_i[r], score);
                    float alpha = expf(m_i[r] - m_curr);
                    float beta  = expf(score - m_curr);

                    d_i[r] = d_i[r] * alpha + beta;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        o_reg[r][p] = o_reg[r][p] * alpha + beta * v_val[p];
                    }
                    m_i[r] = m_curr;
                }
            }
        }
        __syncthreads();
    }

    // Write normalized output
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int m = m_start + warp_id * 4 + r;
        if (m < num_tokens) {
            float inv_d = (d_i[r] > 0.0f) ? (1.0f / d_i[r]) : 0.0f;
            half out_h[4];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                out_h[p] = __float2half(o_reg[r][p] * inv_d);
            }
            size_t out_idx = static_cast<size_t>(m) * hidden_size + static_cast<size_t>(q_h) * head_dim + d_base;
            *reinterpret_cast<uint2*>(attn_out + out_idx) = *reinterpret_cast<const uint2*>(out_h);
        }
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
    attention_gqa_flash_decode_stage1_kernel<<<grid_stage1, 128, 0, stream>>>(
        q, layer_k_cache, layer_v_cache, splitk_partial_out, splitk_partial_lse, d_control,
        num_q_heads, num_kv_heads, head_dim, max_seq_len
    );
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
    dim3 grid(num_q_heads, (num_tokens + PREFILL_BR - 1) / PREFILL_BR);
    attention_gqa_prefill_flash_kernel<<<grid, 128, 0, stream>>>(
        q, layer_k_cache, layer_v_cache, attn_out, d_control, num_tokens,
        num_q_heads, num_kv_heads, head_dim, max_seq_len, hidden_size
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
