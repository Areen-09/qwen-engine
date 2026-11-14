#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "quant_primitives.cuh"
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

using namespace nvcuda;

// ----------------------------------------------------------------------------
// Shared Memory Union for Bank-Conflict-Free 2D-Tiled Tensor Core Prefill GEMM
// Total size: 10,240 bytes (10.0 KB -> high occupancy on sm_86)
// ----------------------------------------------------------------------------
union PrefillSmem {
    struct {
        alignas(16) half s_A[PREFILL_WMMA_BM][PREFILL_SMEM_STRIDE]; // [64][40] = 5,120 B
        alignas(16) half s_B[PREFILL_WMMA_BN][PREFILL_SMEM_STRIDE]; // [64][40] = 5,120 B
    } mainloop;                                                      // 10,240 B
    alignas(16) float s_C[4][16][18];                                // [4][16][18] = 4,608 B
};

namespace {

__device__ __forceinline__ void load_a_tile(
    PrefillSmem& smem,
    const half* __restrict__ input_x,
    int m_block,
    int k_offset,
    int M,
    int K,
    int tid
) {
    int a_row = tid >> 1;
    int a_col = (tid & 1) * 16;
    int global_m = m_block + a_row;
    if (global_m < M) {
        const uint4* in_u4 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = __ldg(&in_u4[0]);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = __ldg(&in_u4[1]);
    } else {
        *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ void dequant_b_row_pair(
    PrefillSmem& smem,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    int b_row,
    int b_col,
    int local_n,
    int max_n,
    int kt,
    int K,
    int num_groups_per_row
) {
    if (local_n < max_n) {
        int word_idx = kt * 4 + (b_col >> 3);
        int group_idx = kt >> 2;
        half sc = __ldg(&scales[static_cast<size_t>(local_n) * num_groups_per_row + group_idx]);
        const uint32_t* w_ptr = qweight + static_cast<size_t>(local_n) * (K / 8) + word_idx;
        uint32_t p0 = __ldg(&w_ptr[0]);
        uint32_t p1 = __ldg(&w_ptr[1]);

        half2 h2_0[4], h2_1[4];
        unpack_w4_to_half2_quad(p0, sc, h2_0);
        unpack_w4_to_half2_quad(p1, sc, h2_1);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *reinterpret_cast<const uint4*>(h2_0);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *reinterpret_cast<const uint4*>(h2_1);
    } else {
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = make_uint4(0, 0, 0, 0);
    }
}

__device__ __forceinline__ void wmma_step(
    const PrefillSmem& smem,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&acc)[2][2],
    int warp_m,
    int warp_n
) {
    #pragma unroll
    for (int k_sub = 0; k_sub < 32; k_sub += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2];

        wmma::load_matrix_sync(frag_a[0], &smem.mainloop.s_A[warp_m * 32 + 0][k_sub], PREFILL_SMEM_STRIDE);
        wmma::load_matrix_sync(frag_a[1], &smem.mainloop.s_A[warp_m * 32 + 16][k_sub], PREFILL_SMEM_STRIDE);

        wmma::load_matrix_sync(frag_b[0], &smem.mainloop.s_B[warp_n * 32 + 0][k_sub], PREFILL_SMEM_STRIDE);
        wmma::load_matrix_sync(frag_b[1], &smem.mainloop.s_B[warp_n * 32 + 16][k_sub], PREFILL_SMEM_STRIDE);

        wmma::mma_sync(acc[0][0], frag_a[0], frag_b[0], acc[0][0]);
        wmma::mma_sync(acc[0][1], frag_a[0], frag_b[1], acc[0][1]);
        wmma::mma_sync(acc[1][0], frag_a[1], frag_b[0], acc[1][0]);
        wmma::mma_sync(acc[1][1], frag_a[1], frag_b[1], acc[1][1]);
    }
}

} // namespace

// ----------------------------------------------------------------------------
// W4A16 2D-Tiled Tensor Core Prefill GEMM Kernel
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_w4a16_kernel(
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const half* __restrict__ input_x,
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int M,
    int N,
    int K
) {
    __shared__ PrefillSmem smem;

    int n_block = blockIdx.x * PREFILL_WMMA_BN;
    int m_block = blockIdx.y * PREFILL_WMMA_BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    int num_kt = K / PREFILL_WMMA_BK;
    int num_groups_per_row = K / QUANT_GROUP_SIZE;
    int b_row = tid >> 1;
    int b_col = (tid & 1) * 16;
    int global_n = n_block + b_row;

    for (int kt = 0; kt < num_kt; ++kt) {
        load_a_tile(smem, input_x, m_block, kt * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_row_pair(smem, qweight, scales, b_row, b_col, global_n, N, kt, K, num_groups_per_row);
        __syncthreads();

        wmma_step(smem, acc, warp_m, warp_n);
        __syncthreads();
    }

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int r = lane_id / 16; r < 16; r += 2) {
                int c = lane_id % 16;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                int global_col = n_block + warp_n * 32 + ni * 16 + c;

                if (global_row < M && global_col < N) {
                    float val = smem.s_C[warp_id][r][c];
                    size_t out_idx = static_cast<size_t>(global_row) * N + global_col;
                    if (residual != nullptr) {
                        val += __half2float(__ldg(&residual[out_idx]));
                    }
                    output_y[out_idx] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// Single-Launch Merged QKV Tensor Core Prefill GEMM (W4A16)
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_qkv_w4a16_kernel(
    QKVProjectionView view,
    const half* __restrict__ input_x,
    int M,
    int in_features,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int K = in_features;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int num_groups = K / QUANT_GROUP_SIZE;

    __shared__ PrefillSmem smem;

    int n_block = blockIdx.x * PREFILL_WMMA_BN;
    int m_block = blockIdx.y * PREFILL_WMMA_BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    const uint32_t* qweight = nullptr;
    const half* scales = nullptr;
    half* output = nullptr;
    int col_offset = 0;
    int out_stride = 0;

    if (n_block < q_dim) {
        qweight = reinterpret_cast<const uint32_t*>(view.q_weight);
        scales  = view.q_scale;
        output  = view.d_q;
        col_offset = 0;
        out_stride = q_dim;
    } else if (n_block < q_dim + kv_dim) {
        qweight = reinterpret_cast<const uint32_t*>(view.k_weight);
        scales  = view.k_scale;
        output  = view.d_k;
        col_offset = q_dim;
        out_stride = kv_dim;
    } else {
        qweight = reinterpret_cast<const uint32_t*>(view.v_weight);
        scales  = view.v_scale;
        output  = view.d_v;
        col_offset = q_dim + kv_dim;
        out_stride = kv_dim;
    }

    int num_kt = K / PREFILL_WMMA_BK;
    int b_row = tid >> 1;
    int b_col = (tid & 1) * 16;
    int local_r = (n_block - col_offset) + b_row;

    for (int kt = 0; kt < num_kt; ++kt) {
        load_a_tile(smem, input_x, m_block, kt * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_row_pair(smem, qweight, scales, b_row, b_col, local_r, out_stride, kt, K, num_groups);
        __syncthreads();

        wmma_step(smem, acc, warp_m, warp_n);
        __syncthreads();
    }

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int r = lane_id / 16; r < 16; r += 2) {
                int c = lane_id % 16;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                int rel_col = (n_block - col_offset) + warp_n * 32 + ni * 16 + c;

                if (global_row < M && rel_col < out_stride) {
                    float val = smem.s_C[warp_id][r][c];
                    output[static_cast<size_t>(global_row) * out_stride + rel_col] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// Fused Dual Gate+Up GEMM + SwiGLU Tensor Core Prefill Kernel (W4A16)
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_dual_gate_up_swiglu_w4a16_kernel(
    const uint32_t* __restrict__ gate_up_qweight,
    const half* __restrict__ gate_up_scales,
    const half* __restrict__ input_x,
    half* __restrict__ output_mlp,
    int M,
    int hidden_size,
    int intermediate_size
) {
    int K = hidden_size;
    int total_n = 2 * intermediate_size;
    int num_groups = K / QUANT_GROUP_SIZE;

    __shared__ PrefillSmem smem;

    int n_block = blockIdx.x * PREFILL_WMMA_BN;
    int m_block = blockIdx.y * PREFILL_WMMA_BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    int num_kt = K / PREFILL_WMMA_BK;
    int b_row = tid >> 1;
    int b_col = (tid & 1) * 16;
    int global_n = n_block + b_row;

    for (int kt = 0; kt < num_kt; ++kt) {
        load_a_tile(smem, input_x, m_block, kt * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_row_pair(smem, gate_up_qweight, gate_up_scales, b_row, b_col, global_n, total_n, kt, K, num_groups);
        __syncthreads();

        wmma_step(smem, acc, warp_m, warp_n);
        __syncthreads();
    }

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            int n_sub_base = n_block + warp_n * 32 + ni * 16;

            #pragma unroll
            for (int r = lane_id / 8; r < 16; r += 4) {
                int p = lane_id % 8;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                int feat_idx = (n_sub_base / 2) + p;

                if (global_row < M && feat_idx < intermediate_size) {
                    float gate = smem.s_C[warp_id][r][2 * p + 0];
                    float up   = smem.s_C[warp_id][r][2 * p + 1];
                    float silu = gate / (1.0f + expf(-gate));
                    output_mlp[static_cast<size_t>(global_row) * intermediate_size + feat_idx] = __float2half(silu * up);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// Host Launch Wrappers
// ----------------------------------------------------------------------------

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
) {
    dim3 grid((N + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
    gemm_prefill_w4a16_kernel<<<grid, 128, 0, stream>>>(
        qweight, scales, input_x, residual, output_y, M, N, K
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemm_prefill_qkv_w4a16(
    const QKVProjectionView& view,
    const half* input_x,
    int M,
    cudaStream_t stream,
    int in_features,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int total_n = (num_q_heads + 2 * num_kv_heads) * head_dim;
    dim3 grid((total_n + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
    gemm_prefill_qkv_w4a16_kernel<<<grid, 128, 0, stream>>>(
        view, input_x, M, in_features, num_q_heads, num_kv_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemm_prefill_dual_gate_up_swiglu_w4a16(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const half* input_x,
    half* output_mlp,
    int M,
    cudaStream_t stream,
    int hidden_size,
    int intermediate_size
) {
    int total_n = 2 * intermediate_size;
    dim3 grid((total_n + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
    gemm_prefill_dual_gate_up_swiglu_w4a16_kernel<<<grid, 128, 0, stream>>>(
        gate_up_qweight, gate_up_scales, input_x, output_mlp, M, hidden_size, intermediate_size
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
