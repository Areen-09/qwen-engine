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
// Total size: 10,240 bytes (10.0 KB, well within 48 KB sm_86 limit)
// ----------------------------------------------------------------------------
union PrefillSmem {
    struct {
        alignas(16) half s_A[PREFILL_WMMA_BM][PREFILL_SMEM_STRIDE]; // [64][40] = 5,120 B
        alignas(16) half s_B[PREFILL_WMMA_BN][PREFILL_SMEM_STRIDE]; // [64][40] = 5,120 B
    } mainloop;                                                      // 10,240 B
    alignas(16) float s_C[4][16][18];                                // [4][16][18] = 4,608 B
};

// ----------------------------------------------------------------------------
// FP16 2D-Tiled Tensor Core Prefill GEMM Kernel
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_fp16_kernel(
    const half* __restrict__ weight,
    const half* __restrict__ input_x,
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int M,
    int N,
    int K
) {
    __shared__ PrefillSmem smem;

    int n_block = blockIdx.x * PREFILL_WMMA_BN; // 64 cols
    int m_block = blockIdx.y * PREFILL_WMMA_BM; // 64 rows

    int tid = threadIdx.x; // 0 .. 127
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / 2; // 0 or 1
    int warp_n = warp_id % 2; // 0 or 1

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    int num_kt = K / PREFILL_WMMA_BK; // K / 32

    // Mainloop over reduction dimension K
    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        // Cooperative load s_A [64, 32]: 128 threads load 2048 elements (16 per thread)
        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        // Cooperative load s_B [64, 32]: row-major in weight matrix
        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int global_n = n_block + b_row;

        if (global_n < N) {
            const uint4* w_u4_0 = reinterpret_cast<const uint4*>(weight + static_cast<size_t>(global_n) * K + k_offset + b_col);
            const uint4* w_u4_1 = reinterpret_cast<const uint4*>(weight + static_cast<size_t>(global_n) * K + k_offset + b_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *w_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *w_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        // WMMA computation for k_sub = 0, 16
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
        __syncthreads();
    }

    // Explicit barrier before aliasing smem.s_C in epilogue
    __syncthreads();

    // Epilogue: store accumulators and write output
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
                        val += __half2float(residual[out_idx]);
                    }
                    output_y[out_idx] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// W4A16 2D-Tiled Tensor Core Prefill GEMM Kernel (on-the-fly dequantization)
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

    int num_kt = K / PREFILL_WMMA_BK; // K / 32
    int num_groups_per_row = K / QUANT_GROUP_SIZE; // K / 128

    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        // Load s_A
        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        // Load and dequantize s_B on the fly
        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int global_n = n_block + b_row;

        if (global_n < N) {
            int word_idx0 = kt * 4 + (b_col / 8);
            int word_idx1 = kt * 4 + ((b_col + 8) / 8);
            int group_idx0 = (k_offset + b_col) / QUANT_GROUP_SIZE;
            int group_idx1 = (k_offset + b_col + 8) / QUANT_GROUP_SIZE;

            half sc0 = scales[static_cast<size_t>(global_n) * num_groups_per_row + group_idx0];
            half sc1 = scales[static_cast<size_t>(global_n) * num_groups_per_row + group_idx1];

            uint32_t p0 = qweight[static_cast<size_t>(global_n) * (K / 8) + word_idx0];
            uint32_t p1 = qweight[static_cast<size_t>(global_n) * (K / 8) + word_idx1];

            half2 h2_0[4], h2_1[4];
            unpack_w4_to_half2_quad(p0, sc0, h2_0);
            unpack_w4_to_half2_quad(p1, sc1, h2_1);

            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *reinterpret_cast<uint4*>(h2_0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *reinterpret_cast<uint4*>(h2_1);
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

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
        __syncthreads();
    }

    __syncthreads();

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
                        val += __half2float(residual[out_idx]);
                    }
                    output_y[out_idx] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// Single-Launch Merged QKV Tensor Core Prefill GEMM (FP16)
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_qkv_fp16_kernel(
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

    // Resolve projection target based on n_block
    const half* weight = nullptr;
    half* output = nullptr;
    int col_offset = 0;
    int out_stride = 0;

    if (n_block < q_dim) {
        weight = reinterpret_cast<const half*>(view.q_weight);
        output = view.d_q;
        col_offset = 0;
        out_stride = q_dim;
    } else if (n_block < q_dim + kv_dim) {
        weight = reinterpret_cast<const half*>(view.k_weight);
        output = view.d_k;
        col_offset = q_dim;
        out_stride = kv_dim;
    } else {
        weight = reinterpret_cast<const half*>(view.v_weight);
        output = view.d_v;
        col_offset = q_dim + kv_dim;
        out_stride = kv_dim;
    }

    int num_kt = K / PREFILL_WMMA_BK;

    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int r = (n_block - col_offset) + b_row;

        const uint4* w_u4_0 = reinterpret_cast<const uint4*>(weight + static_cast<size_t>(r) * K + k_offset + b_col);
        const uint4* w_u4_1 = reinterpret_cast<const uint4*>(weight + static_cast<size_t>(r) * K + k_offset + b_col + 8);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *w_u4_0;
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *w_u4_1;
        __syncthreads();

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
        __syncthreads();
    }

    __syncthreads();

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int r = lane_id / 16; r < 16; r += 2) {
                int c = lane_id % 16;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                int rel_col = (n_block - col_offset) + warp_n * 32 + ni * 16 + c;

                if (global_row < M) {
                    float val = smem.s_C[warp_id][r][c];
                    output[static_cast<size_t>(global_row) * out_stride + rel_col] = __float2half(val);
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

    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int r = (n_block - col_offset) + b_row;

        int word_idx0 = kt * 4 + (b_col / 8);
        int word_idx1 = kt * 4 + ((b_col + 8) / 8);
        int group_idx0 = (k_offset + b_col) / QUANT_GROUP_SIZE;
        int group_idx1 = (k_offset + b_col + 8) / QUANT_GROUP_SIZE;

        half sc0 = scales[static_cast<size_t>(r) * num_groups + group_idx0];
        half sc1 = scales[static_cast<size_t>(r) * num_groups + group_idx1];

        uint32_t p0 = qweight[static_cast<size_t>(r) * (K / 8) + word_idx0];
        uint32_t p1 = qweight[static_cast<size_t>(r) * (K / 8) + word_idx1];

        half2 h2_0[4], h2_1[4];
        unpack_w4_to_half2_quad(p0, sc0, h2_0);
        unpack_w4_to_half2_quad(p1, sc1, h2_1);

        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *reinterpret_cast<uint4*>(h2_0);
        *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *reinterpret_cast<uint4*>(h2_1);
        __syncthreads();

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
        __syncthreads();
    }

    __syncthreads();

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int r = lane_id / 16; r < 16; r += 2) {
                int c = lane_id % 16;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                int rel_col = (n_block - col_offset) + warp_n * 32 + ni * 16 + c;

                if (global_row < M) {
                    float val = smem.s_C[warp_id][r][c];
                    output[static_cast<size_t>(global_row) * out_stride + rel_col] = __float2half(val);
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------------
// Fused Dual Gate+Up GEMM + SwiGLU Tensor Core Prefill Kernel (FP16)
// ----------------------------------------------------------------------------

__global__ void gemm_prefill_dual_gate_up_swiglu_fp16_kernel(
    const half* __restrict__ gate_up_weight,
    const half* __restrict__ input_x,
    half* __restrict__ output_mlp,
    int M,
    int hidden_size,
    int intermediate_size
) {
    int K = hidden_size;
    int total_n = 2 * intermediate_size;

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

    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int global_n = n_block + b_row;

        if (global_n < total_n) {
            const uint4* w_u4_0 = reinterpret_cast<const uint4*>(gate_up_weight + static_cast<size_t>(global_n) * K + k_offset + b_col);
            const uint4* w_u4_1 = reinterpret_cast<const uint4*>(gate_up_weight + static_cast<size_t>(global_n) * K + k_offset + b_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *w_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *w_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

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
        __syncthreads();
    }

    __syncthreads();

    // SwiGLU activation epilogue
    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            int n_sub_base = n_block + warp_n * 32 + ni * 16;

            #pragma unroll
            for (int r = lane_id / 8; r < 16; r += 4) {
                int p = lane_id % 8; // 8 pairs across 16 cols
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

    for (int kt = 0; kt < num_kt; ++kt) {
        int k_offset = kt * PREFILL_WMMA_BK;

        int a_row = tid / 2;
        int a_col = (tid % 2) * 16;
        int global_m = m_block + a_row;

        if (global_m < M) {
            const uint4* in_u4_0 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col);
            const uint4* in_u4_1 = reinterpret_cast<const uint4*>(input_x + static_cast<size_t>(global_m) * K + k_offset + a_col + 8);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = *in_u4_0;
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = *in_u4_1;
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_A[a_row][a_col + 8]) = make_uint4(0, 0, 0, 0);
        }

        int b_row = tid / 2;
        int b_col = (tid % 2) * 16;
        int global_n = n_block + b_row;

        if (global_n < total_n) {
            int word_idx0 = kt * 4 + (b_col / 8);
            int word_idx1 = kt * 4 + ((b_col + 8) / 8);
            int group_idx0 = (k_offset + b_col) / QUANT_GROUP_SIZE;
            int group_idx1 = (k_offset + b_col + 8) / QUANT_GROUP_SIZE;

            half sc0 = gate_up_scales[static_cast<size_t>(global_n) * num_groups + group_idx0];
            half sc1 = gate_up_scales[static_cast<size_t>(global_n) * num_groups + group_idx1];

            uint32_t p0 = gate_up_qweight[static_cast<size_t>(global_n) * (K / 8) + word_idx0];
            uint32_t p1 = gate_up_qweight[static_cast<size_t>(global_n) * (K / 8) + word_idx1];

            half2 h2_0[4], h2_1[4];
            unpack_w4_to_half2_quad(p0, sc0, h2_0);
            unpack_w4_to_half2_quad(p1, sc1, h2_1);

            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = *reinterpret_cast<uint4*>(h2_0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = *reinterpret_cast<uint4*>(h2_1);
        } else {
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col])     = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&smem.mainloop.s_B[b_row][b_col + 8]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

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
        __syncthreads();
    }

    __syncthreads();

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

void launch_gemm_prefill_fp16(
    const half* weight,
    const half* input_x,
    const half* residual,
    half* output_y,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    dim3 grid((N + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
    gemm_prefill_fp16_kernel<<<grid, 128, 0, stream>>>(
        weight, input_x, residual, output_y, M, N, K
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemm_prefill_qkv_fp16(
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
    gemm_prefill_qkv_fp16_kernel<<<grid, 128, 0, stream>>>(
        view, input_x, M, in_features, num_q_heads, num_kv_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemm_prefill_dual_gate_up_swiglu_fp16(
    const half* gate_up_weight,
    const half* input_x,
    half* output_mlp,
    int M,
    cudaStream_t stream,
    int hidden_size,
    int intermediate_size
) {
    int total_n = 2 * intermediate_size;
    dim3 grid((total_n + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
    gemm_prefill_dual_gate_up_swiglu_fp16_kernel<<<grid, 128, 0, stream>>>(
        gate_up_weight, input_x, output_mlp, M, hidden_size, intermediate_size
    );
    CUDA_KERNEL_CHECK();
}

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
