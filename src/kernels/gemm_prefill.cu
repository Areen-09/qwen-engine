#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "quant_primitives.cuh"
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

namespace qwen {

using namespace nvcuda;

// ----------------------------------------------------------------------------
// Shared Memory Union for Double-Buffered 2D-Tiled Tensor Core Prefill GEMM
// 64x64: 2 * (64*40 + 64*40) * 2 B = 20,480 B (3 blocks/SM on sm_75, 4-5 on sm_86)
// 32x32: 2 * (32*40 + 32*40) * 2 B = 10,240 B (6 blocks/SM on sm_75, 9 on sm_86)
// ----------------------------------------------------------------------------
template <int BM, int BN>
union PrefillSmem {
    struct {
        alignas(16) half s_A[BM][PREFILL_SMEM_STRIDE];
        alignas(16) half s_B[BN][PREFILL_SMEM_STRIDE];
    } mainloop[PREFILL_SMEM_STAGES];
    alignas(16) float s_C[4][16][18];
};

namespace {

template <int DIM>
struct TileRegBuf;

template <>
struct TileRegBuf<64> {
    uint4 v[2];
};

template <>
struct TileRegBuf<32> {
    uint4 v[1];
};

template <int BM>
__device__ __forceinline__ void load_a_to_reg(
    TileRegBuf<BM>& reg,
    const half* __restrict__ input_x,
    int m_block,
    int k_offset,
    int M,
    int K,
    int tid
) {
    if constexpr (BM == 64) {
        int a_row = tid >> 1;
        int a_col = (tid & 1) * 16;
        int global_m = m_block + a_row;
        if (global_m < M) {
            const uint4* in_u4 = reinterpret_cast<const uint4*>(
                input_x + static_cast<size_t>(global_m) * K + k_offset + a_col
            );
            reg.v[0] = __ldg(&in_u4[0]);
            reg.v[1] = __ldg(&in_u4[1]);
        } else {
            reg.v[0] = make_uint4(0, 0, 0, 0);
            reg.v[1] = make_uint4(0, 0, 0, 0);
        }
    } else {
        static_assert(BM == 32, "Supported BM are 32 and 64");
        int a_row = tid >> 2;
        int a_col = (tid & 3) * 8;
        int global_m = m_block + a_row;
        if (global_m < M) {
            const uint4* in_u4 = reinterpret_cast<const uint4*>(
                input_x + static_cast<size_t>(global_m) * K + k_offset + a_col
            );
            reg.v[0] = __ldg(&in_u4[0]);
        } else {
            reg.v[0] = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int BM, int BN>
__device__ __forceinline__ void commit_a_from_reg(
    PrefillSmem<BM, BN>& smem,
    int stage,
    const TileRegBuf<BM>& reg,
    int tid
) {
    if constexpr (BM == 64) {
        int a_row = tid >> 1;
        int a_col = (tid & 1) * 16;
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_A[a_row][a_col])     = reg.v[0];
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_A[a_row][a_col + 8]) = reg.v[1];
    } else {
        int a_row = tid >> 2;
        int a_col = (tid & 3) * 8;
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_A[a_row][a_col]) = reg.v[0];
    }
}

template <int BN>
__device__ __forceinline__ void dequant_b_to_reg(
    TileRegBuf<BN>& reg,
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    int local_n_base,
    int max_n,
    int kt,
    int K,
    int num_groups_per_row,
    int tid
) {
    if constexpr (BN == 64) {
        int b_row = tid >> 1;
        int b_col = (tid & 1) * 16;
        int local_n = local_n_base + b_row;
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
            reg.v[0] = *reinterpret_cast<const uint4*>(h2_0);
            reg.v[1] = *reinterpret_cast<const uint4*>(h2_1);
        } else {
            reg.v[0] = make_uint4(0, 0, 0, 0);
            reg.v[1] = make_uint4(0, 0, 0, 0);
        }
    } else {
        static_assert(BN == 32, "Supported BN are 32 and 64");
        int b_row = tid >> 2;
        int b_col = (tid & 3) * 8;
        int local_n = local_n_base + b_row;
        if (local_n < max_n) {
            int word_idx = kt * 4 + (b_col >> 3);
            int group_idx = kt >> 2;
            half sc = __ldg(&scales[static_cast<size_t>(local_n) * num_groups_per_row + group_idx]);
            const uint32_t* w_ptr = qweight + static_cast<size_t>(local_n) * (K / 8) + word_idx;
            uint32_t p0 = __ldg(&w_ptr[0]);

            half2 h2_0[4];
            unpack_w4_to_half2_quad(p0, sc, h2_0);
            reg.v[0] = *reinterpret_cast<const uint4*>(h2_0);
        } else {
            reg.v[0] = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int BM, int BN>
__device__ __forceinline__ void commit_b_from_reg(
    PrefillSmem<BM, BN>& smem,
    int stage,
    const TileRegBuf<BN>& reg,
    int tid
) {
    if constexpr (BN == 64) {
        int b_row = tid >> 1;
        int b_col = (tid & 1) * 16;
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_B[b_row][b_col])     = reg.v[0];
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_B[b_row][b_col + 8]) = reg.v[1];
    } else {
        int b_row = tid >> 2;
        int b_col = (tid & 3) * 8;
        *reinterpret_cast<uint4*>(&smem.mainloop[stage].s_B[b_row][b_col]) = reg.v[0];
    }
}

template <int BM, int BN>
__device__ __forceinline__ void wmma_step(
    const PrefillSmem<BM, BN>& smem,
    int stage,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&acc)[BM / 32][BN / 32],
    int warp_m,
    int warp_n
) {
    constexpr int WM = BM / 32;
    constexpr int WN = BN / 32;

    #pragma unroll
    for (int k_sub = 0; k_sub < PREFILL_WMMA_BK; k_sub += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[WM];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[WN];

        #pragma unroll
        for (int mi = 0; mi < WM; ++mi) {
            wmma::load_matrix_sync(
                frag_a[mi],
                &smem.mainloop[stage].s_A[warp_m * (WM * 16) + mi * 16][k_sub],
                PREFILL_SMEM_STRIDE
            );
        }
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            wmma::load_matrix_sync(
                frag_b[ni],
                &smem.mainloop[stage].s_B[warp_n * (WN * 16) + ni * 16][k_sub],
                PREFILL_SMEM_STRIDE
            );
        }

        #pragma unroll
        for (int mi = 0; mi < WM; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < WN; ++ni) {
                wmma::mma_sync(acc[mi][ni], frag_a[mi], frag_b[ni], acc[mi][ni]);
            }
        }
    }
}

template <int BM, int BN>
__device__ __forceinline__ void store_epilogue_tile(
    PrefillSmem<BM, BN>& smem,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&acc)[BM / 32][BN / 32],
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int m_block,
    int n_out_offset,
    int M,
    int out_stride,
    int warp_id,
    int lane_id,
    int warp_m,
    int warp_n
) {
    constexpr int WM = BM / 32;
    constexpr int WN = BN / 32;

    bool can_u4 = ((out_stride & 7) == 0) &&
                  ((reinterpret_cast<uintptr_t>(output_y) & 15) == 0) &&
                  (residual == nullptr || (reinterpret_cast<uintptr_t>(residual) & 15) == 0);
    bool can_h2 = ((out_stride & 1) == 0) &&
                  ((reinterpret_cast<uintptr_t>(output_y) & 3) == 0) &&
                  (residual == nullptr || (reinterpret_cast<uintptr_t>(residual) & 3) == 0);

    int r = lane_id >> 1;          // 0..15
    int c_base = (lane_id & 1) * 8; // 0 or 8

    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            int global_row = m_block + warp_m * (WM * 16) + mi * 16 + r;
            int global_col = n_out_offset + warp_n * (WN * 16) + ni * 16 + c_base;

            if (global_row < M && global_col < out_stride) {
                size_t row_off = static_cast<size_t>(global_row) * out_stride + global_col;
                const float* s_ptr = &smem.s_C[warp_id][r][c_base];

                if (can_u4 && (global_col + 8 <= out_stride)) {
                    half2 h2[4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        h2[k] = __float22half2_rn(make_float2(s_ptr[2 * k], s_ptr[2 * k + 1]));
                    }
                    if (residual != nullptr) {
                        uint4 res_u4 = __ldg(reinterpret_cast<const uint4*>(&residual[row_off]));
                        const half2* res_h2 = reinterpret_cast<const half2*>(&res_u4);
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            h2[k] = __hadd2(h2[k], res_h2[k]);
                        }
                    }
                    *reinterpret_cast<uint4*>(&output_y[row_off]) = *reinterpret_cast<const uint4*>(h2);
                } else if (can_h2 && (global_col + 8 <= out_stride)) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        half2 h = __float22half2_rn(make_float2(s_ptr[2 * k], s_ptr[2 * k + 1]));
                        if (residual != nullptr) {
                            half2 res_h = __ldg(reinterpret_cast<const half2*>(&residual[row_off + 2 * k]));
                            h = __hadd2(h, res_h);
                        }
                        *reinterpret_cast<half2*>(&output_y[row_off + 2 * k]) = h;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        if (global_col + k < out_stride) {
                            float val = s_ptr[k];
                            if (residual != nullptr) {
                                val += __half2float(__ldg(&residual[row_off + k]));
                            }
                            output_y[row_off + k] = __float2half(val);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

inline bool select_small_tile(int M, int N, int sm_count) {
    if (M <= 128 && sm_count > 0) {
        int blocks_64 = ((M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM) *
                        ((N + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN);
        return blocks_64 < 2 * sm_count;
    }
    return false;
}

} // namespace

// ----------------------------------------------------------------------------
// W4A16 2D-Tiled Double-Buffered Tensor Core Prefill GEMM Kernel
// ----------------------------------------------------------------------------

template <int BM, int BN>
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
    __shared__ PrefillSmem<BM, BN> smem;

    int n_block = blockIdx.x * BN;
    int m_block = blockIdx.y * BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    constexpr int WM = BM / 32;
    constexpr int WN = BN / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    int num_kt = K / PREFILL_WMMA_BK;
    int num_groups_per_row = K / QUANT_GROUP_SIZE;

    TileRegBuf<BM> a_reg;
    TileRegBuf<BN> b_reg;

    load_a_to_reg<BM>(a_reg, input_x, m_block, 0, M, K, tid);
    dequant_b_to_reg<BN>(b_reg, qweight, scales, n_block, N, 0, K, num_groups_per_row, tid);
    commit_a_from_reg<BM, BN>(smem, 0, a_reg, tid);
    commit_b_from_reg<BM, BN>(smem, 0, b_reg, tid);
    __syncthreads();

    for (int kt = 0; kt < num_kt - 1; ++kt) {
        int cur_stage  = kt & 1;
        int next_stage = cur_stage ^ 1;

        load_a_to_reg<BM>(a_reg, input_x, m_block, (kt + 1) * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_to_reg<BN>(b_reg, qweight, scales, n_block, N, kt + 1, K, num_groups_per_row, tid);

        wmma_step<BM, BN>(smem, cur_stage, acc, warp_m, warp_n);

        commit_a_from_reg<BM, BN>(smem, next_stage, a_reg, tid);
        commit_b_from_reg<BM, BN>(smem, next_stage, b_reg, tid);
        __syncthreads();
    }

    wmma_step<BM, BN>(smem, (num_kt - 1) & 1, acc, warp_m, warp_n);
    __syncthreads();

    store_epilogue_tile<BM, BN>(
        smem, acc, residual, output_y, m_block, n_block, M, N,
        warp_id, lane_id, warp_m, warp_n
    );
}

// ----------------------------------------------------------------------------
// Single-Launch Merged QKV Double-Buffered Tensor Core Prefill GEMM (W4A16)
// ----------------------------------------------------------------------------

template <int BM, int BN>
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

    __shared__ PrefillSmem<BM, BN> smem;

    int n_block = blockIdx.x * BN;
    int m_block = blockIdx.y * BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    constexpr int WM = BM / 32;
    constexpr int WN = BN / 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
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
    int local_n_base = n_block - col_offset;

    TileRegBuf<BM> a_reg;
    TileRegBuf<BN> b_reg;

    load_a_to_reg<BM>(a_reg, input_x, m_block, 0, M, K, tid);
    dequant_b_to_reg<BN>(b_reg, qweight, scales, local_n_base, out_stride, 0, K, num_groups, tid);
    commit_a_from_reg<BM, BN>(smem, 0, a_reg, tid);
    commit_b_from_reg<BM, BN>(smem, 0, b_reg, tid);
    __syncthreads();

    for (int kt = 0; kt < num_kt - 1; ++kt) {
        int cur_stage  = kt & 1;
        int next_stage = cur_stage ^ 1;

        load_a_to_reg<BM>(a_reg, input_x, m_block, (kt + 1) * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_to_reg<BN>(b_reg, qweight, scales, local_n_base, out_stride, kt + 1, K, num_groups, tid);

        wmma_step<BM, BN>(smem, cur_stage, acc, warp_m, warp_n);

        commit_a_from_reg<BM, BN>(smem, next_stage, a_reg, tid);
        commit_b_from_reg<BM, BN>(smem, next_stage, b_reg, tid);
        __syncthreads();
    }

    wmma_step<BM, BN>(smem, (num_kt - 1) & 1, acc, warp_m, warp_n);
    __syncthreads();

    store_epilogue_tile<BM, BN>(
        smem, acc, nullptr, output, m_block, local_n_base, M, out_stride,
        warp_id, lane_id, warp_m, warp_n
    );
}

// ----------------------------------------------------------------------------
// Fused Dual Gate+Up GEMM + SwiGLU Double-Buffered Prefill Kernel (W4A16)
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
    constexpr int BM = PREFILL_WMMA_BM;
    constexpr int BN = PREFILL_WMMA_BN;

    int K = hidden_size;
    int total_n = 2 * intermediate_size;
    int num_groups = K / QUANT_GROUP_SIZE;

    __shared__ PrefillSmem<BM, BN> smem;

    int n_block = blockIdx.x * BN;
    int m_block = blockIdx.y * BM;

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

    TileRegBuf<BM> a_reg;
    TileRegBuf<BN> b_reg;

    load_a_to_reg<BM>(a_reg, input_x, m_block, 0, M, K, tid);
    dequant_b_to_reg<BN>(b_reg, gate_up_qweight, gate_up_scales, n_block, total_n, 0, K, num_groups, tid);
    commit_a_from_reg<BM, BN>(smem, 0, a_reg, tid);
    commit_b_from_reg<BM, BN>(smem, 0, b_reg, tid);
    __syncthreads();

    for (int kt = 0; kt < num_kt - 1; ++kt) {
        int cur_stage  = kt & 1;
        int next_stage = cur_stage ^ 1;

        load_a_to_reg<BM>(a_reg, input_x, m_block, (kt + 1) * PREFILL_WMMA_BK, M, K, tid);
        dequant_b_to_reg<BN>(b_reg, gate_up_qweight, gate_up_scales, n_block, total_n, kt + 1, K, num_groups, tid);

        wmma_step<BM, BN>(smem, cur_stage, acc, warp_m, warp_n);

        commit_a_from_reg<BM, BN>(smem, next_stage, a_reg, tid);
        commit_b_from_reg<BM, BN>(smem, next_stage, b_reg, tid);
        __syncthreads();
    }

    wmma_step<BM, BN>(smem, (num_kt - 1) & 1, acc, warp_m, warp_n);
    __syncthreads();

    bool can_u4 = ((intermediate_size & 7) == 0) &&
                  ((reinterpret_cast<uintptr_t>(output_mlp) & 15) == 0);

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::store_matrix_sync(&smem.s_C[warp_id][0][0], acc[mi][ni], 18, wmma::mem_row_major);
            __syncwarp();

            int n_sub_base = n_block + warp_n * 32 + ni * 16;
            int feat_base = n_sub_base / 2; // 8 SwiGLU outputs per 16 gate/up columns

            if (can_u4 && feat_base + 8 <= intermediate_size) {
                // 32 threads: 2 threads per row, only even lane computes all 8 outputs for coalesced uint4 write
                int r = lane_id >> 1;
                int global_row = m_block + warp_m * 32 + mi * 16 + r;
                if ((lane_id & 1) == 0 && global_row < M) {
                    half2 out_h2[4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        float g0 = smem.s_C[warp_id][r][4 * k + 0];
                        float u0 = smem.s_C[warp_id][r][4 * k + 1];
                        float g1 = smem.s_C[warp_id][r][4 * k + 2];
                        float u1 = smem.s_C[warp_id][r][4 * k + 3];
                        float s0 = (g0 / (1.0f + expf(-g0))) * u0;
                        float s1 = (g1 / (1.0f + expf(-g1))) * u1;
                        out_h2[k] = __float22half2_rn(make_float2(s0, s1));
                    }
                    *reinterpret_cast<uint4*>(
                        &output_mlp[static_cast<size_t>(global_row) * intermediate_size + feat_base]
                    ) = *reinterpret_cast<const uint4*>(out_h2);
                }
            } else {
                #pragma unroll
                for (int r = lane_id / 8; r < 16; r += 4) {
                    int p = lane_id % 8;
                    int global_row = m_block + warp_m * 32 + mi * 16 + r;
                    int feat_idx = feat_base + p;

                    if (global_row < M && feat_idx < intermediate_size) {
                        float gate = smem.s_C[warp_id][r][2 * p + 0];
                        float up   = smem.s_C[warp_id][r][2 * p + 1];
                        float silu = gate / (1.0f + expf(-gate));
                        output_mlp[static_cast<size_t>(global_row) * intermediate_size + feat_idx] = __float2half(silu * up);
                    }
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
    cudaStream_t stream,
    int sm_count
) {
    if (select_small_tile(M, N, sm_count)) {
        dim3 grid((N + PREFILL_SMALL_BN - 1) / PREFILL_SMALL_BN, (M + PREFILL_SMALL_BM - 1) / PREFILL_SMALL_BM);
        gemm_prefill_w4a16_kernel<PREFILL_SMALL_BM, PREFILL_SMALL_BN><<<grid, 128, 0, stream>>>(
            qweight, scales, input_x, residual, output_y, M, N, K
        );
    } else {
        dim3 grid((N + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
        gemm_prefill_w4a16_kernel<PREFILL_WMMA_BM, PREFILL_WMMA_BN><<<grid, 128, 0, stream>>>(
            qweight, scales, input_x, residual, output_y, M, N, K
        );
    }
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
    int head_dim,
    int sm_count
) {
    int total_n = (num_q_heads + 2 * num_kv_heads) * head_dim;
    if (select_small_tile(M, total_n, sm_count)) {
        dim3 grid((total_n + PREFILL_SMALL_BN - 1) / PREFILL_SMALL_BN, (M + PREFILL_SMALL_BM - 1) / PREFILL_SMALL_BM);
        gemm_prefill_qkv_w4a16_kernel<PREFILL_SMALL_BM, PREFILL_SMALL_BN><<<grid, 128, 0, stream>>>(
            view, input_x, M, in_features, num_q_heads, num_kv_heads, head_dim
        );
    } else {
        dim3 grid((total_n + PREFILL_WMMA_BN - 1) / PREFILL_WMMA_BN, (M + PREFILL_WMMA_BM - 1) / PREFILL_WMMA_BM);
        gemm_prefill_qkv_w4a16_kernel<PREFILL_WMMA_BM, PREFILL_WMMA_BN><<<grid, 128, 0, stream>>>(
            view, input_x, M, in_features, num_q_heads, num_kv_heads, head_dim
        );
    }
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
