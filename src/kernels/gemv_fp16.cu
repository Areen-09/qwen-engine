#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// Vectorized FP16 GEMV Kernel (1 Warp per Output Row)
// ----------------------------------------------------------------------------

__global__ void gemv_fp16_kernel(
    const half* __restrict__ weight,
    const half* __restrict__ input_x,
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int out_features,
    int in_features
) {
    extern __shared__ half s_x[];

    // Cooperative loading of input_x into shared memory
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        s_x[i] = input_x[i];
    }
    __syncthreads();

    // 1 warp per output row
    int warp_id_in_block = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + warp_id_in_block;

    if (row >= out_features) return;

    const half* w_row = weight + static_cast<size_t>(row) * in_features;

    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_row);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(s_x);
    int num_u4 = in_features / 8;

    float acc = 0.0f;
    for (int i = lane_id; i < num_u4; i += 32) {
        uint4 w_val = w_u4[i];
        uint4 x_val = x_u4[i];

        const half2* w_h2 = reinterpret_cast<const half2*>(&w_val);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_val);

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 w_f2 = __half22float2(w_h2[p]);
            float2 x_f2 = __half22float2(x_h2[p]);
            acc += w_f2.x * x_f2.x + w_f2.y * x_f2.y;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        if (residual != nullptr) {
            acc += __half2float(residual[row]);
        }
        output_y[row] = __float2half(acc);
    }
}

// ----------------------------------------------------------------------------
// Single-Launch Virtual-Row QKV FP16 GEMV Kernel
// ----------------------------------------------------------------------------

__global__ void gemv_qkv_fp16_kernel(
    QKVProjectionView view,
    const half* __restrict__ input_x
) {
    extern __shared__ half s_x[];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += blockDim.x) {
        s_x[i] = input_x[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int row = blockIdx.x * warps_per_block + warp_id;

    constexpr int TOTAL_ROWS = Q_DIM + 2 * KV_DIM; // 2048
    if (row >= TOTAL_ROWS) return;

    const half* weight = nullptr;
    half* output = nullptr;
    int r = 0;

    if (row < Q_DIM) { // 0 .. 1535
        weight = reinterpret_cast<const half*>(view.q_weight);
        output = view.d_q;
        r = row;
    } else if (row < Q_DIM + KV_DIM) { // 1536 .. 1791
        weight = reinterpret_cast<const half*>(view.k_weight);
        output = view.d_k;
        r = row - Q_DIM;
    } else { // 1792 .. 2047
        weight = reinterpret_cast<const half*>(view.v_weight);
        output = view.d_v;
        r = row - (Q_DIM + KV_DIM);
    }

    const half* w_row = weight + static_cast<size_t>(r) * HIDDEN_SIZE;
    const uint4* w_u4 = reinterpret_cast<const uint4*>(w_row);
    const uint4* x_u4 = reinterpret_cast<const uint4*>(s_x);
    constexpr int NUM_U4 = HIDDEN_SIZE / 8; // 192

    float acc = 0.0f;
    for (int i = lane_id; i < NUM_U4; i += 32) {
        uint4 w_val = w_u4[i];
        uint4 x_val = x_u4[i];

        const half2* w_h2 = reinterpret_cast<const half2*>(&w_val);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_val);

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 w_f2 = __half22float2(w_h2[p]);
            float2 x_f2 = __half22float2(x_h2[p]);
            acc += w_f2.x * x_f2.x + w_f2.y * x_f2.y;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        output[r] = __float2half(acc);
    }
}

// ----------------------------------------------------------------------------
// Fused Dual Gate+Up GEMV + SwiGLU Kernel (FP16)
// ----------------------------------------------------------------------------

__global__ void gemv_dual_gate_up_swiglu_fp16_kernel(
    const half* __restrict__ gate_up_weight,
    const half* __restrict__ input_x,
    half* __restrict__ output_mlp,
    int in_features,
    int intermediate_size
) {
    extern __shared__ half s_x[];

    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        s_x[i] = input_x[i];
    }
    __syncthreads();

    int warp_id_in_block = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int r = blockIdx.x * warps_per_block + warp_id_in_block;

    if (r >= intermediate_size) return;

    const half* gate_row = gate_up_weight + (static_cast<size_t>(2 * r) + 0) * in_features;
    const half* up_row   = gate_up_weight + (static_cast<size_t>(2 * r) + 1) * in_features;

    const uint4* gate_u4 = reinterpret_cast<const uint4*>(gate_row);
    const uint4* up_u4   = reinterpret_cast<const uint4*>(up_row);
    const uint4* x_u4    = reinterpret_cast<const uint4*>(s_x);

    int num_u4 = in_features / 8;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (int i = lane_id; i < num_u4; i += 32) {
        uint4 g_val = gate_u4[i];
        uint4 u_val = up_u4[i];
        uint4 x_val = x_u4[i];

        const half2* g_h2 = reinterpret_cast<const half2*>(&g_val);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_val);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_val);

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 x_f2 = __half22float2(x_h2[p]);
            float2 g_f2 = __half22float2(g_h2[p]);
            float2 u_f2 = __half22float2(u_h2[p]);

            gate_acc += g_f2.x * x_f2.x + g_f2.y * x_f2.y;
            up_acc   += u_f2.x * x_f2.x + u_f2.y * x_f2.y;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        gate_acc += __shfl_down_sync(0xFFFFFFFF, gate_acc, offset);
        up_acc   += __shfl_down_sync(0xFFFFFFFF, up_acc, offset);
    }

    if (lane_id == 0) {
        float silu_gate = gate_acc / (1.0f + expf(-gate_acc));
        output_mlp[r] = __float2half(silu_gate * up_acc);
    }
}

void launch_gemv_fp16(
    const half* weight,
    const half* input_x,
    const half* residual,
    half* output_y,
    int out_features,
    int in_features,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    int grid_size = (out_features + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t smem_bytes = static_cast<size_t>(in_features) * sizeof(half);

    gemv_fp16_kernel<<<grid_size, BLOCK_THREADS, smem_bytes, stream>>>(
        weight, input_x, residual, output_y, out_features, in_features
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_qkv_fp16(
    const QKVProjectionView& view,
    const half* input_x,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int TOTAL_ROWS = Q_DIM + 2 * KV_DIM; // 2048
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    constexpr int GRID = TOTAL_ROWS / WARPS_PER_BLOCK; // 512
    size_t smem_bytes = static_cast<size_t>(HIDDEN_SIZE) * sizeof(half);

    gemv_qkv_fp16_kernel<<<GRID, BLOCK_THREADS, smem_bytes, stream>>>(
        view, input_x
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_dual_gate_up_swiglu_fp16(
    const half* gate_up_weight,
    const half* input_x,
    half* output_mlp,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    int grid_size = (intermediate_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t smem_bytes = static_cast<size_t>(hidden_size) * sizeof(half);

    gemv_dual_gate_up_swiglu_fp16_kernel<<<grid_size, BLOCK_THREADS, smem_bytes, stream>>>(
        gate_up_weight, input_x, output_mlp, hidden_size, intermediate_size
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
