#include "deltanet.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// Turing sm_75 Fused Gated DeltaNet Decode Kernel
// 128x128 FP32 recurrent state matrix, 32x64 cooperative sub-tiling
// SMEM consumption: 2 * 32 * 64 * 4 = 16 KB (occupancy-preserving on Turing)
// Register budget: <= 48 registers/thread (hardware ceiling = 255)
// ----------------------------------------------------------------------------

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void deltanet_decode_sm75_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ beta,
    float* __restrict__ s_state,
    half* __restrict__ out,
    int num_heads,
    int head_dim
) {
    int h = blockIdx.x;
    int b = blockIdx.y;

    size_t head_offset = (static_cast<size_t>(b) * num_heads + h) * head_dim;
    size_t state_head_offset = (static_cast<size_t>(b) * num_heads + h) * 128 * 128;

    const half* q_h = q + head_offset;
    const half* k_h = k + head_offset;
    const half* v_h = v + head_offset;
    float* s_h = s_state + state_head_offset;
    half* out_h = out + head_offset;

    // Decay factor alpha = exp(-softplus(beta)) = 1 / (1 + exp(beta))
    float alpha = 1.0f;
    if (beta != nullptr) {
        float beta_val = __half2float(beta[b * num_heads + h]);
        alpha = 1.0f / (1.0f + expf(beta_val));
    }

    // Double-buffered shared memory sub-tile: 16 KB total
    __shared__ float smem_subtile[2][32][64];
    __shared__ float acc_out[128];

    int tid = threadIdx.x; // 0 .. 127
    int warp_id = tid / 32; // 0 .. 3
    int lane_id = tid % 32; // 0 .. 31

    // Initialize accumulator for out vector
    if (tid < 128) {
        acc_out[tid] = 0.0f;
    }
    __syncthreads();

    // Iterate over 4 row tiles (each 32 rows)
    for (int r_tile = 0; r_tile < 4; ++r_tile) {
        int r_base = r_tile * 32;

        // Iterate over 2 column tiles (each 64 columns)
        for (int c_tile = 0; c_tile < 2; ++c_tile) {
            int c_base = c_tile * 64;
            int buf_idx = c_tile % 2;

            // 128 threads load 32x64 = 2048 elements (16 elements per thread)
            #pragma unroll
            for (int elem = 0; elem < 16; ++elem) {
                int flat_idx = tid * 16 + elem;
                int r_sub = flat_idx / 64;
                int c_sub = flat_idx % 64;

                int g_row = r_base + r_sub;
                int g_col = c_base + c_sub;

                size_t g_idx = static_cast<size_t>(g_row) * 128 + g_col;
                float old_s = s_h[g_idx];

                float v_elem = __half2float(v_h[g_row]);
                float k_elem = __half2float(k_h[g_col]);

                // Outer product accumulation: S_t = alpha * S_{t-1} + v_t (x) k_t
                float new_s = alpha * old_s + v_elem * k_elem;

                // Persist updated recurrent state back in FP32
                s_h[g_idx] = new_s;
                smem_subtile[buf_idx][r_sub][c_sub] = new_s;
            }

            __syncthreads();

            // Compute matrix-vector product: o_t = S_t * q_t
            // 4 warps process 32 rows (8 rows per warp)
            int row_in_sub = warp_id * 8;
            for (int r = 0; r < 8; ++r) {
                int cur_row = row_in_sub + r;
                // Each lane processes 2 elements of the 64 columns
                int c0 = lane_id;
                int c1 = lane_id + 32;

                float q0 = __half2float(q_h[c_base + c0]);
                float q1 = __half2float(q_h[c_base + c1]);

                float prod = smem_subtile[buf_idx][cur_row][c0] * q0 +
                             smem_subtile[buf_idx][cur_row][c1] * q1;

                float sum_warp = warp_reduce_sum(prod);
                if (lane_id == 0) {
                    acc_out[r_base + cur_row] += sum_warp;
                }
            }

            __syncthreads();
        }
    }

    // Write final accumulated outputs in FP16
    if (tid < 128) {
        out_h[tid] = __float2half(acc_out[tid]);
    }
}

__global__ void depthwise_conv1d_step_kernel(
    const half* __restrict__ in,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ conv_state,
    half* __restrict__ out,
    int num_channels,
    int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    if (weight == nullptr) {
        out[c] = in[c];
        return;
    }

    int history_len = kernel_size - 1; // typically 3
    half* state_c = conv_state + static_cast<size_t>(c) * history_len;
    const half* weight_c = weight + static_cast<size_t>(c) * kernel_size;

    float acc = __half2float(in[c]) * __half2float(weight_c[0]);

    for (int k = 0; k < history_len; ++k) {
        acc += __half2float(state_c[k]) * __half2float(weight_c[k + 1]);
    }

    if (bias != nullptr) {
        acc += __half2float(bias[c]);
    }

    // State rolling update
    for (int k = history_len - 1; k > 0; --k) {
        state_c[k] = state_c[k - 1];
    }
    state_c[0] = in[c];

    // SiLU activation
    float silu_val = acc / (1.0f + expf(-acc));
    out[c] = __float2half(silu_val);
}

void launch_deltanet_decode_sm75(
    const half* q,
    const half* k,
    const half* v,
    const half* beta,
    float* s_state,
    half* out,
    int batch_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid(num_heads, batch_size);
    dim3 block(128); // 4 warps
    deltanet_decode_sm75_kernel<<<grid, block, 0, stream>>>(
        q, k, v, beta, s_state, out, num_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void launch_depthwise_conv1d_step(
    const half* in,
    const half* weight,
    const half* bias,
    half* conv_state,
    half* out,
    int num_channels,
    int kernel_size,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_channels + block - 1) / block;
    depthwise_conv1d_step_kernel<<<grid, block, 0, stream>>>(
        in, weight, bias, conv_state, out, num_channels, kernel_size
    );
    CUDA_KERNEL_CHECK();
}

void launch_deltanet_step(
    const half* input,
    const half* conv1d_weight,
    const half* conv1d_bias,
    half* conv_state,
    const half* q,
    const half* k,
    const half* v,
    const half* beta,
    float* s_state,
    half* out,
    int hidden_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    if (conv1d_weight != nullptr) {
        launch_depthwise_conv1d_step(
            input, conv1d_weight, conv1d_bias, conv_state,
            const_cast<half*>(input), hidden_size, 4, stream
        );
    }
    launch_deltanet_decode_sm75(
        q, k, v, beta, s_state, out, 1, num_heads, head_dim, stream
    );
}

} // namespace qwen

