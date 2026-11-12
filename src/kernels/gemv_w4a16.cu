#include "config.cuh"
#include "kernels.cuh"
#include "tensor.cuh"
#include "quant_primitives.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// 4-Row-per-Warp W4A8 (__dp4a) GEMV Kernel (decode)
// ----------------------------------------------------------------------------

__global__ void gemv_w4a8_kernel(
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const int8_t* __restrict__ input_q8,
    const float* __restrict__ input_scale,
    const int32_t* __restrict__ input_sum8,
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int out_features,
    int in_features
) {
    extern __shared__ alignas(16) uint8_t s_mem[];
    int8_t* s_x_q8 = reinterpret_cast<int8_t*>(s_mem);
    int32_t* s_x_sum8 = reinterpret_cast<int32_t*>(s_mem + in_features);
    float* s_x_scale = reinterpret_cast<float*>(s_mem + in_features + (in_features / 8) * sizeof(int32_t));

    // Cooperative shared memory load of pre-quantized Q8_1 activations
    int num_u4 = in_features / 16;
    const uint4* in_q8_u4 = reinterpret_cast<const uint4*>(input_q8);
    uint4* s_q8_u4 = reinterpret_cast<uint4*>(s_x_q8);
    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        s_q8_u4[i] = in_q8_u4[i];
    }
    int num_sum8 = in_features / 8;
    for (int i = threadIdx.x; i < num_sum8; i += blockDim.x) {
        s_x_sum8[i] = input_sum8[i];
    }
    int num_groups = in_features / QUANT_GROUP_SIZE;
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        s_x_scale[i] = input_scale[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int base_row = (blockIdx.x * 4 + warp_id) * 4;

    int num_warp_steps = in_features / 256;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_warp_steps; ++s) {
        int group_idx = 2 * s + (lane_id >> 4);
        int sub_lane = lane_id & 15;
        int sum8_idx = group_idx * 16 + sub_lane;

        int32_t x_sum8 = s_x_sum8[sum8_idx];
        const int2* x_q8_ptr = reinterpret_cast<const int2*>(s_x_q8 + group_idx * 128 + sub_lane * 8);
        int2 x_pair = *x_q8_ptr;
        int32_t x_lo = x_pair.x;
        int32_t x_hi = x_pair.y;
        float x_d = s_x_scale[group_idx];

        // Row 0
        if (base_row + 0 < out_features) {
            const uint32_t* row_qw0 = qweight + static_cast<size_t>(base_row + 0) * (in_features / 8);
            const half* row_scales0 = scales + static_cast<size_t>(base_row + 0) * num_groups;
            uint32_t p0 = row_qw0[s * 32 + lane_id];
            int32_t dot0 = dp4a_w4_q8_oct(p0, x_lo, x_hi, x_sum8);
            float w_d0 = __half2float(row_scales0[group_idx]);
            acc0 += static_cast<float>(dot0) * (w_d0 * x_d);
        }
        // Row 1
        if (base_row + 1 < out_features) {
            const uint32_t* row_qw1 = qweight + static_cast<size_t>(base_row + 1) * (in_features / 8);
            const half* row_scales1 = scales + static_cast<size_t>(base_row + 1) * num_groups;
            uint32_t p1 = row_qw1[s * 32 + lane_id];
            int32_t dot1 = dp4a_w4_q8_oct(p1, x_lo, x_hi, x_sum8);
            float w_d1 = __half2float(row_scales1[group_idx]);
            acc1 += static_cast<float>(dot1) * (w_d1 * x_d);
        }
        // Row 2
        if (base_row + 2 < out_features) {
            const uint32_t* row_qw2 = qweight + static_cast<size_t>(base_row + 2) * (in_features / 8);
            const half* row_scales2 = scales + static_cast<size_t>(base_row + 2) * num_groups;
            uint32_t p2 = row_qw2[s * 32 + lane_id];
            int32_t dot2 = dp4a_w4_q8_oct(p2, x_lo, x_hi, x_sum8);
            float w_d2 = __half2float(row_scales2[group_idx]);
            acc2 += static_cast<float>(dot2) * (w_d2 * x_d);
        }
        // Row 3
        if (base_row + 3 < out_features) {
            const uint32_t* row_qw3 = qweight + static_cast<size_t>(base_row + 3) * (in_features / 8);
            const half* row_scales3 = scales + static_cast<size_t>(base_row + 3) * num_groups;
            uint32_t p3 = row_qw3[s * 32 + lane_id];
            int32_t dot3 = dp4a_w4_q8_oct(p3, x_lo, x_hi, x_sum8);
            float w_d3 = __half2float(row_scales3[group_idx]);
            acc3 += static_cast<float>(dot3) * (w_d3 * x_d);
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        acc3 += __shfl_down_sync(0xFFFFFFFF, acc3, offset);
    }

    if (lane_id == 0) {
        if (base_row + 0 < out_features) {
            if (residual != nullptr) acc0 += __half2float(residual[base_row + 0]);
            output_y[base_row + 0] = __float2half(acc0);
        }
        if (base_row + 1 < out_features) {
            if (residual != nullptr) acc1 += __half2float(residual[base_row + 1]);
            output_y[base_row + 1] = __float2half(acc1);
        }
        if (base_row + 2 < out_features) {
            if (residual != nullptr) acc2 += __half2float(residual[base_row + 2]);
            output_y[base_row + 2] = __float2half(acc2);
        }
        if (base_row + 3 < out_features) {
            if (residual != nullptr) acc3 += __half2float(residual[base_row + 3]);
            output_y[base_row + 3] = __float2half(acc3);
        }
    }
}

// ----------------------------------------------------------------------------
// Single-Launch Virtual-Row QKV GEMV Kernel (W4A8 __dp4a)
// ----------------------------------------------------------------------------

__global__ void gemv_qkv_w4a8_kernel(
    QKVProjectionView view,
    const int8_t* __restrict__ input_q8,
    const float* __restrict__ input_scale,
    const int32_t* __restrict__ input_sum8,
    int in_features,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int num_groups = in_features / QUANT_GROUP_SIZE;
    int num_warp_steps = in_features / 256;

    extern __shared__ alignas(16) uint8_t s_mem[];
    int8_t* s_x_q8 = reinterpret_cast<int8_t*>(s_mem);
    int32_t* s_x_sum8 = reinterpret_cast<int32_t*>(s_mem + in_features);
    float* s_x_scale = reinterpret_cast<float*>(s_mem + in_features + (in_features / 8) * sizeof(int32_t));

    // Staging activations into shared memory
    int num_u4 = in_features / 16;
    const uint4* in_q8_u4 = reinterpret_cast<const uint4*>(input_q8);
    uint4* s_q8_u4 = reinterpret_cast<uint4*>(s_x_q8);
    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        s_q8_u4[i] = in_q8_u4[i];
    }
    int num_sum8 = in_features / 8;
    for (int i = threadIdx.x; i < num_sum8; i += blockDim.x) {
        s_x_sum8[i] = input_sum8[i];
    }
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        s_x_scale[i] = input_scale[i];
    }
    __syncthreads();

    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int q_blocks = q_dim / 16;
    int k_blocks = kv_dim / 16;

    const uint32_t* qweight = nullptr;
    const half* scales = nullptr;
    half* output = nullptr;
    int row_offset = 0;

    if (blockIdx.x < q_blocks) {
        qweight = reinterpret_cast<const uint32_t*>(view.q_weight);
        scales  = view.q_scale;
        output  = view.d_q;
        row_offset = 0;
    } else if (blockIdx.x < q_blocks + k_blocks) {
        qweight = reinterpret_cast<const uint32_t*>(view.k_weight);
        scales  = view.k_scale;
        output  = view.d_k;
        row_offset = q_dim;
    } else {
        qweight = reinterpret_cast<const uint32_t*>(view.v_weight);
        scales  = view.v_scale;
        output  = view.d_v;
        row_offset = q_dim + kv_dim;
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int base_virtual_row = (blockIdx.x * 4 + warp_id) * 4;
    int r0 = base_virtual_row + 0 - row_offset;
    int r1 = base_virtual_row + 1 - row_offset;
    int r2 = base_virtual_row + 2 - row_offset;
    int r3 = base_virtual_row + 3 - row_offset;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_warp_steps; ++s) {
        int group_idx = 2 * s + (lane_id >> 4);
        int sub_lane = lane_id & 15;
        int sum8_idx = group_idx * 16 + sub_lane;

        int32_t x_sum8 = s_x_sum8[sum8_idx];
        const int2* x_q8_ptr = reinterpret_cast<const int2*>(s_x_q8 + group_idx * 128 + sub_lane * 8);
        int2 x_pair = *x_q8_ptr;
        int32_t x_lo = x_pair.x;
        int32_t x_hi = x_pair.y;
        float x_d = s_x_scale[group_idx];

        const uint32_t* row_qw0 = qweight + static_cast<size_t>(r0) * (in_features / 8);
        const half* row_scales0 = scales + static_cast<size_t>(r0) * num_groups;
        uint32_t p0 = row_qw0[s * 32 + lane_id];
        int32_t dot0 = dp4a_w4_q8_oct(p0, x_lo, x_hi, x_sum8);
        float w_d0 = __half2float(row_scales0[group_idx]);
        acc0 += static_cast<float>(dot0) * (w_d0 * x_d);

        const uint32_t* row_qw1 = qweight + static_cast<size_t>(r1) * (in_features / 8);
        const half* row_scales1 = scales + static_cast<size_t>(r1) * num_groups;
        uint32_t p1 = row_qw1[s * 32 + lane_id];
        int32_t dot1 = dp4a_w4_q8_oct(p1, x_lo, x_hi, x_sum8);
        float w_d1 = __half2float(row_scales1[group_idx]);
        acc1 += static_cast<float>(dot1) * (w_d1 * x_d);

        const uint32_t* row_qw2 = qweight + static_cast<size_t>(r2) * (in_features / 8);
        const half* row_scales2 = scales + static_cast<size_t>(r2) * num_groups;
        uint32_t p2 = row_qw2[s * 32 + lane_id];
        int32_t dot2 = dp4a_w4_q8_oct(p2, x_lo, x_hi, x_sum8);
        float w_d2 = __half2float(row_scales2[group_idx]);
        acc2 += static_cast<float>(dot2) * (w_d2 * x_d);

        const uint32_t* row_qw3 = qweight + static_cast<size_t>(r3) * (in_features / 8);
        const half* row_scales3 = scales + static_cast<size_t>(r3) * num_groups;
        uint32_t p3 = row_qw3[s * 32 + lane_id];
        int32_t dot3 = dp4a_w4_q8_oct(p3, x_lo, x_hi, x_sum8);
        float w_d3 = __half2float(row_scales3[group_idx]);
        acc3 += static_cast<float>(dot3) * (w_d3 * x_d);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        acc3 += __shfl_down_sync(0xFFFFFFFF, acc3, offset);
    }

    if (lane_id == 0) {
        output[r0] = __float2half(acc0);
        output[r1] = __float2half(acc1);
        output[r2] = __float2half(acc2);
        output[r3] = __float2half(acc3);
    }
}

// ----------------------------------------------------------------------------
// 2-Output (4-Row) per Warp Fused SwiGLU W4A8 Kernel
// ----------------------------------------------------------------------------

__global__ void gemv_dual_gate_up_swiglu_w4a8_kernel(
    const uint32_t* __restrict__ gate_up_qweight,
    const half* __restrict__ gate_up_scales,
    const int8_t* __restrict__ input_q8,
    const float* __restrict__ input_scale,
    const int32_t* __restrict__ input_sum8,
    half* __restrict__ output_mlp,
    int in_features,
    int intermediate_size
) {
    int num_groups = in_features / QUANT_GROUP_SIZE;
    int num_warp_steps = in_features / 256;

    extern __shared__ alignas(16) uint8_t s_mem[];
    int8_t* s_x_q8 = reinterpret_cast<int8_t*>(s_mem);
    int32_t* s_x_sum8 = reinterpret_cast<int32_t*>(s_mem + in_features);
    float* s_x_scale = reinterpret_cast<float*>(s_mem + in_features + (in_features / 8) * sizeof(int32_t));

    int num_u4 = in_features / 16;
    const uint4* in_q8_u4 = reinterpret_cast<const uint4*>(input_q8);
    uint4* s_q8_u4 = reinterpret_cast<uint4*>(s_x_q8);
    for (int i = threadIdx.x; i < num_u4; i += blockDim.x) {
        s_q8_u4[i] = in_q8_u4[i];
    }
    int num_sum8 = in_features / 8;
    for (int i = threadIdx.x; i < num_sum8; i += blockDim.x) {
        s_x_sum8[i] = input_sum8[i];
    }
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x) {
        s_x_scale[i] = input_scale[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_global = blockIdx.x * 4 + warp_id;
    int out_feat0 = 2 * warp_global + 0;
    int out_feat1 = 2 * warp_global + 1;

    // gate0 = 4*warp_global + 0, up0 = 4*warp_global + 1
    // gate1 = 4*warp_global + 2, up1 = 4*warp_global + 3
    int row_gate0 = 4 * warp_global + 0;
    int row_up0   = 4 * warp_global + 1;
    int row_gate1 = 4 * warp_global + 2;
    int row_up1   = 4 * warp_global + 3;

    float acc_gate0 = 0.0f, acc_up0 = 0.0f, acc_gate1 = 0.0f, acc_up1 = 0.0f;

    for (int s = 0; s < num_warp_steps; ++s) {
        int group_idx = 2 * s + (lane_id >> 4);
        int sub_lane = lane_id & 15;
        int sum8_idx = group_idx * 16 + sub_lane;

        int32_t x_sum8 = s_x_sum8[sum8_idx];
        const int2* x_q8_ptr = reinterpret_cast<const int2*>(s_x_q8 + group_idx * 128 + sub_lane * 8);
        int2 x_pair = *x_q8_ptr;
        int32_t x_lo = x_pair.x;
        int32_t x_hi = x_pair.y;
        float x_d = s_x_scale[group_idx];

        if (out_feat0 < intermediate_size) {
            const uint32_t* row_qw_g0 = gate_up_qweight + static_cast<size_t>(row_gate0) * (in_features / 8);
            const half* row_sc_g0 = gate_up_scales + static_cast<size_t>(row_gate0) * num_groups;
            uint32_t pg0 = row_qw_g0[s * 32 + lane_id];
            int32_t dot_g0 = dp4a_w4_q8_oct(pg0, x_lo, x_hi, x_sum8);
            float wd_g0 = __half2float(row_sc_g0[group_idx]);
            acc_gate0 += static_cast<float>(dot_g0) * (wd_g0 * x_d);

            const uint32_t* row_qw_u0 = gate_up_qweight + static_cast<size_t>(row_up0) * (in_features / 8);
            const half* row_sc_u0 = gate_up_scales + static_cast<size_t>(row_up0) * num_groups;
            uint32_t pu0 = row_qw_u0[s * 32 + lane_id];
            int32_t dot_u0 = dp4a_w4_q8_oct(pu0, x_lo, x_hi, x_sum8);
            float wd_u0 = __half2float(row_sc_u0[group_idx]);
            acc_up0 += static_cast<float>(dot_u0) * (wd_u0 * x_d);
        }

        if (out_feat1 < intermediate_size) {
            const uint32_t* row_qw_g1 = gate_up_qweight + static_cast<size_t>(row_gate1) * (in_features / 8);
            const half* row_sc_g1 = gate_up_scales + static_cast<size_t>(row_gate1) * num_groups;
            uint32_t pg1 = row_qw_g1[s * 32 + lane_id];
            int32_t dot_g1 = dp4a_w4_q8_oct(pg1, x_lo, x_hi, x_sum8);
            float wd_g1 = __half2float(row_sc_g1[group_idx]);
            acc_gate1 += static_cast<float>(dot_g1) * (wd_g1 * x_d);

            const uint32_t* row_qw_u1 = gate_up_qweight + static_cast<size_t>(row_up1) * (in_features / 8);
            const half* row_sc_u1 = gate_up_scales + static_cast<size_t>(row_up1) * num_groups;
            uint32_t pu1 = row_qw_u1[s * 32 + lane_id];
            int32_t dot_u1 = dp4a_w4_q8_oct(pu1, x_lo, x_hi, x_sum8);
            float wd_u1 = __half2float(row_sc_u1[group_idx]);
            acc_up1 += static_cast<float>(dot_u1) * (wd_u1 * x_d);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_gate0 += __shfl_down_sync(0xFFFFFFFF, acc_gate0, offset);
        acc_up0   += __shfl_down_sync(0xFFFFFFFF, acc_up0, offset);
        acc_gate1 += __shfl_down_sync(0xFFFFFFFF, acc_gate1, offset);
        acc_up1   += __shfl_down_sync(0xFFFFFFFF, acc_up1, offset);
    }

    if (lane_id == 0) {
        if (out_feat0 < intermediate_size) {
            float silu0 = acc_gate0 / (1.0f + expf(-acc_gate0));
            output_mlp[out_feat0] = __float2half(silu0 * acc_up0);
        }
        if (out_feat1 < intermediate_size) {
            float silu1 = acc_gate1 / (1.0f + expf(-acc_gate1));
            output_mlp[out_feat1] = __float2half(silu1 * acc_up1);
        }
    }
}

// ----------------------------------------------------------------------------
// Standalone W4A16 GEMV Kernel (cooperative on-the-fly Q8_1 quantization)
// ----------------------------------------------------------------------------

__global__ void gemv_w4a16_standalone_kernel(
    const uint32_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const half* __restrict__ input_x,
    const half* __restrict__ residual,
    half* __restrict__ output_y,
    int out_features,
    int in_features
) {
    extern __shared__ alignas(16) uint8_t s_mem[];
    int8_t* s_x_q8 = reinterpret_cast<int8_t*>(s_mem);
    int32_t* s_x_sum8 = reinterpret_cast<int32_t*>(s_mem + in_features);
    float* s_x_scale = reinterpret_cast<float*>(s_mem + in_features + (in_features / 8) * sizeof(int32_t));

    int half_warp_id = threadIdx.x >> 4;
    int sub_lane = threadIdx.x & 15;
    int num_groups = in_features / QUANT_GROUP_SIZE;

    // Cooperatively quantize input_x into shared memory
    for (int g = half_warp_id; g < num_groups; g += 8) {
        const half* g_src = input_x + g * QUANT_GROUP_SIZE + sub_lane * 8;
        float vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            vals[j] = __half2float(g_src[j]);
        }
        quantize_group128_fp16_to_q8_1(vals, g, sub_lane, s_x_q8, s_x_scale, s_x_sum8);
    }
    __syncthreads();

    // 4 rows per warp accumulation
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int base_row = (blockIdx.x * 4 + warp_id) * 4;
    int num_warp_steps = in_features / 256;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int s = 0; s < num_warp_steps; ++s) {
        int group_idx = 2 * s + (lane_id >> 4);
        int sub_lane_idx = lane_id & 15;
        int sum8_idx = group_idx * 16 + sub_lane_idx;

        int32_t x_sum8 = s_x_sum8[sum8_idx];
        const int2* x_q8_ptr = reinterpret_cast<const int2*>(s_x_q8 + group_idx * 128 + sub_lane_idx * 8);
        int2 x_pair = *x_q8_ptr;
        int32_t x_lo = x_pair.x;
        int32_t x_hi = x_pair.y;
        float x_d = s_x_scale[group_idx];

        if (base_row + 0 < out_features) {
            const uint32_t* row_qw0 = qweight + static_cast<size_t>(base_row + 0) * (in_features / 8);
            const half* row_scales0 = scales + static_cast<size_t>(base_row + 0) * num_groups;
            uint32_t p0 = row_qw0[s * 32 + lane_id];
            int32_t dot0 = dp4a_w4_q8_oct(p0, x_lo, x_hi, x_sum8);
            float w_d0 = __half2float(row_scales0[group_idx]);
            acc0 += static_cast<float>(dot0) * (w_d0 * x_d);
        }
        if (base_row + 1 < out_features) {
            const uint32_t* row_qw1 = qweight + static_cast<size_t>(base_row + 1) * (in_features / 8);
            const half* row_scales1 = scales + static_cast<size_t>(base_row + 1) * num_groups;
            uint32_t p1 = row_qw1[s * 32 + lane_id];
            int32_t dot1 = dp4a_w4_q8_oct(p1, x_lo, x_hi, x_sum8);
            float w_d1 = __half2float(row_scales1[group_idx]);
            acc1 += static_cast<float>(dot1) * (w_d1 * x_d);
        }
        if (base_row + 2 < out_features) {
            const uint32_t* row_qw2 = qweight + static_cast<size_t>(base_row + 2) * (in_features / 8);
            const half* row_scales2 = scales + static_cast<size_t>(base_row + 2) * num_groups;
            uint32_t p2 = row_qw2[s * 32 + lane_id];
            int32_t dot2 = dp4a_w4_q8_oct(p2, x_lo, x_hi, x_sum8);
            float w_d2 = __half2float(row_scales2[group_idx]);
            acc2 += static_cast<float>(dot2) * (w_d2 * x_d);
        }
        if (base_row + 3 < out_features) {
            const uint32_t* row_qw3 = qweight + static_cast<size_t>(base_row + 3) * (in_features / 8);
            const half* row_scales3 = scales + static_cast<size_t>(base_row + 3) * num_groups;
            uint32_t p3 = row_qw3[s * 32 + lane_id];
            int32_t dot3 = dp4a_w4_q8_oct(p3, x_lo, x_hi, x_sum8);
            float w_d3 = __half2float(row_scales3[group_idx]);
            acc3 += static_cast<float>(dot3) * (w_d3 * x_d);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        acc3 += __shfl_down_sync(0xFFFFFFFF, acc3, offset);
    }

    if (lane_id == 0) {
        if (base_row + 0 < out_features) {
            if (residual != nullptr) acc0 += __half2float(residual[base_row + 0]);
            output_y[base_row + 0] = __float2half(acc0);
        }
        if (base_row + 1 < out_features) {
            if (residual != nullptr) acc1 += __half2float(residual[base_row + 1]);
            output_y[base_row + 1] = __float2half(acc1);
        }
        if (base_row + 2 < out_features) {
            if (residual != nullptr) acc2 += __half2float(residual[base_row + 2]);
            output_y[base_row + 2] = __float2half(acc2);
        }
        if (base_row + 3 < out_features) {
            if (residual != nullptr) acc3 += __half2float(residual[base_row + 3]);
            output_y[base_row + 3] = __float2half(acc3);
        }
    }
}

// ----------------------------------------------------------------------------
// Standalone Fused SwiGLU W4A16 Kernel (cooperative on-the-fly Q8_1 quantization)
// ----------------------------------------------------------------------------

__global__ void gemv_dual_gate_up_swiglu_w4a16_standalone_kernel(
    const uint32_t* __restrict__ gate_up_qweight,
    const half* __restrict__ gate_up_scales,
    const half* __restrict__ input_x,
    half* __restrict__ output_mlp,
    int in_features,
    int intermediate_size
) {
    int num_groups = in_features / QUANT_GROUP_SIZE;
    int num_warp_steps = in_features / 256;

    extern __shared__ alignas(16) uint8_t s_mem[];
    int8_t* s_x_q8 = reinterpret_cast<int8_t*>(s_mem);
    int32_t* s_x_sum8 = reinterpret_cast<int32_t*>(s_mem + in_features);
    float* s_x_scale = reinterpret_cast<float*>(s_mem + in_features + (in_features / 8) * sizeof(int32_t));

    int half_warp_id = threadIdx.x >> 4;
    int sub_lane = threadIdx.x & 15;

    for (int g = half_warp_id; g < num_groups; g += 8) {
        const half* g_src = input_x + g * QUANT_GROUP_SIZE + sub_lane * 8;
        float vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            vals[j] = __half2float(g_src[j]);
        }
        quantize_group128_fp16_to_q8_1(vals, g, sub_lane, s_x_q8, s_x_scale, s_x_sum8);
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_global = blockIdx.x * 4 + warp_id;
    int out_feat0 = 2 * warp_global + 0;
    int out_feat1 = 2 * warp_global + 1;

    int row_gate0 = 4 * warp_global + 0;
    int row_up0   = 4 * warp_global + 1;
    int row_gate1 = 4 * warp_global + 2;
    int row_up1   = 4 * warp_global + 3;

    float acc_gate0 = 0.0f, acc_up0 = 0.0f, acc_gate1 = 0.0f, acc_up1 = 0.0f;

    for (int s = 0; s < num_warp_steps; ++s) {
        int group_idx = 2 * s + (lane_id >> 4);
        int sub_lane_idx = lane_id & 15;
        int sum8_idx = group_idx * 16 + sub_lane_idx;

        int32_t x_sum8 = s_x_sum8[sum8_idx];
        const int2* x_q8_ptr = reinterpret_cast<const int2*>(s_x_q8 + group_idx * 128 + sub_lane_idx * 8);
        int2 x_pair = *x_q8_ptr;
        int32_t x_lo = x_pair.x;
        int32_t x_hi = x_pair.y;
        float x_d = s_x_scale[group_idx];

        if (out_feat0 < intermediate_size) {
            const uint32_t* row_qw_g0 = gate_up_qweight + static_cast<size_t>(row_gate0) * (in_features / 8);
            const half* row_sc_g0 = gate_up_scales + static_cast<size_t>(row_gate0) * num_groups;
            uint32_t pg0 = row_qw_g0[s * 32 + lane_id];
            int32_t dot_g0 = dp4a_w4_q8_oct(pg0, x_lo, x_hi, x_sum8);
            float wd_g0 = __half2float(row_sc_g0[group_idx]);
            acc_gate0 += static_cast<float>(dot_g0) * (wd_g0 * x_d);

            const uint32_t* row_qw_u0 = gate_up_qweight + static_cast<size_t>(row_up0) * (in_features / 8);
            const half* row_sc_u0 = gate_up_scales + static_cast<size_t>(row_up0) * num_groups;
            uint32_t pu0 = row_qw_u0[s * 32 + lane_id];
            int32_t dot_u0 = dp4a_w4_q8_oct(pu0, x_lo, x_hi, x_sum8);
            float wd_u0 = __half2float(row_sc_u0[group_idx]);
            acc_up0 += static_cast<float>(dot_u0) * (wd_u0 * x_d);
        }

        if (out_feat1 < intermediate_size) {
            const uint32_t* row_qw_g1 = gate_up_qweight + static_cast<size_t>(row_gate1) * (in_features / 8);
            const half* row_sc_g1 = gate_up_scales + static_cast<size_t>(row_gate1) * num_groups;
            uint32_t pg1 = row_qw_g1[s * 32 + lane_id];
            int32_t dot_g1 = dp4a_w4_q8_oct(pg1, x_lo, x_hi, x_sum8);
            float wd_g1 = __half2float(row_sc_g1[group_idx]);
            acc_gate1 += static_cast<float>(dot_g1) * (wd_g1 * x_d);

            const uint32_t* row_qw_u1 = gate_up_qweight + static_cast<size_t>(row_up1) * (in_features / 8);
            const half* row_sc_u1 = gate_up_scales + static_cast<size_t>(row_up1) * num_groups;
            uint32_t pu1 = row_qw_u1[s * 32 + lane_id];
            int32_t dot_u1 = dp4a_w4_q8_oct(pu1, x_lo, x_hi, x_sum8);
            float wd_u1 = __half2float(row_sc_u1[group_idx]);
            acc_up1 += static_cast<float>(dot_u1) * (wd_u1 * x_d);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_gate0 += __shfl_down_sync(0xFFFFFFFF, acc_gate0, offset);
        acc_up0   += __shfl_down_sync(0xFFFFFFFF, acc_up0, offset);
        acc_gate1 += __shfl_down_sync(0xFFFFFFFF, acc_gate1, offset);
        acc_up1   += __shfl_down_sync(0xFFFFFFFF, acc_up1, offset);
    }

    if (lane_id == 0) {
        if (out_feat0 < intermediate_size) {
            float silu0 = acc_gate0 / (1.0f + expf(-acc_gate0));
            output_mlp[out_feat0] = __float2half(silu0 * acc_up0);
        }
        if (out_feat1 < intermediate_size) {
            float silu1 = acc_gate1 / (1.0f + expf(-acc_gate1));
            output_mlp[out_feat1] = __float2half(silu1 * acc_up1);
        }
    }
}

// ----------------------------------------------------------------------------
// Launch Wrappers
// ----------------------------------------------------------------------------

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
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int ROWS_PER_BLOCK = 16;
    int grid = (out_features + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    size_t smem = static_cast<size_t>(in_features) * 49 / 32;

    gemv_w4a8_kernel<<<grid, BLOCK_THREADS, smem, stream>>>(
        qweight, scales, input_q8, input_scale, input_sum8, residual, output_y, out_features, in_features
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_qkv_w4a8(
    const QKVProjectionView& view,
    const int8_t* input_q8,
    const float* input_scale,
    const int32_t* input_sum8,
    cudaStream_t stream,
    int in_features,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    constexpr int BLOCK_THREADS = 128;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int total_rows = q_dim + 2 * kv_dim;
    constexpr int ROWS_PER_BLOCK = 16;
    int grid = (total_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    size_t smem = static_cast<size_t>(in_features) * 49 / 32;

    gemv_qkv_w4a8_kernel<<<grid, BLOCK_THREADS, smem, stream>>>(
        view, input_q8, input_scale, input_sum8, in_features, num_q_heads, num_kv_heads, head_dim
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_dual_gate_up_swiglu_w4a8(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const int8_t* input_q8,
    const float* input_scale,
    const int32_t* input_sum8,
    half* output_mlp,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int OUT_PER_BLOCK = 8; // 4 warps * 2 intermediate outputs = 8
    int grid = (intermediate_size + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK;
    size_t smem = static_cast<size_t>(hidden_size) * 49 / 32;

    gemv_dual_gate_up_swiglu_w4a8_kernel<<<grid, BLOCK_THREADS, smem, stream>>>(
        gate_up_qweight, gate_up_scales, input_q8, input_scale, input_sum8, output_mlp,
        hidden_size, intermediate_size
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_w4a16(
    const uint32_t* qweight,
    const half* scales,
    const half* input_x,
    const half* residual,
    half* output_y,
    int out_features,
    int in_features,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int ROWS_PER_BLOCK = 16;
    int grid = (out_features + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    size_t smem = static_cast<size_t>(in_features) * 49 / 32;

    gemv_w4a16_standalone_kernel<<<grid, BLOCK_THREADS, smem, stream>>>(
        qweight, scales, input_x, residual, output_y, out_features, in_features
    );
    CUDA_KERNEL_CHECK();
}

void launch_gemv_dual_gate_up_swiglu_w4a16(
    const uint32_t* gate_up_qweight,
    const half* gate_up_scales,
    const half* input_x,
    half* output_mlp,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 128;
    constexpr int OUT_PER_BLOCK = 8;
    int grid = (intermediate_size + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK;
    size_t smem = static_cast<size_t>(hidden_size) * 49 / 32;

    gemv_dual_gate_up_swiglu_w4a16_standalone_kernel<<<grid, BLOCK_THREADS, smem, stream>>>(
        gate_up_qweight, gate_up_scales, input_x, output_mlp, hidden_size, intermediate_size
    );
    CUDA_KERNEL_CHECK();
}

} // namespace qwen
