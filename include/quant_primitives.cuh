#pragma once

#include "config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace qwen {

// Computes precomputed zero-point offset: sum8 = -8 * sum_{j=0}^{7} x_{q, k+j}
// x_lo packs {x0, x1, x2, x3} and x_hi packs {x4, x5, x6, x7} as signed int8 bytes.
// Using __dp4a with 0x01010101 treats bytes as signed int8.
__device__ __forceinline__ int32_t compute_q8_sum8(int32_t x_lo, int32_t x_hi) {
    int32_t sum = __dp4a(0x01010101, x_hi, 0);
    sum = __dp4a(0x01010101, x_lo, sum);
    return -8 * sum;
}

// Evaluates sum_{j=0}^7 (q_j - 8) * x_{q,j} for 8 elements in 4 integer instructions.
// packed: uint32_t word packing 8 unsigned nibbles in .qwenbin v1 format:
// q0 | (q4 << 4) | (q1 << 8) | (q5 << 12) | (q2 << 16) | (q6 << 20) | (q3 << 24) | (q7 << 28)
// Extract:
// w_lo = packed & 0x0F0F0F0Fu -> {q0, q1, q2, q3} in [0, 15]
// w_hi = (packed >> 4) & 0x0F0F0F0Fu -> {q4, q5, q6, q7} in [0, 15]
// x_lo: {x0, x1, x2, x3} (signed int8x4)
// x_hi: {x4, x5, x6, x7} (signed int8x4)
// x_sum8: -8 * sum_{j=0}^7 x_{q,j} passed as seed to first __dp4a.
__device__ __forceinline__ int32_t dp4a_w4_q8_oct(uint32_t packed, int32_t x_lo, int32_t x_hi, int32_t x_sum8) {
    int32_t w_lo = packed & 0x0F0F0F0Fu;
    int32_t w_hi = (packed >> 4) & 0x0F0F0F0Fu;
    return __dp4a(w_hi, x_hi, __dp4a(w_lo, x_lo, x_sum8));
}

// Shared 16-lane half-warp helper (8 elements per lane = 128 elements per group)
// Reduces max_abs via 4 half-warp __shfl_xor_sync steps, computes scale d = max(max_abs / 127.0f, 1e-8f),
// quantizes vals[0..7] to int8, packs (x_lo, x_hi), computes sum8, and writes out.
__device__ __forceinline__ void quantize_group128_fp16_to_q8_1(
    const float vals[8],
    int group_idx,
    int sub_lane,
    int8_t* out_q8,
    float* out_scale,
    int32_t* out_sum8
) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        max_abs = fmaxf(max_abs, fabsf(vals[j]));
    }
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xFFFFFFFF, max_abs, mask));
    }

    float d = fmaxf(max_abs / 127.0f, 1e-8f);
    float inv_d = 1.0f / d;

    int8_t q[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int v = __float2int_rn(vals[j] * inv_d);
        v = max(-127, min(127, v));
        q[j] = static_cast<int8_t>(v);
    }

    int32_t x_lo = (static_cast<uint32_t>(static_cast<uint8_t>(q[0]))) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[1])) << 8) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[2])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[3])) << 24);

    int32_t x_hi = (static_cast<uint32_t>(static_cast<uint8_t>(q[4]))) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[5])) << 8) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[6])) << 16) |
                   (static_cast<uint32_t>(static_cast<uint8_t>(q[7])) << 24);

    if (out_q8 != nullptr) {
        int2* dst = reinterpret_cast<int2*>(out_q8 + group_idx * 128 + sub_lane * 8);
        *dst = make_int2(x_lo, x_hi);
    }

    int32_t sum8 = compute_q8_sum8(x_lo, x_hi);
    if (out_sum8 != nullptr) {
        out_sum8[group_idx * 16 + sub_lane] = sum8;
    }

    if (out_scale != nullptr && sub_lane == 0) {
        out_scale[group_idx] = d;
    }
}

// Unpacks packed uint32 into four sequential half2 pairs for coalesced uint4 writes
__device__ __forceinline__ void unpack_w4_to_half2_quad(
    uint32_t packed,
    half scale,
    half2 out_h2[4]
) {
    int q0 = (packed >> 0)  & 0xF;
    int q4 = (packed >> 4)  & 0xF;
    int q1 = (packed >> 8)  & 0xF;
    int q5 = (packed >> 12) & 0xF;
    int q2 = (packed >> 16) & 0xF;
    int q6 = (packed >> 20) & 0xF;
    int q3 = (packed >> 24) & 0xF;
    int q7 = (packed >> 28) & 0xF;

    float s = __half2float(scale);
    float w0 = (static_cast<float>(q0) - 8.0f) * s;
    float w1 = (static_cast<float>(q1) - 8.0f) * s;
    float w2 = (static_cast<float>(q2) - 8.0f) * s;
    float w3 = (static_cast<float>(q3) - 8.0f) * s;
    float w4 = (static_cast<float>(q4) - 8.0f) * s;
    float w5 = (static_cast<float>(q5) - 8.0f) * s;
    float w6 = (static_cast<float>(q6) - 8.0f) * s;
    float w7 = (static_cast<float>(q7) - 8.0f) * s;

    out_h2[0] = make_half2(__float2half(w0), __float2half(w1));
    out_h2[1] = make_half2(__float2half(w2), __float2half(w3));
    out_h2[2] = make_half2(__float2half(w4), __float2half(w5));
    out_h2[3] = make_half2(__float2half(w6), __float2half(w7));
}

} // namespace qwen

