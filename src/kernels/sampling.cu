#include "config.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace qwen {

// ----------------------------------------------------------------------------
// Stateless Philox-4x32-10 PRNG
// ----------------------------------------------------------------------------

__device__ __forceinline__ float generate_uniform_float(uint64_t seed, uint32_t subsequence, uint32_t offset) {
    uint32_t ctr0 = subsequence;
    uint32_t ctr1 = offset;
    uint32_t ctr2 = 0;
    uint32_t ctr3 = 0;

    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    constexpr uint64_t PHILOX_M4x32_0 = 0xD2511F53ULL;
    constexpr uint64_t PHILOX_M4x32_1 = 0xCD9E8D57ULL;
    constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9U;
    constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85U;

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint64_t prod0 = PHILOX_M4x32_0 * ctr0;
        uint64_t prod1 = PHILOX_M4x32_1 * ctr2;

        uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(prod0);
        uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(prod1);

        uint32_t next_ctr0 = hi1 ^ ctr1 ^ key0;
        uint32_t next_ctr1 = lo1;
        uint32_t next_ctr2 = hi0 ^ ctr3 ^ key1;
        uint32_t next_ctr3 = lo0;

        ctr0 = next_ctr0;
        ctr1 = next_ctr1;
        ctr2 = next_ctr2;
        ctr3 = next_ctr3;

        key0 += PHILOX_W32_0;
        key1 += PHILOX_W32_1;
    }

    // Convert ctr0 to uniform float in (1e-7, 1 - 1e-7)
    float u = (static_cast<float>(ctr0) + 1.0f) / 4294967298.0f;
    return fminf(fmaxf(u, 1e-7f), 1.0f - 1e-7f);
}

// ----------------------------------------------------------------------------
// Stage 1 Sampling Kernel (256 Blocks of 256 Threads over 151936 Logits)
// ----------------------------------------------------------------------------

__global__ void sample_stage1_kernel(
    const half* __restrict__ logits,
    float* __restrict__ sample_scratch_val,
    int32_t* __restrict__ sample_scratch_idx,
    const DeviceControlBlock* __restrict__ d_control,
    int vocab_size,
    int32_t index_offset = 0
) {
    float temp = 0.0f;
    uint64_t seed = 0;
    uint32_t seq_pos = 0;
    if (d_control != nullptr) {
        temp = d_control->temperature;
        seed = d_control->rng_seed;
        seq_pos = static_cast<uint32_t>(d_control->seq_pos);
    }

    float max_val = -1e30f;
    int32_t best_idx = 0;

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = global_tid; i < vocab_size; i += total_threads) {
        float val = __half2float(logits[i]);

        if (temp > 1e-5f) {
            float u = generate_uniform_float(seed, static_cast<uint32_t>(i + index_offset), seq_pos);
            // Gumbel noise = -log(-log(u))
            float gumbel = -logf(-logf(u));
            val = (val / temp) + gumbel;
        }

        if (val > max_val) {
            max_val = val;
            best_idx = i;
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int32_t other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
        if (other_val > max_val || (other_val == max_val && other_idx < best_idx)) {
            max_val = other_val;
            best_idx = other_idx;
        }
    }

    __shared__ float s_warp_max[8];
    __shared__ int32_t s_warp_idx[8];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_max[warp_id] = max_val;
        s_warp_idx[warp_id] = best_idx;
    }
    __syncthreads();

    // Warp 0 reduces the 8 warp winners
    if (warp_id == 0 && lane_id < 8) {
        float w_val = s_warp_max[lane_id];
        int32_t w_idx = s_warp_idx[lane_id];

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            float o_val = __shfl_down_sync(0xFF, w_val, offset);
            int32_t o_idx = __shfl_down_sync(0xFF, w_idx, offset);
            if (o_val > w_val || (o_val == w_val && o_idx < w_idx)) {
                w_val = o_val;
                w_idx = o_idx;
            }
        }

        if (lane_id == 0) {
            sample_scratch_val[blockIdx.x] = w_val;
            sample_scratch_idx[blockIdx.x] = w_idx;
        }
    }
}

// ----------------------------------------------------------------------------
// Stage 2 Sampling & Advance Kernel (1 Block of 256 Threads)
// ----------------------------------------------------------------------------

__global__ void sample_stage2_and_advance_kernel(
    const float* __restrict__ sample_scratch_val,
    const int32_t* __restrict__ sample_scratch_idx,
    DeviceControlBlock* __restrict__ d_control,
    int32_t* __restrict__ d_mapped_pinned_token
) {
    int tid = threadIdx.x; // 0 .. 255
    float val = sample_scratch_val[tid];
    int32_t idx = sample_scratch_idx[tid];

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float o_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int32_t o_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (o_val > val || (o_val == val && o_idx < idx)) {
            val = o_val;
            idx = o_idx;
        }
    }

    __shared__ float s_warp_max[8];
    __shared__ int32_t s_warp_idx[8];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_max[warp_id] = val;
        s_warp_idx[warp_id] = idx;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < 8) {
        float w_val = s_warp_max[lane_id];
        int32_t w_idx = s_warp_idx[lane_id];

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            float o_val = __shfl_down_sync(0xFF, w_val, offset);
            int32_t o_idx = __shfl_down_sync(0xFF, w_idx, offset);
            if (o_val > w_val || (o_val == w_val && o_idx < w_idx)) {
                w_val = o_val;
                w_idx = o_idx;
            }
        }

        if (lane_id == 0) {
            // Write winning token to DeviceControlBlock and pinned host pointer
            if (d_control != nullptr) {
                d_control->token_id = w_idx;
                d_control->seq_pos += 1;
                d_control->rng_seed += 0x9E3779B97F4A7C15ULL;
            }
            if (d_mapped_pinned_token != nullptr) {
                *d_mapped_pinned_token = w_idx;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Host Launch Function
// ----------------------------------------------------------------------------

void launch_sample_and_advance(
    const half* logits,
    float* sample_scratch_val,
    int32_t* sample_scratch_idx,
    DeviceControlBlock* d_control,
    int32_t* d_mapped_pinned_token,
    cudaStream_t stream,
    int vocab_size
) {
    constexpr int GRID_SIZE = 256;
    constexpr int BLOCK_SIZE = 256;

    sample_stage1_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        logits, sample_scratch_val, sample_scratch_idx, d_control, vocab_size, 0
    );

    sample_stage2_and_advance_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        sample_scratch_val, sample_scratch_idx, d_control, d_mapped_pinned_token
    );
}

} // namespace qwen

