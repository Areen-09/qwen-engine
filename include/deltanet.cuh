#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qwen {

// Cooperative 2D sub-tiled DeltaNet decode kernel (16 KB SMEM on sm_75)
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
);

// Depthwise Conv1d sliding-window step kernel
void launch_depthwise_conv1d_step(
    const half* in,
    const half* weight,
    const half* bias,
    half* conv_state,
    half* out,
    int num_channels,
    int kernel_size,
    cudaStream_t stream
);

} // namespace qwen

