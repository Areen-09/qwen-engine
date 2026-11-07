#pragma once

#include <cuda_runtime.h>
#include "tensor.cuh"

namespace qwen {

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int target_device)
        : target_device_(target_device), prev_device_(-1) {
        CUDA_CHECK(cudaGetDevice(&prev_device_));
        if (prev_device_ != target_device_) {
            CUDA_CHECK(cudaSetDevice(target_device_));
        }
    }

    ~CudaDeviceGuard() {
        if (prev_device_ != -1 && prev_device_ != target_device_) {
            cudaSetDevice(prev_device_);
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int target_device_;
    int prev_device_;
};

} // namespace qwen

