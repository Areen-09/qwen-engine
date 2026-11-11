#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);      \
            fflush(stderr);                                                   \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

namespace qwen {

// Persistent GPU-resident control block driving zero-update CUDA Graph execution
struct alignas(256) DeviceControlBlock {
    int32_t  token_id;
    int32_t  seq_pos;
    float    temperature;
    uint32_t reserved;
    uint64_t rng_seed;
};

struct PinnedCandidate {
    float val;
    int32_t idx;
};

// Static single-cudaMalloc memory arena
class GpuArena {
public:
    explicit GpuArena(size_t total_bytes)
        : total_bytes_(total_bytes), used_bytes_(0), base_ptr_(nullptr) {
        
        size_t free_bytes = 0;
        size_t total_dev_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_dev_bytes));

        constexpr size_t HEADROOM = 64ULL << 20; // 64 MB driver headroom
        if (free_bytes < total_bytes + HEADROOM) {
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "GpuArena: Insufficient discrete GPU VRAM. Required: %.2f MB (+64 MB headroom), Free: %.2f MB, Total: %.2f MB. "
                     "Consider running in W4A16 mode or closing background GPU applications.",
                     (double)total_bytes / (1024.0 * 1024.0),
                     (double)free_bytes / (1024.0 * 1024.0),
                     (double)total_dev_bytes / (1024.0 * 1024.0));
            fprintf(stderr, "%s\n", msg);
            throw std::runtime_error(msg);
        }

        CUDA_CHECK(cudaMalloc(&base_ptr_, total_bytes_));
        CUDA_CHECK(cudaMemset(base_ptr_, 0, total_bytes_));
    }

    ~GpuArena() {
        if (base_ptr_) {
            cudaFree(base_ptr_);
            base_ptr_ = nullptr;
        }
    }

    GpuArena(const GpuArena&) = delete;
    GpuArena& operator=(const GpuArena&) = delete;

    GpuArena(GpuArena&& other) noexcept
        : total_bytes_(other.total_bytes_),
          used_bytes_(other.used_bytes_),
          base_ptr_(other.base_ptr_) {
        other.base_ptr_ = nullptr;
        other.used_bytes_ = 0;
        other.total_bytes_ = 0;
    }

    GpuArena& operator=(GpuArena&& other) noexcept {
        if (this != &other) {
            if (base_ptr_) {
                cudaFree(base_ptr_);
            }
            total_bytes_ = other.total_bytes_;
            used_bytes_ = other.used_bytes_;
            base_ptr_ = other.base_ptr_;
            other.base_ptr_ = nullptr;
            other.used_bytes_ = 0;
            other.total_bytes_ = 0;
        }
        return *this;
    }

    void* allocate(size_t bytes, size_t alignment = 256) {
        size_t current_addr = reinterpret_cast<size_t>(base_ptr_) + used_bytes_;
        size_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
        size_t new_used = (aligned_addr - reinterpret_cast<size_t>(base_ptr_)) + bytes;

        if (new_used > total_bytes_) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "GpuArena: Out of memory in arena allocation. Requested: %zu bytes, Used: %zu / %zu bytes",
                     bytes, used_bytes_, total_bytes_);
            fprintf(stderr, "%s\n", msg);
            throw std::runtime_error(msg);
        }

        used_bytes_ = new_used;
        return reinterpret_cast<void*>(aligned_addr);
    }

    size_t used_bytes() const { return used_bytes_; }
    size_t capacity_bytes() const { return total_bytes_; }
    void* base_ptr() const { return base_ptr_; }

private:
    size_t total_bytes_;
    size_t used_bytes_;
    void*  base_ptr_;
};

} // namespace qwen

