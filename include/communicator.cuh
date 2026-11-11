#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>

namespace qwen {

enum class DataType : uint32_t {
    FP16 = 0,
    FP32 = 1,
    INT32 = 2
};

struct HostStagingChannel {
    void* host_buf[2][2]{ {nullptr, nullptr}, {nullptr, nullptr} };
    alignas(64) std::atomic<int> arrival_epoch[2]{0, 0};
    alignas(64) std::atomic<int> completion_epoch[2]{0, 0};
    alignas(64) std::atomic<int> exit_epoch[2]{2, 2};
};

void avx2_vector_add_half(half* dst, const half* src, size_t count);

class ICommunicator {
public:
    virtual ~ICommunicator() = default;

    // In-place All-Reduce summation across active devices
    virtual void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type, cudaStream_t stream) = 0;

    // Cross-device stream synchronization barrier
    virtual void sync(cudaStream_t stream) = 0;

    virtual int rank() const = 0;
    virtual int world_size() const = 0;
    virtual bool is_peer_accessible() const = 0;
};

// Factory producing communicator backend (Zero-dependency Direct P2P or optional Linux NCCL)
std::unique_ptr<ICommunicator> create_communicator(
    int rank,
    int world_size,
    const std::vector<int>& device_ids,
    bool force_staging = false,
    std::shared_ptr<HostStagingChannel> shared_channel = nullptr
);

} // namespace qwen

