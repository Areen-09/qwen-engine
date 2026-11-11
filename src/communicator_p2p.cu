#include "communicator.cuh"
#include "cuda_guard.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <cstring>
#include <thread>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace qwen {

// Kernel for elementwise sum reduction between local and peer buffers (Direct P2P)
template <typename T>
__global__ void vector_add_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = dst[idx] + src[idx];
    }
}

template <>
__global__ void vector_add_kernel<half>(half* __restrict__ dst, const half* __restrict__ src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
    }
}

// ----------------------------------------------------------------------------
// AVX2 Vectorized Half-Precision Summation on Host
// ----------------------------------------------------------------------------
#if defined(__AVX2__) || defined(_M_AMD64)
void avx2_vector_add_half(half* dst, const half* src, size_t count) {
    size_t i = 0;
    size_t count8 = (count / 8) * 8;
    for (; i < count8; i += 8) {
        __m128i h_dst = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
        __m128i h_src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f_dst = _mm256_cvtph_ps(h_dst);
        __m256 f_src = _mm256_cvtph_ps(h_src);
        __m256 f_sum = _mm256_add_ps(f_dst, f_src);
        __m128i h_sum = _mm256_cvtps_ph(f_sum, 0); // 0 = round to nearest even
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h_sum);
    }
    for (; i < count; ++i) {
        float s = __half2float(dst[i]) + __half2float(src[i]);
        dst[i] = __float2half(s);
    }
}
#else
void avx2_vector_add_half(half* dst, const half* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float s = __half2float(dst[i]) + __half2float(src[i]);
        dst[i] = __float2half(s);
    }
}
#endif

// ----------------------------------------------------------------------------
// Adaptive Two-Phase Spin-Yield Helper
// ----------------------------------------------------------------------------
template <typename Predicate>
inline void spin_yield_wait(Predicate&& pred) {
    for (int spin = 0; spin < 256; ++spin) {
        if (pred()) return;
        #if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
        _mm_pause();
        #endif
    }
    while (!pred()) {
        std::this_thread::yield();
    }
}

struct P2PDirectState {
    std::mutex mtx;
    void* device_peer_buf[2]{nullptr, nullptr};
    cudaEvent_t peer_events[2]{nullptr, nullptr};
};

static P2PDirectState g_p2p_direct_state;
static std::shared_ptr<HostStagingChannel> g_default_host_channel = nullptr;
static std::mutex g_host_channel_init_mtx;

class P2PCommunicator : public ICommunicator {
public:
    P2PCommunicator(
        int rank,
        int world_size,
        const std::vector<int>& device_ids,
        bool force_staging,
        std::shared_ptr<HostStagingChannel> shared_channel
    ) : rank_(rank),
        world_size_(world_size),
        device_ids_(device_ids),
        peer_accessible_(false),
        d_peer_staging_(nullptr),
        local_event_(nullptr),
        channel_(shared_channel),
        epoch_(0) {

        if (world_size_ <= 1) {
            peer_accessible_ = true;
            return;
        }

        int my_dev = device_ids_[rank_];
        int peer_dev = device_ids_[1 - rank_];

        CudaDeviceGuard guard(my_dev);

        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, my_dev, peer_dev);

        if (can_access && !force_staging) {
            cudaError_t err = cudaDeviceEnablePeerAccess(peer_dev, 0);
            if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
                peer_accessible_ = true;
            }
        }

        size_t max_bytes = 16ULL << 20; // 16 MB staging buffer per slot
        if (peer_accessible_) {
            CUDA_CHECK(cudaMalloc(&d_peer_staging_, max_bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&local_event_, cudaEventDisableTiming));

            std::unique_lock<std::mutex> lock(g_p2p_direct_state.mtx);
            g_p2p_direct_state.device_peer_buf[rank_] = d_peer_staging_;
            g_p2p_direct_state.peer_events[rank_] = local_event_;
            std::cout << "[P2PCommunicator] Rank " << rank_
                      << " enabled direct BAR1 P2P with device " << peer_dev << "." << std::endl;
        } else {
            // Setup HostStagingChannel
            std::unique_lock<std::mutex> lock(g_host_channel_init_mtx);
            if (!channel_) {
                if (!g_default_host_channel) {
                    g_default_host_channel = std::make_shared<HostStagingChannel>();
                }
                channel_ = g_default_host_channel;
            }

            for (int e = 0; e < 2; ++e) {
                if (!channel_->host_buf[e][rank_]) {
                    CUDA_CHECK(cudaHostAlloc(&channel_->host_buf[e][rank_], max_bytes, cudaHostAllocMapped));
                }
            }

            std::cout << "[P2PCommunicator] Rank " << rank_
                      << " initialized double-buffered host staging with AVX2 All-Reduce." << std::endl;
        }
    }

    ~P2PCommunicator() override {
        if (world_size_ <= 1) return;

        CudaDeviceGuard guard(device_ids_[rank_]);
        if (d_peer_staging_) {
            cudaFree(d_peer_staging_);
            d_peer_staging_ = nullptr;
        }
        if (local_event_) {
            cudaEventDestroy(local_event_);
            local_event_ = nullptr;
        }
        // Note: host_buf allocations are freed upon process exit or can be freed if channel unique
    }

    void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type, cudaStream_t stream) override {
        if (world_size_ <= 1) return;

        size_t elem_size = (type == DataType::FP16) ? sizeof(half) :
                           (type == DataType::FP32) ? sizeof(float) : sizeof(int32_t);
        size_t bytes = count * elem_size;

        if (peer_accessible_) {
            // Direct P2P All-Reduce path (CUDA Graph capture compatible)
            int peer_rank = 1 - rank_;
            void* peer_staging = g_p2p_direct_state.device_peer_buf[peer_rank];
            cudaEvent_t peer_event = g_p2p_direct_state.peer_events[peer_rank];

            if (!peer_staging || !peer_event) {
                return;
            }

            CUDA_CHECK(cudaMemcpyPeerAsync(peer_staging, device_ids_[peer_rank],
                                           sendrecvbuf, device_ids_[rank_],
                                           bytes, stream));
            CUDA_CHECK(cudaEventRecord(local_event_, stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream, peer_event, 0));

            int block = 256;
            int grid = static_cast<int>((count + block - 1) / block);
            if (type == DataType::FP16) {
                vector_add_kernel<half><<<grid, block, 0, stream>>>(
                    reinterpret_cast<half*>(sendrecvbuf),
                    reinterpret_cast<const half*>(d_peer_staging_),
                    count
                );
            } else if (type == DataType::FP32) {
                vector_add_kernel<float><<<grid, block, 0, stream>>>(
                    reinterpret_cast<float*>(sendrecvbuf),
                    reinterpret_cast<const float*>(d_peer_staging_),
                    count
                );
            } else {
                vector_add_kernel<int32_t><<<grid, block, 0, stream>>>(
                    reinterpret_cast<int32_t*>(sendrecvbuf),
                    reinterpret_cast<const int32_t*>(d_peer_staging_),
                    count
                );
            }
        } else {
            if (!channel_ || !channel_->host_buf[epoch_][0] || !channel_->host_buf[epoch_][1]) {
                return;
            }

            // 0. Ensure epoch's buffers are vacated from previous cycle
            spin_yield_wait([&]() {
                return channel_->exit_epoch[epoch_].load(std::memory_order_acquire) == 2;
            });

            // 1. Asynchronously copy local device buffer to host pinned buffer
            CUDA_CHECK(cudaMemcpyAsync(channel_->host_buf[epoch_][rank_], sendrecvbuf, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // 2. Rendezvous arrival
            channel_->arrival_epoch[epoch_].fetch_add(1, std::memory_order_release);
            spin_yield_wait([&]() {
                return channel_->arrival_epoch[epoch_].load(std::memory_order_acquire) == 2;
            });

            // 3. Reduction & broadcast
            if (rank_ == 0) {
                channel_->exit_epoch[epoch_].store(0, std::memory_order_relaxed);

                if (type == DataType::FP16) {
                    avx2_vector_add_half(
                        reinterpret_cast<half*>(channel_->host_buf[epoch_][0]),
                        reinterpret_cast<const half*>(channel_->host_buf[epoch_][1]),
                        count
                    );
                } else if (type == DataType::FP32) {
                    float* h0 = reinterpret_cast<float*>(channel_->host_buf[epoch_][0]);
                    const float* h1 = reinterpret_cast<const float*>(channel_->host_buf[epoch_][1]);
                    for (size_t i = 0; i < count; ++i) h0[i] += h1[i];
                }

                // Broadcast reduced sum to rank 1 host buffer
                std::memcpy(channel_->host_buf[epoch_][1], channel_->host_buf[epoch_][0], bytes);

                // Publish completion
                channel_->completion_epoch[epoch_].store(1, std::memory_order_release);
            } else {
                // Rank 1 waits for Rank 0 completion
                spin_yield_wait([&]() {
                    return channel_->completion_epoch[epoch_].load(std::memory_order_acquire) == 1;
                });
            }

            // 4. Async copy reduced sum back to local device buffer
            CUDA_CHECK(cudaMemcpyAsync(sendrecvbuf, channel_->host_buf[epoch_][rank_], bytes, cudaMemcpyHostToDevice, stream));

            // 5. Exit handshake
            channel_->exit_epoch[epoch_].fetch_add(1, std::memory_order_release);
            if (rank_ == 0) {
                spin_yield_wait([&]() {
                    return channel_->exit_epoch[epoch_].load(std::memory_order_acquire) == 2;
                });
                channel_->arrival_epoch[epoch_].store(0, std::memory_order_relaxed);
                channel_->completion_epoch[epoch_].store(0, std::memory_order_relaxed);
            }

            epoch_ = 1 - epoch_;
        }
    }

    void sync(cudaStream_t stream) override {
        if (world_size_ <= 1) return;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    bool is_peer_accessible() const override { return peer_accessible_; }

private:
    int rank_;
    int world_size_;
    std::vector<int> device_ids_;
    bool peer_accessible_;
    void* d_peer_staging_;
    cudaEvent_t local_event_;
    std::shared_ptr<HostStagingChannel> channel_;
    int epoch_;
};

std::unique_ptr<ICommunicator> create_communicator(
    int rank,
    int world_size,
    const std::vector<int>& device_ids,
    bool force_staging,
    std::shared_ptr<HostStagingChannel> shared_channel
) {
    return std::make_unique<P2PCommunicator>(rank, world_size, device_ids, force_staging, shared_channel);
}

} // namespace qwen
