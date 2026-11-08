#ifdef _WIN32

#include "communicator.cuh"
#include "cuda_guard.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace qwen {

// Kernel for elementwise sum reduction between local and peer buffers
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

struct WinP2PSharedState {
    std::mutex mtx;
    std::condition_variable cv;
    int step_counter{0};
    void* host_staging[2]{nullptr, nullptr};
    void* device_peer_buf[2]{nullptr, nullptr};
    cudaEvent_t peer_events[2]{nullptr, nullptr};
    size_t current_count{0};
    DataType current_type{DataType::FP16};
};

static WinP2PSharedState g_win_p2p_state;

class WindowsP2PCommunicator : public ICommunicator {
public:
    WindowsP2PCommunicator(
        int rank,
        int world_size,
        const std::vector<int>& device_ids,
        bool force_staging
    ) : rank_(rank),
        world_size_(world_size),
        device_ids_(device_ids),
        peer_accessible_(false),
        d_peer_staging_(nullptr),
        h_host_staging_(nullptr),
        local_event_(nullptr) {

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
                cudaGetLastError(); // clear error if already enabled
                peer_accessible_ = true;
            }
        }

        // Allocate 64 MB staging buffer
        size_t max_bytes = 64ULL << 20;
        if (peer_accessible_) {
            CUDA_CHECK(cudaMalloc(&d_peer_staging_, max_bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&local_event_, cudaEventDisableTiming));

            std::unique_lock<std::mutex> lock(g_win_p2p_state.mtx);
            g_win_p2p_state.device_peer_buf[rank_] = d_peer_staging_;
            g_win_p2p_state.peer_events[rank_] = local_event_;
            std::cout << "[WindowsP2PCommunicator] Rank " << rank_
                      << " enabled direct BAR1 P2P with device " << peer_dev << " (TCC mode)." << std::endl;
        } else {
            CUDA_CHECK(cudaHostAlloc(&h_host_staging_, max_bytes, cudaHostAllocMapped));
            std::unique_lock<std::mutex> lock(g_win_p2p_state.mtx);
            g_win_p2p_state.host_staging[rank_] = h_host_staging_;
            std::cout << "[WindowsP2PCommunicator] Rank " << rank_
                      << " fallback to double-buffered host staging (WDDM mode)." << std::endl;
        }
    }

    ~WindowsP2PCommunicator() override {
        if (world_size_ <= 1) return;

        CudaDeviceGuard guard(device_ids_[rank_]);
        if (d_peer_staging_) {
            cudaFree(d_peer_staging_);
            d_peer_staging_ = nullptr;
        }
        if (h_host_staging_) {
            cudaFreeHost(h_host_staging_);
            h_host_staging_ = nullptr;
        }
        if (local_event_) {
            cudaEventDestroy(local_event_);
            local_event_ = nullptr;
        }
    }

    void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type, cudaStream_t stream) override {
        if (world_size_ <= 1) return;

        size_t elem_size = (type == DataType::FP16) ? sizeof(half) :
                           (type == DataType::FP32) ? sizeof(float) : sizeof(int32_t);
        size_t bytes = count * elem_size;

        if (peer_accessible_) {
            // Direct P2P All-Reduce path (CUDA Graph capture compatible)
            int peer_rank = 1 - rank_;
            void* peer_staging = g_win_p2p_state.device_peer_buf[peer_rank];
            cudaEvent_t peer_event = g_win_p2p_state.peer_events[peer_rank];

            // 1. Copy local buffer into peer's staging buffer across PCIe BAR1
            CUDA_CHECK(cudaMemcpyPeerAsync(peer_staging, device_ids_[peer_rank],
                                           sendrecvbuf, device_ids_[rank_],
                                           bytes, stream));

            // 2. Record completion event on local stream
            CUDA_CHECK(cudaEventRecord(local_event_, stream));

            // 3. Wait on peer's copy completion
            CUDA_CHECK(cudaStreamWaitEvent(stream, peer_event, 0));

            // 4. Elementwise add peer's contribution into local sendrecvbuf
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
            // WDDM Host-pinned staging buffer fallback (host synchronized)
            CUDA_CHECK(cudaMemcpyAsync(h_host_staging_, sendrecvbuf, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Host rendezvous barrier
            {
                std::unique_lock<std::mutex> lock(g_win_p2p_state.mtx);
                int cur_step = g_win_p2p_state.step_counter;
                g_win_p2p_state.current_count = count;
                g_win_p2p_state.current_type = type;

                if (rank_ == 0) {
                    g_win_p2p_state.step_counter++;
                    g_win_p2p_state.cv.wait(lock, [&]() {
                        return g_win_p2p_state.step_counter == cur_step + 2;
                    });
                } else {
                    // Rank 1 performs the summation in host memory
                    void* h0 = g_win_p2p_state.host_staging[0];
                    void* h1 = g_win_p2p_state.host_staging[1];

                    if (type == DataType::FP16) {
                        half* h0_ptr = reinterpret_cast<half*>(h0);
                        half* h1_ptr = reinterpret_cast<half*>(h1);
                        for (size_t i = 0; i < count; ++i) {
                            float s = __half2float(h0_ptr[i]) + __half2float(h1_ptr[i]);
                            h0_ptr[i] = __float2half(s);
                            h1_ptr[i] = h0_ptr[i];
                        }
                    } else if (type == DataType::FP32) {
                        float* h0_ptr = reinterpret_cast<float*>(h0);
                        float* h1_ptr = reinterpret_cast<float*>(h1);
                        for (size_t i = 0; i < count; ++i) {
                            float s = h0_ptr[i] + h1_ptr[i];
                            h0_ptr[i] = s;
                            h1_ptr[i] = s;
                        }
                    }
                    g_win_p2p_state.step_counter++;
                    g_win_p2p_state.cv.notify_all();
                }
            }

            // Copy sum back to device
            CUDA_CHECK(cudaMemcpyAsync(sendrecvbuf, h_host_staging_, bytes, cudaMemcpyHostToDevice, stream));
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
    void* h_host_staging_;
    cudaEvent_t local_event_;
};

std::unique_ptr<ICommunicator> create_communicator(
    int rank,
    int world_size,
    const std::vector<int>& device_ids,
    bool force_staging
) {
    return std::make_unique<WindowsP2PCommunicator>(rank, world_size, device_ids, force_staging);
}

} // namespace qwen

#endif // _WIN32

