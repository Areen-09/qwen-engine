#ifndef _WIN32

#include "communicator.cuh"
#include <nccl.h>
#include <stdexcept>
#include <iostream>

namespace qwen {

#define NCCL_CHECK(cmd) do { \
    ncclResult_t res = cmd; \
    if (res != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(res)); \
    } \
} while(0)

class NcclCommunicator : public ICommunicator {
public:
    NcclCommunicator(
        int rank,
        int world_size,
        const std::vector<int>& device_ids,
        bool /*force_staging*/
    ) : rank_(rank), world_size_(world_size), device_ids_(device_ids), comm_(nullptr) {
        if (world_size_ <= 1) return;

        cudaSetDevice(device_ids_[rank_]);
        // Simple static unique ID coordination for single-node 2-GPU
        static ncclUniqueId id;
        static bool id_initialized = false;
        if (!id_initialized) {
            ncclGetUniqueId(&id);
            id_initialized = true;
        }

        NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, id, rank_));
    }

    ~NcclCommunicator() override {
        if (comm_) {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
    }

    void all_reduce_sum(void* sendrecvbuf, size_t count, DataType type, cudaStream_t stream) override {
        if (world_size_ <= 1 || !comm_) return;

        ncclDataType_t nccl_type = ncclFloat16;
        if (type == DataType::FP32) nccl_type = ncclFloat32;
        else if (type == DataType::INT32) nccl_type = ncclInt32;

        NCCL_CHECK(ncclAllReduce(sendrecvbuf, sendrecvbuf, count, nccl_type, ncclSum, comm_, stream));
    }

    void sync(cudaStream_t stream) override {
        cudaStreamSynchronize(stream);
    }

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    bool is_peer_accessible() const override { return true; }

private:
    int rank_;
    int world_size_;
    std::vector<int> device_ids_;
    ncclComm_t comm_;
};

std::unique_ptr<ICommunicator> create_communicator(
    int rank,
    int world_size,
    const std::vector<int>& device_ids,
    bool force_staging
) {
    return std::make_unique<NcclCommunicator>(rank, world_size, device_ids, force_staging);
}

} // namespace qwen

#endif // !_WIN32

