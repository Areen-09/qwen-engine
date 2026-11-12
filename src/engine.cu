#include "engine.cuh"
#include "kernels.cuh"
#include "file_map.hpp"
#include "cuda_guard.cuh"
#include "exceptions.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <thread>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace qwen {

template <typename Predicate>
static inline void engine_spin_yield(Predicate&& pred) {
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

static std::string get_rank1_path(const std::string& path) {
    std::string s = path;
    size_t pos = s.find("rank0");
    if (pos != std::string::npos) {
        s.replace(pos, 5, "rank1");
        return s;
    }
    pos = s.find("rank_0");
    if (pos != std::string::npos) {
        s.replace(pos, 6, "rank_1");
        return s;
    }
    std::string dir = "";
    size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
        dir = s.substr(0, slash + 1);
    }
    std::string candidate = dir + "rank1.qwenbin";
    std::ifstream f(candidate, std::ios::binary);
    if (f.good()) return candidate;

    return path;
}

__global__ void residual_add_kernel(half* __restrict__ dst, const half* __restrict__ src, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
    }
}

static void launch_residual_add(half* dst, const half* src, int count, cudaStream_t stream) {
    int block = 256;
    int grid = (count + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, stream>>>(dst, src, count);
}

QwenEngine::QwenEngine(const EngineConfig& config)
    : qwenbin_path_(config.model_path),
      device_ids_(config.device_ids),
      force_wddm_staging_(config.force_wddm_staging),
      max_seq_len_(config.max_seq_len),
      quant_mode_(QuantMode::FP16),
      is_generating_(false),
      file_map_(nullptr),
      mapped_data_(nullptr),
      mapped_size_(0),
      arena_(nullptr),
      stream_(nullptr),
      d_embed_weight_(nullptr),
      d_embed_scale_(nullptr),
      d_lm_head_weight_(nullptr),
      d_lm_head_scale_(nullptr),
      d_final_norm_(nullptr),
      d_k_cache_(nullptr),
      d_v_cache_(nullptr),
      num_gqa_layers_(0),
      d_deltanet_state_(nullptr),
      d_conv1d_state_(nullptr),
      num_deltanet_layers_(0),
      d_control_(nullptr),
      h_pinned_token_(nullptr),
      d_mapped_pinned_token_(nullptr),
      h_pinned_cand_(nullptr),
      d_mapped_pinned_cand_(nullptr),
      decode_graph_(nullptr),
      decode_graph_exec_(nullptr),
      graph_captured_(false),
      seq_pos_(0),
      last_token_id_(0),
      temperature_(0.0f),
      weights_bytes_(0),
      kv_cache_bytes_(0),
      workspace_bytes_(0) {

    if (device_ids_.empty()) {
        device_ids_ = {0};
    }

    CudaDeviceGuard guard(device_id());
    CUDA_CHECK(cudaStreamCreate(&stream_));
    load_model_file(qwenbin_path_);
    setup_arena();
    capture_decode_graph();
}

QwenEngine::QwenEngine(const std::string& qwenbin_path, int device_id, size_t max_seq_len)
    : QwenEngine(EngineConfig{qwenbin_path, "", {device_id}, max_seq_len, false}) {}

QwenEngine::QwenEngine(const std::string& qwenbin_path, const std::vector<int>& device_ids, size_t max_seq_len)
    : QwenEngine(EngineConfig{qwenbin_path, "", device_ids, max_seq_len, false}) {}

QwenEngine::~QwenEngine() {
    if (worker_thread_.joinable()) {
        dispatch_worker(WorkerOp::SHUTDOWN);
        worker_thread_.request_stop();
        worker_thread_.join();
    }
    CudaDeviceGuard guard(device_id());
    if (decode_graph_exec_) {
        cudaGraphExecDestroy(decode_graph_exec_);
        decode_graph_exec_ = nullptr;
    }
    if (decode_graph_) {
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
    }
    if (h_pinned_cand_) {
        cudaFreeHost(h_pinned_cand_);
        h_pinned_cand_ = nullptr;
    }
    if (h_pinned_token_) {
        cudaFreeHost(h_pinned_token_);
        h_pinned_token_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    file_map_.reset();
    mapped_data_ = nullptr;
    mapped_size_ = 0;
}

void QwenEngine::load_model_file(const std::string& qwenbin_path) {
    file_map_ = std::make_unique<PlatformFileMap>(qwenbin_path);
    mapped_data_ = static_cast<const uint8_t*>(file_map_->data());
    mapped_size_ = file_map_->size();

    if (mapped_size_ < sizeof(uint32_t) * 2) {
        throw QwenError("QwenEngine: File too small to contain valid header in " + qwenbin_path);
    }

    uint32_t magic = *reinterpret_cast<const uint32_t*>(mapped_data_);
    uint32_t version = *reinterpret_cast<const uint32_t*>(mapped_data_ + sizeof(uint32_t));

    if (magic != 0x5157454E) {
        throw QwenError("QwenEngine: Invalid magic in " + qwenbin_path);
    }

    if (version == 2) {
        const QwenBinHeaderV2* h2 = reinterpret_cast<const QwenBinHeaderV2*>(mapped_data_);
        if (mapped_size_ < h2->layer_table_offset + h2->num_layers * sizeof(LayerDescriptor)) {
            throw QwenError("QwenEngine: Truncated .qwenbin v2 container");
        }
        const LayerDescriptor* layers = reinterpret_cast<const LayerDescriptor*>(mapped_data_ + h2->layer_table_offset);
        model_config_ = ModelConfig(*h2, layers);
        quant_mode_ = static_cast<QuantMode>(h2->quant_type);
        // Retain user-configured max_seq_len_ (default 4096), capping at container max_seq_len if smaller.
        // Prevents allocating excessive KV cache when GGUF specifies a 256k theoretical context window.
        if (max_seq_len_ == 0 || max_seq_len_ > 32768) {
            max_seq_len_ = (h2->max_seq_len > 0 && h2->max_seq_len <= 8192) ? h2->max_seq_len : 4096;
        } else if (h2->max_seq_len > 0 && max_seq_len_ > h2->max_seq_len) {
            max_seq_len_ = h2->max_seq_len;
        }
    } else if (version == 1) {
        const QWQwenHeader* h1 = reinterpret_cast<const QWQwenHeader*>(mapped_data_);
        QwenBinHeaderV2 synth_h2{};
        synth_h2.magic = h1->magic;
        synth_h2.version = 2;
        synth_h2.hidden_size = h1->hidden_size;
        synth_h2.intermediate_size = h1->intermediate_size;
        synth_h2.num_layers = h1->num_layers;
        synth_h2.num_attention_heads = h1->num_q_heads;
        synth_h2.num_kv_heads = h1->num_kv_heads;
        synth_h2.head_dim = h1->head_dim;
        synth_h2.vocab_size = h1->vocab_size;
        synth_h2.max_seq_len = MAX_SEQ_LEN;
        synth_h2.quant_type = h1->quant_mode;
        synth_h2.tie_word_embeddings = 1;
        synth_h2.rms_norm_eps = RMS_NORM_EPS;
        synth_h2.rope_theta = ROPE_THETA;
        synth_h2.layer_table_offset = 0;
        synth_h2.embed_tokens_offset = h1->embed_weight_offset;
        synth_h2.final_norm_offset = h1->final_norm_offset;
        synth_h2.lm_head_offset = h1->embed_weight_offset;
        synth_h2.weights_offset = 4096;
        synth_h2.reserved[0] = h1->embed_scale_offset;

        std::vector<LayerDescriptor> synth_layers(h1->num_layers);
        for (uint32_t l = 0; l < h1->num_layers; ++l) {
            synth_layers[l].type = LAYER_GQA;
            synth_layers[l].reserved = 0;
            synth_layers[l].norm1_weight_offset = h1->layer_offsets[l][LayerTensorIdx::INPUT_LAYERNORM];
            synth_layers[l].norm2_weight_offset = h1->layer_offsets[l][LayerTensorIdx::POST_ATTENTION_LAYERNORM];
            synth_layers[l].qkv_weight_offset = h1->layer_offsets[l][LayerTensorIdx::Q_WEIGHT];
            synth_layers[l].q_bias_offset = h1->layer_offsets[l][LayerTensorIdx::Q_BIAS];
            synth_layers[l].k_bias_offset = h1->layer_offsets[l][LayerTensorIdx::K_BIAS];
            synth_layers[l].v_bias_offset = h1->layer_offsets[l][LayerTensorIdx::V_BIAS];
            synth_layers[l].attn_out_weight_offset = h1->layer_offsets[l][LayerTensorIdx::O_WEIGHT];
            synth_layers[l].mlp_gate_offset = h1->layer_offsets[l][LayerTensorIdx::GATE_UP_WEIGHT];
            synth_layers[l].mlp_up_offset = h1->layer_offsets[l][LayerTensorIdx::GATE_UP_WEIGHT];
            synth_layers[l].mlp_down_offset = h1->layer_offsets[l][LayerTensorIdx::DOWN_WEIGHT];
            synth_layers[l].k_weight_offset = h1->layer_offsets[l][LayerTensorIdx::K_WEIGHT];
            synth_layers[l].v_weight_offset = h1->layer_offsets[l][LayerTensorIdx::V_WEIGHT];
            synth_layers[l].q_scale_offset = h1->layer_offsets[l][LayerTensorIdx::Q_SCALE];
            synth_layers[l].k_scale_offset = h1->layer_offsets[l][LayerTensorIdx::K_SCALE];
            synth_layers[l].v_scale_offset = h1->layer_offsets[l][LayerTensorIdx::V_SCALE];
            synth_layers[l].attn_out_scale_offset = h1->layer_offsets[l][LayerTensorIdx::O_SCALE];
            synth_layers[l].mlp_gate_scale_offset = h1->layer_offsets[l][LayerTensorIdx::GATE_UP_SCALE];
            synth_layers[l].mlp_up_scale_offset = h1->layer_offsets[l][LayerTensorIdx::GATE_UP_SCALE];
            synth_layers[l].mlp_down_scale_offset = h1->layer_offsets[l][LayerTensorIdx::DOWN_SCALE];
        }
        model_config_ = ModelConfig(synth_h2, synth_layers.data());
        quant_mode_ = static_cast<QuantMode>(h1->quant_mode);
        max_seq_len_ = MAX_SEQ_LEN;
    } else {
        throw QwenError("QwenEngine: Unsupported format version " + std::to_string(version));
    }
}

void QwenEngine::setup_arena() {
    int n_layers = model_config_.num_layers();
    layers_.resize(n_layers);
    gqa_layer_idx_.assign(n_layers, -1);
    num_gqa_layers_ = 0;
    num_deltanet_layers_ = 0;

    for (int l = 0; l < n_layers; ++l) {
        if (model_config_.layer(l).type == LAYER_GQA) {
            gqa_layer_idx_[l] = static_cast<int>(num_gqa_layers_++);
        } else {
            num_deltanet_layers_++;
        }
    }

    uint64_t w_offset = model_config_.header().weights_offset;
    weights_bytes_ = (mapped_size_ > w_offset) ? (mapped_size_ - w_offset) : 0;

    // 1. KV Cache allocation (GQA layers only)
    size_t kv_elements = num_gqa_layers_ * model_config_.num_kv_heads() * max_seq_len_ * model_config_.head_dim();
    kv_cache_bytes_ = 2 * kv_elements * sizeof(half);

    // 2. DeltaNet State allocation (FP32)
    size_t deltanet_state_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * 128 * 128;
    size_t deltanet_state_bytes = deltanet_state_elements * sizeof(float);

    // 3. Conv1d History State allocation (FP16, history_len = 3)
    size_t conv1d_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * model_config_.head_dim() * 3;
    size_t conv1d_bytes = conv1d_elements * sizeof(half);

    // 4. Dynamic Workspace Buffers (~128 MB)
    size_t max_dim = static_cast<size_t>(std::max(model_config_.hidden_size(), model_config_.intermediate_size()));
    workspace_bytes_ = PREFILL_MAX_CHUNK * max_dim * 2 * 2 + model_config_.vocab_size() * 2 + (64ULL << 20);
    if (workspace_bytes_ < (128ULL << 20)) {
        workspace_bytes_ = 128ULL << 20;
    }

    size_t dcb_bytes = 256;
    size_t total_arena_bytes = weights_bytes_ + kv_cache_bytes_ + deltanet_state_bytes + conv1d_bytes + workspace_bytes_ + dcb_bytes;

    std::cout << "[QwenEngine] Initializing GPU Arena (" << n_layers << " layers: "
              << num_gqa_layers_ << " GQA, " << num_deltanet_layers_ << " DeltaNet)..." << std::endl;
    std::cout << "  - Weights Region:   " << (weights_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - Selective KV Cache: " << (kv_cache_bytes_ / (1024.0 * 1024.0)) << " MB ("
              << max_seq_len_ << " max tokens)" << std::endl;
    if (num_deltanet_layers_ > 0) {
        std::cout << "  - DeltaNet Recurrent State: " << (deltanet_state_bytes / (1024.0 * 1024.0)) << " MB (FP32)" << std::endl;
        std::cout << "  - Conv1d History Cache:    " << (conv1d_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    }
    std::cout << "  - Workspace Region: " << (workspace_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - Total Allocation: " << (total_arena_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    arena_ = std::make_unique<GpuArena>(total_arena_bytes);

    void* d_weights_base = arena_->allocate(weights_bytes_, 256);
    CUDA_CHECK(cudaMemcpy(d_weights_base, mapped_data_ + w_offset, weights_bytes_, cudaMemcpyHostToDevice));

    auto get_d_ptr = [d_weights_base, w_offset](uint64_t offset) -> const void* {
        if (offset == 0) return nullptr;
        return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(d_weights_base) + (offset - w_offset));
    };

    d_embed_weight_ = get_d_ptr(model_config_.header().embed_tokens_offset);
    d_embed_scale_  = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().reserved[0]));
    if (model_config_.header().lm_head_offset != 0 && model_config_.header().lm_head_offset != model_config_.header().embed_tokens_offset) {
        d_lm_head_weight_ = get_d_ptr(model_config_.header().lm_head_offset);
        d_lm_head_scale_  = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().reserved[1]));
    } else {
        d_lm_head_weight_ = d_embed_weight_;
        d_lm_head_scale_  = d_embed_scale_;
    }
    d_final_norm_   = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().final_norm_offset));

    for (int l = 0; l < n_layers; ++l) {
        const auto& desc = model_config_.layer(l);
        layers_[l].input_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm1_weight_offset));
        layers_[l].post_attention_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm2_weight_offset));

        if (desc.type == LAYER_GQA) {
            layers_[l].q_weight = get_d_ptr(desc.qkv_weight_offset);
            layers_[l].q_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.q_bias_offset));
            layers_[l].k_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.k_bias_offset));
            layers_[l].v_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.v_bias_offset));
            layers_[l].o_weight = get_d_ptr(desc.attn_out_weight_offset);
            layers_[l].k_weight = get_d_ptr(desc.k_weight_offset);
            layers_[l].v_weight = get_d_ptr(desc.v_weight_offset);
            layers_[l].q_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.q_scale_offset));
            layers_[l].k_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.k_scale_offset));
            layers_[l].v_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.v_scale_offset));
            layers_[l].o_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.attn_out_scale_offset));
        } else {
            layers_[l].conv1d_weight    = reinterpret_cast<const half*>(get_d_ptr(desc.conv1d_weight_offset));
            layers_[l].conv1d_bias      = reinterpret_cast<const half*>(get_d_ptr(desc.conv1d_bias_offset));
            layers_[l].in_proj_weight   = get_d_ptr(desc.in_proj_weight_offset);
            layers_[l].decay_weight     = reinterpret_cast<const half*>(get_d_ptr(desc.decay_weight_offset));
            layers_[l].delta_out_weight = get_d_ptr(desc.delta_out_weight_offset);
            layers_[l].delta_out_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.attn_out_scale_offset));
        }

        layers_[l].gate_up_weight = get_d_ptr(desc.mlp_gate_offset);
        layers_[l].gate_up_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_gate_scale_offset));
        layers_[l].down_weight    = get_d_ptr(desc.mlp_down_offset);
        layers_[l].down_scale     = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_down_scale_offset));
    }

    if (num_gqa_layers_ > 0) {
        d_k_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements * sizeof(half), 256));
        d_v_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements * sizeof(half), 256));
        CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, kv_elements * sizeof(half), stream_));
        CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, kv_elements * sizeof(half), stream_));
    }

    if (num_deltanet_layers_ > 0) {
        d_deltanet_state_ = reinterpret_cast<float*>(arena_->allocate(deltanet_state_bytes, 256));
        d_conv1d_state_   = reinterpret_cast<half*>(arena_->allocate(conv1d_bytes, 256));
        CUDA_CHECK(cudaMemsetAsync(d_deltanet_state_, 0, deltanet_state_bytes, stream_));
        CUDA_CHECK(cudaMemsetAsync(d_conv1d_state_, 0, conv1d_bytes, stream_));
    }

    int h_dim = model_config_.hidden_size();
    int i_dim = model_config_.intermediate_size();
    int q_dim = model_config_.num_attention_heads() * model_config_.head_dim();
    int kv_dim = model_config_.num_kv_heads() * model_config_.head_dim();

    ws_.d_act_0         = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    ws_.d_act_1         = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    ws_.d_q             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * q_dim * sizeof(half), 256));
    ws_.d_k             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * kv_dim * sizeof(half), 256));
    ws_.d_v             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * kv_dim * sizeof(half), 256));
    ws_.d_attn_out      = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    ws_.d_mlp_act       = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * i_dim * sizeof(half), 256));
    ws_.d_splitk_out    = reinterpret_cast<float*>(arena_->allocate(model_config_.num_attention_heads() * FLASH_DECODE_SPLITS * model_config_.head_dim() * sizeof(float), 256));
    ws_.d_splitk_lse    = reinterpret_cast<float*>(arena_->allocate(model_config_.num_attention_heads() * FLASH_DECODE_SPLITS * 2 * sizeof(float), 256));
    ws_.d_logits        = reinterpret_cast<half*>(arena_->allocate(model_config_.vocab_size() * sizeof(half), 256));
    ws_.d_sample_val    = reinterpret_cast<float*>(arena_->allocate(256 * sizeof(float), 256));
    ws_.d_sample_idx    = reinterpret_cast<int32_t*>(arena_->allocate(256 * sizeof(int32_t), 256));
    ws_.d_prompt_tokens = reinterpret_cast<int32_t*>(arena_->allocate(max_seq_len_ * sizeof(int32_t), 256));

    size_t q8_offset = 65536;
    size_t q8_bytes = ((static_cast<size_t>(i_dim) + 255) / 256) * 256;
    size_t scale_offset = q8_offset + q8_bytes;
    size_t num_groups = (static_cast<size_t>(i_dim) + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE;
    size_t scale_bytes = ((num_groups * sizeof(float) + 255) / 256) * 256;
    size_t sum8_offset = scale_offset + scale_bytes;

    ws_.d_act_q8    = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + q8_offset);
    ws_.d_act_scale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + scale_offset);
    ws_.d_act_sum8  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + sum8_offset);

    d_control_ = reinterpret_cast<DeviceControlBlock*>(arena_->allocate(sizeof(DeviceControlBlock), 256));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_token_), sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mapped_pinned_token_), h_pinned_token_, 0));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_cand_), sizeof(PinnedCandidate), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mapped_pinned_cand_), h_pinned_cand_, 0));
    h_pinned_cand_->val = -1e30f;
    h_pinned_cand_->idx = -1;

    DeviceControlBlock init_dcb = { 0, 0, 0.0f, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &init_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

    if (device_ids_.size() > 1) {
        staging_channel_ = std::make_shared<HostStagingChannel>();
        comm_ = create_communicator(0, 2, device_ids_, force_wddm_staging_, staging_channel_);
        worker_thread_ = std::jthread([this](std::stop_token st) { worker_loop(st); });
        wait_worker_done();
        worker_ack_.store(0, std::memory_order_relaxed);
    } else {
        comm_ = create_communicator(0, 1, device_ids_, force_wddm_staging_);
    }
}

void QwenEngine::reset_session() {
    if (device_ids_.size() > 1) {
        dispatch_worker(WorkerOp::RESET);
    }
    CudaDeviceGuard guard(device_id());
    cached_tokens_.clear();
    seq_pos_ = 0;
    last_token_id_ = 0;
    DeviceControlBlock reset_dcb = { 0, 0, temperature_, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &reset_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
    *h_pinned_token_ = 0;
    if (h_pinned_cand_) {
        h_pinned_cand_->val = -1e30f;
        h_pinned_cand_->idx = -1;
    }

    if (d_deltanet_state_ && num_deltanet_layers_ > 0) {
        size_t state_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * 128 * 128;
        CUDA_CHECK(cudaMemsetAsync(d_deltanet_state_, 0, state_elements * sizeof(float), stream_));
    }
    if (d_conv1d_state_ && num_deltanet_layers_ > 0) {
        size_t conv1d_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * model_config_.head_dim() * 3;
        CUDA_CHECK(cudaMemsetAsync(d_conv1d_state_, 0, conv1d_elements * sizeof(half), stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (device_ids_.size() > 1) {
        wait_worker_done();
    }
}

void QwenEngine::execute_gqa_decode_layer(int l) {
    int gqa_idx = gqa_layer_idx_[l];
    size_t kv_stride = static_cast<size_t>(model_config_.num_kv_heads()) * max_seq_len_ * model_config_.head_dim();
    half* layer_k = d_k_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);
    half* layer_v = d_v_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);

    int h_dim = model_config_.hidden_size();
    int num_q_heads = model_config_.num_attention_heads();
    int num_kv_heads = model_config_.num_kv_heads();
    int head_dim = model_config_.head_dim();
    int rotary_dim = (head_dim == 256) ? 64 : head_dim;
    float rope_theta = model_config_.rope_theta();
    half* residual_ptr = (device_ids_.size() > 1) ? nullptr : ws_.d_act_0;
    half* output_ptr   = (device_ids_.size() > 1) ? ws_.d_act_1 : ws_.d_act_0;

    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, 1, stream_, h_dim);

        QKVProjectionView qkv_view = {
            layers_[l].q_weight, nullptr, ws_.d_q,
            layers_[l].k_weight, nullptr, ws_.d_k,
            layers_[l].v_weight, nullptr, ws_.d_v
        };
        launch_gemv_qkv_fp16(qkv_view, ws_.d_act_1, stream_, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            ws_.d_q, ws_.d_k, ws_.d_v,
            layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
            layer_k, layer_v, d_control_, 1, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_flash_decode(
            ws_.d_q, layer_k, layer_v,
            ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
        );

        launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, residual_ptr, output_ptr, h_dim, num_q_heads * head_dim, stream_);
    } else {
        launch_rmsnorm_q8_1(
            ws_.d_act_0, layers_[l].input_layernorm, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            h_dim
        );

        QKVProjectionView qkv_view = {
            layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
            layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
            layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
        };
        launch_gemv_qkv_w4a8(qkv_view, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            ws_.d_q, ws_.d_k, ws_.d_v,
            layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
            layer_k, layer_v, d_control_, 1, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_flash_decode(
            ws_.d_q, layer_k, layer_v,
            ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8
        );

        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(layers_[l].o_weight),
            layers_[l].o_scale,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            residual_ptr, output_ptr,
            h_dim, num_q_heads * head_dim, stream_
        );
    }
}

void QwenEngine::execute_deltanet_decode_layer(int l) {
    int h_dim = model_config_.hidden_size();
    int num_heads = model_config_.num_attention_heads();
    int head_dim = model_config_.head_dim();
    int q_dim = num_heads * head_dim;
    half* residual_ptr = (device_ids_.size() > 1) ? nullptr : ws_.d_act_0;
    half* output_ptr   = (device_ids_.size() > 1) ? ws_.d_act_1 : ws_.d_act_0;

    // 1. RMSNorm
    launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, 1, stream_, h_dim);

    // 2. Depthwise Conv1d Step
    int conv_idx = 0;
    for (int prev = 0; prev < l; ++prev) {
        if (model_config_.layer(prev).type == LAYER_DELTANET) conv_idx++;
    }
    half* layer_conv_state = d_conv1d_state_ + (static_cast<size_t>(conv_idx) * q_dim * 3);
    launch_depthwise_conv1d_step(
        ws_.d_act_1, layers_[l].conv1d_weight, layers_[l].conv1d_bias,
        layer_conv_state, ws_.d_q, q_dim, 4, stream_
    );

    // 3. DeltaNet Decode State Step
    float* layer_s_state = d_deltanet_state_ + (static_cast<size_t>(conv_idx) * num_heads * 128 * 128);
    launch_deltanet_decode_sm75(
        ws_.d_q, ws_.d_q, ws_.d_q, layers_[l].decay_weight,
        layer_s_state, ws_.d_attn_out, 1,
        num_heads, head_dim, stream_
    );

    // 4. Out projection
    if (quant_mode_ == QuantMode::FP16) {
        launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].delta_out_weight), ws_.d_attn_out, residual_ptr, output_ptr, h_dim, q_dim, stream_);
    } else {
        launch_gemv_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].delta_out_weight), layers_[l].delta_out_scale, ws_.d_attn_out, residual_ptr, output_ptr, h_dim, q_dim, stream_);
    }
}

void QwenEngine::execute_decode_pass() {
    int h_dim = model_config_.hidden_size();
    int i_dim = model_config_.intermediate_size();
    int n_layers = model_config_.num_layers();

    // 1. Embedding lookup
    if (quant_mode_ == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(d_embed_weight_), nullptr, ws_.d_act_0, d_control_, 1, stream_, h_dim
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, nullptr, ws_.d_act_0, d_control_, 1, stream_, h_dim
        );
    }

    // 2. Transformer layers
    for (int l = 0; l < n_layers; ++l) {
        if (model_config_.layer(l).type == LAYER_GQA) {
            execute_gqa_decode_layer(l);
        } else {
            execute_deltanet_decode_layer(l);
        }

        // Tensor Parallel All-Reduce 1 (Attention / DeltaNet Out Projection)
        if (device_ids_.size() > 1 && comm_) {
            comm_->all_reduce_sum(ws_.d_act_1, h_dim, DataType::FP16, stream_);
            launch_residual_add(ws_.d_act_0, ws_.d_act_1, h_dim, stream_);
        }

        // SwiGLU MLP
        half* mlp_residual_ptr = (device_ids_.size() > 1) ? nullptr : ws_.d_act_0;
        half* mlp_output_ptr   = (device_ids_.size() > 1) ? ws_.d_act_1 : ws_.d_act_0;

        if (quant_mode_ == QuantMode::FP16) {
            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, 1, stream_, h_dim);
            launch_gemv_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, h_dim, i_dim, stream_);
            launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, mlp_residual_ptr, mlp_output_ptr, h_dim, i_dim, stream_);
        } else {
            launch_rmsnorm_q8_1(
                ws_.d_act_0, layers_[l].post_attention_layernorm, nullptr,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
                h_dim
            );
            launch_gemv_dual_gate_up_swiglu_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight),
                layers_[l].gate_up_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_mlp_act, h_dim, i_dim, stream_
            );
            launch_quantize_fp16_to_q8_1(ws_.d_mlp_act, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, i_dim, stream_);
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].down_weight),
                layers_[l].down_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                mlp_residual_ptr, mlp_output_ptr,
                h_dim, i_dim, stream_
            );
        }

        // Tensor Parallel All-Reduce 2 (MLP Down Projection)
        if (device_ids_.size() > 1 && comm_) {
            comm_->all_reduce_sum(ws_.d_act_1, h_dim, DataType::FP16, stream_);
            launch_residual_add(ws_.d_act_0, ws_.d_act_1, h_dim, stream_);
        }
    }

    // 3. Final RMSNorm + LM Head
    int vocab_size = model_config_.vocab_size();
    int rank_vocab = (device_ids_.size() > 1) ? (vocab_size / 2) : vocab_size;

    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(ws_.d_act_0, d_final_norm_, ws_.d_act_1, 1, stream_, h_dim);
        launch_gemv_fp16(reinterpret_cast<const half*>(d_lm_head_weight_), ws_.d_act_1, nullptr, ws_.d_logits, rank_vocab, h_dim, stream_);
    } else {
        launch_rmsnorm_q8_1(
            ws_.d_act_0, d_final_norm_, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            h_dim
        );
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(d_lm_head_weight_),
            d_lm_head_scale_,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            nullptr, ws_.d_logits, rank_vocab, h_dim, stream_
        );
    }

    // 4. Sample and Advance
    if (device_ids_.size() > 1) {
        launch_sample_candidate(
            ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_cand_, 0, rank_vocab, stream_
        );
    } else {
        launch_sample_and_advance(
            ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_,
            vocab_size
        );
    }
}

void QwenEngine::capture_decode_graph() {
    CudaDeviceGuard guard(device_id());

    // In multi-GPU mode without direct P2P access (WDDM staging), host polling cannot be captured in a CUDA Graph
    if (device_ids_.size() > 1) {
        std::cout << "[QwenEngine] Notice: Multi-GPU TP=2 mode active. Operating in asynchronous direct stream mode." << std::endl;
        graph_captured_ = false;
        return;
    }

    std::cout << "[QwenEngine] Capturing zero-update CUDA Graph for autoregressive decoding..." << std::endl;

    // Warm-up pass before capture
    execute_decode_pass();
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    execute_decode_pass();
    CUDA_CHECK(cudaStreamEndCapture(stream_, &decode_graph_));

    CUDA_CHECK(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
    graph_captured_ = true;

    reset_session();
    std::cout << "[QwenEngine] CUDA Graph successfully instantiated." << std::endl;
}

void QwenEngine::execute_gqa_prefill_layer(int l, int chunk_start, int chunk_len) {
    int gqa_idx = gqa_layer_idx_[l];
    size_t kv_stride = static_cast<size_t>(model_config_.num_kv_heads()) * max_seq_len_ * model_config_.head_dim();
    half* layer_k = d_k_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);
    half* layer_v = d_v_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);

    int h_dim = model_config_.hidden_size();
    int num_q_heads = model_config_.num_attention_heads();
    int num_kv_heads = model_config_.num_kv_heads();
    int head_dim = model_config_.head_dim();
    int rotary_dim = (head_dim == 256) ? 64 : head_dim;
    float rope_theta = model_config_.rope_theta();

    if (quant_mode_ == QuantMode::FP16) {
        QKVProjectionView qkv_view = {
            layers_[l].q_weight, nullptr, ws_.d_q,
            layers_[l].k_weight, nullptr, ws_.d_k,
            layers_[l].v_weight, nullptr, ws_.d_v
        };
        launch_gemm_prefill_qkv_fp16(qkv_view, ws_.d_act_1, chunk_len, stream_, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            ws_.d_q, ws_.d_k, ws_.d_v,
            layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
            layer_k, layer_v, d_control_, chunk_len, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_prefill(
            ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), num_q_heads * head_dim
        );

        if (device_ids_.size() > 1) {
            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, nullptr, ws_.d_act_1, chunk_len, h_dim, num_q_heads * head_dim, stream_);
            comm_->all_reduce_sum(ws_.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, stream_);
            launch_residual_add(ws_.d_act_0, ws_.d_act_1, chunk_len * h_dim, stream_);
        } else {
            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, num_q_heads * head_dim, stream_);
        }
    } else {
        QKVProjectionView qkv_view = {
            layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
            layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
            layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
        };
        launch_gemm_prefill_qkv_w4a16(qkv_view, ws_.d_act_1, chunk_len, stream_, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            ws_.d_q, ws_.d_k, ws_.d_v,
            layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
            layer_k, layer_v, d_control_, chunk_len, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_prefill(
            ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), num_q_heads * head_dim
        );

        if (device_ids_.size() > 1) {
            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale, ws_.d_attn_out, nullptr, ws_.d_act_1, chunk_len, h_dim, num_q_heads * head_dim, stream_);
            comm_->all_reduce_sum(ws_.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, stream_);
            launch_residual_add(ws_.d_act_0, ws_.d_act_1, chunk_len * h_dim, stream_);
        } else {
            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale, ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, num_q_heads * head_dim, stream_);
        }
    }
}

void QwenEngine::execute_deltanet_prefill_layer(int l, int chunk_start, int chunk_len) {
    int h_dim = model_config_.hidden_size();
    int num_heads = model_config_.num_attention_heads();
    int head_dim = model_config_.head_dim();
    int q_dim = num_heads * head_dim;

    int conv_idx = 0;
    for (int prev = 0; prev < l; ++prev) {
        if (model_config_.layer(prev).type == LAYER_DELTANET) conv_idx++;
    }
    half* layer_conv_state = d_conv1d_state_ + (static_cast<size_t>(conv_idx) * q_dim * 3);
    float* layer_s_state = d_deltanet_state_ + (static_cast<size_t>(conv_idx) * num_heads * 128 * 128);

    for (int t = 0; t < chunk_len; ++t) {
        const half* tok_act = ws_.d_act_1 + static_cast<size_t>(t) * h_dim;
        half* tok_res = ws_.d_act_0 + static_cast<size_t>(t) * h_dim;
        half* tok_out = (device_ids_.size() > 1) ? (ws_.d_act_1 + static_cast<size_t>(t) * h_dim) : tok_res;

        launch_depthwise_conv1d_step(tok_act, layers_[l].conv1d_weight, layers_[l].conv1d_bias, layer_conv_state, ws_.d_q, q_dim, 4, stream_);
        launch_deltanet_decode_sm75(ws_.d_q, ws_.d_q, ws_.d_q, layers_[l].decay_weight, layer_s_state, ws_.d_attn_out, 1, num_heads, head_dim, stream_);
        if (quant_mode_ == QuantMode::FP16) {
            launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].delta_out_weight), ws_.d_attn_out, (device_ids_.size() > 1 ? nullptr : tok_res), tok_out, h_dim, q_dim, stream_);
        } else {
            launch_gemv_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].delta_out_weight), layers_[l].delta_out_scale, ws_.d_attn_out, (device_ids_.size() > 1 ? nullptr : tok_res), tok_out, h_dim, q_dim, stream_);
        }
    }

    if (device_ids_.size() > 1) {
        comm_->all_reduce_sum(ws_.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, stream_);
        launch_residual_add(ws_.d_act_0, ws_.d_act_1, chunk_len * h_dim, stream_);
    }
}

void QwenEngine::execute_prefill_chunk(int chunk_start, int chunk_len) {
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    int h_dim = model_config_.hidden_size();
    int i_dim = model_config_.intermediate_size();
    int n_layers = model_config_.num_layers();

    if (quant_mode_ == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(d_embed_weight_), ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_, h_dim
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_, h_dim
        );
    }

    for (int l = 0; l < n_layers; ++l) {
        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, chunk_len, stream_, h_dim);

        if (model_config_.layer(l).type == LAYER_GQA) {
            execute_gqa_prefill_layer(l, chunk_start, chunk_len);
        } else {
            execute_deltanet_prefill_layer(l, chunk_start, chunk_len);
        }

        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_, h_dim);

        if (quant_mode_ == QuantMode::FP16) {
            launch_gemm_prefill_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, h_dim, i_dim);
            if (device_ids_.size() > 1) {
                launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, nullptr, ws_.d_act_1, chunk_len, h_dim, i_dim, stream_);
                comm_->all_reduce_sum(ws_.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, stream_);
                launch_residual_add(ws_.d_act_0, ws_.d_act_1, chunk_len * h_dim, stream_);
            } else {
                launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, i_dim, stream_);
            }
        } else {
            launch_gemm_prefill_dual_gate_up_swiglu_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale, ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, h_dim, i_dim);
            if (device_ids_.size() > 1) {
                launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale, ws_.d_mlp_act, nullptr, ws_.d_act_1, chunk_len, h_dim, i_dim, stream_);
                comm_->all_reduce_sum(ws_.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, stream_);
                launch_residual_add(ws_.d_act_0, ws_.d_act_1, chunk_len * h_dim, stream_);
            } else {
                launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale, ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, i_dim, stream_);
            }
        }
    }
}

int32_t QwenEngine::prefill(const std::vector<int32_t>& prompt_tokens, float temperature) {
    CudaDeviceGuard guard(device_id());
    if (prompt_tokens.empty()) return 0;
    if (prompt_tokens.size() >= max_seq_len_) {
        throw ContextWindowExceededError("QwenEngine: Prompt length exceeds MAX_SEQ_LEN");
    }

    temperature_ = temperature;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->temperature), &temperature_, sizeof(float), cudaMemcpyHostToDevice, stream_));

    if (device_ids_.size() > 1) {
        CudaDeviceGuard guard1(rank1_.device_id);
        CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->temperature), &temperature_, sizeof(float), cudaMemcpyHostToDevice, rank1_.stream));
    }

    size_t prefix_len = 0;
    while (prefix_len < cached_tokens_.size() &&
           prefix_len < prompt_tokens.size() &&
           cached_tokens_[prefix_len] == prompt_tokens[prefix_len]) {
        prefix_len++;
    }

    int total_tokens = static_cast<int>(prompt_tokens.size());
    CUDA_CHECK(cudaMemcpyAsync(
        ws_.d_prompt_tokens,
        prompt_tokens.data(),
        total_tokens * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream_
    ));

    if (device_ids_.size() > 1) {
        CudaDeviceGuard guard1(rank1_.device_id);
        CUDA_CHECK(cudaMemcpyAsync(
            rank1_.ws.d_prompt_tokens,
            prompt_tokens.data(),
            total_tokens * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            rank1_.stream
        ));
    }

    for (int chunk_start = static_cast<int>(prefix_len); chunk_start < total_tokens; chunk_start += PREFILL_MAX_CHUNK) {
        int chunk_len = std::min(PREFILL_MAX_CHUNK, total_tokens - chunk_start);
        if (device_ids_.size() > 1) {
            worker_chunk_start_.store(chunk_start, std::memory_order_relaxed);
            worker_chunk_len_.store(chunk_len, std::memory_order_relaxed);
            dispatch_worker(WorkerOp::PREFILL_CHUNK);
        }
        execute_prefill_chunk(chunk_start, chunk_len);
        if (device_ids_.size() > 1) {
            wait_worker_done();
        }
    }

    int last_pos = total_tokens - 1;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &last_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    int h_dim = model_config_.hidden_size();
    half* d_last_hidden = ws_.d_act_0 + static_cast<size_t>((total_tokens - 1) % PREFILL_MAX_CHUNK) * h_dim;

    if (device_ids_.size() > 1) {
        int rank_vocab = model_config_.vocab_size() / 2;
        if (quant_mode_ == QuantMode::FP16) {
            launch_rmsnorm_fp16(d_last_hidden, d_final_norm_, ws_.d_act_1, 1, stream_, h_dim);
            launch_gemv_fp16(reinterpret_cast<const half*>(d_lm_head_weight_), ws_.d_act_1, nullptr, ws_.d_logits, rank_vocab, h_dim, stream_);
        } else {
            launch_rmsnorm_q8_1(
                d_last_hidden, d_final_norm_, nullptr,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
                h_dim
            );
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(d_lm_head_weight_),
                d_lm_head_scale_,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                nullptr, ws_.d_logits, rank_vocab, h_dim, stream_
            );
        }

        launch_sample_candidate(
            ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_cand_, 0, rank_vocab, stream_
        );
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Rank 1 final token LM head & candidate
        {
            CudaDeviceGuard guard1(rank1_.device_id);
            CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->seq_pos), &last_pos, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
            half* d_last_hidden1 = rank1_.ws.d_act_0 + static_cast<size_t>((total_tokens - 1) % PREFILL_MAX_CHUNK) * h_dim;
            if (rank1_.quant_mode == QuantMode::FP16) {
                launch_rmsnorm_fp16(d_last_hidden1, rank1_.d_final_norm, rank1_.ws.d_act_1, 1, rank1_.stream, h_dim);
                launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.d_lm_head_weight), rank1_.ws.d_act_1, nullptr, rank1_.ws.d_logits, rank_vocab, h_dim, rank1_.stream);
            } else {
                launch_rmsnorm_q8_1(
                    d_last_hidden1, rank1_.d_final_norm, nullptr,
                    rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, rank1_.stream,
                    h_dim
                );
                launch_gemv_w4a8(
                    reinterpret_cast<const uint32_t*>(rank1_.d_lm_head_weight),
                    rank1_.d_lm_head_scale,
                    rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8,
                    nullptr, rank1_.ws.d_logits, rank_vocab, h_dim, rank1_.stream
                );
            }
            launch_sample_candidate(
                rank1_.ws.d_logits, rank1_.ws.d_sample_val, rank1_.ws.d_sample_idx, rank1_.d_control, rank1_.d_mapped_pinned_cand, rank_vocab, rank_vocab, rank1_.stream
            );
            CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
            if (rank1_.h_pinned_cand) {
                rank1_.top_val.store(rank1_.h_pinned_cand->val, std::memory_order_release);
                rank1_.top_idx.store(rank1_.h_pinned_cand->idx, std::memory_order_release);
            }
        }

        float v0 = h_pinned_cand_ ? h_pinned_cand_->val : -1e30f;
        float v1 = rank1_.top_val.load(std::memory_order_acquire);
        int32_t idx0 = h_pinned_cand_ ? h_pinned_cand_->idx : 0;
        int32_t idx1 = rank1_.top_idx.load(std::memory_order_acquire);
        int32_t chosen = (v0 >= v1) ? idx0 : idx1;

        last_token_id_ = chosen;
        cached_tokens_ = prompt_tokens;
        cached_tokens_.push_back(last_token_id_);
        seq_pos_ = total_tokens;

        if (h_pinned_token_) *h_pinned_token_ = chosen;
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->token_id), &chosen, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        {
            CudaDeviceGuard guard1(rank1_.device_id);
            if (rank1_.h_pinned_token) *rank1_.h_pinned_token = chosen;
            CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->token_id), &chosen, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
            CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
            CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
        }

        return last_token_id_;
    }

    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(d_last_hidden, d_final_norm_, ws_.d_act_1, 1, stream_, h_dim);
        launch_gemv_fp16(reinterpret_cast<const half*>(d_lm_head_weight_), ws_.d_act_1, nullptr, ws_.d_logits, model_config_.vocab_size(), h_dim, stream_);
    } else {
        launch_rmsnorm_q8_1(
            d_last_hidden, d_final_norm_, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            h_dim
        );
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(d_lm_head_weight_),
            d_lm_head_scale_,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            nullptr, ws_.d_logits, model_config_.vocab_size(), h_dim, stream_
        );
    }

    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_,
        model_config_.vocab_size()
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    last_token_id_ = *h_pinned_token_;
    cached_tokens_ = prompt_tokens;
    cached_tokens_.push_back(last_token_id_);
    seq_pos_ = total_tokens;

    return last_token_id_;
}

int32_t QwenEngine::decode_step() {
    CudaDeviceGuard guard(device_id());
    if (seq_pos_ >= static_cast<int32_t>(max_seq_len_)) {
        throw ContextWindowExceededError("QwenEngine: Context length reached MAX_SEQ_LEN");
    }

    if (device_ids_.size() > 1) {
        dispatch_worker(WorkerOp::DECODE_PASS);
        execute_decode_pass();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        wait_worker_done();

        float v0 = h_pinned_cand_ ? h_pinned_cand_->val : -1e30f;
        float v1 = rank1_.top_val.load(std::memory_order_acquire);
        int32_t idx0 = h_pinned_cand_ ? h_pinned_cand_->idx : 0;
        int32_t idx1 = rank1_.top_idx.load(std::memory_order_acquire);

        int32_t chosen = (v0 >= v1) ? idx0 : idx1;
        last_token_id_ = chosen;
        cached_tokens_.push_back(chosen);
        seq_pos_++;

        if (h_pinned_token_) *h_pinned_token_ = chosen;
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->token_id), &chosen, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Update rank 1 control block
        {
            CudaDeviceGuard guard1(rank1_.device_id);
            if (rank1_.h_pinned_token) *rank1_.h_pinned_token = chosen;
            CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->token_id), &chosen, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
            CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
            CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
        }

        return chosen;
    }

    if (graph_captured_ && decode_graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
    } else {
        execute_decode_pass();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    last_token_id_ = *h_pinned_token_;
    cached_tokens_.push_back(last_token_id_);
    seq_pos_++;

    return last_token_id_;
}

void QwenEngine::rollback_to(int32_t target_seq_pos) {
    CudaDeviceGuard guard(device_id());
    if (target_seq_pos < 0 || target_seq_pos > seq_pos_) {
        throw QwenError("QwenEngine::rollback_to: target_seq_pos out of bounds [0, " +
                        std::to_string(seq_pos_) + "]: " + std::to_string(target_seq_pos));
    }

    if (num_deltanet_layers_ > 0) {
        // DeltaNet states accumulate in-place; replay prefix from 0 up to target_seq_pos
        std::vector<int32_t> prefix_tokens(cached_tokens_.begin(), cached_tokens_.begin() + target_seq_pos);
        reset_session();
        if (!prefix_tokens.empty()) {
            prefill(prefix_tokens, temperature_);
        }
    } else {
        if (device_ids_.size() > 1) {
            worker_target_pos_.store(target_seq_pos, std::memory_order_relaxed);
            dispatch_worker(WorkerOp::ROLLBACK);
        }
        seq_pos_ = target_seq_pos;
        cached_tokens_.resize(target_seq_pos);
        last_token_id_ = (target_seq_pos > 0) ? cached_tokens_[target_seq_pos - 1] : TOKEN_ENDOFTEXT;

        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->token_id), &last_token_id_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (device_ids_.size() > 1) {
            wait_worker_done();
        }
    }
}

bool QwenEngine::try_acquire_session() {
    bool expected = false;
    return is_generating_.compare_exchange_strong(expected, true);
}

void QwenEngine::release_session() {
    is_generating_.store(false);
}

StageProfile QwenEngine::profile_stages(const std::vector<int32_t>& prompt_tokens) {
    CudaDeviceGuard guard(device_id());
    if (prompt_tokens.empty()) {
        throw QwenError("QwenEngine::profile_stages: Empty prompt tokens");
    }

    StageProfile profile = {};
    reset_session();
    prefill(prompt_tokens);
    return profile;
}

ArenaStats QwenEngine::get_memory_stats() const {
    CudaDeviceGuard guard(device_id());
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    ArenaStats stats;
    stats.weights_bytes = weights_bytes_;
    stats.kv_cache_bytes = kv_cache_bytes_;
    stats.workspace_bytes = workspace_bytes_;
    stats.total_allocated_bytes = arena_ ? arena_->used_bytes() : 0;
    stats.device_free_bytes = free_bytes;
    stats.device_total_bytes = total_bytes;
    stats.mode = quant_mode_;
    return stats;
}

// ----------------------------------------------------------------------------
// Companion Rank 1 Lifecycle & Execution
// ----------------------------------------------------------------------------

void QwenEngine::dispatch_worker(WorkerOp op) {
    worker_ack_.store(0, std::memory_order_release);
    worker_op_.store(static_cast<uint32_t>(op), std::memory_order_release);
}

void QwenEngine::wait_worker_done() {
    engine_spin_yield([this]() {
        return worker_ack_.load(std::memory_order_acquire) != 0;
    });
}

void QwenEngine::init_rank1() {
    rank1_.device_id = device_ids_[1];
    rank1_.rank = 1;
    CUDA_CHECK(cudaSetDevice(rank1_.device_id));
    CUDA_CHECK(cudaStreamCreate(&rank1_.stream));

    rank1_.qwenbin_path = get_rank1_path(qwenbin_path_);
    rank1_.file_map = std::make_unique<PlatformFileMap>(rank1_.qwenbin_path);
    rank1_.mapped_data = static_cast<const uint8_t*>(rank1_.file_map->data());
    rank1_.mapped_size = rank1_.file_map->size();

    if (rank1_.mapped_size < sizeof(uint32_t) * 2) {
        throw QwenError("Rank 1: File too small in " + rank1_.qwenbin_path);
    }
    const QwenBinHeaderV2* h2 = reinterpret_cast<const QwenBinHeaderV2*>(rank1_.mapped_data);
    const LayerDescriptor* layers = reinterpret_cast<const LayerDescriptor*>(rank1_.mapped_data + h2->layer_table_offset);
    rank1_.model_config = ModelConfig(*h2, layers);
    rank1_.quant_mode = static_cast<QuantMode>(h2->quant_type);

    rank1_.comm = create_communicator(1, 2, device_ids_, force_wddm_staging_, staging_channel_);

    int n_layers = rank1_.model_config.num_layers();
    rank1_.layers.resize(n_layers);
    rank1_.gqa_layer_idx.assign(n_layers, -1);
    rank1_.num_gqa_layers = 0;
    rank1_.num_deltanet_layers = 0;

    for (int l = 0; l < n_layers; ++l) {
        if (rank1_.model_config.layer(l).type == LAYER_GQA) {
            rank1_.gqa_layer_idx[l] = static_cast<int>(rank1_.num_gqa_layers++);
        } else {
            rank1_.num_deltanet_layers++;
        }
    }

    uint64_t w_offset = rank1_.model_config.header().weights_offset;
    size_t rank1_weights_bytes = (rank1_.mapped_size > w_offset) ? (rank1_.mapped_size - w_offset) : 0;

    size_t kv_elements = rank1_.num_gqa_layers * rank1_.model_config.num_kv_heads() * max_seq_len_ * rank1_.model_config.head_dim();
    size_t kv_cache_bytes = 2 * kv_elements * sizeof(half);

    size_t deltanet_state_elements = rank1_.num_deltanet_layers * rank1_.model_config.num_attention_heads() * 128 * 128;
    size_t deltanet_state_bytes = deltanet_state_elements * sizeof(float);

    size_t conv1d_elements = rank1_.num_deltanet_layers * rank1_.model_config.num_attention_heads() * rank1_.model_config.head_dim() * 3;
    size_t conv1d_bytes = conv1d_elements * sizeof(half);

    size_t max_dim = static_cast<size_t>(std::max(rank1_.model_config.hidden_size(), rank1_.model_config.intermediate_size()));
    size_t workspace_bytes = PREFILL_MAX_CHUNK * max_dim * 2 * 2 + rank1_.model_config.vocab_size() * 2 + (64ULL << 20);
    if (workspace_bytes < (128ULL << 20)) {
        workspace_bytes = 128ULL << 20;
    }
    size_t dcb_bytes = 256;
    size_t total_arena_bytes = rank1_weights_bytes + kv_cache_bytes + deltanet_state_bytes + conv1d_bytes + workspace_bytes + dcb_bytes;

    rank1_.arena = std::make_unique<GpuArena>(total_arena_bytes);

    void* d_weights_base = rank1_.arena->allocate(rank1_weights_bytes, 256);
    CUDA_CHECK(cudaMemcpy(d_weights_base, rank1_.mapped_data + w_offset, rank1_weights_bytes, cudaMemcpyHostToDevice));

    auto get_d_ptr = [d_weights_base, w_offset](uint64_t offset) -> const void* {
        if (offset == 0) return nullptr;
        return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(d_weights_base) + (offset - w_offset));
    };

    rank1_.d_embed_weight = get_d_ptr(rank1_.model_config.header().embed_tokens_offset);
    rank1_.d_embed_scale  = reinterpret_cast<const half*>(get_d_ptr(rank1_.model_config.header().reserved[0]));
    if (rank1_.model_config.header().lm_head_offset != 0 && rank1_.model_config.header().lm_head_offset != rank1_.model_config.header().embed_tokens_offset) {
        rank1_.d_lm_head_weight = get_d_ptr(rank1_.model_config.header().lm_head_offset);
        rank1_.d_lm_head_scale  = reinterpret_cast<const half*>(get_d_ptr(rank1_.model_config.header().reserved[1]));
    } else {
        rank1_.d_lm_head_weight = rank1_.d_embed_weight;
        rank1_.d_lm_head_scale  = rank1_.d_embed_scale;
    }
    rank1_.d_final_norm   = reinterpret_cast<const half*>(get_d_ptr(rank1_.model_config.header().final_norm_offset));

    for (int l = 0; l < n_layers; ++l) {
        const auto& desc = rank1_.model_config.layer(l);
        rank1_.layers[l].input_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm1_weight_offset));
        rank1_.layers[l].post_attention_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm2_weight_offset));

        if (desc.type == LAYER_GQA) {
            rank1_.layers[l].q_weight = get_d_ptr(desc.qkv_weight_offset);
            rank1_.layers[l].q_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.q_bias_offset));
            rank1_.layers[l].k_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.k_bias_offset));
            rank1_.layers[l].v_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.v_bias_offset));
            rank1_.layers[l].o_weight = get_d_ptr(desc.attn_out_weight_offset);
            rank1_.layers[l].k_weight = get_d_ptr(desc.k_weight_offset);
            rank1_.layers[l].v_weight = get_d_ptr(desc.v_weight_offset);
            rank1_.layers[l].q_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.q_scale_offset));
            rank1_.layers[l].k_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.k_scale_offset));
            rank1_.layers[l].v_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.v_scale_offset));
            rank1_.layers[l].o_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.attn_out_scale_offset));
        } else {
            rank1_.layers[l].conv1d_weight    = reinterpret_cast<const half*>(get_d_ptr(desc.conv1d_weight_offset));
            rank1_.layers[l].conv1d_bias      = reinterpret_cast<const half*>(get_d_ptr(desc.conv1d_bias_offset));
            rank1_.layers[l].in_proj_weight   = get_d_ptr(desc.in_proj_weight_offset);
            rank1_.layers[l].decay_weight     = reinterpret_cast<const half*>(get_d_ptr(desc.decay_weight_offset));
            rank1_.layers[l].delta_out_weight = get_d_ptr(desc.delta_out_weight_offset);
            rank1_.layers[l].delta_out_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.attn_out_scale_offset));
        }

        rank1_.layers[l].gate_up_weight = get_d_ptr(desc.mlp_gate_offset);
        rank1_.layers[l].gate_up_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_gate_scale_offset));
        rank1_.layers[l].down_weight    = get_d_ptr(desc.mlp_down_offset);
        rank1_.layers[l].down_scale     = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_down_scale_offset));
    }

    if (rank1_.num_gqa_layers > 0) {
        rank1_.d_k_cache = reinterpret_cast<half*>(rank1_.arena->allocate(kv_elements * sizeof(half), 256));
        rank1_.d_v_cache = reinterpret_cast<half*>(rank1_.arena->allocate(kv_elements * sizeof(half), 256));
        CUDA_CHECK(cudaMemsetAsync(rank1_.d_k_cache, 0, kv_elements * sizeof(half), rank1_.stream));
        CUDA_CHECK(cudaMemsetAsync(rank1_.d_v_cache, 0, kv_elements * sizeof(half), rank1_.stream));
    }

    if (rank1_.num_deltanet_layers > 0) {
        rank1_.d_deltanet_state = reinterpret_cast<float*>(rank1_.arena->allocate(deltanet_state_bytes, 256));
        rank1_.d_conv1d_state   = reinterpret_cast<half*>(rank1_.arena->allocate(conv1d_bytes, 256));
        CUDA_CHECK(cudaMemsetAsync(rank1_.d_deltanet_state, 0, deltanet_state_bytes, rank1_.stream));
        CUDA_CHECK(cudaMemsetAsync(rank1_.d_conv1d_state, 0, conv1d_bytes, rank1_.stream));
    }

    int h_dim = rank1_.model_config.hidden_size();
    int i_dim = rank1_.model_config.intermediate_size();
    int q_dim = rank1_.model_config.num_attention_heads() * rank1_.model_config.head_dim();
    int kv_dim = rank1_.model_config.num_kv_heads() * rank1_.model_config.head_dim();

    rank1_.ws.d_act_0         = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    rank1_.ws.d_act_1         = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    rank1_.ws.d_q             = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * q_dim * sizeof(half), 256));
    rank1_.ws.d_k             = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * kv_dim * sizeof(half), 256));
    rank1_.ws.d_v             = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * kv_dim * sizeof(half), 256));
    rank1_.ws.d_attn_out      = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * h_dim * sizeof(half), 256));
    rank1_.ws.d_mlp_act       = reinterpret_cast<half*>(rank1_.arena->allocate(PREFILL_MAX_CHUNK * i_dim * sizeof(half), 256));
    rank1_.ws.d_splitk_out    = reinterpret_cast<float*>(rank1_.arena->allocate(rank1_.model_config.num_attention_heads() * FLASH_DECODE_SPLITS * rank1_.model_config.head_dim() * sizeof(float), 256));
    rank1_.ws.d_splitk_lse    = reinterpret_cast<float*>(rank1_.arena->allocate(rank1_.model_config.num_attention_heads() * FLASH_DECODE_SPLITS * 2 * sizeof(float), 256));
    rank1_.ws.d_logits        = reinterpret_cast<half*>(rank1_.arena->allocate(rank1_.model_config.vocab_size() * sizeof(half), 256));
    rank1_.ws.d_sample_val    = reinterpret_cast<float*>(rank1_.arena->allocate(256 * sizeof(float), 256));
    rank1_.ws.d_sample_idx    = reinterpret_cast<int32_t*>(rank1_.arena->allocate(256 * sizeof(int32_t), 256));
    rank1_.ws.d_prompt_tokens = reinterpret_cast<int32_t*>(rank1_.arena->allocate(max_seq_len_ * sizeof(int32_t), 256));

    size_t q8_offset = 65536;
    size_t q8_bytes = ((static_cast<size_t>(i_dim) + 255) / 256) * 256;
    size_t scale_offset = q8_offset + q8_bytes;
    size_t num_groups = (static_cast<size_t>(i_dim) + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE;
    size_t scale_bytes = ((num_groups * sizeof(float) + 255) / 256) * 256;
    size_t sum8_offset = scale_offset + scale_bytes;

    rank1_.ws.d_act_q8    = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(rank1_.ws.d_act_1) + q8_offset);
    rank1_.ws.d_act_scale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rank1_.ws.d_act_1) + scale_offset);
    rank1_.ws.d_act_sum8  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(rank1_.ws.d_act_1) + sum8_offset);

    rank1_.d_control = reinterpret_cast<DeviceControlBlock*>(rank1_.arena->allocate(sizeof(DeviceControlBlock), 256));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&rank1_.h_pinned_token), sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&rank1_.d_mapped_pinned_token), rank1_.h_pinned_token, 0));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&rank1_.h_pinned_cand), sizeof(PinnedCandidate), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&rank1_.d_mapped_pinned_cand), rank1_.h_pinned_cand, 0));
    rank1_.h_pinned_cand->val = -1e30f;
    rank1_.h_pinned_cand->idx = -1;

    DeviceControlBlock init_dcb = { 0, 0, 0.0f, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(rank1_.d_control, &init_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
}

void QwenEngine::cleanup_rank1() {
    if (rank1_.device_id >= 0) {
        cudaSetDevice(rank1_.device_id);
    }
    if (rank1_.h_pinned_cand) {
        cudaFreeHost(rank1_.h_pinned_cand);
        rank1_.h_pinned_cand = nullptr;
    }
    if (rank1_.h_pinned_token) {
        cudaFreeHost(rank1_.h_pinned_token);
        rank1_.h_pinned_token = nullptr;
    }
    if (rank1_.stream) {
        cudaStreamDestroy(rank1_.stream);
        rank1_.stream = nullptr;
    }
    rank1_.comm.reset();
    rank1_.arena.reset();
    rank1_.file_map.reset();
    rank1_.mapped_data = nullptr;
    rank1_.mapped_size = 0;
}

void QwenEngine::worker_loop(std::stop_token st) {
    try {
        init_rank1();
    } catch (const std::exception& e) {
        std::cerr << "[QwenEngine Rank 1 Error] " << e.what() << std::endl;
    }
    // Acknowledge initialization
    worker_ack_.store(1, std::memory_order_release);

    while (!st.stop_requested()) {
        engine_spin_yield([this, &st]() {
            return st.stop_requested() || worker_op_.load(std::memory_order_acquire) != static_cast<uint32_t>(WorkerOp::IDLE);
        });
        if (st.stop_requested()) break;

        WorkerOp op = static_cast<WorkerOp>(worker_op_.load(std::memory_order_acquire));
        if (op == WorkerOp::SHUTDOWN) {
            break;
        }

        switch (op) {
            case WorkerOp::DECODE_PASS:
                execute_rank1_decode_pass();
                break;
            case WorkerOp::PREFILL_CHUNK:
                execute_rank1_prefill_chunk(worker_chunk_start_.load(std::memory_order_relaxed), worker_chunk_len_.load(std::memory_order_relaxed));
                break;
            case WorkerOp::RESET: {
                CudaDeviceGuard guard(rank1_.device_id);
                DeviceControlBlock reset_dcb = { 0, 0, temperature_, 0, 1337ULL };
                CUDA_CHECK(cudaMemcpy(rank1_.d_control, &reset_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
                *rank1_.h_pinned_token = 0;
                if (rank1_.h_pinned_cand) {
                    rank1_.h_pinned_cand->val = -1e30f;
                    rank1_.h_pinned_cand->idx = -1;
                }
                if (rank1_.d_deltanet_state && rank1_.num_deltanet_layers > 0) {
                    size_t state_elements = rank1_.num_deltanet_layers * rank1_.model_config.num_attention_heads() * 128 * 128;
                    CUDA_CHECK(cudaMemsetAsync(rank1_.d_deltanet_state, 0, state_elements * sizeof(float), rank1_.stream));
                }
                if (rank1_.d_conv1d_state && rank1_.num_deltanet_layers > 0) {
                    size_t conv1d_elements = rank1_.num_deltanet_layers * rank1_.model_config.num_attention_heads() * rank1_.model_config.head_dim() * 3;
                    CUDA_CHECK(cudaMemsetAsync(rank1_.d_conv1d_state, 0, conv1d_elements * sizeof(half), rank1_.stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
                break;
            }
            case WorkerOp::ROLLBACK: {
                CudaDeviceGuard guard(rank1_.device_id);
                int32_t target_pos = worker_target_pos_.load(std::memory_order_relaxed);
                int32_t last_token = (target_pos > 0) ? cached_tokens_[target_pos - 1] : TOKEN_ENDOFTEXT;
                CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->seq_pos), &target_pos, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
                CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->token_id), &last_token, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));
                CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
                break;
            }
            default:
                break;
        }

        worker_op_.store(static_cast<uint32_t>(WorkerOp::IDLE), std::memory_order_release);
        worker_ack_.store(1, std::memory_order_release);
    }

    cleanup_rank1();
}

void QwenEngine::execute_rank1_gqa_decode_layer(int l) {
    int gqa_idx = rank1_.gqa_layer_idx[l];
    size_t kv_stride = static_cast<size_t>(rank1_.model_config.num_kv_heads()) * max_seq_len_ * rank1_.model_config.head_dim();
    half* layer_k = rank1_.d_k_cache + (static_cast<size_t>(gqa_idx) * kv_stride);
    half* layer_v = rank1_.d_v_cache + (static_cast<size_t>(gqa_idx) * kv_stride);

    int h_dim = rank1_.model_config.hidden_size();
    int num_q_heads = rank1_.model_config.num_attention_heads();
    int num_kv_heads = rank1_.model_config.num_kv_heads();
    int head_dim = rank1_.model_config.head_dim();
    int rotary_dim = (head_dim == 256) ? 64 : head_dim;
    float rope_theta = rank1_.model_config.rope_theta();

    half* residual_ptr = nullptr;
    half* output_ptr   = rank1_.ws.d_act_1;

    if (rank1_.quant_mode == QuantMode::FP16) {
        launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.layers[l].input_layernorm, rank1_.ws.d_act_1, 1, rank1_.stream, h_dim);

        QKVProjectionView qkv_view = {
            rank1_.layers[l].q_weight, nullptr, rank1_.ws.d_q,
            rank1_.layers[l].k_weight, nullptr, rank1_.ws.d_k,
            rank1_.layers[l].v_weight, nullptr, rank1_.ws.d_v
        };
        launch_gemv_qkv_fp16(qkv_view, rank1_.ws.d_act_1, rank1_.stream, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            rank1_.ws.d_q, rank1_.ws.d_k, rank1_.ws.d_v,
            rank1_.layers[l].q_bias, rank1_.layers[l].k_bias, rank1_.layers[l].v_bias,
            layer_k, layer_v, rank1_.d_control, 1, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_flash_decode(
            rank1_.ws.d_q, layer_k, layer_v,
            rank1_.ws.d_splitk_out, rank1_.ws.d_splitk_lse, rank1_.ws.d_attn_out, rank1_.d_control, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
        );

        launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.layers[l].o_weight), rank1_.ws.d_attn_out, residual_ptr, output_ptr, h_dim, num_q_heads * head_dim, rank1_.stream);
    } else {
        launch_rmsnorm_q8_1(
            rank1_.ws.d_act_0, rank1_.layers[l].input_layernorm, nullptr,
            rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, rank1_.stream,
            h_dim
        );

        QKVProjectionView qkv_view = {
            rank1_.layers[l].q_weight, rank1_.layers[l].q_scale, rank1_.ws.d_q,
            rank1_.layers[l].k_weight, rank1_.layers[l].k_scale, rank1_.ws.d_k,
            rank1_.layers[l].v_weight, rank1_.layers[l].v_scale, rank1_.ws.d_v
        };
        launch_gemv_qkv_w4a8(qkv_view, rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, rank1_.stream, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            rank1_.ws.d_q, rank1_.ws.d_k, rank1_.ws.d_v,
            rank1_.layers[l].q_bias, rank1_.layers[l].k_bias, rank1_.layers[l].v_bias,
            layer_k, layer_v, rank1_.d_control, 1, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_flash_decode(
            rank1_.ws.d_q, layer_k, layer_v,
            rank1_.ws.d_splitk_out, rank1_.ws.d_splitk_lse, rank1_.ws.d_attn_out, rank1_.d_control, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8
        );

        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(rank1_.layers[l].o_weight),
            rank1_.layers[l].o_scale,
            rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8,
            residual_ptr, output_ptr,
            h_dim, num_q_heads * head_dim, rank1_.stream
        );
    }
}

void QwenEngine::execute_rank1_deltanet_decode_layer(int l) {
    int h_dim = rank1_.model_config.hidden_size();
    int num_heads = rank1_.model_config.num_attention_heads();
    int head_dim = rank1_.model_config.head_dim();
    int q_dim = num_heads * head_dim;

    half* residual_ptr = nullptr;
    half* output_ptr   = rank1_.ws.d_act_1;

    // 1. RMSNorm
    launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.layers[l].input_layernorm, rank1_.ws.d_act_1, 1, rank1_.stream, h_dim);

    // 2. Depthwise Conv1d Step
    int conv_idx = 0;
    for (int prev = 0; prev < l; ++prev) {
        if (rank1_.model_config.layer(prev).type == LAYER_DELTANET) conv_idx++;
    }
    half* layer_conv_state = rank1_.d_conv1d_state + (static_cast<size_t>(conv_idx) * q_dim * 3);
    launch_depthwise_conv1d_step(
        rank1_.ws.d_act_1, rank1_.layers[l].conv1d_weight, rank1_.layers[l].conv1d_bias,
        layer_conv_state, rank1_.ws.d_q, q_dim, 4, rank1_.stream
    );

    // 3. DeltaNet Decode State Step
    float* layer_s_state = rank1_.d_deltanet_state + (static_cast<size_t>(conv_idx) * num_heads * 128 * 128);
    launch_deltanet_decode_sm75(
        rank1_.ws.d_q, rank1_.ws.d_q, rank1_.ws.d_q, rank1_.layers[l].decay_weight,
        layer_s_state, rank1_.ws.d_attn_out, 1,
        num_heads, head_dim, rank1_.stream
    );

    // 4. Out projection
    if (rank1_.quant_mode == QuantMode::FP16) {
        launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.layers[l].delta_out_weight), rank1_.ws.d_attn_out, residual_ptr, output_ptr, h_dim, q_dim, rank1_.stream);
    } else {
        launch_gemv_w4a16(reinterpret_cast<const uint32_t*>(rank1_.layers[l].delta_out_weight), rank1_.layers[l].delta_out_scale, rank1_.ws.d_attn_out, residual_ptr, output_ptr, h_dim, q_dim, rank1_.stream);
    }
}

void QwenEngine::execute_rank1_decode_pass() {
    CudaDeviceGuard guard(rank1_.device_id);
    int h_dim = rank1_.model_config.hidden_size();
    int i_dim = rank1_.model_config.intermediate_size();
    int n_layers = rank1_.model_config.num_layers();

    // 1. Embedding lookup
    if (rank1_.quant_mode == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(rank1_.d_embed_weight), nullptr, rank1_.ws.d_act_0, rank1_.d_control, 1, rank1_.stream, h_dim
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(rank1_.d_embed_weight), rank1_.d_embed_scale, nullptr, rank1_.ws.d_act_0, rank1_.d_control, 1, rank1_.stream, h_dim
        );
    }

    // 2. Transformer layers
    for (int l = 0; l < n_layers; ++l) {
        if (rank1_.model_config.layer(l).type == LAYER_GQA) {
            execute_rank1_gqa_decode_layer(l);
        } else {
            execute_rank1_deltanet_decode_layer(l);
        }

        // Tensor Parallel All-Reduce 1
        if (rank1_.comm) {
            rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, h_dim, DataType::FP16, rank1_.stream);
            launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, h_dim, rank1_.stream);
        }

        // SwiGLU MLP
        half* mlp_residual_ptr = nullptr;
        half* mlp_output_ptr   = rank1_.ws.d_act_1;

        if (rank1_.quant_mode == QuantMode::FP16) {
            launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.layers[l].post_attention_layernorm, rank1_.ws.d_act_1, 1, rank1_.stream, h_dim);
            launch_gemv_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(rank1_.layers[l].gate_up_weight), rank1_.ws.d_act_1, rank1_.ws.d_mlp_act, h_dim, i_dim, rank1_.stream);
            launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.layers[l].down_weight), rank1_.ws.d_mlp_act, mlp_residual_ptr, mlp_output_ptr, h_dim, i_dim, rank1_.stream);
        } else {
            launch_rmsnorm_q8_1(
                rank1_.ws.d_act_0, rank1_.layers[l].post_attention_layernorm, nullptr,
                rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, rank1_.stream,
                h_dim
            );
            launch_gemv_dual_gate_up_swiglu_w4a8(
                reinterpret_cast<const uint32_t*>(rank1_.layers[l].gate_up_weight),
                rank1_.layers[l].gate_up_scale,
                rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8,
                rank1_.ws.d_mlp_act, h_dim, i_dim, rank1_.stream
            );
            launch_quantize_fp16_to_q8_1(rank1_.ws.d_mlp_act, rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, i_dim, rank1_.stream);
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(rank1_.layers[l].down_weight),
                rank1_.layers[l].down_scale,
                rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8,
                mlp_residual_ptr, mlp_output_ptr,
                h_dim, i_dim, rank1_.stream
            );
        }

        // Tensor Parallel All-Reduce 2
        if (rank1_.comm) {
            rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, h_dim, DataType::FP16, rank1_.stream);
            launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, h_dim, rank1_.stream);
        }
    }

    // 3. Final RMSNorm + LM Head
    int rank_vocab = rank1_.model_config.vocab_size() / 2;
    if (rank1_.quant_mode == QuantMode::FP16) {
        launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.d_final_norm, rank1_.ws.d_act_1, 1, rank1_.stream, h_dim);
        launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.d_lm_head_weight), rank1_.ws.d_act_1, nullptr, rank1_.ws.d_logits, rank_vocab, h_dim, rank1_.stream);
    } else {
        launch_rmsnorm_q8_1(
            rank1_.ws.d_act_0, rank1_.d_final_norm, nullptr,
            rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8, rank1_.stream,
            h_dim
        );
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(rank1_.d_lm_head_weight),
            rank1_.d_lm_head_scale,
            rank1_.ws.d_act_q8, rank1_.ws.d_act_scale, rank1_.ws.d_act_sum8,
            nullptr, rank1_.ws.d_logits, rank_vocab, h_dim, rank1_.stream
        );
    }

    // 4. Sample Candidate for Rank 1 with index_offset = rank_vocab
    launch_sample_candidate(
        rank1_.ws.d_logits, rank1_.ws.d_sample_val, rank1_.ws.d_sample_idx, rank1_.d_control, rank1_.d_mapped_pinned_cand, rank_vocab, rank_vocab, rank1_.stream
    );
    CUDA_CHECK(cudaStreamSynchronize(rank1_.stream));
    if (rank1_.h_pinned_cand) {
        rank1_.top_val.store(rank1_.h_pinned_cand->val, std::memory_order_release);
        rank1_.top_idx.store(rank1_.h_pinned_cand->idx, std::memory_order_release);
    }
}

void QwenEngine::execute_rank1_gqa_prefill_layer(int l, int chunk_start, int chunk_len) {
    int gqa_idx = rank1_.gqa_layer_idx[l];
    size_t kv_stride = static_cast<size_t>(rank1_.model_config.num_kv_heads()) * max_seq_len_ * rank1_.model_config.head_dim();
    half* layer_k = rank1_.d_k_cache + (static_cast<size_t>(gqa_idx) * kv_stride);
    half* layer_v = rank1_.d_v_cache + (static_cast<size_t>(gqa_idx) * kv_stride);

    int h_dim = rank1_.model_config.hidden_size();
    int num_q_heads = rank1_.model_config.num_attention_heads();
    int num_kv_heads = rank1_.model_config.num_kv_heads();
    int head_dim = rank1_.model_config.head_dim();
    int rotary_dim = (head_dim == 256) ? 64 : head_dim;
    float rope_theta = rank1_.model_config.rope_theta();

    if (rank1_.quant_mode == QuantMode::FP16) {
        QKVProjectionView qkv_view = {
            rank1_.layers[l].q_weight, nullptr, rank1_.ws.d_q,
            rank1_.layers[l].k_weight, nullptr, rank1_.ws.d_k,
            rank1_.layers[l].v_weight, nullptr, rank1_.ws.d_v
        };
        launch_gemm_prefill_qkv_fp16(qkv_view, rank1_.ws.d_act_1, chunk_len, rank1_.stream, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            rank1_.ws.d_q, rank1_.ws.d_k, rank1_.ws.d_v,
            rank1_.layers[l].q_bias, rank1_.layers[l].k_bias, rank1_.layers[l].v_bias,
            layer_k, layer_v, rank1_.d_control, chunk_len, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_prefill(
            rank1_.ws.d_q, layer_k, layer_v, rank1_.ws.d_attn_out, rank1_.d_control, chunk_len, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), num_q_heads * head_dim
        );

        launch_gemm_prefill_fp16(reinterpret_cast<const half*>(rank1_.layers[l].o_weight), rank1_.ws.d_attn_out, nullptr, rank1_.ws.d_act_1, chunk_len, h_dim, num_q_heads * head_dim, rank1_.stream);
        rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, rank1_.stream);
        launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, chunk_len * h_dim, rank1_.stream);
    } else {
        QKVProjectionView qkv_view = {
            rank1_.layers[l].q_weight, rank1_.layers[l].q_scale, rank1_.ws.d_q,
            rank1_.layers[l].k_weight, rank1_.layers[l].k_scale, rank1_.ws.d_k,
            rank1_.layers[l].v_weight, rank1_.layers[l].v_scale, rank1_.ws.d_v
        };
        launch_gemm_prefill_qkv_w4a16(qkv_view, rank1_.ws.d_act_1, chunk_len, rank1_.stream, h_dim, num_q_heads, num_kv_heads, head_dim);

        launch_qkv_bias_rope_kv_append(
            rank1_.ws.d_q, rank1_.ws.d_k, rank1_.ws.d_v,
            rank1_.layers[l].q_bias, rank1_.layers[l].k_bias, rank1_.layers[l].v_bias,
            layer_k, layer_v, rank1_.d_control, chunk_len, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_),
            rotary_dim, rope_theta
        );

        launch_attention_gqa_prefill(
            rank1_.ws.d_q, layer_k, layer_v, rank1_.ws.d_attn_out, rank1_.d_control, chunk_len, rank1_.stream,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), num_q_heads * head_dim
        );

        launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(rank1_.layers[l].o_weight), rank1_.layers[l].o_scale, rank1_.ws.d_attn_out, nullptr, rank1_.ws.d_act_1, chunk_len, h_dim, num_q_heads * head_dim, rank1_.stream);
        rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, rank1_.stream);
        launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, chunk_len * h_dim, rank1_.stream);
    }
}

void QwenEngine::execute_rank1_deltanet_prefill_layer(int l, int chunk_start, int chunk_len) {
    int h_dim = rank1_.model_config.hidden_size();
    int num_heads = rank1_.model_config.num_attention_heads();
    int head_dim = rank1_.model_config.head_dim();
    int q_dim = num_heads * head_dim;

    int conv_idx = 0;
    for (int prev = 0; prev < l; ++prev) {
        if (rank1_.model_config.layer(prev).type == LAYER_DELTANET) conv_idx++;
    }
    half* layer_conv_state = rank1_.d_conv1d_state + (static_cast<size_t>(conv_idx) * q_dim * 3);
    float* layer_s_state = rank1_.d_deltanet_state + (static_cast<size_t>(conv_idx) * num_heads * 128 * 128);

    for (int t = 0; t < chunk_len; ++t) {
        const half* tok_act = rank1_.ws.d_act_1 + static_cast<size_t>(t) * h_dim;
        half* tok_out = rank1_.ws.d_act_1 + static_cast<size_t>(t) * h_dim;

        launch_depthwise_conv1d_step(tok_act, rank1_.layers[l].conv1d_weight, rank1_.layers[l].conv1d_bias, layer_conv_state, rank1_.ws.d_q, q_dim, 4, rank1_.stream);
        launch_deltanet_decode_sm75(rank1_.ws.d_q, rank1_.ws.d_q, rank1_.ws.d_q, rank1_.layers[l].decay_weight, layer_s_state, rank1_.ws.d_attn_out, 1, num_heads, head_dim, rank1_.stream);
        if (rank1_.quant_mode == QuantMode::FP16) {
            launch_gemv_fp16(reinterpret_cast<const half*>(rank1_.layers[l].delta_out_weight), rank1_.ws.d_attn_out, nullptr, tok_out, h_dim, q_dim, rank1_.stream);
        } else {
            launch_gemv_w4a16(reinterpret_cast<const uint32_t*>(rank1_.layers[l].delta_out_weight), rank1_.layers[l].delta_out_scale, rank1_.ws.d_attn_out, nullptr, tok_out, h_dim, q_dim, rank1_.stream);
        }
    }

    rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, rank1_.stream);
    launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, chunk_len * h_dim, rank1_.stream);
}

void QwenEngine::execute_rank1_prefill_chunk(int chunk_start, int chunk_len) {
    CudaDeviceGuard guard(rank1_.device_id);
    CUDA_CHECK(cudaMemcpyAsync(&(rank1_.d_control->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, rank1_.stream));

    int h_dim = rank1_.model_config.hidden_size();
    int i_dim = rank1_.model_config.intermediate_size();
    int n_layers = rank1_.model_config.num_layers();

    if (rank1_.quant_mode == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(rank1_.d_embed_weight), rank1_.ws.d_prompt_tokens + chunk_start, rank1_.ws.d_act_0, nullptr, chunk_len, rank1_.stream, h_dim
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(rank1_.d_embed_weight), rank1_.d_embed_scale, rank1_.ws.d_prompt_tokens + chunk_start, rank1_.ws.d_act_0, nullptr, chunk_len, rank1_.stream, h_dim
        );
    }

    for (int l = 0; l < n_layers; ++l) {
        launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.layers[l].input_layernorm, rank1_.ws.d_act_1, chunk_len, rank1_.stream, h_dim);

        if (rank1_.model_config.layer(l).type == LAYER_GQA) {
            execute_rank1_gqa_prefill_layer(l, chunk_start, chunk_len);
        } else {
            execute_rank1_deltanet_prefill_layer(l, chunk_start, chunk_len);
        }

        launch_rmsnorm_fp16(rank1_.ws.d_act_0, rank1_.layers[l].post_attention_layernorm, rank1_.ws.d_act_1, chunk_len, rank1_.stream, h_dim);

        if (rank1_.quant_mode == QuantMode::FP16) {
            launch_gemm_prefill_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(rank1_.layers[l].gate_up_weight), rank1_.ws.d_act_1, rank1_.ws.d_mlp_act, chunk_len, rank1_.stream, h_dim, i_dim);
            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(rank1_.layers[l].down_weight), rank1_.ws.d_mlp_act, nullptr, rank1_.ws.d_act_1, chunk_len, h_dim, i_dim, rank1_.stream);
            rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, rank1_.stream);
            launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, chunk_len * h_dim, rank1_.stream);
        } else {
            launch_gemm_prefill_dual_gate_up_swiglu_w4a16(reinterpret_cast<const uint32_t*>(rank1_.layers[l].gate_up_weight), rank1_.layers[l].gate_up_scale, rank1_.ws.d_act_1, rank1_.ws.d_mlp_act, chunk_len, rank1_.stream, h_dim, i_dim);
            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(rank1_.layers[l].down_weight), rank1_.layers[l].down_scale, rank1_.ws.d_mlp_act, nullptr, rank1_.ws.d_act_1, chunk_len, h_dim, i_dim, rank1_.stream);
            rank1_.comm->all_reduce_sum(rank1_.ws.d_act_1, static_cast<size_t>(chunk_len) * h_dim, DataType::FP16, rank1_.stream);
            launch_residual_add(rank1_.ws.d_act_0, rank1_.ws.d_act_1, chunk_len * h_dim, rank1_.stream);
        }
    }
}

} // namespace qwen
