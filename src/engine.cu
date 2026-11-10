#include "engine.cuh"
#include "kernels.cuh"
#include "file_map.hpp"
#include "cuda_guard.cuh"
#include "exceptions.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace qwen {

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

QwenEngine::QwenEngine(const std::string& qwenbin_path, int device_id)
    : QwenEngine(EngineConfig{qwenbin_path, "", {device_id}, 4096, false}) {}

QwenEngine::QwenEngine(const std::string& qwenbin_path, const std::vector<int>& device_ids)
    : QwenEngine(EngineConfig{qwenbin_path, "", device_ids, 4096, false}) {}

QwenEngine::~QwenEngine() {
    CudaDeviceGuard guard(device_id());
    if (decode_graph_exec_) {
        cudaGraphExecDestroy(decode_graph_exec_);
        decode_graph_exec_ = nullptr;
    }
    if (decode_graph_) {
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
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
        max_seq_len_ = h2->max_seq_len;
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
    int my_rank = (device_ids_.size() > 1 && device_ids_[0] == 1) ? 1 : 0;
    comm_ = create_communicator(my_rank, static_cast<int>(device_ids_.size()), device_ids_, force_wddm_staging_);

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

    DeviceControlBlock init_dcb = { 0, 0, 0.0f, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &init_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
}

void QwenEngine::reset_session() {
    CudaDeviceGuard guard(device_id());
    cached_tokens_.clear();
    seq_pos_ = 0;
    last_token_id_ = 0;
    DeviceControlBlock reset_dcb = { 0, 0, temperature_, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &reset_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
    *h_pinned_token_ = 0;

    if (d_deltanet_state_ && num_deltanet_layers_ > 0) {
        size_t state_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * 128 * 128;
        CUDA_CHECK(cudaMemsetAsync(d_deltanet_state_, 0, state_elements * sizeof(float), stream_));
    }
    if (d_conv1d_state_ && num_deltanet_layers_ > 0) {
        size_t conv1d_elements = num_deltanet_layers_ * model_config_.num_attention_heads() * model_config_.head_dim() * 3;
        CUDA_CHECK(cudaMemsetAsync(d_conv1d_state_, 0, conv1d_elements * sizeof(half), stream_));
    }
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

void QwenEngine::execute_gqa_decode_layer(int l) {
    int gqa_idx = gqa_layer_idx_[l];
    size_t kv_stride = static_cast<size_t>(model_config_.num_kv_heads()) * max_seq_len_ * model_config_.head_dim();
    half* layer_k = d_k_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);
    half* layer_v = d_v_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);

    int h_dim = model_config_.hidden_size();
    int num_q_heads = model_config_.num_attention_heads();
    int num_kv_heads = model_config_.num_kv_heads();
    int head_dim = model_config_.head_dim();
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
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
        );

        launch_attention_gqa_flash_decode(
            ws_.d_q, layer_k, layer_v,
            ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
        );

        launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, residual_ptr, output_ptr, h_dim, h_dim, stream_);
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
            num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
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
            h_dim, h_dim, stream_
        );
    }
}

void QwenEngine::execute_deltanet_decode_layer(int l) {
    int h_dim = model_config_.hidden_size();
    int q_dim = model_config_.num_attention_heads() * model_config_.head_dim();
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
    float* layer_s_state = d_deltanet_state_ + (static_cast<size_t>(conv_idx) * model_config_.num_attention_heads() * 128 * 128);
    launch_deltanet_decode_sm75(
        ws_.d_q, ws_.d_q, ws_.d_q, layers_[l].decay_weight,
        layer_s_state, ws_.d_attn_out, 1,
        model_config_.num_attention_heads(), model_config_.head_dim(), stream_
    );

    // 4. Out projection
    launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].delta_out_weight), ws_.d_attn_out, residual_ptr, output_ptr, h_dim, q_dim, stream_);
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

    // 3. Final RMSNorm
    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(ws_.d_act_0, d_final_norm_, ws_.d_act_1, 1, stream_, h_dim);
        launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, model_config_.vocab_size(), h_dim, stream_);
    } else {
        launch_rmsnorm_q8_1(
            ws_.d_act_0, d_final_norm_, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            h_dim
        );
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(d_embed_weight_),
            d_embed_scale_,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            nullptr, ws_.d_logits, model_config_.vocab_size(), h_dim, stream_
        );
    }

    // 4. Sample and Advance
    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_,
        model_config_.vocab_size()
    );
}

void QwenEngine::capture_decode_graph() {
    CudaDeviceGuard guard(device_id());

    // In multi-GPU mode without direct P2P access (WDDM staging), host polling cannot be captured in a CUDA Graph
    if (device_ids_.size() > 1 && comm_ && !comm_->is_peer_accessible()) {
        std::cout << "[QwenEngine] Warning: Windows WDDM staging detected on multi-GPU. Bypassing CUDA Graph capture." << std::endl;
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

void QwenEngine::execute_prefill_chunk(int chunk_start, int chunk_len) {
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    int h_dim = model_config_.hidden_size();
    int i_dim = model_config_.intermediate_size();
    int n_layers = model_config_.num_layers();
    int num_q_heads = model_config_.num_attention_heads();
    int num_kv_heads = model_config_.num_kv_heads();
    int head_dim = model_config_.head_dim();

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
            int gqa_idx = gqa_layer_idx_[l];
            size_t kv_stride = static_cast<size_t>(model_config_.num_kv_heads()) * max_seq_len_ * model_config_.head_dim();
            half* layer_k = d_k_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);
            half* layer_v = d_v_cache_ + (static_cast<size_t>(gqa_idx) * kv_stride);

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
                    num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
                );

                launch_attention_gqa_prefill(
                    ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
                    num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), h_dim
                );

                launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, h_dim, stream_);
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
                    num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_)
                );

                launch_attention_gqa_prefill(
                    ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
                    num_q_heads, num_kv_heads, head_dim, static_cast<int>(max_seq_len_), h_dim
                );

                launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale, ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, h_dim, stream_);
            }
        } else {
            // DeltaNet prefill token loop
            int conv_idx = 0;
            for (int prev = 0; prev < l; ++prev) {
                if (model_config_.layer(prev).type == LAYER_DELTANET) conv_idx++;
            }
            int q_dim = model_config_.num_attention_heads() * model_config_.head_dim();
            half* layer_conv_state = d_conv1d_state_ + (static_cast<size_t>(conv_idx) * q_dim * 3);
            float* layer_s_state = d_deltanet_state_ + (static_cast<size_t>(conv_idx) * model_config_.num_attention_heads() * 128 * 128);

            for (int t = 0; t < chunk_len; ++t) {
                const half* tok_act = ws_.d_act_1 + static_cast<size_t>(t) * h_dim;
                half* tok_res = ws_.d_act_0 + static_cast<size_t>(t) * h_dim;

                launch_depthwise_conv1d_step(tok_act, layers_[l].conv1d_weight, layers_[l].conv1d_bias, layer_conv_state, ws_.d_q, q_dim, 4, stream_);
                launch_deltanet_decode_sm75(ws_.d_q, ws_.d_q, ws_.d_q, layers_[l].decay_weight, layer_s_state, ws_.d_attn_out, 1, model_config_.num_attention_heads(), model_config_.head_dim(), stream_);
                launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].delta_out_weight), ws_.d_attn_out, tok_res, tok_res, h_dim, q_dim, stream_);
            }
        }

        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_, h_dim);

        if (quant_mode_ == QuantMode::FP16) {
            launch_gemm_prefill_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, h_dim, i_dim);
            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, i_dim, stream_);
        } else {
            launch_gemm_prefill_dual_gate_up_swiglu_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale, ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, h_dim, i_dim);
            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale, ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, h_dim, i_dim, stream_);
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

    for (int chunk_start = static_cast<int>(prefix_len); chunk_start < total_tokens; chunk_start += PREFILL_MAX_CHUNK) {
        int chunk_len = std::min(PREFILL_MAX_CHUNK, total_tokens - chunk_start);
        execute_prefill_chunk(chunk_start, chunk_len);
    }

    int last_pos = total_tokens - 1;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &last_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    int h_dim = model_config_.hidden_size();
    half* d_last_hidden = ws_.d_act_0 + static_cast<size_t>((total_tokens - 1) % PREFILL_MAX_CHUNK) * h_dim;

    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(d_last_hidden, d_final_norm_, ws_.d_act_1, 1, stream_, h_dim);
        launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, model_config_.vocab_size(), h_dim, stream_);
    } else {
        launch_rmsnorm_q8_1(
            d_last_hidden, d_final_norm_, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            h_dim
        );
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(d_embed_weight_),
            d_embed_scale_,
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
        seq_pos_ = target_seq_pos;
        cached_tokens_.resize(target_seq_pos);
        last_token_id_ = (target_seq_pos > 0) ? cached_tokens_[target_seq_pos - 1] : TOKEN_ENDOFTEXT;

        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->token_id), &last_token_id_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
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

} // namespace qwen
