#include "engine.cuh"
#include "kernels.cuh"
#include "file_map.hpp"
#include "cuda_guard.cuh"
#include "exceptions.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace qwen {

namespace {

inline size_t align_up(size_t n, size_t alignment = 256) {
    return ((n + alignment - 1) / alignment) * alignment;
}

} // namespace

QwenEngine::QwenEngine(const EngineConfig& config)
    : qwenbin_path_(config.model_path),
      device_id_(config.device_id),
      max_seq_len_(config.max_seq_len),
      quant_mode_(QuantMode::W4A16_G128),
      weights_16byte_aligned_(true),
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

    CudaDeviceGuard guard(device_id_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    load_model_file(qwenbin_path_);
    setup_arena();
    capture_decode_graph();
}

QwenEngine::QwenEngine(const std::string& qwenbin_path, int device_id, size_t max_seq_len)
    : QwenEngine(EngineConfig{qwenbin_path, "", device_id, max_seq_len}) {}

QwenEngine::~QwenEngine() {
    CudaDeviceGuard guard(device_id_);
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

    if (mapped_size_ < sizeof(QwenBinHeaderV2)) {
        throw QwenError("QwenEngine: File too small to contain valid .qwenbin v2 header in " + qwenbin_path);
    }

    uint32_t magic = *reinterpret_cast<const uint32_t*>(mapped_data_);
    uint32_t version = *reinterpret_cast<const uint32_t*>(mapped_data_ + sizeof(uint32_t));

    if (magic != 0x5157454E) {
        throw QwenError("QwenEngine: Invalid magic in " + qwenbin_path);
    }

    if (version == 2) {
        const QwenBinHeaderV2* h2 = reinterpret_cast<const QwenBinHeaderV2*>(mapped_data_);
        if (h2->quant_type != static_cast<uint32_t>(QuantMode::W4A16_G128)) {
            throw QwenError("QwenEngine: Only W4A16_G128 quantization is supported (found quant_type " +
                            std::to_string(h2->quant_type) + ")");
        }
        if (h2->hidden_size != HIDDEN_SIZE ||
            h2->intermediate_size != INTERMEDIATE_SIZE ||
            h2->num_layers != NUM_LAYERS ||
            h2->num_attention_heads != NUM_Q_HEADS ||
            h2->num_kv_heads != NUM_KV_HEADS ||
            h2->head_dim != HEAD_DIM ||
            h2->vocab_size != VOCAB_SIZE) {
            throw QwenError("QwenEngine: Model dimensions do not match Qwen 2.5-1.5B-Instruct");
        }
        if (mapped_size_ < h2->layer_table_offset + static_cast<size_t>(h2->num_layers) * sizeof(LayerDescriptor)) {
            throw QwenError("QwenEngine: Truncated .qwenbin v2 container");
        }

        const LayerDescriptor* layers = reinterpret_cast<const LayerDescriptor*>(mapped_data_ + h2->layer_table_offset);
        for (uint32_t l = 0; l < h2->num_layers; ++l) {
            if (layers[l].type != LAYER_GQA) {
                throw QwenError("QwenEngine: Unsupported non-GQA layer type at layer " + std::to_string(l));
            }
        }

        model_config_ = ModelConfig(*h2, layers);
        quant_mode_ = QuantMode::W4A16_G128;

        if (max_seq_len_ == 0 || max_seq_len_ > 32768) {
            max_seq_len_ = (h2->max_seq_len > 0 && h2->max_seq_len <= 8192) ? h2->max_seq_len : MAX_SEQ_LEN;
        } else if (h2->max_seq_len > 0 && max_seq_len_ > h2->max_seq_len) {
            max_seq_len_ = h2->max_seq_len;
        }
    } else if (version == 1) {
        if (mapped_size_ < sizeof(QWQwenHeader)) {
            throw QwenError("QwenEngine: File too small to contain valid .qwenbin v1 header in " + qwenbin_path);
        }
        const QWQwenHeader* h1 = reinterpret_cast<const QWQwenHeader*>(mapped_data_);
        if (h1->quant_mode != static_cast<uint32_t>(QuantMode::W4A16_G128)) {
            throw QwenError("QwenEngine: Only W4A16_G128 quantization is supported (found quant_mode " +
                            std::to_string(h1->quant_mode) + ")");
        }
        if (h1->hidden_size != HIDDEN_SIZE ||
            h1->intermediate_size != INTERMEDIATE_SIZE ||
            h1->num_layers != NUM_LAYERS ||
            h1->num_q_heads != NUM_Q_HEADS ||
            h1->num_kv_heads != NUM_KV_HEADS ||
            h1->head_dim != HEAD_DIM ||
            h1->vocab_size != VOCAB_SIZE) {
            throw QwenError("QwenEngine: Model dimensions do not match Qwen 2.5-1.5B-Instruct");
        }

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
        quant_mode_ = QuantMode::W4A16_G128;
        if (max_seq_len_ == 0 || max_seq_len_ > MAX_SEQ_LEN) {
            max_seq_len_ = MAX_SEQ_LEN;
        }
    } else {
        throw QwenError("QwenEngine: Unsupported format version " + std::to_string(version));
    }
}

void QwenEngine::setup_arena() {
    layers_.resize(NUM_LAYERS);

    uint64_t w_offset = model_config_.header().weights_offset;
    weights_bytes_ = (mapped_size_ > w_offset) ? (mapped_size_ - w_offset) : 0;

    // 1. KV Cache allocation
    size_t kv_elements = static_cast<size_t>(NUM_LAYERS) * NUM_KV_HEADS * max_seq_len_ * HEAD_DIM;
    kv_cache_bytes_ = 2 * kv_elements * sizeof(half);

    // 2. Exact Workspace Buffers (no 128 MB floor)
    size_t act_buf_bytes = align_up(static_cast<size_t>(PREFILL_MAX_CHUNK) * HIDDEN_SIZE * sizeof(half));
    size_t q_bytes = align_up(static_cast<size_t>(PREFILL_MAX_CHUNK) * Q_DIM * sizeof(half));
    size_t kv_proj_bytes = align_up(static_cast<size_t>(PREFILL_MAX_CHUNK) * KV_DIM * sizeof(half));
    size_t mlp_act_bytes = align_up(static_cast<size_t>(PREFILL_MAX_CHUNK) * INTERMEDIATE_SIZE * sizeof(half));
    size_t splitk_out_bytes = align_up(static_cast<size_t>(NUM_Q_HEADS) * FLASH_DECODE_SPLITS * HEAD_DIM * sizeof(float));
    size_t splitk_lse_bytes = align_up(static_cast<size_t>(NUM_Q_HEADS) * FLASH_DECODE_SPLITS * 2 * sizeof(float));
    size_t logits_bytes = align_up(static_cast<size_t>(VOCAB_SIZE) * sizeof(half));
    size_t sample_val_bytes = align_up(256 * sizeof(float));
    size_t sample_idx_bytes = align_up(256 * sizeof(int32_t));
    size_t prompt_tok_bytes = align_up(max_seq_len_ * sizeof(int32_t));
    size_t q8_scratch_bytes = 16384; // Dedicated 16 KB aligned Q8_1 scratch (separate from d_act_1)

    workspace_bytes_ =
        act_buf_bytes * 3 + // d_act_0, d_act_1, d_attn_out
        q_bytes +
        kv_proj_bytes * 2 + // d_k, d_v
        mlp_act_bytes +
        splitk_out_bytes +
        splitk_lse_bytes +
        logits_bytes +
        sample_val_bytes +
        sample_idx_bytes +
        prompt_tok_bytes +
        q8_scratch_bytes;

    size_t dcb_bytes = align_up(sizeof(DeviceControlBlock), 256);
    size_t arena_slack = 1ULL << 20; // 1 MB alignment padding headroom
    size_t total_arena_bytes = align_up(weights_bytes_) + align_up(kv_elements * sizeof(half)) * 2 +
                               workspace_bytes_ + dcb_bytes + arena_slack;

    std::cout << "[QwenEngine] Initializing Single-GPU W4A16 Arena (Qwen 2.5-1.5B, " << NUM_LAYERS << " GQA layers)..." << std::endl;
    std::cout << "  - Weights Region:   " << (weights_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - KV Cache:         " << (kv_cache_bytes_ / (1024.0 * 1024.0)) << " MB ("
              << max_seq_len_ << " max tokens)" << std::endl;
    std::cout << "  - Workspace Region: " << (workspace_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - Total Allocation: " << (total_arena_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    arena_ = std::make_unique<GpuArena>(total_arena_bytes);

    void* d_weights_base = arena_->allocate(weights_bytes_, 256);
    CUDA_CHECK(cudaMemcpy(d_weights_base, mapped_data_ + w_offset, weights_bytes_, cudaMemcpyHostToDevice));

    weights_16byte_aligned_ = true;
    auto get_d_ptr = [d_weights_base, w_offset, this](uint64_t offset, bool check_16b = false) -> const void* {
        if (offset == 0) return nullptr;
        uint64_t rel = offset - w_offset;
        if (check_16b && (rel % 16 != 0)) {
            weights_16byte_aligned_ = false;
        }
        return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(d_weights_base) + rel);
    };

    d_embed_weight_ = get_d_ptr(model_config_.header().embed_tokens_offset, true);
    d_embed_scale_  = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().reserved[0]));
    if (model_config_.header().lm_head_offset != 0 && model_config_.header().lm_head_offset != model_config_.header().embed_tokens_offset) {
        d_lm_head_weight_ = get_d_ptr(model_config_.header().lm_head_offset, true);
        d_lm_head_scale_  = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().reserved[1]));
    } else {
        d_lm_head_weight_ = d_embed_weight_;
        d_lm_head_scale_  = d_embed_scale_;
    }
    d_final_norm_ = reinterpret_cast<const half*>(get_d_ptr(model_config_.header().final_norm_offset));

    for (int l = 0; l < NUM_LAYERS; ++l) {
        const auto& desc = model_config_.layer(l);
        layers_[l].input_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm1_weight_offset));
        layers_[l].post_attention_layernorm = reinterpret_cast<const half*>(get_d_ptr(desc.norm2_weight_offset));

        layers_[l].q_weight = get_d_ptr(desc.qkv_weight_offset, true);
        layers_[l].q_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.q_bias_offset));
        layers_[l].k_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.k_bias_offset));
        layers_[l].v_bias   = reinterpret_cast<const half*>(get_d_ptr(desc.v_bias_offset));
        layers_[l].o_weight = get_d_ptr(desc.attn_out_weight_offset, true);
        layers_[l].k_weight = get_d_ptr(desc.k_weight_offset, true);
        layers_[l].v_weight = get_d_ptr(desc.v_weight_offset, true);
        layers_[l].q_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.q_scale_offset));
        layers_[l].k_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.k_scale_offset));
        layers_[l].v_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.v_scale_offset));
        layers_[l].o_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.attn_out_scale_offset));

        layers_[l].gate_up_weight = get_d_ptr(desc.mlp_gate_offset, true);
        layers_[l].gate_up_scale  = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_gate_scale_offset));
        layers_[l].down_weight    = get_d_ptr(desc.mlp_down_offset, true);
        layers_[l].down_scale     = reinterpret_cast<const half*>(get_d_ptr(desc.mlp_down_scale_offset));
    }

    if (!weights_16byte_aligned_) {
        std::cout << "[QwenEngine] Notice: Unaligned .qwenbin v2 weight offset detected; using uint32_t GEMV fallback." << std::endl;
    }

    d_k_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements * sizeof(half), 256));
    d_v_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements * sizeof(half), 256));
    CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, kv_elements * sizeof(half), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, kv_elements * sizeof(half), stream_));

    ws_.d_act_0         = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * HIDDEN_SIZE * sizeof(half), 256));
    ws_.d_act_1         = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * HIDDEN_SIZE * sizeof(half), 256));
    ws_.d_q             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * Q_DIM * sizeof(half), 256));
    ws_.d_k             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * KV_DIM * sizeof(half), 256));
    ws_.d_v             = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * KV_DIM * sizeof(half), 256));
    ws_.d_attn_out      = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * HIDDEN_SIZE * sizeof(half), 256));
    ws_.d_mlp_act       = reinterpret_cast<half*>(arena_->allocate(PREFILL_MAX_CHUNK * INTERMEDIATE_SIZE * sizeof(half), 256));
    ws_.d_splitk_out    = reinterpret_cast<float*>(arena_->allocate(NUM_Q_HEADS * FLASH_DECODE_SPLITS * HEAD_DIM * sizeof(float), 256));
    ws_.d_splitk_lse    = reinterpret_cast<float*>(arena_->allocate(NUM_Q_HEADS * FLASH_DECODE_SPLITS * 2 * sizeof(float), 256));
    ws_.d_logits        = reinterpret_cast<half*>(arena_->allocate(VOCAB_SIZE * sizeof(half), 256));
    ws_.d_sample_val    = reinterpret_cast<float*>(arena_->allocate(256 * sizeof(float), 256));
    ws_.d_sample_idx    = reinterpret_cast<int32_t*>(arena_->allocate(256 * sizeof(int32_t), 256));
    ws_.d_prompt_tokens = reinterpret_cast<int32_t*>(arena_->allocate(max_seq_len_ * sizeof(int32_t), 256));

    // Dedicated 16 KB aligned Q8_1 scratch buffer (separate from d_act_1)
    uint8_t* d_q8_base = reinterpret_cast<uint8_t*>(arena_->allocate(q8_scratch_bytes, 256));
    size_t q8_bytes = align_up(INTERMEDIATE_SIZE, 256); // 8960 -> 8960 B (multiple of 256)
    size_t num_groups = (INTERMEDIATE_SIZE + QUANT_GROUP_SIZE - 1) / QUANT_GROUP_SIZE; // 70
    size_t scale_bytes = align_up(num_groups * sizeof(float), 256); // 280 -> 512 B

    ws_.d_act_q8    = reinterpret_cast<int8_t*>(d_q8_base);
    ws_.d_act_scale = reinterpret_cast<float*>(d_q8_base + q8_bytes);
    ws_.d_act_sum8  = reinterpret_cast<int32_t*>(d_q8_base + q8_bytes + scale_bytes);

    d_control_ = reinterpret_cast<DeviceControlBlock*>(arena_->allocate(sizeof(DeviceControlBlock), 256));

    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_token_), sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mapped_pinned_token_), h_pinned_token_, 0));

    DeviceControlBlock init_dcb = { 0, 0, 0.0f, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &init_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
}

void QwenEngine::reset_session() {
    CudaDeviceGuard guard(device_id_);
    cached_tokens_.clear();
    seq_pos_ = 0;
    last_token_id_ = 0;
    DeviceControlBlock reset_dcb = { 0, 0, temperature_, 0, 1337ULL };
    CUDA_CHECK(cudaMemcpy(d_control_, &reset_dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));
    *h_pinned_token_ = 0;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void QwenEngine::execute_gqa_decode_layer(int l) {
    size_t kv_stride = static_cast<size_t>(NUM_KV_HEADS) * max_seq_len_ * HEAD_DIM;
    half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * kv_stride);
    half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * kv_stride);
    float rope_theta = model_config_.rope_theta();

    launch_rmsnorm_q8_1(
        ws_.d_act_0, layers_[l].input_layernorm, nullptr,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
        HIDDEN_SIZE
    );

    QKVProjectionView qkv_view = {
        layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
        layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
        layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
    };
    launch_gemv_qkv_w4a8(qkv_view, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
                         HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);

    launch_qkv_bias_rope_kv_append(
        ws_.d_q, ws_.d_k, ws_.d_v,
        layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
        layer_k, layer_v, d_control_, 1, stream_,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
        HEAD_DIM, rope_theta
    );

    launch_attention_gqa_flash_decode(
        ws_.d_q, layer_k, layer_v,
        ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8
    );

    launch_gemv_w4a8(
        reinterpret_cast<const uint32_t*>(layers_[l].o_weight),
        layers_[l].o_scale,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
        ws_.d_act_0, ws_.d_act_0,
        HIDDEN_SIZE, Q_DIM, stream_
    );
}

void QwenEngine::execute_decode_pass() {
    launch_embed_lookup_w4a16(
        reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_,
        nullptr, ws_.d_act_0, d_control_, 1, stream_, HIDDEN_SIZE
    );

    for (int l = 0; l < NUM_LAYERS; ++l) {
        execute_gqa_decode_layer(l);

        launch_rmsnorm_q8_1(
            ws_.d_act_0, layers_[l].post_attention_layernorm, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
            HIDDEN_SIZE
        );
        launch_gemv_dual_gate_up_swiglu_w4a8(
            reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight),
            layers_[l].gate_up_scale,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            ws_.d_mlp_act, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
        );
        launch_quantize_fp16_to_q8_1(ws_.d_mlp_act, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, INTERMEDIATE_SIZE, stream_);
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(layers_[l].down_weight),
            layers_[l].down_scale,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            ws_.d_act_0, ws_.d_act_0,
            HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
        );
    }

    launch_rmsnorm_q8_1(
        ws_.d_act_0, d_final_norm_, nullptr,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
        HIDDEN_SIZE
    );
    launch_gemv_w4a8(
        reinterpret_cast<const uint32_t*>(d_lm_head_weight_),
        d_lm_head_scale_,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
        nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_
    );

    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_,
        VOCAB_SIZE
    );
}

void QwenEngine::capture_decode_graph() {
    CudaDeviceGuard guard(device_id_);

    std::cout << "[QwenEngine] Capturing zero-update CUDA Graph for autoregressive decoding..." << std::endl;

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
    (void)chunk_start;
    size_t kv_stride = static_cast<size_t>(NUM_KV_HEADS) * max_seq_len_ * HEAD_DIM;
    half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * kv_stride);
    half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * kv_stride);
    float rope_theta = model_config_.rope_theta();

    QKVProjectionView qkv_view = {
        layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
        layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
        layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
    };
    launch_gemm_prefill_qkv_w4a16(qkv_view, ws_.d_act_1, chunk_len, stream_,
                                  HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);

    launch_qkv_bias_rope_kv_append(
        ws_.d_q, ws_.d_k, ws_.d_v,
        layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
        layer_k, layer_v, d_control_, chunk_len, stream_,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
        HEAD_DIM, rope_theta
    );

    launch_attention_gqa_prefill(
        ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_), Q_DIM
    );

    launch_gemm_prefill_w4a16(
        reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale,
        ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, Q_DIM, stream_
    );
}

void QwenEngine::execute_prefill_chunk(int chunk_start, int chunk_len) {
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    launch_embed_lookup_w4a16(
        reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_,
        ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_, HIDDEN_SIZE
    );

    for (int l = 0; l < NUM_LAYERS; ++l) {
        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, chunk_len, stream_, HIDDEN_SIZE);
        execute_gqa_prefill_layer(l, chunk_start, chunk_len);
        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_, HIDDEN_SIZE);
        launch_gemm_prefill_dual_gate_up_swiglu_w4a16(
            reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale,
            ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, HIDDEN_SIZE, INTERMEDIATE_SIZE
        );
        launch_gemm_prefill_w4a16(
            reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale,
            ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
        );
    }
}

int32_t QwenEngine::prefill(const std::vector<int32_t>& prompt_tokens, float temperature) {
    CudaDeviceGuard guard(device_id_);
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

    half* d_last_hidden = ws_.d_act_0 + static_cast<size_t>((total_tokens - 1) % PREFILL_MAX_CHUNK) * HIDDEN_SIZE;

    launch_rmsnorm_q8_1(
        d_last_hidden, d_final_norm_, nullptr,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
        HIDDEN_SIZE
    );
    launch_gemv_w4a8(
        reinterpret_cast<const uint32_t*>(d_lm_head_weight_),
        d_lm_head_scale_,
        ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
        nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_
    );

    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_,
        VOCAB_SIZE
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    last_token_id_ = *h_pinned_token_;
    cached_tokens_ = prompt_tokens;
    cached_tokens_.push_back(last_token_id_);
    seq_pos_ = total_tokens;

    return last_token_id_;
}

int32_t QwenEngine::decode_step() {
    CudaDeviceGuard guard(device_id_);
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
    CudaDeviceGuard guard(device_id_);
    if (target_seq_pos < 0 || target_seq_pos > seq_pos_) {
        throw QwenError("QwenEngine::rollback_to: target_seq_pos out of bounds [0, " +
                        std::to_string(seq_pos_) + "]: " + std::to_string(target_seq_pos));
    }

    seq_pos_ = target_seq_pos;
    cached_tokens_.resize(target_seq_pos);
    last_token_id_ = (target_seq_pos > 0) ? cached_tokens_[target_seq_pos - 1] : TOKEN_ENDOFTEXT;

    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &seq_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->token_id), &last_token_id_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool QwenEngine::try_acquire_session() {
    bool expected = false;
    return is_generating_.compare_exchange_strong(expected, true);
}

void QwenEngine::release_session() {
    is_generating_.store(false);
}

StageProfile QwenEngine::profile_stages(const std::vector<int32_t>& prompt_tokens) {
    CudaDeviceGuard guard(device_id_);
    if (prompt_tokens.empty()) {
        throw QwenError("QwenEngine::profile_stages: Empty prompt tokens");
    }
    if (prompt_tokens.size() >= max_seq_len_) {
        throw ContextWindowExceededError("QwenEngine: Prompt length exceeds MAX_SEQ_LEN");
    }

    StageProfile profile = {};
    reset_session();

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    auto time_stage = [&](float& accum_ms, auto&& fn) {
        CUDA_CHECK(cudaEventRecord(ev_start, stream_));
        fn();
        CUDA_CHECK(cudaEventRecord(ev_stop, stream_));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        accum_ms += ms;
    };

    int total_tokens = static_cast<int>(prompt_tokens.size());
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_prompt_tokens, prompt_tokens.data(),
                               total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    // 1. Timed Prefill Pass
    for (int chunk_start = 0; chunk_start < total_tokens; chunk_start += PREFILL_MAX_CHUNK) {
        int chunk_len = std::min(PREFILL_MAX_CHUNK, total_tokens - chunk_start);
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

        time_stage(profile.prefill.embed_ms, [&]() {
            launch_embed_lookup_w4a16(
                reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_,
                ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_, HIDDEN_SIZE
            );
        });

        size_t kv_stride = static_cast<size_t>(NUM_KV_HEADS) * max_seq_len_ * HEAD_DIM;
        float rope_theta = model_config_.rope_theta();

        for (int l = 0; l < NUM_LAYERS; ++l) {
            half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * kv_stride);
            half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * kv_stride);

            time_stage(profile.prefill.rmsnorm_ms, [&]() {
                launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, chunk_len, stream_, HIDDEN_SIZE);
            });
            time_stage(profile.prefill.qkv_ms, [&]() {
                QKVProjectionView qkv_view = {
                    layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                    layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                    layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
                };
                launch_gemm_prefill_qkv_w4a16(qkv_view, ws_.d_act_1, chunk_len, stream_,
                                              HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);
            });
            time_stage(profile.prefill.rope_kv_ms, [&]() {
                launch_qkv_bias_rope_kv_append(
                    ws_.d_q, ws_.d_k, ws_.d_v,
                    layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                    layer_k, layer_v, d_control_, chunk_len, stream_,
                    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
                    HEAD_DIM, rope_theta
                );
            });
            time_stage(profile.prefill.attn_ms, [&]() {
                launch_attention_gqa_prefill(
                    ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_,
                    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_), Q_DIM
                );
            });
            time_stage(profile.prefill.o_proj_ms, [&]() {
                launch_gemm_prefill_w4a16(
                    reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale,
                    ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, Q_DIM, stream_
                );
            });
            time_stage(profile.prefill.post_norm_ms, [&]() {
                launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_, HIDDEN_SIZE);
            });
            time_stage(profile.prefill.gate_up_swiglu_ms, [&]() {
                launch_gemm_prefill_dual_gate_up_swiglu_w4a16(
                    reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale,
                    ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_, HIDDEN_SIZE, INTERMEDIATE_SIZE
                );
            });
            time_stage(profile.prefill.down_proj_ms, [&]() {
                launch_gemm_prefill_w4a16(
                    reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale,
                    ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
                );
            });
        }
    }

    int last_pos = total_tokens - 1;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &last_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    half* d_last_hidden = ws_.d_act_0 + static_cast<size_t>((total_tokens - 1) % PREFILL_MAX_CHUNK) * HIDDEN_SIZE;

    time_stage(profile.prefill.final_norm_ms, [&]() {
        launch_rmsnorm_q8_1(d_last_hidden, d_final_norm_, nullptr, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_, HIDDEN_SIZE);
    });
    time_stage(profile.prefill.lm_head_ms, [&]() {
        launch_gemv_w4a8(reinterpret_cast<const uint32_t*>(d_lm_head_weight_), d_lm_head_scale_,
                         ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                         nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    });
    time_stage(profile.prefill.sample_ms, [&]() {
        launch_sample_and_advance(ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_, VOCAB_SIZE);
    });

    profile.prefill.total_ms =
        profile.prefill.embed_ms + profile.prefill.rmsnorm_ms + profile.prefill.qkv_ms +
        profile.prefill.rope_kv_ms + profile.prefill.attn_ms + profile.prefill.o_proj_ms +
        profile.prefill.post_norm_ms + profile.prefill.gate_up_swiglu_ms +
        profile.prefill.down_proj_ms + profile.prefill.final_norm_ms +
        profile.prefill.lm_head_ms + profile.prefill.sample_ms;

    seq_pos_ = total_tokens;
    last_token_id_ = *h_pinned_token_;

    // 2. Timed Ungraphed Decode Pass (1 token)
    time_stage(profile.decode.embed_ms, [&]() {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_,
            nullptr, ws_.d_act_0, d_control_, 1, stream_, HIDDEN_SIZE
        );
    });

    size_t kv_stride = static_cast<size_t>(NUM_KV_HEADS) * max_seq_len_ * HEAD_DIM;
    float rope_theta = model_config_.rope_theta();

    for (int l = 0; l < NUM_LAYERS; ++l) {
        half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * kv_stride);
        half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * kv_stride);

        time_stage(profile.decode.rmsnorm_ms, [&]() {
            launch_rmsnorm_q8_1(ws_.d_act_0, layers_[l].input_layernorm, nullptr,
                                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_, HIDDEN_SIZE);
        });
        time_stage(profile.decode.qkv_ms, [&]() {
            QKVProjectionView qkv_view = {
                layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
            };
            launch_gemv_qkv_w4a8(qkv_view, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_,
                                 HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM);
        });
        time_stage(profile.decode.rope_kv_ms, [&]() {
            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, 1, stream_,
                NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
                HEAD_DIM, rope_theta
            );
        });
        time_stage(profile.decode.attn_ms, [&]() {
            launch_attention_gqa_flash_decode(
                ws_.d_q, layer_k, layer_v,
                ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
                NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, static_cast<int>(max_seq_len_),
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8
            );
        });
        time_stage(profile.decode.o_proj_ms, [&]() {
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, Q_DIM, stream_
            );
        });
        time_stage(profile.decode.post_norm_ms, [&]() {
            launch_rmsnorm_q8_1(ws_.d_act_0, layers_[l].post_attention_layernorm, nullptr,
                                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_, HIDDEN_SIZE);
        });
        time_stage(profile.decode.gate_up_swiglu_ms, [&]() {
            launch_gemv_dual_gate_up_swiglu_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_mlp_act, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
            );
        });
        time_stage(profile.decode.quant_down_ms, [&]() {
            launch_quantize_fp16_to_q8_1(ws_.d_mlp_act, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, INTERMEDIATE_SIZE, stream_);
        });
        time_stage(profile.decode.down_proj_ms, [&]() {
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
            );
        });
    }

    time_stage(profile.decode.final_norm_ms, [&]() {
        launch_rmsnorm_q8_1(ws_.d_act_0, d_final_norm_, nullptr, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_, HIDDEN_SIZE);
    });
    time_stage(profile.decode.lm_head_ms, [&]() {
        launch_gemv_w4a8(reinterpret_cast<const uint32_t*>(d_lm_head_weight_), d_lm_head_scale_,
                         ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                         nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    });
    time_stage(profile.decode.sample_ms, [&]() {
        launch_sample_and_advance(ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_, VOCAB_SIZE);
    });

    profile.decode.total_ms =
        profile.decode.embed_ms + profile.decode.rmsnorm_ms + profile.decode.qkv_ms +
        profile.decode.rope_kv_ms + profile.decode.attn_ms + profile.decode.o_proj_ms +
        profile.decode.post_norm_ms + profile.decode.gate_up_swiglu_ms +
        profile.decode.quant_down_ms + profile.decode.down_proj_ms +
        profile.decode.final_norm_ms + profile.decode.lm_head_ms +
        profile.decode.sample_ms;

    // Approximate FLOPs & weight bandwidth metrics
    double prefill_flops = 2.0 * static_cast<double>(total_tokens) * 1.54e9;
    profile.prefill_tflops = (profile.prefill.total_ms > 0.0f)
        ? static_cast<float>((prefill_flops / (profile.prefill.total_ms * 1e-3)) / 1e12)
        : 0.0f;
    profile.decode_bw_gbs = (profile.decode.total_ms > 0.0f)
        ? static_cast<float>((static_cast<double>(weights_bytes_) / (profile.decode.total_ms * 1e-3)) / 1e9)
        : 0.0f;

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    reset_session();
    return profile;
}

ArenaStats QwenEngine::get_memory_stats() const {
    CudaDeviceGuard guard(device_id_);
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
