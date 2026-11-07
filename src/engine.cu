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

QwenEngine::QwenEngine(const std::string& qwenbin_path, int device_id)
    : qwenbin_path_(qwenbin_path),
      device_id_(device_id),
      quant_mode_(QuantMode::FP16),
      is_generating_(false),
      file_map_(nullptr),
      mapped_data_(nullptr),
      mapped_size_(0),
      stream_(nullptr),
      d_embed_weight_(nullptr),
      d_embed_scale_(nullptr),
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
    load_model_file(qwenbin_path);
    setup_arena();
    capture_decode_graph();
}

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

    const QWQwenHeader* header = reinterpret_cast<const QWQwenHeader*>(mapped_data_);
    if (header->magic != 0x5157454E || header->version != 1) {
        throw QwenError("QwenEngine: Invalid header or version in " + qwenbin_path);
    }

    quant_mode_ = static_cast<QuantMode>(header->quant_mode);
}

void QwenEngine::setup_arena() {
    const QWQwenHeader* header = reinterpret_cast<const QWQwenHeader*>(mapped_data_);

    // 1. Calculate weights size
    // In .qwenbin, tensor payloads begin immediately after the 4096-byte header
    weights_bytes_ = mapped_size_ - 4096;

    // 2. Static KV Cache: [28, 2, 4096, 128] for K and V
    size_t kv_elements_per_cache = static_cast<size_t>(NUM_LAYERS) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM;
    kv_cache_bytes_ = 2 * kv_elements_per_cache * sizeof(half); // K cache + V cache = 114.69 MB

    // 3. Workspace Buffers (~10.2 MB inside 32 MB arena block)
    workspace_bytes_ = 32ULL << 20;

    // 4. DeviceControlBlock: 256 bytes
    size_t dcb_bytes = 256;

    size_t total_arena_bytes = weights_bytes_ + kv_cache_bytes_ + workspace_bytes_ + dcb_bytes;

    std::cout << "[QwenEngine] Initializing GPU Arena..." << std::endl;
    std::cout << "  - Weights Region:   " << (weights_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - Static KV Cache:  " << (kv_cache_bytes_ / (1024.0 * 1024.0)) << " MB ("
              << MAX_SEQ_LEN << " tokens)" << std::endl;
    std::cout << "  - Workspace Region: " << (workspace_bytes_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  - Total Allocation: " << (total_arena_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    arena_ = std::make_unique<GpuArena>(total_arena_bytes);

    // Sub-allocate weights in GPU VRAM and copy in a single bulk transfer
    void* d_weights_base = arena_->allocate(weights_bytes_, 256);
    CUDA_CHECK(cudaMemcpy(d_weights_base, mapped_data_ + 4096, weights_bytes_, cudaMemcpyHostToDevice));

    // Resolve device pointers using header byte offsets
    auto get_d_ptr = [d_weights_base](uint64_t offset) -> const void* {
        if (offset == 0) return nullptr;
        return reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(d_weights_base) + (offset - 4096));
    };

    d_embed_weight_ = get_d_ptr(header->embed_weight_offset);
    d_embed_scale_  = reinterpret_cast<const half*>(get_d_ptr(header->embed_scale_offset));
    d_final_norm_   = reinterpret_cast<const half*>(get_d_ptr(header->final_norm_offset));

    for (int l = 0; l < NUM_LAYERS; ++l) {
        layers_[l].input_layernorm = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::INPUT_LAYERNORM]));
        layers_[l].q_weight        = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::Q_WEIGHT]);
        layers_[l].q_scale         = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::Q_SCALE]));
        layers_[l].q_bias          = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::Q_BIAS]));
        layers_[l].k_weight        = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::K_WEIGHT]);
        layers_[l].k_scale         = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::K_SCALE]));
        layers_[l].k_bias          = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::K_BIAS]));
        layers_[l].v_weight        = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::V_WEIGHT]);
        layers_[l].v_scale         = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::V_SCALE]));
        layers_[l].v_bias          = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::V_BIAS]));
        layers_[l].o_weight        = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::O_WEIGHT]);
        layers_[l].o_scale         = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::O_SCALE]));
        layers_[l].post_attention_layernorm = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::POST_ATTENTION_LAYERNORM]));
        layers_[l].gate_up_weight  = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::GATE_UP_WEIGHT]);
        layers_[l].gate_up_scale   = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::GATE_UP_SCALE]));
        layers_[l].down_weight     = get_d_ptr(header->layer_offsets[l][LayerTensorIdx::DOWN_WEIGHT]);
        layers_[l].down_scale      = reinterpret_cast<const half*>(get_d_ptr(header->layer_offsets[l][LayerTensorIdx::DOWN_SCALE]));
    }

    // Sub-allocate static KV Cache
    d_k_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements_per_cache * sizeof(half), 256));
    d_v_cache_ = reinterpret_cast<half*>(arena_->allocate(kv_elements_per_cache * sizeof(half), 256));

    // Sub-allocate workspace buffers
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
    ws_.d_prompt_tokens = reinterpret_cast<int32_t*>(arena_->allocate(MAX_SEQ_LEN * sizeof(int32_t), 256));

    // Carve Q8_1 scratch buffers within ws_.d_act_1 at fixed offsets (zero extra arena allocations)
    // ws_.d_act_1 has size 786,432 bytes; single-token decode uses 3,072 bytes.
    // offsets: +65536 (9,216 B), +74752 (512 B), +75264 (4,608 B)
    ws_.d_act_q8    = reinterpret_cast<int8_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 65536);
    ws_.d_act_scale = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 74752);
    ws_.d_act_sum8  = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(ws_.d_act_1) + 75264);

    // Sub-allocate DeviceControlBlock
    d_control_ = reinterpret_cast<DeviceControlBlock*>(arena_->allocate(sizeof(DeviceControlBlock), 256));

    // Allocate mapped pinned host pointer for zero-copy device-to-host token reads
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_token_), sizeof(int32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mapped_pinned_token_), h_pinned_token_, 0));

    // Initialize DeviceControlBlock
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
}

void QwenEngine::execute_decode_pass() {
    // 1. Embedding lookup (reads d_control->token_id directly from device memory)
    if (quant_mode_ == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(d_embed_weight_), nullptr, ws_.d_act_0, d_control_, 1, stream_
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, nullptr, ws_.d_act_0, d_control_, 1, stream_
        );
    }

    // 2. Transformer layers 0 .. 27
    for (int l = 0; l < NUM_LAYERS; ++l) {
        half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);
        half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);

        if (quant_mode_ == QuantMode::FP16) {
            // Input RMSNorm
            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, 1, stream_);

            // Merged QKV Projections
            QKVProjectionView qkv_view = {
                layers_[l].q_weight, nullptr, ws_.d_q,
                layers_[l].k_weight, nullptr, ws_.d_k,
                layers_[l].v_weight, nullptr, ws_.d_v
            };
            launch_gemv_qkv_fp16(qkv_view, ws_.d_act_1, stream_);

            // QKV Bias + NeoX RoPE + KV Cache Append
            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, 1, stream_
            );

            // GQA FlashDecoding
            launch_attention_gqa_flash_decode(
                ws_.d_q, layer_k, layer_v,
                ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_
            );

            // O-Proj + Residual Add into d_act_0
            launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, HIDDEN_SIZE, stream_);

            // Post-Attention RMSNorm
            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, 1, stream_);

            // Fused Dual Gate+Up + SwiGLU
            launch_gemv_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, stream_);

            // Down-Proj + Residual Add into d_act_0
            launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
        } else {
            // Input RMSNorm with fused Q8_1 quantization
            launch_rmsnorm_q8_1(
                ws_.d_act_0, layers_[l].input_layernorm, nullptr,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
            );

            // Merged QKV Projections (W4A8 __dp4a single launch)
            QKVProjectionView qkv_view = {
                layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
            };
            launch_gemv_qkv_w4a8(qkv_view, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_);

            // QKV Bias + NeoX RoPE + KV Cache Append
            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, 1, stream_
            );

            // GQA FlashDecoding with fused Q8_1 epilogue for O-Proj
            launch_attention_gqa_flash_decode(
                ws_.d_q, layer_k, layer_v,
                ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8
            );

            // O-Proj + Residual Add into d_act_0 (W4A8 __dp4a)
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].o_weight),
                layers_[l].o_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_act_0, ws_.d_act_0,
                HIDDEN_SIZE, HIDDEN_SIZE, stream_
            );

            // Post-Attention RMSNorm with fused Q8_1 quantization
            launch_rmsnorm_q8_1(
                ws_.d_act_0, layers_[l].post_attention_layernorm, nullptr,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
            );

            // Fused Dual Gate+Up + SwiGLU (W4A8 __dp4a)
            launch_gemv_dual_gate_up_swiglu_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight),
                layers_[l].gate_up_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_mlp_act, stream_
            );

            // Quantize d_mlp_act [8960 half] to Q8_1 for Down-Proj
            launch_quantize_fp16_to_q8_1(
                ws_.d_mlp_act, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                INTERMEDIATE_SIZE, stream_
            );

            // Down-Proj + Residual Add into d_act_0 (W4A8 __dp4a)
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(layers_[l].down_weight),
                layers_[l].down_scale,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                ws_.d_act_0, ws_.d_act_0,
                HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
            );
        }
    }

    // 3. Final RMSNorm
    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(ws_.d_act_0, d_final_norm_, ws_.d_act_1, 1, stream_);
        // 4. Tied LM Head GEMV
        launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    } else {
        launch_rmsnorm_q8_1(
            ws_.d_act_0, d_final_norm_, nullptr,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
        );
        // 4. Tied LM Head GEMV (W4A8 __dp4a)
        launch_gemv_w4a8(
            reinterpret_cast<const uint32_t*>(d_embed_weight_),
            d_embed_scale_,
            ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
            nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_
        );
    }

    // 5. Sample and Advance (increments d_control->seq_pos, writes d_control->token_id and *h_pinned_token_)
    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_
    );
}

void QwenEngine::capture_decode_graph() {
    CudaDeviceGuard guard(device_id_);
    std::cout << "[QwenEngine] Capturing zero-update CUDA Graph for autoregressive decoding..." << std::endl;

    // Warm-up pass before capture
    execute_decode_pass();
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Capture the entire decode forward pass
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    execute_decode_pass();
    CUDA_CHECK(cudaStreamEndCapture(stream_, &decode_graph_));

    CUDA_CHECK(cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
    graph_captured_ = true;

    // Reset control block after capture warm-up
    reset_session();
    std::cout << "[QwenEngine] CUDA Graph successfully instantiated." << std::endl;
}

void QwenEngine::execute_prefill_chunk(int chunk_start, int chunk_len) {
    // Set device control block sequence position for this chunk
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    // 1. Embedding lookup for chunk_len tokens
    if (quant_mode_ == QuantMode::FP16) {
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(d_embed_weight_), ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_
        );
    } else {
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_
        );
    }

    // 2. Transformer layers 0 .. 27
    for (int l = 0; l < NUM_LAYERS; ++l) {
        half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);
        half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);

        launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, chunk_len, stream_);

        if (quant_mode_ == QuantMode::FP16) {
            QKVProjectionView qkv_view = {
                layers_[l].q_weight, nullptr, ws_.d_q,
                layers_[l].k_weight, nullptr, ws_.d_k,
                layers_[l].v_weight, nullptr, ws_.d_v
            };
            launch_gemm_prefill_qkv_fp16(qkv_view, ws_.d_act_1, chunk_len, stream_);

            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, chunk_len, stream_
            );

            launch_attention_gqa_prefill(
                ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_
            );

            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, HIDDEN_SIZE, stream_);

            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_);

            launch_gemm_prefill_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_);

            launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
        } else {
            QKVProjectionView qkv_view = {
                layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
            };
            launch_gemm_prefill_qkv_w4a16(qkv_view, ws_.d_act_1, chunk_len, stream_);

            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, chunk_len, stream_
            );

            launch_attention_gqa_prefill(
                ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_
            );

            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale, ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, HIDDEN_SIZE, stream_);

            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_);

            launch_gemm_prefill_dual_gate_up_swiglu_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale, ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_);

            launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale, ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
        }
    }
}

int32_t QwenEngine::prefill(const std::vector<int32_t>& prompt_tokens, float temperature) {
    CudaDeviceGuard guard(device_id_);
    if (prompt_tokens.empty()) return 0;
    if (prompt_tokens.size() >= MAX_SEQ_LEN) {
        throw ContextWindowExceededError("QwenEngine: Prompt length exceeds MAX_SEQ_LEN");
    }

    temperature_ = temperature;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->temperature), &temperature_, sizeof(float), cudaMemcpyHostToDevice, stream_));

    // Stateful KV prefix matching
    size_t prefix_len = 0;
    while (prefix_len < cached_tokens_.size() &&
           prefix_len < prompt_tokens.size() &&
           cached_tokens_[prefix_len] == prompt_tokens[prefix_len]) {
        prefix_len++;
    }

    int total_tokens = static_cast<int>(prompt_tokens.size());
    int start_pos = static_cast<int>(prefix_len);

    if (start_pos < total_tokens) {
        // Copy uncached prompt suffix once via single cudaMemcpyAsync
        CUDA_CHECK(cudaMemcpyAsync(
            ws_.d_prompt_tokens + start_pos,
            prompt_tokens.data() + start_pos,
            (total_tokens - start_pos) * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream_
        ));

        // Process prompt in chunks of PREFILL_MAX_CHUNK = 256 with zero per-chunk host synchronization
        for (int chunk_start = start_pos; chunk_start < total_tokens; chunk_start += PREFILL_MAX_CHUNK) {
            int chunk_len = min(PREFILL_MAX_CHUNK, total_tokens - chunk_start);
            execute_prefill_chunk(chunk_start, chunk_len);
        }
    }

    // Evaluate final token (index total_tokens - 1) for the first generated token
    int last_chunk_len = min(PREFILL_MAX_CHUNK, total_tokens - ((total_tokens - 1) / PREFILL_MAX_CHUNK) * PREFILL_MAX_CHUNK);
    int last_token_row = last_chunk_len - 1;
    half* last_act = ws_.d_act_0 + static_cast<size_t>(last_token_row) * HIDDEN_SIZE;

    if (quant_mode_ == QuantMode::FP16) {
        // Final RMSNorm
        launch_rmsnorm_fp16(last_act, d_final_norm_, ws_.d_act_1, 1, stream_);
        // LM Head Projection
        launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    } else {
        // Final RMSNorm with Q8_1 quantization
        launch_rmsnorm_q8_1(last_act, d_final_norm_, nullptr, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_);
        // LM Head Projection with W4A8 __dp4a
        launch_gemv_w4a8(reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    }

    // Set sequence position for the new token in d_control
    int final_seq_pos = total_tokens - 1;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &final_seq_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    // Sample next token and advance
    launch_sample_and_advance(
        ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    last_token_id_ = *h_pinned_token_;
    cached_tokens_ = prompt_tokens;
    cached_tokens_.push_back(last_token_id_);
    seq_pos_ = total_tokens; // next slot to write

    return last_token_id_;
}

int32_t QwenEngine::decode_step() {
    CudaDeviceGuard guard(device_id_);
    if (seq_pos_ >= MAX_SEQ_LEN) {
        throw ContextWindowExceededError("QwenEngine: Context length reached MAX_SEQ_LEN (4096)");
    }

    // Autoregressive single step execution via zero-update captured CUDA Graph
    CUDA_CHECK(cudaGraphLaunch(decode_graph_exec_, stream_));
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

    StageProfile profile = {};

    // Structure for easy timing
    struct GpuTimer {
        cudaEvent_t st, sp;
        GpuTimer() {
            CUDA_CHECK(cudaEventCreate(&st));
            CUDA_CHECK(cudaEventCreate(&sp));
        }
        ~GpuTimer() {
            cudaEventDestroy(st);
            cudaEventDestroy(sp);
        }
        void record_start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(st, s)); }
        void record_stop(cudaStream_t s)  { CUDA_CHECK(cudaEventRecord(sp, s)); }
        float elapsed_ms() {
            CUDA_CHECK(cudaEventSynchronize(sp));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, st, sp));
            return ms;
        }
    };

    reset_session();

    // -------------------------------------------------------------
    // 1. Profile Prefill Stages
    // -------------------------------------------------------------
    int total_tokens = static_cast<int>(prompt_tokens.size());
    CUDA_CHECK(cudaMemcpyAsync(
        ws_.d_prompt_tokens,
        prompt_tokens.data(),
        total_tokens * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream_
    ));

    GpuTimer t_prefill_total;
    GpuTimer t_prefill_embed;
    GpuTimer t_prefill_in_norm[NUM_LAYERS];
    GpuTimer t_prefill_qkv[NUM_LAYERS];
    GpuTimer t_prefill_rope[NUM_LAYERS];
    GpuTimer t_prefill_attn[NUM_LAYERS];
    GpuTimer t_prefill_oproj[NUM_LAYERS];
    GpuTimer t_prefill_post_norm[NUM_LAYERS];
    GpuTimer t_prefill_swiglu[NUM_LAYERS];
    GpuTimer t_prefill_down[NUM_LAYERS];
    GpuTimer t_prefill_final_norm;
    GpuTimer t_prefill_lm_head;
    GpuTimer t_prefill_sample;

    t_prefill_total.record_start(stream_);

    for (int chunk_start = 0; chunk_start < total_tokens; chunk_start += PREFILL_MAX_CHUNK) {
        int chunk_len = min(PREFILL_MAX_CHUNK, total_tokens - chunk_start);
        CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &chunk_start, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

        if (chunk_start == 0) t_prefill_embed.record_start(stream_);
        if (quant_mode_ == QuantMode::FP16) {
            launch_embed_lookup_fp16(
                reinterpret_cast<const half*>(d_embed_weight_), ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_
            );
        } else {
            launch_embed_lookup_w4a16(
                reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, ws_.d_prompt_tokens + chunk_start, ws_.d_act_0, nullptr, chunk_len, stream_
            );
        }
        if (chunk_start == 0) t_prefill_embed.record_stop(stream_);

        for (int l = 0; l < NUM_LAYERS; ++l) {
            half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);
            half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);

            t_prefill_in_norm[l].record_start(stream_);
            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, chunk_len, stream_);
            t_prefill_in_norm[l].record_stop(stream_);

            t_prefill_qkv[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                QKVProjectionView qkv_view = {
                    layers_[l].q_weight, nullptr, ws_.d_q,
                    layers_[l].k_weight, nullptr, ws_.d_k,
                    layers_[l].v_weight, nullptr, ws_.d_v
                };
                launch_gemm_prefill_qkv_fp16(qkv_view, ws_.d_act_1, chunk_len, stream_);
            } else {
                QKVProjectionView qkv_view = {
                    layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                    layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                    layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
                };
                launch_gemm_prefill_qkv_w4a16(qkv_view, ws_.d_act_1, chunk_len, stream_);
            }
            t_prefill_qkv[l].record_stop(stream_);

            t_prefill_rope[l].record_start(stream_);
            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, chunk_len, stream_
            );
            t_prefill_rope[l].record_stop(stream_);

            t_prefill_attn[l].record_start(stream_);
            launch_attention_gqa_prefill(
                ws_.d_q, layer_k, layer_v, ws_.d_attn_out, d_control_, chunk_len, stream_
            );
            t_prefill_attn[l].record_stop(stream_);

            t_prefill_oproj[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, HIDDEN_SIZE, stream_);
            } else {
                launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].o_weight), layers_[l].o_scale, ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, HIDDEN_SIZE, stream_);
            }
            t_prefill_oproj[l].record_stop(stream_);

            t_prefill_post_norm[l].record_start(stream_);
            launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, chunk_len, stream_);
            t_prefill_post_norm[l].record_stop(stream_);

            t_prefill_swiglu[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemm_prefill_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_);
            } else {
                launch_gemm_prefill_dual_gate_up_swiglu_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight), layers_[l].gate_up_scale, ws_.d_act_1, ws_.d_mlp_act, chunk_len, stream_);
            }
            t_prefill_swiglu[l].record_stop(stream_);

            t_prefill_down[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemm_prefill_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
            } else {
                launch_gemm_prefill_w4a16(reinterpret_cast<const uint32_t*>(layers_[l].down_weight), layers_[l].down_scale, ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, chunk_len, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
            }
            t_prefill_down[l].record_stop(stream_);
        }
    }

    int last_chunk_len = min(PREFILL_MAX_CHUNK, total_tokens - ((total_tokens - 1) / PREFILL_MAX_CHUNK) * PREFILL_MAX_CHUNK);
    int last_token_row = last_chunk_len - 1;
    half* last_act = ws_.d_act_0 + static_cast<size_t>(last_token_row) * HIDDEN_SIZE;

    t_prefill_final_norm.record_start(stream_);
    if (quant_mode_ == QuantMode::FP16) {
        launch_rmsnorm_fp16(last_act, d_final_norm_, ws_.d_act_1, 1, stream_);
    } else {
        launch_rmsnorm_q8_1(last_act, d_final_norm_, nullptr, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_);
    }
    t_prefill_final_norm.record_stop(stream_);

    t_prefill_lm_head.record_start(stream_);
    if (quant_mode_ == QuantMode::FP16) {
        launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    } else {
        launch_gemv_w4a8(reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
    }
    t_prefill_lm_head.record_stop(stream_);

    int final_seq_pos = total_tokens - 1;
    CUDA_CHECK(cudaMemcpyAsync(&(d_control_->seq_pos), &final_seq_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    t_prefill_sample.record_start(stream_);
    launch_sample_and_advance(ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_);
    t_prefill_sample.record_stop(stream_);

    t_prefill_total.record_stop(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    profile.prefill.embed_ms           = t_prefill_embed.elapsed_ms();
    profile.prefill.final_norm_ms      = t_prefill_final_norm.elapsed_ms();
    profile.prefill.lm_head_ms         = t_prefill_lm_head.elapsed_ms();
    profile.prefill.sample_ms          = t_prefill_sample.elapsed_ms();
    profile.prefill.quant_down_ms      = 0.0f;
    profile.prefill.total_ms           = t_prefill_total.elapsed_ms();

    for (int l = 0; l < NUM_LAYERS; ++l) {
        profile.prefill.rmsnorm_ms         += t_prefill_in_norm[l].elapsed_ms();
        profile.prefill.qkv_ms             += t_prefill_qkv[l].elapsed_ms();
        profile.prefill.rope_kv_ms         += t_prefill_rope[l].elapsed_ms();
        profile.prefill.attn_ms            += t_prefill_attn[l].elapsed_ms();
        profile.prefill.o_proj_ms          += t_prefill_oproj[l].elapsed_ms();
        profile.prefill.post_norm_ms       += t_prefill_post_norm[l].elapsed_ms();
        profile.prefill.gate_up_swiglu_ms  += t_prefill_swiglu[l].elapsed_ms();
        profile.prefill.down_proj_ms       += t_prefill_down[l].elapsed_ms();
    }

    // Prefill TFLOPS computation
    double prefill_sec = profile.prefill.total_ms / 1000.0;
    double prompt_n = static_cast<double>(total_tokens);
    double linear_flops = 2.0 * prompt_n * (28.0 * 46792704.0) + 2.0 * 233373696.0;
    profile.prefill_tflops = (prefill_sec > 0.0) ? static_cast<float>((linear_flops / prefill_sec) / 1e12) : 0.0f;

    // -------------------------------------------------------------
    // 2. Profile Decode Stages (Averaged over 5 decode steps)
    // -------------------------------------------------------------
    constexpr int PROFILE_DECODE_STEPS = 5;

    for (int step = 0; step < PROFILE_DECODE_STEPS; ++step) {
        GpuTimer t_dec_total;
        GpuTimer t_dec_embed;
        GpuTimer t_dec_in_norm[NUM_LAYERS];
        GpuTimer t_dec_qkv[NUM_LAYERS];
        GpuTimer t_dec_rope[NUM_LAYERS];
        GpuTimer t_dec_attn[NUM_LAYERS];
        GpuTimer t_dec_oproj[NUM_LAYERS];
        GpuTimer t_dec_post_norm[NUM_LAYERS];
        GpuTimer t_dec_swiglu[NUM_LAYERS];
        GpuTimer t_dec_quant_down[NUM_LAYERS];
        GpuTimer t_dec_down[NUM_LAYERS];
        GpuTimer t_dec_final_norm;
        GpuTimer t_dec_lm_head;
        GpuTimer t_dec_sample;

        t_dec_total.record_start(stream_);

        t_dec_embed.record_start(stream_);
        if (quant_mode_ == QuantMode::FP16) {
            launch_embed_lookup_fp16(
                reinterpret_cast<const half*>(d_embed_weight_), nullptr, ws_.d_act_0, d_control_, 1, stream_
            );
        } else {
            launch_embed_lookup_w4a16(
                reinterpret_cast<const uint32_t*>(d_embed_weight_), d_embed_scale_, nullptr, ws_.d_act_0, d_control_, 1, stream_
            );
        }
        t_dec_embed.record_stop(stream_);

        for (int l = 0; l < NUM_LAYERS; ++l) {
            half* layer_k = d_k_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);
            half* layer_v = d_v_cache_ + (static_cast<size_t>(l) * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM);

            t_dec_in_norm[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].input_layernorm, ws_.d_act_1, 1, stream_);
            } else {
                launch_rmsnorm_q8_1(
                    ws_.d_act_0, layers_[l].input_layernorm, nullptr,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
                );
            }
            t_dec_in_norm[l].record_stop(stream_);

            t_dec_qkv[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                QKVProjectionView qkv_view = {
                    layers_[l].q_weight, nullptr, ws_.d_q,
                    layers_[l].k_weight, nullptr, ws_.d_k,
                    layers_[l].v_weight, nullptr, ws_.d_v
                };
                launch_gemv_qkv_fp16(qkv_view, ws_.d_act_1, stream_);
            } else {
                QKVProjectionView qkv_view = {
                    layers_[l].q_weight, layers_[l].q_scale, ws_.d_q,
                    layers_[l].k_weight, layers_[l].k_scale, ws_.d_k,
                    layers_[l].v_weight, layers_[l].v_scale, ws_.d_v
                };
                launch_gemv_qkv_w4a8(qkv_view, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_);
            }
            t_dec_qkv[l].record_stop(stream_);

            t_dec_rope[l].record_start(stream_);
            launch_qkv_bias_rope_kv_append(
                ws_.d_q, ws_.d_k, ws_.d_v,
                layers_[l].q_bias, layers_[l].k_bias, layers_[l].v_bias,
                layer_k, layer_v, d_control_, 1, stream_
            );
            t_dec_rope[l].record_stop(stream_);

            t_dec_attn[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_attention_gqa_flash_decode(
                    ws_.d_q, layer_k, layer_v,
                    ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_
                );
            } else {
                launch_attention_gqa_flash_decode(
                    ws_.d_q, layer_k, layer_v,
                    ws_.d_splitk_out, ws_.d_splitk_lse, ws_.d_attn_out, d_control_, stream_,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8
                );
            }
            t_dec_attn[l].record_stop(stream_);

            t_dec_oproj[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].o_weight), ws_.d_attn_out, ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, HIDDEN_SIZE, stream_);
            } else {
                launch_gemv_w4a8(
                    reinterpret_cast<const uint32_t*>(layers_[l].o_weight),
                    layers_[l].o_scale,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                    ws_.d_act_0, ws_.d_act_0,
                    HIDDEN_SIZE, HIDDEN_SIZE, stream_
                );
            }
            t_dec_oproj[l].record_stop(stream_);

            t_dec_post_norm[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_rmsnorm_fp16(ws_.d_act_0, layers_[l].post_attention_layernorm, ws_.d_act_1, 1, stream_);
            } else {
                launch_rmsnorm_q8_1(
                    ws_.d_act_0, layers_[l].post_attention_layernorm, nullptr,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
                );
            }
            t_dec_post_norm[l].record_stop(stream_);

            t_dec_swiglu[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemv_dual_gate_up_swiglu_fp16(reinterpret_cast<const half*>(layers_[l].gate_up_weight), ws_.d_act_1, ws_.d_mlp_act, stream_);
            } else {
                launch_gemv_dual_gate_up_swiglu_w4a8(
                    reinterpret_cast<const uint32_t*>(layers_[l].gate_up_weight),
                    layers_[l].gate_up_scale,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                    ws_.d_mlp_act, stream_
                );
            }
            t_dec_swiglu[l].record_stop(stream_);

            t_dec_quant_down[l].record_start(stream_);
            if (quant_mode_ != QuantMode::FP16) {
                launch_quantize_fp16_to_q8_1(
                    ws_.d_mlp_act, ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                    INTERMEDIATE_SIZE, stream_
                );
            }
            t_dec_quant_down[l].record_stop(stream_);

            t_dec_down[l].record_start(stream_);
            if (quant_mode_ == QuantMode::FP16) {
                launch_gemv_fp16(reinterpret_cast<const half*>(layers_[l].down_weight), ws_.d_mlp_act, ws_.d_act_0, ws_.d_act_0, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_);
            } else {
                launch_gemv_w4a8(
                    reinterpret_cast<const uint32_t*>(layers_[l].down_weight),
                    layers_[l].down_scale,
                    ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                    ws_.d_act_0, ws_.d_act_0,
                    HIDDEN_SIZE, INTERMEDIATE_SIZE, stream_
                );
            }
            t_dec_down[l].record_stop(stream_);
        }

        t_dec_final_norm.record_start(stream_);
        if (quant_mode_ == QuantMode::FP16) {
            launch_rmsnorm_fp16(ws_.d_act_0, d_final_norm_, ws_.d_act_1, 1, stream_);
        } else {
            launch_rmsnorm_q8_1(
                ws_.d_act_0, d_final_norm_, nullptr,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8, stream_
            );
        }
        t_dec_final_norm.record_stop(stream_);

        t_dec_lm_head.record_start(stream_);
        if (quant_mode_ == QuantMode::FP16) {
            launch_gemv_fp16(reinterpret_cast<const half*>(d_embed_weight_), ws_.d_act_1, nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_);
        } else {
            launch_gemv_w4a8(
                reinterpret_cast<const uint32_t*>(d_embed_weight_),
                d_embed_scale_,
                ws_.d_act_q8, ws_.d_act_scale, ws_.d_act_sum8,
                nullptr, ws_.d_logits, VOCAB_SIZE, HIDDEN_SIZE, stream_
            );
        }
        t_dec_lm_head.record_stop(stream_);

        t_dec_sample.record_start(stream_);
        launch_sample_and_advance(ws_.d_logits, ws_.d_sample_val, ws_.d_sample_idx, d_control_, d_mapped_pinned_token_, stream_);
        t_dec_sample.record_stop(stream_);

        t_dec_total.record_stop(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        profile.decode.embed_ms      += t_dec_embed.elapsed_ms();
        profile.decode.final_norm_ms += t_dec_final_norm.elapsed_ms();
        profile.decode.lm_head_ms    += t_dec_lm_head.elapsed_ms();
        profile.decode.sample_ms     += t_dec_sample.elapsed_ms();
        profile.decode.total_ms      += t_dec_total.elapsed_ms();

        for (int l = 0; l < NUM_LAYERS; ++l) {
            profile.decode.rmsnorm_ms         += t_dec_in_norm[l].elapsed_ms();
            profile.decode.qkv_ms             += t_dec_qkv[l].elapsed_ms();
            profile.decode.rope_kv_ms         += t_dec_rope[l].elapsed_ms();
            profile.decode.attn_ms            += t_dec_attn[l].elapsed_ms();
            profile.decode.o_proj_ms          += t_dec_oproj[l].elapsed_ms();
            profile.decode.post_norm_ms       += t_dec_post_norm[l].elapsed_ms();
            profile.decode.gate_up_swiglu_ms  += t_dec_swiglu[l].elapsed_ms();
            profile.decode.quant_down_ms      += t_dec_quant_down[l].elapsed_ms();
            profile.decode.down_proj_ms       += t_dec_down[l].elapsed_ms();
        }
    }

    // Average across PROFILE_DECODE_STEPS
    float inv_steps = 1.0f / static_cast<float>(PROFILE_DECODE_STEPS);
    profile.decode.embed_ms          *= inv_steps;
    profile.decode.rmsnorm_ms        *= inv_steps;
    profile.decode.qkv_ms            *= inv_steps;
    profile.decode.rope_kv_ms        *= inv_steps;
    profile.decode.attn_ms           *= inv_steps;
    profile.decode.o_proj_ms         *= inv_steps;
    profile.decode.post_norm_ms      *= inv_steps;
    profile.decode.gate_up_swiglu_ms *= inv_steps;
    profile.decode.quant_down_ms     *= inv_steps;
    profile.decode.down_proj_ms      *= inv_steps;
    profile.decode.final_norm_ms     *= inv_steps;
    profile.decode.lm_head_ms        *= inv_steps;
    profile.decode.sample_ms         *= inv_steps;
    profile.decode.total_ms          *= inv_steps;

    // Decode memory bandwidth calculation
    double decode_sec = profile.decode.total_ms / 1000.0;
    double weight_bytes = static_cast<double>(weights_bytes_);
    double avg_seq_len = prompt_tokens.size() + (PROFILE_DECODE_STEPS / 2.0);
    double kv_bytes = NUM_LAYERS * 2 * avg_seq_len * HEAD_DIM * sizeof(half);
    double total_bytes = weight_bytes + kv_bytes;
    profile.decode_bw_gbs = (decode_sec > 0.0) ? static_cast<float>((total_bytes / decode_sec) / 1e9) : 0.0f;

    // Reset session before returning
    reset_session();

    return profile;
}

ArenaStats QwenEngine::get_memory_stats() const {
    CudaDeviceGuard guard(device_id_);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    ArenaStats stats;
    stats.weights_bytes         = weights_bytes_;
    stats.kv_cache_bytes        = kv_cache_bytes_;
    stats.workspace_bytes       = workspace_bytes_;
    stats.total_allocated_bytes = arena_->used_bytes();
    stats.device_free_bytes     = free_b;
    stats.device_total_bytes    = total_b;
    stats.mode                  = quant_mode_;
    return stats;
}

} // namespace qwen
