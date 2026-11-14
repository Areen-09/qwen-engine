#pragma once

#include "config.cuh"
#include "tensor.cuh"
#include "file_map.hpp"
#include "cuda_guard.cuh"
#include "exceptions.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <memory>
#include <atomic>

namespace qwen {

struct LayerWeights {
    // Normalization
    const half*     input_layernorm{nullptr};
    const half*     post_attention_layernorm{nullptr};

    // GQA Projections (W4A16 G128)
    const void*     q_weight{nullptr};
    const half*     q_scale{nullptr};
    const half*     q_bias{nullptr};
    const void*     k_weight{nullptr};
    const half*     k_scale{nullptr};
    const half*     k_bias{nullptr};
    const void*     v_weight{nullptr};
    const half*     v_scale{nullptr};
    const half*     v_bias{nullptr};
    const void*     o_weight{nullptr};
    const half*     o_scale{nullptr};

    // MLP SwiGLU Projections (W4A16 G128)
    const void*     gate_up_weight{nullptr};
    const half*     gate_up_scale{nullptr};
    const void*     down_weight{nullptr};
    const half*     down_scale{nullptr};
};

struct WorkspaceBuffers {
    half*     d_act_0{nullptr};          // Dynamic hidden activations
    half*     d_act_1{nullptr};          // Dynamic hidden activations
    half*     d_q{nullptr};              // Query projection
    half*     d_k{nullptr};              // Key projection
    half*     d_v{nullptr};              // Value projection
    half*     d_attn_out{nullptr};       // Attention output
    half*     d_mlp_act{nullptr};        // MLP intermediate activations
    float*    d_splitk_out{nullptr};     // GQA Split-K partial outputs
    float*    d_splitk_lse{nullptr};     // GQA Split-K max/lse
    half*     d_logits{nullptr};         // Vocab projection logits
    float*    d_sample_val{nullptr};     // Sampler values
    int32_t*  d_sample_idx{nullptr};     // Sampler indices
    int32_t*  d_prompt_tokens{nullptr};  // Prompt tokens buffer

    // Dedicated 16 KB aligned Q8_1 scratch buffers (separate from d_act_1)
    int8_t*   d_act_q8{nullptr};
    float*    d_act_scale{nullptr};
    int32_t*  d_act_sum8{nullptr};
};

struct StageBreakdown {
    float embed_ms;
    float rmsnorm_ms;
    float qkv_ms;
    float rope_kv_ms;
    float attn_ms;
    float o_proj_ms;
    float post_norm_ms;
    float gate_up_swiglu_ms;
    float quant_down_ms;
    float down_proj_ms;
    float final_norm_ms;
    float lm_head_ms;
    float sample_ms;
    float total_ms;
};

struct StageProfile {
    StageBreakdown prefill;
    StageBreakdown decode;
    float prefill_tflops;
    float decode_bw_gbs;
};

struct ArenaStats {
    size_t    weights_bytes;
    size_t    kv_cache_bytes;
    size_t    workspace_bytes;
    size_t    total_allocated_bytes;
    size_t    device_free_bytes;
    size_t    device_total_bytes;
    QuantMode mode;
};

struct EngineConfig {
    std::string model_path;
    std::string tokenizer_path;
    int device_id{0};
    size_t max_seq_len{4096};
};

class QwenEngine {
public:
    explicit QwenEngine(const EngineConfig& config);
    explicit QwenEngine(const std::string& qwenbin_path, int device_id = 0, size_t max_seq_len = 4096);
    ~QwenEngine();

    QwenEngine(const QwenEngine&) = delete;
    QwenEngine& operator=(const QwenEngine&) = delete;

    void reset_session();

    // Sequence rollback to target_seq_pos: resets seq_pos_, updates d_control_->seq_pos and token_id
    void rollback_to(int32_t target_seq_pos);

    // Session lease concurrency control
    bool try_acquire_session();
    void release_session();
    bool is_generating() const { return is_generating_.load(); }

    int device_id() const { return device_id_; }

    // Multi-token prefill with prefix caching: returns the first newly generated token
    int32_t prefill(const std::vector<int32_t>& prompt_tokens, float temperature = 0.0f);

    // Single-step autoregressive decode via zero-update captured CUDA Graph
    int32_t decode_step();

    int32_t current_seq_pos() const { return seq_pos_; }
    int32_t current_seq_len() const { return seq_pos_; }
    int32_t current_token_id() const { return last_token_id_; }
    const std::vector<int32_t>& cached_tokens() const { return cached_tokens_; }

    // Profile breakdown across prefill and ungraphed decode stages
    StageProfile profile_stages(const std::vector<int32_t>& prompt_tokens);

    ArenaStats get_memory_stats() const;

    // Public execution primitives
    void execute_decode_pass();
    void execute_prefill_chunk(int chunk_start, int chunk_len);

    void execute_gqa_decode_layer(int l);
    void execute_gqa_prefill_layer(int l, int chunk_start, int chunk_len);

    QuantMode quant_mode() const { return quant_mode_; }
    cudaStream_t stream() const { return stream_; }
    const ModelConfig& config() const { return model_config_; }
    bool weights_16byte_aligned() const { return weights_16byte_aligned_; }

    // Accessors for unit testing
    const LayerWeights& layer_weights(int l) const { return layers_[l]; }
    const void* embed_weight() const { return d_embed_weight_; }
    const half* embed_scale() const { return d_embed_scale_; }
    const void* lm_head_weight() const { return d_lm_head_weight_; }
    const half* lm_head_scale() const { return d_lm_head_scale_; }
    const half* final_norm() const { return d_final_norm_; }
    const WorkspaceBuffers& workspace() const { return ws_; }
    DeviceControlBlock* control_block() const { return d_control_; }
    int32_t* pinned_token_ptr() const { return h_pinned_token_; }
    half* kv_cache_k() const { return d_k_cache_; }
    half* kv_cache_v() const { return d_v_cache_; }
    size_t num_gqa_layers() const { return NUM_LAYERS; }

private:
    void load_model_file(const std::string& qwenbin_path);
    void setup_arena();
    void capture_decode_graph();

    std::string qwenbin_path_;
    int device_id_{0};
    size_t max_seq_len_{4096};
    QuantMode quant_mode_{QuantMode::W4A16_G128};
    bool weights_16byte_aligned_{true};
    std::atomic<bool> is_generating_{false};

    ModelConfig model_config_;

    // Cross-platform zero-copy file mapping
    std::unique_ptr<PlatformFileMap> file_map_;
    const uint8_t* mapped_data_{nullptr};
    size_t mapped_size_{0};

    // CUDA Arena & allocations
    std::unique_ptr<GpuArena> arena_;
    cudaStream_t stream_{nullptr};

    // Model weight device pointers
    const void*  d_embed_weight_{nullptr};
    const half*  d_embed_scale_{nullptr};
    const void*  d_lm_head_weight_{nullptr};
    const half*  d_lm_head_scale_{nullptr};
    const half*  d_final_norm_{nullptr};
    std::vector<LayerWeights> layers_;

    // KV Cache: shape [NUM_LAYERS, NUM_KV_HEADS, max_seq_len, HEAD_DIM]
    half* d_k_cache_{nullptr};
    half* d_v_cache_{nullptr};

    // Workspace & Control
    WorkspaceBuffers ws_;
    DeviceControlBlock* d_control_{nullptr};
    int32_t* h_pinned_token_{nullptr};
    int32_t* d_mapped_pinned_token_{nullptr};

    // Captured zero-update CUDA Graph
    cudaGraph_t     decode_graph_{nullptr};
    cudaGraphExec_t decode_graph_exec_{nullptr};
    bool            graph_captured_{false};

    // Session state
    std::vector<int32_t> cached_tokens_;
    int32_t seq_pos_{0};
    int32_t last_token_id_{0};
    float   temperature_{0.0f};

    // Memory tracking stats
    size_t weights_bytes_{0};
    size_t kv_cache_bytes_{0};
    size_t workspace_bytes_{0};
};

} // namespace qwen
