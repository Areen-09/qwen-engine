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
    const half*     input_layernorm;
    const void*     q_weight;
    const half*     q_scale;
    const half*     q_bias;
    const void*     k_weight;
    const half*     k_scale;
    const half*     k_bias;
    const void*     v_weight;
    const half*     v_scale;
    const half*     v_bias;
    const void*     o_weight;
    const half*     o_scale;
    const half*     post_attention_layernorm;
    const void*     gate_up_weight;
    const half*     gate_up_scale;
    const void*     down_weight;
    const half*     down_scale;
};

struct WorkspaceBuffers {
    half*     d_act_0;          // [256, 1536] = 786,432 B
    half*     d_act_1;          // [256, 1536] = 786,432 B
    half*     d_q;              // [256, 1536] = 786,432 B
    half*     d_k;              // [256, 256]  = 131,072 B
    half*     d_v;              // [256, 256]  = 131,072 B
    half*     d_attn_out;       // [256, 1536] = 786,432 B
    half*     d_mlp_act;        // [256, 8960] = 4,587,520 B
    float*    d_splitk_out;     // [12, 16, 128] = 98,304 B
    float*    d_splitk_lse;     // [12, 16, 2] = 1,536 B
    half*     d_logits;         // [151936]    = 303,872 B
    float*    d_sample_val;     // [256]       = 1,024 B
    int32_t*  d_sample_idx;     // [256]       = 1,024 B
    int32_t*  d_prompt_tokens;  // [4096]      = 16,384 B

    // Carved Q8_1 scratch pointers (zero extra arena allocation, inside d_act_1 slack)
    int8_t*   d_act_q8;         // [8960] = 8,960 B (aligned to 9,216 B)
    float*    d_act_scale;      // [70]   = 280 B   (aligned to 512 B)
    int32_t*  d_act_sum8;       // [1120] = 4,480 B (aligned to 4,608 B)
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

class QwenEngine {
public:
    explicit QwenEngine(const std::string& qwenbin_path, int device_id = 0);
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
    int32_t current_token_id() const { return last_token_id_; }
    const std::vector<int32_t>& cached_tokens() const { return cached_tokens_; }

    // Profile breakdown across prefill and decode stages
    StageProfile profile_stages(const std::vector<int32_t>& prompt_tokens);

    ArenaStats get_memory_stats() const;

    // Public execution primitives for uncaptured decode forward pass (and graph capture)
    void execute_decode_pass();
    void execute_prefill_chunk(int chunk_start, int chunk_len);

    QuantMode quant_mode() const { return quant_mode_; }
    cudaStream_t stream() const { return stream_; }

    // Accessors for unit testing
    const LayerWeights& layer_weights(int l) const { return layers_[l]; }
    const void* embed_weight() const { return d_embed_weight_; }
    const half* embed_scale() const { return d_embed_scale_; }
    const half* final_norm() const { return d_final_norm_; }
    const WorkspaceBuffers& workspace() const { return ws_; }
    DeviceControlBlock* control_block() const { return d_control_; }
    int32_t* pinned_token_ptr() const { return h_pinned_token_; }
    half* kv_cache_k() const { return d_k_cache_; }
    half* kv_cache_v() const { return d_v_cache_; }

private:
    void load_model_file(const std::string& qwenbin_path);
    void setup_arena();
    void capture_decode_graph();

    std::string qwenbin_path_;
    int device_id_;
    QuantMode quant_mode_;
    std::atomic<bool> is_generating_{false};

    // Cross-platform zero-copy file mapping
    std::unique_ptr<PlatformFileMap> file_map_;
    const uint8_t* mapped_data_;
    size_t mapped_size_;

    // CUDA Arena & allocations
    std::unique_ptr<GpuArena> arena_;
    cudaStream_t stream_;

    // Model weight device pointers
    const void*  d_embed_weight_;
    const half*  d_embed_scale_;
    const half*  d_final_norm_;
    LayerWeights layers_[NUM_LAYERS];

    // KV Cache device pointers: shape [28, 2, 4096, 128]
    half* d_k_cache_;
    half* d_v_cache_;

    // Workspace & Control
    WorkspaceBuffers ws_;
    DeviceControlBlock* d_control_;
    int32_t* h_pinned_token_; // Host mapped pinned pointer
    int32_t* d_mapped_pinned_token_; // Device pointer to pinned token

    // Captured zero-update CUDA Graph
    cudaGraph_t     decode_graph_;
    cudaGraphExec_t decode_graph_exec_;
    bool            graph_captured_;

    // Session state
    std::vector<int32_t> cached_tokens_;
    int32_t seq_pos_;
    int32_t last_token_id_;
    float   temperature_;

    // Memory tracking stats
    size_t weights_bytes_;
    size_t kv_cache_bytes_;
    size_t workspace_bytes_;
};

} // namespace qwen

