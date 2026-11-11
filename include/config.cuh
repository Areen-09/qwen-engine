#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace qwen {

// Architectural compile-time dimensions for Qwen2.5-1.5B (backward compatibility defaults)
constexpr int32_t NUM_LAYERS          = 28;
constexpr int32_t HIDDEN_SIZE         = 1536;
constexpr int32_t INTERMEDIATE_SIZE   = 8960;
constexpr int32_t NUM_Q_HEADS         = 12;
constexpr int32_t NUM_KV_HEADS        = 2;
constexpr int32_t GQA_GROUP_SIZE      = 6;  // NUM_Q_HEADS / NUM_KV_HEADS
constexpr int32_t HEAD_DIM            = 128;
constexpr int32_t Q_DIM               = 1536; // NUM_Q_HEADS * HEAD_DIM
constexpr int32_t KV_DIM              = 256;  // NUM_KV_HEADS * HEAD_DIM
constexpr int32_t VOCAB_SIZE          = 151936;
constexpr float   RMS_NORM_EPS        = 1e-6f;
constexpr float   ROPE_THETA          = 1000000.0f;
constexpr int32_t MAX_SEQ_LEN         = 4096;

// Hardware & Quantization parameters
constexpr int32_t QUANT_GROUP_SIZE    = 128;
constexpr int32_t PREFILL_MAX_CHUNK   = 256;
constexpr int32_t PREFILL_TILE_M      = 16;
constexpr int32_t PREFILL_TILE_K      = 128;
constexpr int32_t FLASH_DECODE_SPLITS = 16;

// Tensor Core Prefill & Multi-Row Decode Tiling Constants
constexpr int32_t PREFILL_WMMA_BM     = 64;
constexpr int32_t PREFILL_WMMA_BN     = 64;
constexpr int32_t PREFILL_WMMA_BK     = 32;
constexpr int32_t PREFILL_SMEM_STRIDE = 40; // BK=32 + 8 half skew
constexpr int32_t DECODE_ROWS_PER_WARP = 4;
constexpr float   LOG2_ROPE_THETA     = 19.931568569324174f; // log2(1000000.0f)

// Special token IDs
constexpr int32_t TOKEN_PAD           = 151643;
constexpr int32_t TOKEN_ENDOFTEXT     = 151643;
constexpr int32_t TOKEN_IM_START      = 151644;
constexpr int32_t TOKEN_IM_END        = 151645;
constexpr int32_t TOKEN_ENDOFTEXT_QWEN38 = 248044;
constexpr int32_t TOKEN_BOS_QWEN38       = 248044;

// Architectural compile-time dimensions for Qwen3.8-27B
constexpr int32_t QWEN38_NUM_LAYERS        = 64;
constexpr int32_t QWEN38_HIDDEN_SIZE       = 5120;
constexpr int32_t QWEN38_INTERMEDIATE_SIZE = 17408;
constexpr int32_t QWEN38_NUM_Q_HEADS       = 32;
constexpr int32_t QWEN38_NUM_KV_HEADS      = 4;
constexpr int32_t QWEN38_HEAD_DIM          = 256;
constexpr int32_t QWEN38_ROTARY_DIM        = 64;
constexpr int32_t QWEN38_VOCAB_SIZE        = 248320;

enum class QuantMode : uint32_t {
    FP16 = 0,
    W4A16_G128 = 1
};

// Layer weight tensor indices in layer_offsets[28][17] for v1 legacy containers
namespace LayerTensorIdx {
    constexpr int INPUT_LAYERNORM          = 0;
    constexpr int Q_WEIGHT                 = 1;
    constexpr int Q_SCALE                  = 2;
    constexpr int Q_BIAS                   = 3;
    constexpr int K_WEIGHT                 = 4;
    constexpr int K_SCALE                  = 5;
    constexpr int K_BIAS                   = 6;
    constexpr int V_WEIGHT                 = 7;
    constexpr int V_SCALE                  = 8;
    constexpr int V_BIAS                   = 9;
    constexpr int O_WEIGHT                 = 10;
    constexpr int O_SCALE                  = 11;
    constexpr int POST_ATTENTION_LAYERNORM = 12;
    constexpr int GATE_UP_WEIGHT           = 13;
    constexpr int GATE_UP_SCALE            = 14;
    constexpr int DOWN_WEIGHT              = 15;
    constexpr int DOWN_SCALE               = 16;
    constexpr int COUNT                    = 17;
}

// 4096-byte page-aligned header for legacy .qwenbin v1 weight format
struct alignas(4096) QWQwenHeader {
    uint32_t magic;               // 0x5157454E ("QWEN")
    uint32_t version;             // 1
    uint32_t quant_mode;          // 0 = FP16, 1 = W4A16_G128
    uint32_t vocab_size;          // 151936
    uint32_t hidden_size;         // 1536
    uint32_t intermediate_size;   // 8960
    uint32_t num_layers;          // 28
    uint32_t num_q_heads;         // 12
    uint32_t num_kv_heads;        // 2
    uint32_t head_dim;            // 128
    uint32_t quant_group_size;    // 128
    uint32_t reserved0;           // 0

    uint64_t embed_weight_offset;
    uint64_t embed_scale_offset;  // 0 if FP16
    uint64_t final_norm_offset;

    uint64_t layer_offsets[NUM_LAYERS][LayerTensorIdx::COUNT];

    uint8_t padding[216];         // Pad to exactly 4096 bytes
};
static_assert(sizeof(QWQwenHeader) == 4096, "QWQwenHeader must be exactly 4096 bytes");

// ----------------------------------------------------------------------------
// .qwenbin v2 Self-Describing Container Specification
// ----------------------------------------------------------------------------

enum LayerType : uint32_t {
    LAYER_GQA = 0,       // Standard Grouped Query Attention
    LAYER_DELTANET = 1   // Gated DeltaNet Linear Attention
};

struct alignas(64) QwenBinHeaderV2 {
    uint32_t magic;                 // 0x5157454E ("QWEN")
    uint32_t version;               // 2
    uint32_t hidden_size;           // Hidden dimension (e.g. 1536, 2560, 5120)
    uint32_t intermediate_size;     // MLP intermediate dimension
    uint32_t num_layers;            // Total layer count
    uint32_t num_attention_heads;   // Attention query heads
    uint32_t num_kv_heads;          // Key/Value heads
    uint32_t head_dim;              // Head dimension (typically 128)
    uint32_t vocab_size;            // Vocabulary size (e.g. 151936)
    uint32_t max_seq_len;           // Context window ceiling
    uint32_t quant_type;            // 0 = FP16, 1 = W4A16_G128
    uint32_t tie_word_embeddings;   // 1 if lm_head shares embed_tokens, else 0
    float    rms_norm_eps;          // Normalization epsilon (e.g. 1e-6)
    float    rope_theta;            // RoPE base frequency (e.g. 1000000.0)
    uint32_t reserved_flags;        // Alignment padding
    uint32_t padding0;              // Explicit 4-byte padding ensuring 8-byte alignment

    // 64-bit File Offsets
    uint64_t layer_table_offset;    // Byte offset to LayerDescriptor array
    uint64_t embed_tokens_offset;   // Byte offset to input embedding table
    uint64_t final_norm_offset;     // Byte offset to final RMSNorm weights
    uint64_t lm_head_offset;        // Byte offset to output unembedding weights
    uint64_t weights_offset;        // Byte offset to aligned layer weight payload
    uint64_t reserved[6];           // reserved[0] = embed_scale_offset, reserved[1..5] = headroom
};

struct alignas(8) LayerDescriptor {
    uint32_t type;                  // Execution tag (LAYER_GQA or LAYER_DELTANET)
    uint32_t reserved;              // Struct padding ensuring 8-byte alignment

    // Common Normalization Offsets
    uint64_t norm1_weight_offset;   // Pre-attention / Pre-DeltaNet RMSNorm
    uint64_t norm2_weight_offset;   // Pre-MLP RMSNorm

    // GQA Specific Offsets (Valid when type == LAYER_GQA)
    uint64_t qkv_weight_offset;     // Fused or Q projection weights
    uint64_t q_bias_offset;         // Q attention bias
    uint64_t k_bias_offset;         // K attention bias
    uint64_t v_bias_offset;         // V attention bias
    uint64_t attn_out_weight_offset;// Attention output projection

    // DeltaNet Specific Offsets (Valid when type == LAYER_DELTANET)
    uint64_t conv1d_weight_offset;  // Depthwise 1D convolution weights
    uint64_t conv1d_bias_offset;    // Depthwise 1D convolution bias
    uint64_t in_proj_weight_offset; // Input projection (Q, K, V, Z gates)
    uint64_t decay_weight_offset;   // Decay projection weights (beta gate)
    uint64_t delta_out_weight_offset;// DeltaNet output projection

    // MLP Offsets
    uint64_t mlp_gate_offset;       // SwiGLU gate projection
    uint64_t mlp_up_offset;         // SwiGLU up projection
    uint64_t mlp_down_offset;       // SwiGLU down projection

    // Quantization Offsets (Valid when quant_type == W4A16_G128)
    uint64_t quant_scales_offset;   // FP16 scaling factors
    uint64_t quant_zeros_offset;    // Zero points (or 0 for symmetric quantization)

    // Sub-tensor and per-projection offsets
    uint64_t k_weight_offset;
    uint64_t v_weight_offset;
    uint64_t q_scale_offset;
    uint64_t k_scale_offset;
    uint64_t v_scale_offset;
    uint64_t attn_out_scale_offset;
    uint64_t mlp_gate_scale_offset;
    uint64_t mlp_up_scale_offset;
    uint64_t mlp_down_scale_offset;
};
static_assert(sizeof(LayerDescriptor) == 216, "LayerDescriptor must be exactly 216 bytes");

class ModelConfig {
public:
    ModelConfig() : header_{}, layers_{} {}

    explicit ModelConfig(const QwenBinHeaderV2& header, const LayerDescriptor* layers)
        : header_(header), layers_(layers, layers + header.num_layers) {}

    // Model Dimensions
    int32_t hidden_size() const { return static_cast<int32_t>(header_.hidden_size); }
    int32_t intermediate_size() const { return static_cast<int32_t>(header_.intermediate_size); }
    int32_t num_layers() const { return static_cast<int32_t>(header_.num_layers); }
    int32_t num_attention_heads() const { return static_cast<int32_t>(header_.num_attention_heads); }
    int32_t num_kv_heads() const { return static_cast<int32_t>(header_.num_kv_heads); }
    int32_t head_dim() const { return static_cast<int32_t>(header_.head_dim); }
    int32_t vocab_size() const { return static_cast<int32_t>(header_.vocab_size); }
    int32_t max_seq_len() const { return static_cast<int32_t>(header_.max_seq_len); }
    uint32_t quant_type() const { return header_.quant_type; }
    bool tie_word_embeddings() const { return header_.tie_word_embeddings != 0; }
    float rms_norm_eps() const { return header_.rms_norm_eps; }
    float rope_theta() const { return header_.rope_theta; }

    // Layer Descriptors & Offsets
    const LayerDescriptor& layer(size_t idx) const { return layers_.at(idx); }
    const QwenBinHeaderV2& header() const { return header_; }
    const std::vector<LayerDescriptor>& layers() const { return layers_; }

private:
    QwenBinHeaderV2 header_;
    std::vector<LayerDescriptor> layers_;
};

// Binary tokenizer header for tokenizer.bin
struct TokenizerBinHeader {
    uint32_t magic;       // 0x51544F4B ("QTOK")
    uint32_t vocab_size;  // 151646
    uint32_t num_merges;
    uint32_t eos_id;      // 151645
};

// Golden reference test dump header for golden_ref.bin
struct alignas(256) GoldenRefHeader {
    uint32_t magic;               // 0x474F4C44 ("GOLD")
    uint32_t prompt_len;          // 8
    int32_t  prompt_tokens[8];
    int32_t  greedy_tokens[32];
    uint64_t tensor_offsets[13];
    uint8_t  padding[240];        // Pad to 512 bytes (256-aligned)
};
static_assert(sizeof(GoldenRefHeader) == 512, "GoldenRefHeader must be 512 bytes");

} // namespace qwen
