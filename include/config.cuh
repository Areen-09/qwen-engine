#pragma once

#include <cstdint>
#include <cstddef>

namespace qwen {

// Architectural compile-time dimensions for Qwen2.5-1.5B
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

enum class QuantMode : uint32_t {
    FP16 = 0,
    W4A16_G128 = 1
};

// Layer weight tensor indices in layer_offsets[28][17]
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

// 4096-byte page-aligned header for .qwenbin weight format
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

