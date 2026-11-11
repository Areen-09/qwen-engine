#pragma once

#include "config.cuh"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace qwen {

struct ChatMessage {
    std::string role;
    std::string content;
};

class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::string& tokenizer_bin_path);
    explicit QwenTokenizer(const void* buffer, size_t size);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& tokens) const;
    std::string decode_token(int32_t token_id) const;

    // Streaming decoder with UTF-8 multi-byte boundary preservation
    std::string decode_streaming(int32_t token_id, std::string& utf8_carry_buffer) const;

    // ChatML formatting: deterministic <|im_start|>role\ncontent<|im_end|>\n format
    std::vector<int32_t> format_chatml(const std::vector<ChatMessage>& messages) const;

    static bool is_eos(int32_t token_id) {
        return token_id == TOKEN_ENDOFTEXT || token_id == TOKEN_IM_END || token_id == TOKEN_ENDOFTEXT_QWEN38;
    }

    int32_t eos_id() const { return eos_id_; }
    void set_eos_id(int32_t id) { eos_id_ = id; }

    int32_t vocab_size() const { return static_cast<int32_t>(vocab_pieces_.size()); }

private:
    void load_tokenizer_bin(const std::string& path);
    void load_tokenizer_buffer(const void* buffer, size_t size);

    int32_t eos_id_{TOKEN_IM_END};

    // BPE piece table
    std::vector<std::string> vocab_pieces_;
    std::unordered_map<std::string, int32_t> piece_to_id_;

    // Merge pairs: map pair (left_id, right_id) -> merged_id
    struct PairHash {
        std::size_t operator()(const std::pair<int32_t, int32_t>& p) const {
            return (static_cast<uint64_t>(p.first) << 32) | static_cast<uint32_t>(p.second);
        }
    };
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, PairHash> merges_;
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, PairHash> merge_ranks_;
};

} // namespace qwen
