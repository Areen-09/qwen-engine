#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace qwen {

QwenTokenizer::QwenTokenizer(const std::string& tokenizer_bin_path) {
    load_tokenizer_bin(tokenizer_bin_path);
}

QwenTokenizer::QwenTokenizer(const void* buffer, size_t size) {
    load_tokenizer_buffer(buffer, size);
}

void QwenTokenizer::load_tokenizer_buffer(const void* buffer, size_t size) {
    if (!buffer || size < sizeof(TokenizerBinHeader)) {
        throw std::runtime_error("QwenTokenizer: Invalid tokenizer memory buffer");
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    const uint8_t* end = ptr + size;

    const TokenizerBinHeader* header = reinterpret_cast<const TokenizerBinHeader*>(ptr);
    if (header->magic != 0x51544F4B) {
        throw std::runtime_error("QwenTokenizer: Invalid magic number in memory buffer");
    }
    ptr += sizeof(TokenizerBinHeader);

    vocab_pieces_.resize(header->vocab_size);
    for (uint32_t i = 0; i < header->vocab_size; ++i) {
        if (ptr + sizeof(uint16_t) > end) {
            throw std::runtime_error("QwenTokenizer: Unexpected end of buffer reading piece length");
        }
        uint16_t byte_len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);

        if (ptr + byte_len > end) {
            throw std::runtime_error("QwenTokenizer: Unexpected end of buffer reading piece data");
        }
        std::string piece(reinterpret_cast<const char*>(ptr), byte_len);
        ptr += byte_len;

        if (ptr + sizeof(int32_t) > end) {
            throw std::runtime_error("QwenTokenizer: Unexpected end of buffer reading rank");
        }
        ptr += sizeof(int32_t); // rank

        vocab_pieces_[i] = piece;
        piece_to_id_[piece] = static_cast<int32_t>(i);
    }

    for (uint32_t m = 0; m < header->num_merges; ++m) {
        if (ptr + 3 * sizeof(int32_t) > end) {
            throw std::runtime_error("QwenTokenizer: Unexpected end of buffer reading merge pair");
        }
        int32_t left = *reinterpret_cast<const int32_t*>(ptr);
        int32_t right = *reinterpret_cast<const int32_t*>(ptr + sizeof(int32_t));
        int32_t merged = *reinterpret_cast<const int32_t*>(ptr + 2 * sizeof(int32_t));
        ptr += 3 * sizeof(int32_t);

        auto pair = std::make_pair(left, right);
        merges_[pair] = merged;
        merge_ranks_[pair] = static_cast<int32_t>(m);
    }
}

void QwenTokenizer::load_tokenizer_bin(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("QwenTokenizer: Failed to open tokenizer binary: " + path);
    }

    TokenizerBinHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != 0x51544F4B) {
        throw std::runtime_error("QwenTokenizer: Invalid magic number in " + path);
    }

    vocab_pieces_.resize(header.vocab_size);
    for (uint32_t i = 0; i < header.vocab_size; ++i) {
        uint16_t byte_len = 0;
        file.read(reinterpret_cast<char*>(&byte_len), sizeof(byte_len));
        std::string piece(byte_len, '\0');
        if (byte_len > 0) {
            file.read(&piece[0], byte_len);
        }
        int32_t rank = 0;
        file.read(reinterpret_cast<char*>(&rank), sizeof(rank));

        vocab_pieces_[i] = piece;
        piece_to_id_[piece] = static_cast<int32_t>(i);
    }

    for (uint32_t m = 0; m < header.num_merges; ++m) {
        int32_t left = 0, right = 0, merged = 0;
        file.read(reinterpret_cast<char*>(&left), sizeof(left));
        file.read(reinterpret_cast<char*>(&right), sizeof(right));
        file.read(reinterpret_cast<char*>(&merged), sizeof(merged));

        auto pair = std::make_pair(left, right);
        merges_[pair] = merged;
        merge_ranks_[pair] = static_cast<int32_t>(m);
    }
}

std::string QwenTokenizer::decode_token(int32_t token_id) const {
    if (token_id >= 0 && token_id < static_cast<int32_t>(vocab_pieces_.size())) {
        return vocab_pieces_[token_id];
    }
    return "";
}

std::string QwenTokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t t : tokens) {
        result += decode_token(t);
    }
    return result;
}

std::string QwenTokenizer::decode_streaming(int32_t token_id, std::string& utf8_carry_buffer) const {
    if (is_eos(token_id)) {
        std::string flushed = utf8_carry_buffer;
        utf8_carry_buffer.clear();
        return flushed;
    }

    std::string piece = decode_token(token_id);
    utf8_carry_buffer += piece;

    if (utf8_carry_buffer.empty()) {
        return "";
    }

    // Inspect trailing bytes to prevent breaking a multi-byte UTF-8 sequence
    size_t len = utf8_carry_buffer.size();
    size_t incomplete_start = len;

    // Scan backwards up to 4 bytes for a UTF-8 lead byte
    size_t lookback = (len >= 4) ? 4 : len;
    for (size_t k = 1; k <= lookback; ++k) {
        unsigned char b = static_cast<unsigned char>(utf8_carry_buffer[len - k]);
        if ((b & 0x80) == 0x00) {
            // ASCII byte
            break;
        } else if ((b & 0xC0) == 0x80) {
            // Continuation byte, keep searching for lead byte
            continue;
        } else if ((b & 0xE0) == 0xC0) {
            // 2-byte sequence lead
            if (k < 2) incomplete_start = len - k;
            break;
        } else if ((b & 0xF0) == 0xE0) {
            // 3-byte sequence lead
            if (k < 3) incomplete_start = len - k;
            break;
        } else if ((b & 0xF8) == 0xF0) {
            // 4-byte sequence lead
            if (k < 4) incomplete_start = len - k;
            break;
        }
    }

    if (incomplete_start < len) {
        std::string emit = utf8_carry_buffer.substr(0, incomplete_start);
        utf8_carry_buffer = utf8_carry_buffer.substr(incomplete_start);
        return emit;
    }

    std::string emit = utf8_carry_buffer;
    utf8_carry_buffer.clear();
    return emit;
}

std::vector<int32_t> QwenTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> result;
    if (text.empty()) return result;

    // Identify special tokens
    const std::vector<std::string> special_tokens = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>"
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t next_special_pos = std::string::npos;
        std::string matched_special;

        for (const auto& sp : special_tokens) {
            size_t p = text.find(sp, pos);
            if (p != std::string::npos && (next_special_pos == std::string::npos || p < next_special_pos)) {
                next_special_pos = p;
                matched_special = sp;
            }
        }

        // Process standard text chunk before special token
        size_t text_end = (next_special_pos != std::string::npos) ? next_special_pos : text.size();
        if (text_end > pos) {
            std::string chunk = text.substr(pos, text_end - pos);
            // Decompose into initial vocabulary tokens using greedy longest prefix match
            std::vector<int32_t> ids;
            size_t c_pos = 0;
            size_t c_len = chunk.size();
            while (c_pos < c_len) {
                bool matched = false;
                size_t max_lookahead = std::min(static_cast<size_t>(32), c_len - c_pos);
                for (size_t l = max_lookahead; l > 0; --l) {
                    std::string sub = chunk.substr(c_pos, l);
                    auto it = piece_to_id_.find(sub);
                    if (it != piece_to_id_.end()) {
                        ids.push_back(it->second);
                        c_pos += l;
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    c_pos += 1;
                }
            }

            // Apply BPE merges
            while (ids.size() >= 2) {
                int best_rank = 1000000000;
                size_t best_idx = 0;
                int32_t best_merged_id = -1;

                for (size_t i = 0; i + 1 < ids.size(); ++i) {
                    auto pair = std::make_pair(ids[i], ids[i + 1]);
                    auto it = merge_ranks_.find(pair);
                    if (it != merge_ranks_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                        best_merged_id = merges_.at(pair);
                    }
                }

                if (best_merged_id == -1) break;

                // Merge the winning pair
                ids[best_idx] = best_merged_id;
                ids.erase(ids.begin() + best_idx + 1);
            }

            result.insert(result.end(), ids.begin(), ids.end());
        }

        if (next_special_pos != std::string::npos) {
            auto it = piece_to_id_.find(matched_special);
            if (it != piece_to_id_.end()) {
                result.push_back(it->second);
            }
            pos = next_special_pos + matched_special.size();
        } else {
            pos = text.size();
        }
    }

    return result;
}

std::vector<int32_t> QwenTokenizer::format_chatml(const std::vector<ChatMessage>& messages) const {
    std::string prompt;
    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    // Append assistant generation prompt
    prompt += "<|im_start|>assistant\n";
    return encode(prompt);
}

} // namespace qwen
