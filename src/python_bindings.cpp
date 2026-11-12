#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include "engine.cuh"
#include "tokenizer.hpp"
#include "exceptions.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace qwen {

class PyEngineStream {
public:
    PyEngineStream(
        QwenEngine* engine,
        QwenTokenizer* tokenizer,
        int prompt_tokens_len,
        int max_new_tokens = 256,
        bool rollback_on_interrupt = true,
        const std::string& initial_carry = ""
    ) : engine_(engine),
        tokenizer_(tokenizer),
        prompt_tokens_len_(prompt_tokens_len),
        max_new_tokens_(max_new_tokens),
        rollback_on_interrupt_(rollback_on_interrupt),
        generated_count_(0),
        closed_(false),
        eos_reached_(false),
        utf8_carry_buffer_(initial_carry) {
        if (!engine_) {
            throw QwenError("PyEngineStream: Null engine pointer");
        }
        if (!tokenizer_) {
            throw QwenError("PyEngineStream: Null tokenizer pointer");
        }
    }

    ~PyEngineStream() {
        if (!closed_) {
            close(rollback_on_interrupt_);
        }
    }

    std::string next() {
        if (closed_ || eos_reached_) {
            throw nb::stop_iteration();
        }

        while (generated_count_ < max_new_tokens_ && !eos_reached_) {
            int32_t token_id;
            {
                nb::gil_scoped_release release;
                token_id = engine_->decode_step();
            }
            generated_count_++;

            if (tokenizer_->is_eos(token_id)) {
                eos_reached_ = true;
                std::string piece = tokenizer_->decode_streaming(token_id, utf8_carry_buffer_);
                if (!piece.empty()) {
                    return piece;
                }
                throw nb::stop_iteration();
            }

            std::string piece = tokenizer_->decode_streaming(token_id, utf8_carry_buffer_);
            if (!piece.empty()) {
                return piece;
            }
        }

        // Reached max_new_tokens_
        if (!utf8_carry_buffer_.empty()) {
            std::string piece = std::move(utf8_carry_buffer_);
            utf8_carry_buffer_.clear();
            eos_reached_ = true;
            return piece;
        }

        eos_reached_ = true;
        throw nb::stop_iteration();
    }

    void close(bool rollback = true) {
        if (!closed_) {
            closed_ = true;
            if (rollback && engine_) {
                engine_->rollback_to(prompt_tokens_len_);
            }
            if (engine_) {
                engine_->release_session();
            }
        }
    }

    bool is_closed() const { return closed_; }
    int generated_count() const { return generated_count_; }

private:
    QwenEngine* engine_;
    QwenTokenizer* tokenizer_;
    int prompt_tokens_len_;
    int max_new_tokens_;
    bool rollback_on_interrupt_;
    int generated_count_;
    bool closed_;
    bool eos_reached_;
    std::string utf8_carry_buffer_;
};

} // namespace qwen

NB_MODULE(_core, m) {
    m.doc() = "Bare-metal zero-dependency CUDA inference engine and tokenizer for Qwen2.5";

    // Exception hierarchy
    auto qwen_err = nb::exception<qwen::QwenError>(m, "QwenError", PyExc_RuntimeError);
    nb::exception<qwen::ConcurrencyError>(m, "ConcurrencyError", qwen_err);
    nb::exception<qwen::ContextWindowExceededError>(m, "ContextWindowExceededError", qwen_err);
    nb::exception<qwen::CudaOutOfMemoryError>(m, "CudaOutOfMemoryError", qwen_err);
    nb::exception<qwen::ModelNotFoundError>(m, "ModelNotFoundError", PyExc_FileNotFoundError);

    // QuantMode enum
    nb::enum_<qwen::QuantMode>(m, "QuantMode")
        .value("FP16", qwen::QuantMode::FP16)
        .value("W4A16_G128", qwen::QuantMode::W4A16_G128);

    // ArenaStats
    nb::class_<qwen::ArenaStats>(m, "ArenaStats")
        .def_ro("weights_bytes", &qwen::ArenaStats::weights_bytes)
        .def_ro("kv_cache_bytes", &qwen::ArenaStats::kv_cache_bytes)
        .def_ro("workspace_bytes", &qwen::ArenaStats::workspace_bytes)
        .def_ro("total_allocated_bytes", &qwen::ArenaStats::total_allocated_bytes)
        .def_ro("device_free_bytes", &qwen::ArenaStats::device_free_bytes)
        .def_ro("device_total_bytes", &qwen::ArenaStats::device_total_bytes)
        .def_ro("mode", &qwen::ArenaStats::mode)
        .def("__repr__", [](const qwen::ArenaStats& s) {
            return "<ArenaStats total_allocated_mb=" + std::to_string(s.total_allocated_bytes / (1024.0 * 1024.0)) +
                   " weights_mb=" + std::to_string(s.weights_bytes / (1024.0 * 1024.0)) +
                   " kv_cache_mb=" + std::to_string(s.kv_cache_bytes / (1024.0 * 1024.0)) + ">";
        });

    // ChatMessage
    nb::class_<qwen::ChatMessage>(m, "ChatMessage")
        .def(nb::init<const std::string&, const std::string&>(), "role"_a, "content"_a)
        .def_rw("role", &qwen::ChatMessage::role)
        .def_rw("content", &qwen::ChatMessage::content);

    m.def("cuda_device_count", []() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) return 0;
        return count;
    });

    // NativeTokenizer
    auto tok_cls = nb::class_<qwen::QwenTokenizer>(m, "NativeTokenizer");
    tok_cls.def(nb::init<const std::string&>(), "tokenizer_bin_path"_a);
    tok_cls.def_static("from_bytes", [](const nb::bytes& b) {
        return std::make_unique<qwen::QwenTokenizer>(b.data(), b.size());
    }, "buffer"_a);
    tok_cls.def("encode", &qwen::QwenTokenizer::encode, "text"_a);
    tok_cls.def("decode", &qwen::QwenTokenizer::decode, "tokens"_a);
    tok_cls.def("decode_token", &qwen::QwenTokenizer::decode_token, "token_id"_a);
    tok_cls.def("decode_streaming", [](const qwen::QwenTokenizer& tok, int32_t token_id, std::string carry_buffer) {
        std::string carry = std::move(carry_buffer);
        std::string piece = tok.decode_streaming(token_id, carry);
        return std::make_pair(piece, carry);
    }, "token_id"_a, "carry_buffer"_a = "");
    tok_cls.def("format_chatml", [](const qwen::QwenTokenizer& tok, const nb::list& py_messages) {
        std::vector<qwen::ChatMessage> msgs;
        msgs.reserve(py_messages.size());
        for (size_t i = 0; i < py_messages.size(); ++i) {
            nb::handle item = py_messages[i];
            if (nb::isinstance<nb::dict>(item)) {
                nb::dict d = nb::cast<nb::dict>(item);
                std::string role = nb::cast<std::string>(d["role"]);
                std::string content = nb::cast<std::string>(d["content"]);
                msgs.push_back({role, content});
            } else if (nb::isinstance<qwen::ChatMessage>(item)) {
                msgs.push_back(nb::cast<qwen::ChatMessage>(item));
            } else if (nb::isinstance<nb::tuple>(item) || nb::isinstance<nb::list>(item)) {
                nb::sequence s = nb::cast<nb::sequence>(item);
                if (nb::len(s) == 2) {
                    std::string role = nb::cast<std::string>(s[0]);
                    std::string content = nb::cast<std::string>(s[1]);
                    msgs.push_back({role, content});
                }
            }
        }
        return tok.format_chatml(msgs);
    }, "messages"_a);
    tok_cls.def("is_eos", [](const qwen::QwenTokenizer&, int32_t token_id) {
        return qwen::QwenTokenizer::is_eos(token_id);
    }, "token_id"_a);
    tok_cls.def("vocab_size", &qwen::QwenTokenizer::vocab_size);

    // NativeEngine
    nb::class_<qwen::QwenEngine>(m, "NativeEngine")
        .def(nb::init<const std::string&, int, size_t>(), "qwenbin_path"_a, "device_id"_a = 0, "max_seq_len"_a = 4096)
        .def(nb::init<const std::string&, const std::vector<int>&, size_t>(), "model_path"_a, "device_ids"_a, "max_seq_len"_a = 4096)
        .def("prefill", [](qwen::QwenEngine& self, const std::vector<int32_t>& tokens, float temp) {
            nb::gil_scoped_release release;
            return self.prefill(tokens, temp);
        }, "tokens"_a, "temperature"_a = 0.0f)
        .def("decode_step", [](qwen::QwenEngine& self) {
            nb::gil_scoped_release release;
            return self.decode_step();
        })
        .def("rollback_to", &qwen::QwenEngine::rollback_to, "target_seq_pos"_a)
        .def("reset_session", &qwen::QwenEngine::reset_session)
        .def("get_memory_stats", &qwen::QwenEngine::get_memory_stats)
        .def("device_id", &qwen::QwenEngine::device_id)
        .def("num_gqa_layers", &qwen::QwenEngine::num_gqa_layers)
        .def("num_deltanet_layers", &qwen::QwenEngine::num_deltanet_layers)
        .def("current_seq_pos", &qwen::QwenEngine::current_seq_pos)
        .def("current_token_id", &qwen::QwenEngine::current_token_id)
        .def("try_acquire_session", &qwen::QwenEngine::try_acquire_session)
        .def("release_session", &qwen::QwenEngine::release_session)
        .def("is_generating", &qwen::QwenEngine::is_generating);

    // PyEngineStream
    nb::class_<qwen::PyEngineStream>(m, "PyEngineStream")
        .def(nb::init<qwen::QwenEngine*, qwen::QwenTokenizer*, int, int, bool, const std::string&>(),
             "engine"_a, "tokenizer"_a, "prompt_tokens_len"_a,
             "max_new_tokens"_a = 256, "rollback_on_interrupt"_a = true,
             "initial_carry"_a = "",
             nb::keep_alive<0, 1>(), nb::keep_alive<0, 2>())
        .def("__iter__", [](qwen::PyEngineStream& s) -> qwen::PyEngineStream& { return s; })
        .def("__next__", &qwen::PyEngineStream::next)
        .def("close", &qwen::PyEngineStream::close, "rollback"_a = true)
        .def_prop_ro("closed", &qwen::PyEngineStream::is_closed)
        .def_prop_ro("generated_count", &qwen::PyEngineStream::generated_count);
}
