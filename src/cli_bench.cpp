#include "config.cuh"
#include "engine.cuh"
#include "tokenizer.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace qwen;

int main(int argc, char** argv) {
    std::string model_path;
    std::string tokenizer_path = "models/tokenizer.bin";
    int prompt_len = 128;
    int gen_len = 256;
    int warmup_steps = 10;
    int runs = 50;
    bool profile_stages_flag = false;
    std::string custom_prompt = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--prompt-len" && i + 1 < argc) prompt_len = std::stoi(argv[++i]);
        else if (arg == "--gen-len" && i + 1 < argc) gen_len = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup_steps = std::stoi(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
        else if (arg == "--prompt" && i + 1 < argc) custom_prompt = argv[++i];
        else if (arg == "--profile-stages") profile_stages_flag = true;
    }

    if (model_path.empty()) {
        std::cerr << "Usage: cli_bench.exe --model <path_to_qwenbin> [--tokenizer <path>] [--prompt-len <N>] [--gen-len <M>] [--runs <R>]\n";
        return 1;
    }

    std::cout << "====================================================================" << std::endl;
    std::cout << "  Qwen2.5-1.5B Custom CUDA Engine Deterministic Benchmark (sm_86)   " << std::endl;
    std::cout << "====================================================================" << std::endl;
    std::cout << "Model:         " << model_path << std::endl;
    std::cout << "Prompt Length: " << prompt_len << " tokens" << std::endl;
    std::cout << "Gen Length:    " << gen_len << " tokens" << std::endl;
    std::cout << "Warmup Steps:  " << warmup_steps << " steps" << std::endl;
    std::cout << "Timed Runs:    " << runs << " runs" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    QwenEngine engine(model_path);
    ArenaStats mem_stats = engine.get_memory_stats();

    // Prepare prompt tokens
    std::vector<int32_t> prompt_tokens;
    if (!custom_prompt.empty()) {
        QwenTokenizer tokenizer(tokenizer_path);
        prompt_tokens = tokenizer.encode(custom_prompt);
        std::cout << "\n[Prompt]: " << custom_prompt << "\n[Tokens]: " << prompt_tokens.size() << std::endl;
        std::cout << "[Generated Output]:\n";
        int32_t tok = engine.prefill(prompt_tokens, 0.0f);
        std::cout << tokenizer.decode({tok}) << std::flush;
        for (int step = 0; step < gen_len; ++step) {
            tok = engine.decode_step();
            if (tok == 151643 || tok == 151645) break;
            std::cout << tokenizer.decode({tok}) << std::flush;
        }
        std::cout << "\n" << std::endl;
        return 0;
    } else {
        // Synthetic deterministic prompt
        prompt_tokens.resize(prompt_len);
        for (int i = 0; i < prompt_len; ++i) {
            prompt_tokens[i] = (i % 1000) + 100;
        }
    }

    cudaStream_t stream = engine.stream();
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // Warmup phase
    std::cout << "[Benchmark] Warming up (" << warmup_steps << " steps)..." << std::endl;
    engine.reset_session();
    engine.prefill(prompt_tokens, 0.0f);
    for (int w = 0; w < warmup_steps; ++w) {
        engine.decode_step();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed Benchmark Runs
    std::cout << "[Benchmark] Running " << runs << " timed iterations..." << std::endl;
    std::vector<float> ttft_list;
    std::vector<float> token_latencies;
    token_latencies.reserve(runs * gen_len);

    for (int r = 0; r < runs; ++r) {
        engine.reset_session();

        // Measure Prefill TTFT
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        engine.prefill(prompt_tokens, 0.0f);
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float ttft_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ttft_ms, start_event, stop_event));
        ttft_list.push_back(ttft_ms);

        // Measure Autoregressive Token Generation Latencies
        for (int step = 0; step < gen_len; ++step) {
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            engine.decode_step();
            CUDA_CHECK(cudaEventRecord(stop_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event));

            float step_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&step_ms, start_event, stop_event));
            token_latencies.push_back(step_ms);
        }
    }

    // Compute statistics
    std::sort(ttft_list.begin(), ttft_list.end());
    std::sort(token_latencies.begin(), token_latencies.end());

    float median_ttft = ttft_list[ttft_list.size() / 2];
    float p50_itl = token_latencies[static_cast<size_t>(token_latencies.size() * 0.50)];
    float p95_itl = token_latencies[static_cast<size_t>(token_latencies.size() * 0.95)];
    float p99_itl = token_latencies[static_cast<size_t>(token_latencies.size() * 0.99)];

    double total_decode_time_ms = std::accumulate(token_latencies.begin(), token_latencies.end(), 0.0);
    double decode_tps = (static_cast<double>(token_latencies.size()) / total_decode_time_ms) * 1000.0;

    // Calculate Effective Memory Bandwidth
    // Weight bytes read per token + average KV cache bytes read per token
    double weight_bytes = static_cast<double>(mem_stats.weights_bytes);
    // Average sequence length during generation
    double avg_seq_len = prompt_len + (gen_len / 2.0);
    double kv_bytes_per_token = NUM_LAYERS * 2 * avg_seq_len * HEAD_DIM * sizeof(half);
    double total_bytes_per_token = weight_bytes + kv_bytes_per_token;
    double median_itl_sec = static_cast<double>(p50_itl) / 1000.0;
    double effective_bw_gbs = (total_bytes_per_token / median_itl_sec) / (1e9);
    constexpr double THEORETICAL_PEAK_BW = 192.0; // RTX 3050 Laptop 128-bit GDDR6
    double bw_utilization_pct = (effective_bw_gbs / THEORETICAL_PEAK_BW) * 100.0;

    double prefill_sec = median_ttft / 1000.0;
    double prefill_tps = (prefill_sec > 0.0) ? (static_cast<double>(prompt_len) / prefill_sec) : 0.0;
    double linear_flops = 2.0 * static_cast<double>(prompt_len) * (28.0 * 46792704.0) + 2.0 * 233373696.0;
    double prefill_tflops = (prefill_sec > 0.0) ? ((linear_flops / prefill_sec) / 1e12) : 0.0;

    std::cout << "\n====================================================================" << std::endl;
    std::cout << "                         BENCHMARK RESULTS                          " << std::endl;
    std::cout << "====================================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Model Mode:               " << (mem_stats.mode == QuantMode::FP16 ? "FP16 Baseline" : "W4A16 Group-128") << std::endl;
    std::cout << "Total Discrete VRAM Used: " << (mem_stats.total_allocated_bytes / (1024.0 * 1024.0)) << " MB / "
              << (mem_stats.device_total_bytes / (1024.0 * 1024.0)) << " MB ("
              << ((double)mem_stats.total_allocated_bytes / mem_stats.device_total_bytes * 100.0) << "%)" << std::endl;
    std::cout << "  * Weights Region:       " << (mem_stats.weights_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  * Static KV Cache:      " << (mem_stats.kv_cache_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  * Workspace Region:     " << (mem_stats.workspace_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "Time-To-First-Token (TTFT): " << median_ttft << " ms (Prompt: " << prompt_len << " tokens)" << std::endl;
    std::cout << "Prefill Throughput:         " << prefill_tps << " tokens/sec" << std::endl;
    std::cout << "Prefill Compute:            " << prefill_tflops << " TFLOPS" << std::endl;
    std::cout << "Decode Throughput:          " << decode_tps << " tokens/sec" << std::endl;
    std::cout << "Inter-Token Latency (ITL):" << std::endl;
    std::cout << "  * p50 (Median):           " << p50_itl << " ms" << std::endl;
    std::cout << "  * p95:                    " << p95_itl << " ms" << std::endl;
    std::cout << "  * p99:                    " << p99_itl << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "Effective Memory Bandwidth: " << effective_bw_gbs << " GB/s ("
              << bw_utilization_pct << "% of 192 GB/s peak)" << std::endl;
    std::cout << "====================================================================" << std::endl;

    if (profile_stages_flag) {
        std::cout << "\n[Profile] Profiling per-stage kernel breakdowns..." << std::endl;
        StageProfile profile = engine.profile_stages(prompt_tokens);

        auto print_breakdown = [](const char* name, const StageBreakdown& b, float tflops_or_bw, bool is_prefill) {
            std::cout << "\n--------------------------------------------------------------------" << std::endl;
            std::cout << "  " << name << " Stage Breakdown (Total: " << b.total_ms << " ms)" << std::endl;
            std::cout << "--------------------------------------------------------------------" << std::endl;
            std::cout << std::left << std::setw(28) << "  Operator"
                      << std::right << std::setw(12) << "Latency (ms)"
                      << std::setw(14) << "Portion (%)" << std::endl;
            std::cout << "  ------------------------------------------------------------------" << std::endl;

            auto print_row = [&](const char* op, float ms) {
                float pct = (b.total_ms > 0.0f) ? (ms / b.total_ms * 100.0f) : 0.0f;
                std::cout << "  " << std::left << std::setw(26) << op
                          << std::right << std::setw(12) << std::fixed << std::setprecision(3) << ms
                          << std::setw(13) << std::fixed << std::setprecision(1) << pct << "%" << std::endl;
            };

            print_row("Embedding Lookup", b.embed_ms);
            print_row("Input RMSNorm (28x)", b.rmsnorm_ms);
            print_row("QKV Projections (28x)", b.qkv_ms);
            print_row("RoPE + KV Append (28x)", b.rope_kv_ms);
            print_row("GQA Attention (28x)", b.attn_ms);
            print_row("O-Projection (28x)", b.o_proj_ms);
            print_row("Post-Attn RMSNorm (28x)", b.post_norm_ms);
            print_row("Gate+Up SwiGLU (28x)", b.gate_up_swiglu_ms);
            if (!is_prefill) {
                print_row("Quantize Down-Act (28x)", b.quant_down_ms);
            }
            print_row("Down-Projection (28x)", b.down_proj_ms);
            print_row("Final RMSNorm", b.final_norm_ms);
            print_row("LM Head Projection", b.lm_head_ms);
            print_row("Sample & Advance", b.sample_ms);
            std::cout << "  ------------------------------------------------------------------" << std::endl;
            if (is_prefill) {
                std::cout << "  Achieved Prefill TFLOPS:    " << std::fixed << std::setprecision(2) << tflops_or_bw << " TFLOPS" << std::endl;
            } else {
                std::cout << "  Achieved Memory Bandwidth: " << std::fixed << std::setprecision(2) << tflops_or_bw << " GB/s" << std::endl;
            }
        };

        print_breakdown("Prompt Prefill", profile.prefill, profile.prefill_tflops, true);
        print_breakdown("Decode Step", profile.decode, profile.decode_bw_gbs, false);
        std::cout << "====================================================================" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return 0;
}

