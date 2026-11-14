#include "config.cuh"
#include "tensor.cuh"
#include "kernels.cuh"
#include "engine.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>
#include <algorithm>
#include <memory>
#include <random>

using namespace qwen;

// ============================================================================
// Numerical Evaluation Helpers
// ============================================================================

static float compute_max_abs_err(const half* a, const half* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(__half2float(a[i]) - __half2float(b[i]));
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

static float compute_cosine_similarity(const half* a, const half* b, size_t n) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double fa = __half2float(a[i]);
        double fb = __half2float(b[i]);
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

static bool check_has_nan_or_inf(const half* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float f = __half2float(data[i]);
        if (std::isnan(f) || std::isinf(f)) return true;
    }
    return false;
}

// ============================================================================
// Standalone Synthetic Mathematical Suites (Zero File Dependencies)
// ============================================================================

// [Synthetic Suite 1] RMSNorm FP16 Suite
bool run_synthetic_rmsnorm_suite() {
    std::cout << "\n[Synthetic Suite 1] Testing Standalone RMSNorm (FP16, H=1536)..." << std::endl;
    constexpr int H = 1536;
    constexpr int num_tokens = 8;
    size_t total_elements = static_cast<size_t>(num_tokens) * H;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis_in(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dis_gamma(0.8f, 1.2f);

    std::vector<half> h_in(total_elements);
    std::vector<half> h_gamma(H);
    std::vector<half> h_cpu_ref(total_elements);
    std::vector<half> h_gpu_out(total_elements);

    for (size_t i = 0; i < total_elements; ++i) {
        h_in[i] = __float2half(dis_in(gen));
    }
    for (int i = 0; i < H; ++i) {
        h_gamma[i] = __float2half(dis_gamma(gen));
    }

    // CPU reference matching the GPU kernel's two-stage FP16 multiplication
    for (int t = 0; t < num_tokens; ++t) {
        const half* in_tok = h_in.data() + t * H;
        half* out_tok = h_cpu_ref.data() + t * H;
        double sum_sq = 0.0;
        for (int i = 0; i < H; ++i) {
            float v = __half2float(in_tok[i]);
            sum_sq += static_cast<double>(v * v);
        }
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(H) + static_cast<double>(RMS_NORM_EPS)));
        half inv_rms_h = __float2half(inv_rms);
        for (int i = 0; i < H; ++i) {
            half norm_val = __hmul(in_tok[i], inv_rms_h);
            out_tok[i] = __hmul(norm_val, h_gamma[i]);
        }
    }

    half *d_in = nullptr, *d_gamma = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), H * sizeof(half), cudaMemcpyHostToDevice));

    launch_rmsnorm_fp16(d_in, d_gamma, d_out, num_tokens, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out, total_elements * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    float max_err = compute_max_abs_err(h_gpu_out.data(), h_cpu_ref.data(), total_elements);
    bool has_nan = check_has_nan_or_inf(h_gpu_out.data(), total_elements);

    std::cout << "  - Max Abs Error: " << max_err << " (threshold: <= 1e-3)" << std::endl;
    if (has_nan || max_err > 1e-3f) {
        std::cerr << "  FAILED: Synthetic RMSNorm error " << max_err << " exceeds threshold or has NaN/Inf!" << std::endl;
        return false;
    }
    std::cout << "  [PASSED] Synthetic RMSNorm parity verified." << std::endl;
    return true;
}

// [Synthetic Suite 2] Rotary Position Embeddings (NeoX RoPE)
bool run_synthetic_rope_suite() {
    std::cout << "\n[Synthetic Suite 2] Testing Standalone Rotary Position Embeddings (NeoX RoPE)..." << std::endl;
    constexpr int num_tokens = 4;
    constexpr int head_dim = 128;
    constexpr int q_dim = NUM_Q_HEADS * head_dim;    // 12 * 128 = 1536
    constexpr int kv_dim = NUM_KV_HEADS * head_dim;  // 2 * 128 = 256

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<half> h_q(num_tokens * q_dim);
    std::vector<half> h_k(num_tokens * kv_dim);
    std::vector<half> h_v(num_tokens * kv_dim);
    std::vector<half> h_q_ref(num_tokens * q_dim);
    std::vector<half> h_k_ref(num_tokens * kv_dim);

    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = __float2half(dis(gen));
    for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = __float2half(dis(gen));
    for (size_t i = 0; i < h_v.size(); ++i) h_v[i] = __float2half(dis(gen));
    h_q_ref = h_q;
    h_k_ref = h_k;

    std::vector<half> zero_q_bias(q_dim, __float2half(0.0f));
    std::vector<half> zero_kv_bias(kv_dim, __float2half(0.0f));

    // Analytic trigonometric reference on CPU for seq_pos = 0 .. num_tokens-1
    for (int m = 0; m < num_tokens; ++m) {
        int pos = m;
        // Q heads
        for (int h = 0; h < NUM_Q_HEADS; ++h) {
            half* head_ptr = h_q_ref.data() + m * q_dim + h * head_dim;
            for (int coord_i = 0; coord_i < head_dim / 2; ++coord_i) {
                float exponent = -2.0f * static_cast<float>(coord_i) / static_cast<float>(head_dim);
                float theta = std::exp2(exponent * LOG2_ROPE_THETA);
                float angle = static_cast<float>(pos) * theta;
                float sin_val = std::sin(angle);
                float cos_val = std::cos(angle);
                float x0 = __half2float(head_ptr[coord_i]);
                float x1 = __half2float(head_ptr[coord_i + head_dim / 2]);
                head_ptr[coord_i] = __float2half(x0 * cos_val - x1 * sin_val);
                head_ptr[coord_i + head_dim / 2] = __float2half(x0 * sin_val + x1 * cos_val);
            }
        }
        // K heads
        for (int h = 0; h < NUM_KV_HEADS; ++h) {
            half* head_ptr = h_k_ref.data() + m * kv_dim + h * head_dim;
            for (int coord_i = 0; coord_i < head_dim / 2; ++coord_i) {
                float exponent = -2.0f * static_cast<float>(coord_i) / static_cast<float>(head_dim);
                float theta = std::exp2(exponent * LOG2_ROPE_THETA);
                float angle = static_cast<float>(pos) * theta;
                float sin_val = std::sin(angle);
                float cos_val = std::cos(angle);
                float x0 = __half2float(head_ptr[coord_i]);
                float x1 = __half2float(head_ptr[coord_i + head_dim / 2]);
                head_ptr[coord_i] = __float2half(x0 * cos_val - x1 * sin_val);
                head_ptr[coord_i + head_dim / 2] = __float2half(x0 * sin_val + x1 * cos_val);
            }
        }
    }

    half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    half *d_qb = nullptr, *d_kb = nullptr, *d_vb = nullptr;
    half *d_kc = nullptr, *d_vc = nullptr;
    DeviceControlBlock* d_dcb = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_qb, q_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_kb, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vb, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_kc, NUM_KV_HEADS * MAX_SEQ_LEN * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_vc, NUM_KV_HEADS * MAX_SEQ_LEN * head_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dcb, sizeof(DeviceControlBlock)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qb, zero_q_bias.data(), q_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kb, zero_kv_bias.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vb, zero_kv_bias.data(), kv_dim * sizeof(half), cudaMemcpyHostToDevice));

    DeviceControlBlock dcb = {0, 0, 0.0f, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_dcb, &dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

    launch_qkv_bias_rope_kv_append(d_q, d_k, d_v, d_qb, d_kb, d_vb, d_kc, d_vc, d_dcb, num_tokens, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_gpu_q(h_q.size());
    std::vector<half> h_gpu_k(h_k.size());
    CUDA_CHECK(cudaMemcpy(h_gpu_q.data(), d_q, h_q.size() * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gpu_k.data(), d_k, h_k.size() * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_qb); cudaFree(d_kb); cudaFree(d_vb);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_dcb);

    float err_q = compute_max_abs_err(h_gpu_q.data(), h_q_ref.data(), h_q.size());
    float err_k = compute_max_abs_err(h_gpu_k.data(), h_k_ref.data(), h_k.size());
    bool has_nan_q = check_has_nan_or_inf(h_gpu_q.data(), h_q.size());
    bool has_nan_k = check_has_nan_or_inf(h_gpu_k.data(), h_k.size());

    std::cout << "  - Q RoPE Max Abs Error: " << err_q << " (threshold: <= 1e-3)" << std::endl;
    std::cout << "  - K RoPE Max Abs Error: " << err_k << " (threshold: <= 1e-3)" << std::endl;

    if (has_nan_q || has_nan_k || err_q > 1e-3f || err_k > 1e-3f) {
        std::cerr << "  FAILED: Synthetic RoPE error exceeds threshold or has NaN/Inf!" << std::endl;
        return false;
    }
    std::cout << "  [PASSED] Synthetic RoPE trigonometric accuracy verified." << std::endl;
    return true;
}

// [Synthetic Suite 3] Dynamic Dimension Sweeps for GEMV & SwiGLU (FP16 & W4A16)
bool run_synthetic_gemv_swiglu_suite() {
    std::cout << "\n[Synthetic Suite 3] Testing Dynamic Dimension Sweeps for GEMV & SwiGLU..." << std::endl;

    struct DimPair {
        int H;
        int I;
        const char* label;
    };

    std::vector<DimPair> pairs = {
        {1536, 8960,  "1.5B Scale (H=1536, I=8960)"},
        {2560, 10240, "4B Scale   (H=2560, I=10240)"},
        {5120, 27648, "27B Scale  (H=5120, I=27648)"},
        {256,  512,   "Small Scale (H=256, I=512)"}
    };

    bool all_pairs_passed = true;

    for (const auto& pair : pairs) {
        int H = pair.H;
        int I = pair.I;
        std::cout << "\n  --- Dimension Sweep: " << pair.label << " ---" << std::endl;

        std::mt19937 gen(42 + H);
        float scale_factor = 1.0f / std::sqrt(static_cast<float>(H));
        std::uniform_real_distribution<float> dis_x(-scale_factor, scale_factor);
        std::uniform_real_distribution<float> dis_w(-scale_factor, scale_factor);

        // 1. Prepare input_x
        std::vector<half> h_x(H);
        for (int i = 0; i < H; ++i) h_x[i] = __float2half(dis_x(gen));

        half *d_x = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, H * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_out, I * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), H * sizeof(half), cudaMemcpyHostToDevice));

        // ====================================================================
        // W4A16 SwiGLU verification
        // ====================================================================
        {
            int num_groups = H / 128;
            size_t num_qwords = static_cast<size_t>(2 * I) * (H / 8);
            size_t num_scales = static_cast<size_t>(2 * I) * num_groups;

            std::vector<uint32_t> h_qweight(num_qwords);
            std::vector<half> h_scales(num_scales);

            std::uniform_int_distribution<int> dis_nibble(2, 14);
            for (size_t i = 0; i < num_qwords; ++i) {
                uint32_t q0 = dis_nibble(gen);
                uint32_t q1 = dis_nibble(gen);
                uint32_t q2 = dis_nibble(gen);
                uint32_t q3 = dis_nibble(gen);
                uint32_t q4 = dis_nibble(gen);
                uint32_t q5 = dis_nibble(gen);
                uint32_t q6 = dis_nibble(gen);
                uint32_t q7 = dis_nibble(gen);
                h_qweight[i] = (q0) | (q4 << 4) | (q1 << 8) | (q5 << 12) |
                               (q2 << 16) | (q6 << 20) | (q3 << 24) | (q7 << 28);
            }
            float sc_val = 0.05f * scale_factor;
            for (size_t i = 0; i < num_scales; ++i) {
                h_scales[i] = __float2half(sc_val);
            }

            // CPU reference simulating Q8_1 quantization and W4 dot product
            std::vector<float> x_scale(num_groups);
            std::vector<int8_t> x_q(H);
            for (int g = 0; g < num_groups; ++g) {
                float max_abs = 0.0f;
                for (int j = 0; j < 128; ++j) {
                    float val = std::abs(__half2float(h_x[g * 128 + j]));
                    if (val > max_abs) max_abs = val;
                }
                float d = std::max(max_abs / 127.0f, 1e-8f);
                x_scale[g] = d;
                float inv_d = 1.0f / d;
                for (int j = 0; j < 128; ++j) {
                    float val = __half2float(h_x[g * 128 + j]) * inv_d;
                    int v = static_cast<int>(std::round(val));
                    v = std::max(-127, std::min(127, v));
                    x_q[g * 128 + j] = static_cast<int8_t>(v);
                }
            }

            std::vector<half> h_cpu_ref_w4(I);
            for (int r = 0; r < I; ++r) {
                auto eval_row = [&](int row_idx) -> double {
                    const uint32_t* row_qw = h_qweight.data() + static_cast<size_t>(row_idx) * (H / 8);
                    const half* row_sc = h_scales.data() + static_cast<size_t>(row_idx) * num_groups;
                    double acc = 0.0;
                    for (int g = 0; g < num_groups; ++g) {
                        float w_sc = __half2float(row_sc[g]);
                        float group_factor = w_sc * x_scale[g];
                        int32_t group_dot = 0;
                        for (int w = 0; w < 16; ++w) {
                            uint32_t packed = row_qw[g * 16 + w];
                            int q[8];
                            q[0] = (packed >> 0)  & 0xF;
                            q[4] = (packed >> 4)  & 0xF;
                            q[1] = (packed >> 8)  & 0xF;
                            q[5] = (packed >> 12) & 0xF;
                            q[2] = (packed >> 16) & 0xF;
                            q[6] = (packed >> 20) & 0xF;
                            q[3] = (packed >> 24) & 0xF;
                            q[7] = (packed >> 28) & 0xF;
                            for (int j = 0; j < 8; ++j) {
                                int8_t x_val = x_q[g * 128 + w * 8 + j];
                                group_dot += (q[j] - 8) * static_cast<int32_t>(x_val);
                            }
                        }
                        acc += static_cast<double>(group_dot) * static_cast<double>(group_factor);
                    }
                    return acc;
                };

                double g_acc = eval_row(2 * r + 0);
                double u_acc = eval_row(2 * r + 1);
                double silu = g_acc / (1.0 + std::exp(-g_acc));
                h_cpu_ref_w4[r] = __float2half(static_cast<float>(silu * u_acc));
            }

            uint32_t* d_gate_up_w4 = nullptr;
            half* d_gate_up_scales = nullptr;
            CUDA_CHECK(cudaMalloc(&d_gate_up_w4, num_qwords * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_gate_up_scales, num_scales * sizeof(half)));

            CUDA_CHECK(cudaMemcpy(d_gate_up_w4, h_qweight.data(), num_qwords * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_gate_up_scales, h_scales.data(), num_scales * sizeof(half), cudaMemcpyHostToDevice));

            launch_gemv_dual_gate_up_swiglu_w4a16(d_gate_up_w4, d_gate_up_scales, d_x, d_out, H, I, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<half> h_gpu_out_w4(I);
            CUDA_CHECK(cudaMemcpy(h_gpu_out_w4.data(), d_out, I * sizeof(half), cudaMemcpyDeviceToHost));

            cudaFree(d_gate_up_w4);
            cudaFree(d_gate_up_scales);

            float err_w4 = compute_max_abs_err(h_gpu_out_w4.data(), h_cpu_ref_w4.data(), I);
            bool nan_w4 = check_has_nan_or_inf(h_gpu_out_w4.data(), I);
            std::cout << "    [W4A16] Max Abs Error: " << err_w4 << " (threshold: <= 5e-3)" << std::endl;

            if (nan_w4 || err_w4 > 5e-3f) {
                std::cerr << "    FAILED: W4A16 SwiGLU error exceeds 5e-3 or has NaN/Inf on " << pair.label << "!" << std::endl;
                all_pairs_passed = false;
            }
        }

        cudaFree(d_x);
        cudaFree(d_out);
    }

    if (all_pairs_passed) {
        std::cout << "  [PASSED] Dynamic dimension sweeps verified across all scales." << std::endl;
        return true;
    } else {
        std::cerr << "  FAILED: Some dynamic dimension sweeps failed!" << std::endl;
        return false;
    }
}

// ============================================================================
// Golden Reference Parity Suite (Opt-In with --golden <path>)
// ============================================================================

bool run_golden_reference_suite(
    const std::string& golden_path,
    const std::string& w4a16_path
) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "   Golden Reference Verification Suite (W4A16)          " << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Golden Reference: " << golden_path << std::endl;
    std::cout << "W4A16 Model:      " << w4a16_path << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    std::ifstream gfile(golden_path, std::ios::binary);
    if (!gfile.is_open()) {
        std::cerr << "[ERROR] Cannot open golden reference file: " << golden_path << std::endl;
        std::cerr << "Run 'python tools/verify_reference.py' first!" << std::endl;
        return false;
    }

    GoldenRefHeader g_header;
    gfile.read(reinterpret_cast<char*>(&g_header), sizeof(g_header));
    if (g_header.magic != 0x474F4C44) {
        std::cerr << "[ERROR] Invalid golden reference magic: 0x" << std::hex << g_header.magic << std::endl;
        return false;
    }

    auto read_ref_tensor = [&](int idx, size_t num_elements, size_t elem_size) -> std::vector<char> {
        std::vector<char> buf(num_elements * elem_size);
        gfile.seekg(g_header.tensor_offsets[idx]);
        gfile.read(buf.data(), num_elements * elem_size);
        return buf;
    };

    // Load reference activations
    auto ref_embed_raw     = read_ref_tensor(0, 8 * HIDDEN_SIZE, sizeof(half));
    auto ref_in_norm_raw   = read_ref_tensor(1, 8 * HIDDEN_SIZE, sizeof(half));
    auto ref_q_raw_raw     = read_ref_tensor(2, 8 * Q_DIM, sizeof(half));
    auto ref_k_raw_raw     = read_ref_tensor(3, 8 * KV_DIM, sizeof(half));
    auto ref_v_raw_raw     = read_ref_tensor(4, 8 * KV_DIM, sizeof(half));
    auto ref_q_rope_raw    = read_ref_tensor(5, 8 * Q_DIM, sizeof(half));
    auto ref_k_rope_raw    = read_ref_tensor(6, 8 * KV_DIM, sizeof(half));
    auto ref_v_biased_raw  = read_ref_tensor(7, 8 * KV_DIM, sizeof(half));
    auto ref_attn_out_raw  = read_ref_tensor(8, 8 * HIDDEN_SIZE, sizeof(half));
    auto ref_post_norm_raw = read_ref_tensor(9, 8 * HIDDEN_SIZE, sizeof(half));
    auto ref_swiglu_raw    = read_ref_tensor(10, 8 * INTERMEDIATE_SIZE, sizeof(half));
    auto ref_mlp_out_raw   = read_ref_tensor(11, 8 * HIDDEN_SIZE, sizeof(half));
    auto ref_logits_raw    = read_ref_tensor(12, VOCAB_SIZE, sizeof(float));

    const half* ref_embed     = reinterpret_cast<const half*>(ref_embed_raw.data());
    const half* ref_in_norm   = reinterpret_cast<const half*>(ref_in_norm_raw.data());
    const half* ref_q_raw     = reinterpret_cast<const half*>(ref_q_raw_raw.data());
    const half* ref_k_raw     = reinterpret_cast<const half*>(ref_k_raw_raw.data());
    const half* ref_v_raw     = reinterpret_cast<const half*>(ref_v_raw_raw.data());
    const half* ref_q_rope    = reinterpret_cast<const half*>(ref_q_rope_raw.data());
    const half* ref_k_rope    = reinterpret_cast<const half*>(ref_k_rope_raw.data());
    const half* ref_v_biased  = reinterpret_cast<const half*>(ref_v_biased_raw.data());
    const half* ref_attn_out  = reinterpret_cast<const half*>(ref_attn_out_raw.data());
    const half* ref_post_norm = reinterpret_cast<const half*>(ref_post_norm_raw.data());
    const half* ref_swiglu    = reinterpret_cast<const half*>(ref_swiglu_raw.data());
    const half* ref_mlp_out   = reinterpret_cast<const half*>(ref_mlp_out_raw.data());
    const float* ref_logits   = reinterpret_cast<const float*>(ref_logits_raw.data());
    (void)ref_mlp_out;

    bool all_passed = true;

    // Common GPU scratch buffers
    half *d_test_in = nullptr, *d_test_out = nullptr, *d_test_q = nullptr, *d_test_k = nullptr, *d_test_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_test_in, 8 * INTERMEDIATE_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_test_out, 8 * INTERMEDIATE_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_test_q, 8 * Q_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_test_k, 8 * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_test_v, 8 * KV_DIM * sizeof(half)));

    int32_t* d_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_tokens, g_header.prompt_tokens, 8 * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<half> h_test_out(8 * INTERMEDIATE_SIZE);


    // ========================================================================
    // PART B: W4A16 Engine Tests (Suites 1b, 5, 6b, 8b)
    // ========================================================================
    std::cout << "\n>>> [PART B] Loading W4A16 Engine (~903 MB VRAM)..." << std::endl;
    {
        auto engine_w4a16 = std::make_unique<QwenEngine>(w4a16_path);
        cudaStream_t stream = engine_w4a16->stream();

        // [Suite 1b] W4A16 Embedding Lookup
        std::cout << "\n[Suite 1b] Testing W4A16 Embedding Lookup (Cosine Similarity >= 0.990)..." << std::endl;
        launch_embed_lookup_w4a16(
            reinterpret_cast<const uint32_t*>(engine_w4a16->embed_weight()),
            engine_w4a16->embed_scale(),
            d_tokens, d_test_out, nullptr, 8, stream
        );
        CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
        float cos_sim = compute_cosine_similarity(h_test_out.data(), ref_embed, 8 * HIDDEN_SIZE);
        std::cout << "  - W4A16 Cosine Similarity: " << cos_sim << " (threshold: >= 0.990)" << std::endl;
        if (cos_sim < 0.990f) {
            std::cerr << "  FAILED: W4A16 embedding lookup similarity below 0.990!" << std::endl;
            all_passed = false;
        }

        // [Suite 5] Fused W4A16 GEMV & Fused SwiGLU
        std::cout << "\n[Suite 5] Testing Fused W4A16 GEMV & Fused SwiGLU..." << std::endl;
        {
            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemv_w4a16(
                reinterpret_cast<const uint32_t*>(engine_w4a16->layer_weights(0).q_weight),
                engine_w4a16->layer_weights(0).q_scale,
                d_test_in, nullptr, d_test_out, Q_DIM, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_w4_q = compute_cosine_similarity(h_test_out.data(), ref_q_raw, Q_DIM);
            std::cout << "  - W4A16 GEMV (Q-proj) Cosine Similarity: " << cos_w4_q << " (threshold: >= 0.990)" << std::endl;

            CUDA_CHECK(cudaMemcpy(d_test_in, ref_post_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemv_dual_gate_up_swiglu_w4a16(
                reinterpret_cast<const uint32_t*>(engine_w4a16->layer_weights(0).gate_up_weight),
                engine_w4a16->layer_weights(0).gate_up_scale,
                d_test_in, d_test_out, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, INTERMEDIATE_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_w4_swiglu = compute_cosine_similarity(h_test_out.data(), ref_swiglu, INTERMEDIATE_SIZE);
            std::cout << "  - W4A16 Fused SwiGLU Cosine Similarity:  " << cos_w4_swiglu << " (threshold: >= 0.985)" << std::endl;

            // Merged QKV W4A8 test
            int8_t* d_act_q8_test = nullptr;
            float* d_scale_q8_test = nullptr;
            int32_t* d_sum8_q8_test = nullptr;
            CUDA_CHECK(cudaMalloc(&d_act_q8_test, HIDDEN_SIZE * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&d_scale_q8_test, (HIDDEN_SIZE / 128) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sum8_q8_test, (HIDDEN_SIZE / 8) * sizeof(int32_t)));

            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_quantize_fp16_to_q8_1(d_test_in, d_act_q8_test, d_scale_q8_test, d_sum8_q8_test, HIDDEN_SIZE, stream);

            QKVProjectionView qkv_view_w4 = {
                engine_w4a16->layer_weights(0).q_weight, engine_w4a16->layer_weights(0).q_scale, d_test_q,
                engine_w4a16->layer_weights(0).k_weight, engine_w4a16->layer_weights(0).k_scale, d_test_k,
                engine_w4a16->layer_weights(0).v_weight, engine_w4a16->layer_weights(0).v_scale, d_test_v
            };
            launch_gemv_qkv_w4a8(qkv_view_w4, d_act_q8_test, d_scale_q8_test, d_sum8_q8_test, stream);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_q, Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_qkv_q = compute_cosine_similarity(h_test_out.data(), ref_q_raw, Q_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_k, KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_qkv_k = compute_cosine_similarity(h_test_out.data(), ref_k_raw, KV_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_v, KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_qkv_v = compute_cosine_similarity(h_test_out.data(), ref_v_raw, KV_DIM);
            std::cout << "  - W4A8 Merged QKV Cosine Sim: Q=" << cos_qkv_q << " (>=0.990), K=" << cos_qkv_k << " (>=0.985), V=" << cos_qkv_v << " (>=0.985)" << std::endl;

            cudaFree(d_act_q8_test);
            cudaFree(d_scale_q8_test);
            cudaFree(d_sum8_q8_test);

            if (cos_w4_q < 0.990f || cos_w4_swiglu < 0.985f || cos_qkv_q < 0.990f || cos_qkv_k < 0.985f || cos_qkv_v < 0.985f) {
                std::cerr << "  FAILED: W4A16/W4A8 cosine similarity below threshold!" << std::endl;
                all_passed = false;
            }
        }

        // [Suite 6b] Tiled Prefill GEMM W4A16
        std::cout << "\n[Suite 6b] Testing Tiled Prefill GEMM (W4A16, M=8)..." << std::endl;
        {
            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemm_prefill_w4a16(
                reinterpret_cast<const uint32_t*>(engine_w4a16->layer_weights(0).q_weight),
                engine_w4a16->layer_weights(0).q_scale,
                d_test_in, nullptr, d_test_out, 8, Q_DIM, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, 8 * Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_prefill_w4 = compute_cosine_similarity(h_test_out.data(), ref_q_raw, 8 * Q_DIM);
            std::cout << "  - W4A16 Prefill GEMM Cosine Sim:   " << cos_prefill_w4 << " (threshold: >= 0.990)" << std::endl;
            if (cos_prefill_w4 < 0.990f) {
                std::cerr << "  FAILED: W4A16 Prefill GEMM cosine similarity below 0.990!" << std::endl;
                all_passed = false;
            }

            // Prefill Merged QKV W4A16 test
            QKVProjectionView qkv_view_w4 = {
                engine_w4a16->layer_weights(0).q_weight, engine_w4a16->layer_weights(0).q_scale, d_test_q,
                engine_w4a16->layer_weights(0).k_weight, engine_w4a16->layer_weights(0).k_scale, d_test_k,
                engine_w4a16->layer_weights(0).v_weight, engine_w4a16->layer_weights(0).v_scale, d_test_v
            };
            launch_gemm_prefill_qkv_w4a16(qkv_view_w4, d_test_in, 8, stream);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_q, 8 * Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_prefill_qkv_q = compute_cosine_similarity(h_test_out.data(), ref_q_raw, 8 * Q_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_k, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_prefill_qkv_k = compute_cosine_similarity(h_test_out.data(), ref_k_raw, 8 * KV_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_v, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float cos_prefill_qkv_v = compute_cosine_similarity(h_test_out.data(), ref_v_raw, 8 * KV_DIM);
            std::cout << "  - W4A16 Prefill Merged QKV Cosine Sim: Q=" << cos_prefill_qkv_q << " (>=0.990), K=" << cos_prefill_qkv_k << " (>=0.985), V=" << cos_prefill_qkv_v << " (>=0.985)" << std::endl;
            if (cos_prefill_qkv_q < 0.990f || cos_prefill_qkv_k < 0.985f || cos_prefill_qkv_v < 0.985f) {
                std::cerr << "  FAILED: W4A16 Prefill Merged QKV cosine similarity below threshold!" << std::endl;
                all_passed = false;
            }
        }

        // [Suite 8b] End-to-end W4A16 Generation Verification (Graph vs Stream Parity)
        std::cout << "\n[Suite 8b] Testing End-to-End W4A16 Prefill & Decode Generation (Graph vs Stream Parity)..." << std::endl;
        {
            std::vector<int32_t> prompt(g_header.prompt_tokens, g_header.prompt_tokens + 8);

            // Pass 1: Uncaptured stream execution
            engine_w4a16->reset_session();
            int32_t first_stream = engine_w4a16->prefill(prompt, 0.0f);
            std::vector<int32_t> stream_tokens;
            stream_tokens.push_back(first_stream);
            for (int step = 1; step < 16; ++step) {
                engine_w4a16->execute_decode_pass();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                stream_tokens.push_back(*(engine_w4a16->pinned_token_ptr()));
            }

            // Pass 2: Captured CUDA Graph replay
            engine_w4a16->reset_session();
            int32_t first_graph = engine_w4a16->prefill(prompt, 0.0f);
            std::vector<int32_t> graph_tokens;
            graph_tokens.push_back(first_graph);
            for (int step = 1; step < 16; ++step) {
                int32_t next_t = engine_w4a16->decode_step();
                graph_tokens.push_back(next_t);
            }

            int parity_matches = 0;
            bool valid_vocab = true;
            for (int i = 0; i < 16; ++i) {
                if (stream_tokens[i] >= 0 && stream_tokens[i] < VOCAB_SIZE &&
                    graph_tokens[i] >= 0 && graph_tokens[i] < VOCAB_SIZE) {
                    if (stream_tokens[i] == graph_tokens[i]) {
                        parity_matches++;
                    }
                } else {
                    valid_vocab = false;
                }
            }

            std::cout << "  - W4A16 First Generated Token: " << first_graph << " (Stream: " << first_stream << ")" << std::endl;
            std::cout << "  - W4A16 Graph vs Stream Replay Parity: " << parity_matches << " / 16 tokens" << std::endl;

            if (!valid_vocab) {
                std::cerr << "  FAILED: Generated token outside vocabulary range [0, " << VOCAB_SIZE << ")!" << std::endl;
                all_passed = false;
            }
            if (parity_matches < 16) {
                std::cerr << "  FAILED: Graph vs Stream decode execution diverged! (" << parity_matches << "/16)" << std::endl;
                all_passed = false;
            }
        }

        // Verify Q8_1 scratch buffers do not overlap d_act_1
        {
            const auto& ws = engine_w4a16->workspace();
            auto ptr_val = [](const void* p) -> uintptr_t { return reinterpret_cast<uintptr_t>(p); };
            uintptr_t act1_start = ptr_val(ws.d_act_1);
            uintptr_t act1_end   = act1_start + static_cast<size_t>(512) * HIDDEN_SIZE * sizeof(half);
            uintptr_t q8_ptr     = ptr_val(ws.d_act_q8);
            uintptr_t scale_ptr  = ptr_val(ws.d_act_scale);
            uintptr_t sum8_ptr   = ptr_val(ws.d_act_sum8);

            bool overlaps = (q8_ptr >= act1_start && q8_ptr < act1_end) ||
                            (scale_ptr >= act1_start && scale_ptr < act1_end) ||
                            (sum8_ptr >= act1_start && sum8_ptr < act1_end);
            std::cout << "  - Q8_1 Buffer Non-Overlap Check: " << (overlaps ? "OVERLAP" : "OK") << std::endl;
            std::cout << "  - Weights 16-byte Aligned: " << (engine_w4a16->weights_16byte_aligned() ? "YES" : "NO") << std::endl;
            if (overlaps) {
                std::cerr << "  FAILED: Q8_1 scratch buffers overlap d_act_1!" << std::endl;
                all_passed = false;
            }
        }

        // Verify profile_stages() returns positive timings
        {
            std::vector<int32_t> prof_toks = {151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30};
            StageProfile prof = engine_w4a16->profile_stages(prof_toks);
            bool positive = prof.prefill.total_ms > 0.0f && prof.decode.total_ms > 0.0f &&
                            prof.prefill.gate_up_swiglu_ms > 0.0f && prof.decode.gate_up_swiglu_ms > 0.0f &&
                            prof.decode.down_proj_ms > 0.0f && prof.decode.lm_head_ms > 0.0f;
            std::cout << "  - StageProfile Check: " << (positive ? "OK" : "INVALID") << std::endl;
            if (!positive) {
                std::cerr << "  FAILED: profile_stages() returned non-positive stage timing!" << std::endl;
                all_passed = false;
            }
        }

        std::cout << "[Test] Unloading W4A16 engine and freeing ~903 MB VRAM..." << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup common scratch buffers
    cudaFree(d_tokens);
    cudaFree(d_test_in);
    cudaFree(d_test_out);
    cudaFree(d_test_q);
    cudaFree(d_test_k);
    cudaFree(d_test_v);

    return all_passed;
}

// ============================================================================
// Main Dispatcher
// ============================================================================

int main(int argc, char** argv) {
    bool run_golden = false;
    std::string golden_path = "models/golden_ref.bin";
    std::string w4a16_path  = "models/qwen25_1.5b_w4a16.qwenbin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--golden") {
            run_golden = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                golden_path = argv[++i];
            }
        }
        else if (arg == "--w4a16" && i + 1 < argc) w4a16_path = argv[++i];
    }

    std::cout << "========================================================" << std::endl;
    std::cout << "   Qwen 2.5-1.5B W4A16 Kernel & Engine Test Suite       " << std::endl;
    std::cout << "========================================================" << std::endl;

    bool all_passed = true;

    // Unconditionally execute all standalone synthetic unit tests
    std::cout << "\n>>> Executing Standalone Synthetic Mathematical Suites..." << std::endl;

    if (!run_synthetic_rmsnorm_suite()) all_passed = false;
    if (!run_synthetic_rope_suite()) all_passed = false;
    if (!run_synthetic_gemv_swiglu_suite()) all_passed = false;

    // Conditionally execute golden reference suite if requested
    if (run_golden) {
        std::cout << "\n>>> Executing Golden Reference Parity Suite..." << std::endl;
        if (!run_golden_reference_suite(golden_path, w4a16_path)) {
            all_passed = false;
        }
    }

    std::cout << "\n========================================================" << std::endl;
    if (all_passed) {
        std::cout << ">>> ALL TEST SUITES PASSED (EXIT CODE 0) <<<" << std::endl;
        std::cout << "========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << ">>> SOME TEST SUITES FAILED <<<" << std::endl;
        std::cout << "========================================================" << std::endl;
        return 1;
    }
}
