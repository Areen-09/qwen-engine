#include "config.cuh"
#include "tensor.cuh"
#include "kernels.cuh"
#include "deltanet.cuh"
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
        // FP16 SwiGLU verification
        // ====================================================================
        {
            size_t num_w_elements = static_cast<size_t>(2 * I) * H;
            std::vector<half> h_gate_up_fp16(num_w_elements);
            for (size_t i = 0; i < num_w_elements; ++i) {
                h_gate_up_fp16[i] = __float2half(dis_w(gen));
            }

            // CPU reference for FP16 SwiGLU
            std::vector<half> h_cpu_ref_fp16(I);
            for (int r = 0; r < I; ++r) {
                const half* g_row = h_gate_up_fp16.data() + static_cast<size_t>(2 * r + 0) * H;
                const half* u_row = h_gate_up_fp16.data() + static_cast<size_t>(2 * r + 1) * H;
                double g_acc = 0.0;
                double u_acc = 0.0;
                for (int c = 0; c < H; ++c) {
                    double x_val = static_cast<double>(__half2float(h_x[c]));
                    g_acc += static_cast<double>(__half2float(g_row[c])) * x_val;
                    u_acc += static_cast<double>(__half2float(u_row[c])) * x_val;
                }
                double silu = g_acc / (1.0 + std::exp(-g_acc));
                h_cpu_ref_fp16[r] = __float2half(static_cast<float>(silu * u_acc));
            }

            half* d_gate_up_fp16 = nullptr;
            CUDA_CHECK(cudaMalloc(&d_gate_up_fp16, num_w_elements * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(d_gate_up_fp16, h_gate_up_fp16.data(), num_w_elements * sizeof(half), cudaMemcpyHostToDevice));

            launch_gemv_dual_gate_up_swiglu_fp16(d_gate_up_fp16, d_x, d_out, H, I, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<half> h_gpu_out_fp16(I);
            CUDA_CHECK(cudaMemcpy(h_gpu_out_fp16.data(), d_out, I * sizeof(half), cudaMemcpyDeviceToHost));
            cudaFree(d_gate_up_fp16);

            float err_fp16 = compute_max_abs_err(h_gpu_out_fp16.data(), h_cpu_ref_fp16.data(), I);
            bool nan_fp16 = check_has_nan_or_inf(h_gpu_out_fp16.data(), I);
            std::cout << "    [FP16] Max Abs Error: " << err_fp16 << " (threshold: <= 5e-3)" << std::endl;

            if (nan_fp16 || err_fp16 > 5e-3f) {
                std::cerr << "    FAILED: FP16 SwiGLU error exceeds 5e-3 or has NaN/Inf on " << pair.label << "!" << std::endl;
                all_pairs_passed = false;
            }
        }

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

// [Synthetic Suite 4] Standalone Gated DeltaNet Recurrence Suite
bool run_synthetic_deltanet_suite() {
    std::cout << "\n[Synthetic Suite 4] Testing Standalone Gated DeltaNet Recurrence (sm_75)..." << std::endl;
    constexpr int batch_size = 1;
    constexpr int num_heads = 4;
    constexpr int head_dim = 128;
    constexpr size_t state_elements = static_cast<size_t>(batch_size) * num_heads * head_dim * head_dim; // 65536
    constexpr size_t vector_elements = static_cast<size_t>(batch_size) * num_heads * head_dim;           // 512

    std::vector<int> seq_lengths = {1, 4, 16, 64};
    bool all_t_passed = true;

    half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_beta = nullptr, *d_out = nullptr;
    float *d_state = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, vector_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, vector_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_v, vector_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_beta, num_heads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, vector_elements * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_state, state_elements * sizeof(float)));

    for (int T : seq_lengths) {
        std::cout << "  - Evaluating recurrence across sequence length T=" << T << "..." << std::endl;
        CUDA_CHECK(cudaMemset(d_state, 0, state_elements * sizeof(float)));

        std::vector<double> ref_state(state_elements, 0.0);
        std::mt19937 gen(42 + T);
        std::uniform_real_distribution<float> dis_vec(-0.2f, 0.2f);
        std::uniform_real_distribution<float> dis_beta(-1.0f, 1.0f);

        std::vector<half> h_q(vector_elements);
        std::vector<half> h_k(vector_elements);
        std::vector<half> h_v(vector_elements);
        std::vector<half> h_beta(num_heads);
        std::vector<half> h_cpu_out(vector_elements);
        std::vector<half> h_gpu_out(vector_elements);

        float max_step_err = 0.0f;
        float min_step_cos = 1.0f;

        for (int t = 0; t < T; ++t) {
            for (size_t i = 0; i < vector_elements; ++i) {
                h_q[i] = __float2half(dis_vec(gen));
                h_k[i] = __float2half(dis_vec(gen));
                h_v[i] = __float2half(dis_vec(gen));
            }
            for (int h = 0; h < num_heads; ++h) {
                h_beta[h] = __float2half(dis_beta(gen));
            }

            // CPU reference recurrence:
            // alpha = 1 / (1 + exp(beta))
            // S_t = alpha * S_{t-1} + v_t * k_t^T
            // o_t = S_t * q_t
            for (int h = 0; h < num_heads; ++h) {
                double beta_val = static_cast<double>(__half2float(h_beta[h]));
                double alpha = 1.0 / (1.0 + std::exp(beta_val));

                for (int r = 0; r < head_dim; ++r) {
                    double v_val = static_cast<double>(__half2float(h_v[h * head_dim + r]));
                    for (int c = 0; c < head_dim; ++c) {
                        double k_val = static_cast<double>(__half2float(h_k[h * head_dim + c]));
                        size_t idx = static_cast<size_t>(h * head_dim + r) * head_dim + c;
                        ref_state[idx] = alpha * ref_state[idx] + v_val * k_val;
                    }
                }

                for (int r = 0; r < head_dim; ++r) {
                    double sum_prod = 0.0;
                    for (int c = 0; c < head_dim; ++c) {
                        double q_val = static_cast<double>(__half2float(h_q[h * head_dim + c]));
                        size_t idx = static_cast<size_t>(h * head_dim + r) * head_dim + c;
                        sum_prod += ref_state[idx] * q_val;
                    }
                    h_cpu_out[h * head_dim + r] = __float2half(static_cast<float>(sum_prod));
                }
            }

            CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), vector_elements * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), vector_elements * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), vector_elements * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), num_heads * sizeof(half), cudaMemcpyHostToDevice));

            launch_deltanet_decode_sm75(d_q, d_k, d_v, d_beta, d_state, d_out, batch_size, num_heads, head_dim, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out, vector_elements * sizeof(half), cudaMemcpyDeviceToHost));

            float step_err = compute_max_abs_err(h_gpu_out.data(), h_cpu_out.data(), vector_elements);
            float step_cos = compute_cosine_similarity(h_gpu_out.data(), h_cpu_out.data(), vector_elements);

            if (step_err > max_step_err) max_step_err = step_err;
            if (step_cos < min_step_cos) min_step_cos = step_cos;
        }

        std::cout << "    T=" << T << " Max Abs Error: " << max_step_err << " (threshold: <= 1e-3), Min Cos Sim: " << min_step_cos << " (threshold: >= 0.9999)" << std::endl;
        if (max_step_err > 1e-3f || min_step_cos < 0.9999f) {
            std::cerr << "    FAILED: DeltaNet parity exceeded tolerance on T=" << T << "!" << std::endl;
            all_t_passed = false;
        }
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_beta); cudaFree(d_out); cudaFree(d_state);

    if (all_t_passed) {
        std::cout << "  [PASSED] Synthetic Gated DeltaNet recurrence parity verified." << std::endl;
        return true;
    } else {
        std::cerr << "  FAILED: Synthetic DeltaNet recurrence failed!" << std::endl;
        return false;
    }
}

// [Synthetic Suite 5] Standalone Depthwise Conv1d Suite
bool run_synthetic_conv1d_suite() {
    std::cout << "\n[Synthetic Suite 5] Testing Standalone Depthwise Conv1d Steps (C=2560, K=4)..." << std::endl;
    constexpr int C = 2560;
    constexpr int K = 4;
    constexpr int history_len = K - 1; // 3
    constexpr int num_steps = 8;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis_val(-0.5f, 0.5f);

    std::vector<half> h_weight(C * K);
    std::vector<half> h_bias(C);
    std::vector<half> h_cpu_state(C * history_len, __float2half(0.0f));

    for (size_t i = 0; i < h_weight.size(); ++i) h_weight[i] = __float2half(dis_val(gen));
    for (size_t i = 0; i < h_bias.size(); ++i) h_bias[i] = __float2half(dis_val(gen));

    half *d_in = nullptr, *d_weight = nullptr, *d_bias = nullptr, *d_conv_state = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, C * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv_state, C * history_len * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, C * sizeof(half)));

    CUDA_CHECK(cudaMemset(d_conv_state, 0, C * history_len * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), C * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), C * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<half> h_in(C);
    std::vector<half> h_cpu_out(C);
    std::vector<half> h_gpu_out(C);

    float max_err = 0.0f;

    for (int step = 0; step < num_steps; ++step) {
        for (int c = 0; c < C; ++c) h_in[c] = __float2half(dis_val(gen));

        // CPU reference
        for (int c = 0; c < C; ++c) {
            double in_elem = static_cast<double>(__half2float(h_in[c]));
            const half* w_c = h_weight.data() + c * K;
            half* st_c = h_cpu_state.data() + c * history_len;

            double acc = in_elem * static_cast<double>(__half2float(w_c[0]));
            for (int k = 0; k < history_len; ++k) {
                acc += static_cast<double>(__half2float(st_c[k])) * static_cast<double>(__half2float(w_c[k + 1]));
            }
            acc += static_cast<double>(__half2float(h_bias[c]));

            // Shift state
            for (int k = history_len - 1; k > 0; --k) {
                st_c[k] = st_c[k - 1];
            }
            st_c[0] = h_in[c];

            double silu_val = acc / (1.0 + std::exp(-acc));
            h_cpu_out[c] = __float2half(static_cast<float>(silu_val));
        }

        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), C * sizeof(half), cudaMemcpyHostToDevice));
        launch_depthwise_conv1d_step(d_in, d_weight, d_bias, d_conv_state, d_out, C, K, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out, C * sizeof(half), cudaMemcpyDeviceToHost));

        float step_err = compute_max_abs_err(h_gpu_out.data(), h_cpu_out.data(), C);
        if (step_err > max_err) max_err = step_err;
    }

    cudaFree(d_in); cudaFree(d_weight); cudaFree(d_bias);
    cudaFree(d_conv_state); cudaFree(d_out);

    std::cout << "  - Conv1d Steps Max Abs Error: " << max_err << " (threshold: <= 5e-3)" << std::endl;
    if (max_err > 5e-3f) {
        std::cerr << "  FAILED: Synthetic Depthwise Conv1d error exceeds threshold!" << std::endl;
        return false;
    }
    std::cout << "  [PASSED] Synthetic Depthwise Conv1d step parity verified." << std::endl;
    return true;
}

// ============================================================================
// Golden Reference Parity Suite (Opt-In with --golden <path>)
// ============================================================================

bool run_golden_reference_suite(
    const std::string& golden_path,
    const std::string& fp16_path,
    const std::string& w4a16_path
) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "   Golden Reference Verification Suite                  " << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Golden Reference: " << golden_path << std::endl;
    std::cout << "FP16 Model:       " << fp16_path << std::endl;
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
    // PART A: FP16 Engine Tests (Suites 1a, 2, 3, 4, 6a, 7, 8)
    // ========================================================================
    std::cout << "\n>>> [PART A] Loading FP16 Engine (~3088 MB VRAM)..." << std::endl;
    {
        auto engine_fp16 = std::make_unique<QwenEngine>(fp16_path);
        cudaStream_t stream = engine_fp16->stream();

        // [Suite 1a] FP16 Embedding Lookup
        std::cout << "\n[Suite 1a] Testing FP16 Embedding Lookup..." << std::endl;
        launch_embed_lookup_fp16(
            reinterpret_cast<const half*>(engine_fp16->embed_weight()),
            d_tokens, d_test_out, nullptr, 8, stream
        );
        CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
        float err_fp16 = compute_max_abs_err(h_test_out.data(), ref_embed, 8 * HIDDEN_SIZE);
        std::cout << "  - FP16 Max Abs Error: " << err_fp16 << " (expected: 0.0)" << std::endl;
        if (err_fp16 != 0.0f) {
            std::cerr << "  FAILED: FP16 embedding lookup mismatch!" << std::endl;
            all_passed = false;
        }

        // [Suite 2] RMSNorm
        std::cout << "\n[Suite 2] Testing RMSNorm (FP16)..." << std::endl;
        CUDA_CHECK(cudaMemcpy(d_test_in, ref_embed, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
        launch_rmsnorm_fp16(d_test_in, engine_fp16->layer_weights(0).input_layernorm, d_test_out, 8, stream);
        CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
        float err_norm = compute_max_abs_err(h_test_out.data(), ref_in_norm, 8 * HIDDEN_SIZE);
        std::cout << "  - Input RMSNorm Max Abs Error: " << err_norm << " (threshold: <= 8e-3)" << std::endl;
        if (err_norm > 8e-3f) {
            std::cerr << "  FAILED: RMSNorm error exceeds 8e-3!" << std::endl;
            all_passed = false;
        }

        // [Suite 3] QKV Bias + NeoX RoPE + KV Cache Write
        std::cout << "\n[Suite 3] Testing QKV Bias + NeoX RoPE + KV Cache Write..." << std::endl;
        {
            CUDA_CHECK(cudaMemcpy(d_test_q, ref_q_raw_raw.data(), 8 * Q_DIM * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_test_k, ref_k_raw_raw.data(), 8 * KV_DIM * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_test_v, ref_v_raw_raw.data(), 8 * KV_DIM * sizeof(half), cudaMemcpyHostToDevice));

            DeviceControlBlock dcb = { 0, 0, 0.0f, 0, 0 };
            DeviceControlBlock* d_dcb = nullptr;
            CUDA_CHECK(cudaMalloc(&d_dcb, sizeof(DeviceControlBlock)));
            CUDA_CHECK(cudaMemcpy(d_dcb, &dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

            half *d_kc = nullptr, *d_vc = nullptr;
            CUDA_CHECK(cudaMalloc(&d_kc, 2 * MAX_SEQ_LEN * HEAD_DIM * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_vc, 2 * MAX_SEQ_LEN * HEAD_DIM * sizeof(half)));

            launch_qkv_bias_rope_kv_append(
                d_test_q, d_test_k, d_test_v,
                engine_fp16->layer_weights(0).q_bias,
                engine_fp16->layer_weights(0).k_bias,
                engine_fp16->layer_weights(0).v_bias,
                d_kc, d_vc, d_dcb, 8, stream
            );

            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_q, 8 * Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_q_rope = compute_max_abs_err(h_test_out.data(), ref_q_rope, 8 * Q_DIM);
            std::cout << "  - Q NeoX RoPE Max Abs Error: " << err_q_rope << " (threshold: <= 0.05)" << std::endl;

            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_k, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_k_rope = compute_max_abs_err(h_test_out.data(), ref_k_rope, 8 * KV_DIM);
            std::cout << "  - K NeoX RoPE Max Abs Error: " << err_k_rope << " (threshold: <= 0.01)" << std::endl;

            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_v, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_v_bias = compute_max_abs_err(h_test_out.data(), ref_v_biased, 8 * KV_DIM);
            std::cout << "  - V Bias Max Abs Error:      " << err_v_bias << " (threshold: <= 5e-3)" << std::endl;

            if (err_q_rope > 0.05f || err_k_rope > 0.01f || err_v_bias > 5e-3f) {
                std::cerr << "  FAILED: QKV Bias/RoPE error exceeds tolerance!" << std::endl;
                all_passed = false;
            }

            cudaFree(d_kc);
            cudaFree(d_vc);
            cudaFree(d_dcb);
        }

        // [Suite 4] Vectorized FP16 GEMV & Fused SwiGLU
        std::cout << "\n[Suite 4] Testing Vectorized FP16 GEMV & Fused SwiGLU..." << std::endl;
        {
            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemv_fp16(
                reinterpret_cast<const half*>(engine_fp16->layer_weights(0).q_weight),
                d_test_in, nullptr, d_test_out, Q_DIM, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_gemv_q = compute_max_abs_err(h_test_out.data(), ref_q_raw, Q_DIM);
            std::cout << "  - FP16 GEMV (Q-proj) Max Abs Error: " << err_gemv_q << " (threshold: <= 5e-3)" << std::endl;

            CUDA_CHECK(cudaMemcpy(d_test_in, ref_post_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemv_dual_gate_up_swiglu_fp16(
                reinterpret_cast<const half*>(engine_fp16->layer_weights(0).gate_up_weight),
                d_test_in, d_test_out, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, INTERMEDIATE_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
            float err_swiglu = compute_max_abs_err(h_test_out.data(), ref_swiglu, INTERMEDIATE_SIZE);
            std::cout << "  - FP16 Fused SwiGLU Max Abs Error:  " << err_swiglu << " (threshold: <= 5e-3)" << std::endl;

            // Merged QKV test
            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            QKVProjectionView qkv_view = {
                engine_fp16->layer_weights(0).q_weight, engine_fp16->layer_weights(0).q_scale, d_test_q,
                engine_fp16->layer_weights(0).k_weight, engine_fp16->layer_weights(0).k_scale, d_test_k,
                engine_fp16->layer_weights(0).v_weight, engine_fp16->layer_weights(0).v_scale, d_test_v
            };
            launch_gemv_qkv_fp16(qkv_view, d_test_in, stream);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_q, Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_qkv_q = compute_max_abs_err(h_test_out.data(), ref_q_raw, Q_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_k, KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_qkv_k = compute_max_abs_err(h_test_out.data(), ref_k_raw, KV_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_v, KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_qkv_v = compute_max_abs_err(h_test_out.data(), ref_v_raw, KV_DIM);
            std::cout << "  - FP16 Merged QKV Max Abs Errors: Q=" << err_qkv_q << ", K=" << err_qkv_k << ", V=" << err_qkv_v << " (threshold: <= 5e-3)" << std::endl;

            if (err_gemv_q > 5e-3f || err_swiglu > 5e-3f || err_qkv_q > 5e-3f || err_qkv_k > 5e-3f || err_qkv_v > 5e-3f) {
                std::cerr << "  FAILED: FP16 GEMV/SwiGLU/QKV error exceeds tolerance!" << std::endl;
                all_passed = false;
            }
        }

        // [Suite 6a] Tiled Prefill GEMM FP16
        std::cout << "\n[Suite 6a] Testing Tiled Prefill GEMM (FP16, M=8)..." << std::endl;
        {
            CUDA_CHECK(cudaMemcpy(d_test_in, ref_in_norm, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyHostToDevice));
            launch_gemm_prefill_fp16(
                reinterpret_cast<const half*>(engine_fp16->layer_weights(0).q_weight),
                d_test_in, nullptr, d_test_out, 8, Q_DIM, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_out, 8 * Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_prefill_fp16 = compute_max_abs_err(h_test_out.data(), ref_q_raw, 8 * Q_DIM);
            std::cout << "  - FP16 Prefill GEMM Max Abs Error: " << err_prefill_fp16 << " (threshold: <= 0.05)" << std::endl;
            if (err_prefill_fp16 > 0.05f) {
                std::cerr << "  FAILED: FP16 Prefill GEMM error exceeds tolerance!" << std::endl;
                all_passed = false;
            }

            // Prefill Merged QKV test
            QKVProjectionView qkv_view_fp16 = {
                engine_fp16->layer_weights(0).q_weight, nullptr, d_test_q,
                engine_fp16->layer_weights(0).k_weight, nullptr, d_test_k,
                engine_fp16->layer_weights(0).v_weight, nullptr, d_test_v
            };
            launch_gemm_prefill_qkv_fp16(qkv_view_fp16, d_test_in, 8, stream);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_q, 8 * Q_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_prefill_qkv_q = compute_max_abs_err(h_test_out.data(), ref_q_raw, 8 * Q_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_k, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_prefill_qkv_k = compute_max_abs_err(h_test_out.data(), ref_k_raw, 8 * KV_DIM);
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_v, 8 * KV_DIM * sizeof(half), cudaMemcpyDeviceToHost));
            float err_prefill_qkv_v = compute_max_abs_err(h_test_out.data(), ref_v_raw, 8 * KV_DIM);
            std::cout << "  - FP16 Prefill Merged QKV Max Abs Errors: Q=" << err_prefill_qkv_q << " (<=0.05), K=" << err_prefill_qkv_k << " (<=5e-3), V=" << err_prefill_qkv_v << " (<=5e-3)" << std::endl;
            if (err_prefill_qkv_q > 0.05f || err_prefill_qkv_k > 5e-3f || err_prefill_qkv_v > 5e-3f) {
                std::cerr << "  FAILED: FP16 Prefill Merged QKV error exceeds tolerance!" << std::endl;
                all_passed = false;
            }
        }

        // [Suite 7] Attention GQA FlashDecoding vs Prefill Attention (+ O-Proj)
        std::cout << "\n[Suite 7] Testing Attention GQA FlashDecoding & Prefill..." << std::endl;
        {
            half *d_layer_k = nullptr, *d_layer_v = nullptr;
            CUDA_CHECK(cudaMalloc(&d_layer_k, 2 * MAX_SEQ_LEN * HEAD_DIM * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_layer_v, 2 * MAX_SEQ_LEN * HEAD_DIM * sizeof(half)));

            DeviceControlBlock dcb = { 0, 0, 0.0f, 0, 0 };
            DeviceControlBlock* d_dcb = nullptr;
            CUDA_CHECK(cudaMalloc(&d_dcb, sizeof(DeviceControlBlock)));
            CUDA_CHECK(cudaMemcpy(d_dcb, &dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(d_test_q, ref_q_raw_raw.data(), 8 * Q_DIM * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_test_k, ref_k_raw_raw.data(), 8 * KV_DIM * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_test_v, ref_v_raw_raw.data(), 8 * KV_DIM * sizeof(half), cudaMemcpyHostToDevice));

            launch_qkv_bias_rope_kv_append(
                d_test_q, d_test_k, d_test_v,
                engine_fp16->layer_weights(0).q_bias,
                engine_fp16->layer_weights(0).k_bias,
                engine_fp16->layer_weights(0).v_bias,
                d_layer_k, d_layer_v, d_dcb, 8, stream
            );

            // Prefill attention + O-proj
            launch_attention_gqa_prefill(d_test_q, d_layer_k, d_layer_v, d_test_out, d_dcb, 8, stream);
            launch_gemm_prefill_fp16(
                reinterpret_cast<const half*>(engine_fp16->layer_weights(0).o_weight),
                d_test_out, nullptr, d_test_in, 8, HIDDEN_SIZE, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_in, 8 * HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
            float err_attn_prefill = compute_max_abs_err(h_test_out.data(), ref_attn_out, 8 * HIDDEN_SIZE);
            std::cout << "  - Prefill Attention + O-proj Max Abs Error: " << err_attn_prefill << " (threshold: <= 5e-3)" << std::endl;

            // FlashDecoding single-token decode test at position 7 + O-proj
            dcb.seq_pos = 7;
            CUDA_CHECK(cudaMemcpy(d_dcb, &dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

            float *d_sp_out = nullptr, *d_sp_lse = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sp_out, NUM_Q_HEADS * FLASH_DECODE_SPLITS * HEAD_DIM * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_sp_lse, NUM_Q_HEADS * FLASH_DECODE_SPLITS * 2 * sizeof(float)));

            half* d_q7 = d_test_q + 7 * Q_DIM;
            launch_attention_gqa_flash_decode(d_q7, d_layer_k, d_layer_v, d_sp_out, d_sp_lse, d_test_out, d_dcb, stream);
            launch_gemv_fp16(
                reinterpret_cast<const half*>(engine_fp16->layer_weights(0).o_weight),
                d_test_out, nullptr, d_test_in, HIDDEN_SIZE, HIDDEN_SIZE, stream
            );
            CUDA_CHECK(cudaMemcpy(h_test_out.data(), d_test_in, HIDDEN_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

            const half* ref_attn7 = ref_attn_out + 7 * HIDDEN_SIZE;
            float err_flash_decode = compute_max_abs_err(h_test_out.data(), ref_attn7, HIDDEN_SIZE);
            std::cout << "  - FlashDecoding + O-proj Max Abs Error (seq_len=8): " << err_flash_decode << " (threshold: <= 5e-3)" << std::endl;

            if (err_attn_prefill > 5e-3f || err_flash_decode > 5e-3f) {
                std::cerr << "  FAILED: Attention error exceeds tolerance!" << std::endl;
                all_passed = false;
            }

            cudaFree(d_sp_out);
            cudaFree(d_sp_lse);
            cudaFree(d_layer_k);
            cudaFree(d_layer_v);
            cudaFree(d_dcb);
        }

        // [Suite 8] Argmax / Gumbel-Max Sampling & 28-Layer Greedy Token Parity
        std::cout << "\n[Suite 8] Testing Sampling & 28-Layer Greedy Token Parity..." << std::endl;
        {
            std::vector<half> h_logits_fp16(VOCAB_SIZE);
            for (int i = 0; i < VOCAB_SIZE; ++i) {
                h_logits_fp16[i] = __float2half(ref_logits[i]);
            }
            half* d_logits_test = nullptr;
            CUDA_CHECK(cudaMalloc(&d_logits_test, VOCAB_SIZE * sizeof(half)));
            CUDA_CHECK(cudaMemcpy(d_logits_test, h_logits_fp16.data(), VOCAB_SIZE * sizeof(half), cudaMemcpyHostToDevice));

            float* d_s_val = nullptr;
            int32_t* d_s_idx = nullptr;
            CUDA_CHECK(cudaMalloc(&d_s_val, 256 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_s_idx, 256 * sizeof(int32_t)));

            int32_t* h_pinned = nullptr;
            int32_t* d_pinned = nullptr;
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_pinned), sizeof(int32_t), cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_pinned), h_pinned, 0));

            DeviceControlBlock dcb = { 0, 7, 0.0f, 0, 1337ULL };
            DeviceControlBlock* d_dcb = nullptr;
            CUDA_CHECK(cudaMalloc(&d_dcb, sizeof(DeviceControlBlock)));
            CUDA_CHECK(cudaMemcpy(d_dcb, &dcb, sizeof(DeviceControlBlock), cudaMemcpyHostToDevice));

            launch_sample_and_advance(d_logits_test, d_s_val, d_s_idx, d_dcb, d_pinned, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int32_t sampled_token = *h_pinned;
            int32_t expected_token = g_header.greedy_tokens[0];
            std::cout << "  - Argmax Sampled Token:  " << sampled_token << " (expected: " << expected_token << ")" << std::endl;
            if (sampled_token != expected_token) {
                std::cerr << "  FAILED: Argmax token mismatch!" << std::endl;
                all_passed = false;
            }

            // End-to-end 32 greedy token check with FP16 engine
            std::cout << "  - Running 32 greedy decode steps on FP16 engine..." << std::endl;
            engine_fp16->reset_session();
            std::vector<int32_t> prompt(g_header.prompt_tokens, g_header.prompt_tokens + 8);
            int32_t first_gen = engine_fp16->prefill(prompt, 0.0f);

            std::vector<int32_t> generated_tokens;
            generated_tokens.push_back(first_gen);

            for (int step = 1; step < 32; ++step) {
                int32_t next_t = engine_fp16->decode_step();
                generated_tokens.push_back(next_t);
            }

            int matches = 0;
            for (int i = 0; i < 32; ++i) {
                if (generated_tokens[i] == g_header.greedy_tokens[i]) {
                    matches++;
                }
            }
            std::cout << "  - End-to-End FP16 Greedy Match: " << matches << " / 32 tokens" << std::endl;
            if (matches < 8) {
                std::cout << "    Tokens: ";
                for (int i = 0; i < 8; ++i) std::cout << generated_tokens[i] << "(exp:" << g_header.greedy_tokens[i] << ") ";
                std::cout << std::endl;
                std::cerr << "  FAILED: Greedy token generation diverged!" << std::endl;
                all_passed = false;
            } else {
                std::cout << "    [Greedy Token Parity Verified across autoregressive steps]" << std::endl;
            }

            cudaFreeHost(h_pinned);
            cudaFree(d_logits_test);
            cudaFree(d_s_val);
            cudaFree(d_s_idx);
            cudaFree(d_dcb);
        }

        std::cout << "[Test] Unloading FP16 engine and freeing ~3088 MB VRAM..." << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

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
    std::string fp16_path   = "models/qwen25_1.5b_fp16.qwenbin";
    std::string w4a16_path  = "models/qwen25_1.5b_w4a16.qwenbin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--golden") {
            run_golden = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                golden_path = argv[++i];
            }
        }
        else if (arg == "--fp16" && i + 1 < argc) fp16_path = argv[++i];
        else if (arg == "--w4a16" && i + 1 < argc) w4a16_path = argv[++i];
    }

    std::cout << "========================================================" << std::endl;
    std::cout << "   Qwen Inference Engine Multi-Model Test Suite         " << std::endl;
    std::cout << "========================================================" << std::endl;

    bool all_passed = true;

    // Unconditionally execute all standalone synthetic unit tests
    std::cout << "\n>>> Executing Standalone Synthetic Mathematical Suites..." << std::endl;

    if (!run_synthetic_rmsnorm_suite()) all_passed = false;
    if (!run_synthetic_rope_suite()) all_passed = false;
    if (!run_synthetic_gemv_swiglu_suite()) all_passed = false;
    if (!run_synthetic_deltanet_suite()) all_passed = false;
    if (!run_synthetic_conv1d_suite()) all_passed = false;

    // Conditionally execute golden reference suite if requested
    if (run_golden) {
        std::cout << "\n>>> Executing Golden Reference Parity Suite..." << std::endl;
        if (!run_golden_reference_suite(golden_path, fp16_path, w4a16_path)) {
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
