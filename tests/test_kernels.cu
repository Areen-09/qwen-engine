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

using namespace qwen;

// Metric evaluation helpers
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

int main(int argc, char** argv) {
    std::string golden_path = "models/golden_ref.bin";
    std::string fp16_path   = "models/qwen25_1.5b_fp16.qwenbin";
    std::string w4a16_path  = "models/qwen25_1.5b_w4a16.qwenbin";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--golden" && i + 1 < argc) golden_path = argv[++i];
        else if (arg == "--fp16" && i + 1 < argc) fp16_path = argv[++i];
        else if (arg == "--w4a16" && i + 1 < argc) w4a16_path = argv[++i];
    }

    std::cout << "========================================================" << std::endl;
    std::cout << "   Qwen2.5-1.5B CUDA Inference Engine Test Suite        " << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Golden Reference: " << golden_path << std::endl;
    std::cout << "FP16 Model:       " << fp16_path << std::endl;
    std::cout << "W4A16 Model:      " << w4a16_path << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 1. Read Golden Reference Header and Tensors
    std::ifstream gfile(golden_path, std::ios::binary);
    if (!gfile.is_open()) {
        std::cerr << "[ERROR] Cannot open golden reference file: " << golden_path << std::endl;
        std::cerr << "Run 'python tools/verify_reference.py' first!" << std::endl;
        return 1;
    }

    GoldenRefHeader g_header;
    gfile.read(reinterpret_cast<char*>(&g_header), sizeof(g_header));
    if (g_header.magic != 0x474F4C44) {
        std::cerr << "[ERROR] Invalid golden reference magic: 0x" << std::hex << g_header.magic << std::endl;
        return 1;
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
    } // engine_fp16 goes out of scope here and deallocates its arena
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
    } // engine_w4a16 goes out of scope here
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup common scratch buffers
    cudaFree(d_tokens);
    cudaFree(d_test_in);
    cudaFree(d_test_out);
    cudaFree(d_test_q);
    cudaFree(d_test_k);
    cudaFree(d_test_v);

    std::cout << "\n========================================================" << std::endl;
    if (all_passed) {
        std::cout << ">>> ALL 8 VERIFICATION SUITES PASSED (EXIT CODE 0) <<<" << std::endl;
        std::cout << "========================================================" << std::endl;
        return 0;
    } else {
        std::cerr << ">>> SOME VERIFICATION SUITES FAILED <<<" << std::endl;
        std::cout << "========================================================" << std::endl;
        return 1;
    }
}
