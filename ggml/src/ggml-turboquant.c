/**
 * VAL dispatcher — selects scalar vs x86 AVX2+FMA vs (future) NEON / GPU.
 * No hardware intrinsics in this file.
 */

#include "ggml-turboquant.h"
#include "ggml-turboquant-internal.h"
#include "ggml-turboquant-qjl.h"

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

static int g_backend = 0;
static int g_inited  = 0;

void ggml_turboquant_init(void) {
    if (g_inited) {
        return;
    }
    g_inited = 1;
#if defined(__x86_64__) || defined(_M_X64)
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) {
        g_backend = 1;
        return;
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    if (ggml_turboquant_neon32_available()) {
        g_backend = 2;
        return;
    }
#endif
    if (ggml_turboquant_gpu_available()) {
        g_backend = 3;
        return;
    }
    g_backend = 0;
}

int ggml_turboquant_backend_id(void) {
    ggml_turboquant_init();
    return g_backend;
}

void ggml_turboquant_vec_normalize_l2(float * x, int n) {
    ggml_turboquant_init();
#if defined(__x86_64__) || defined(_M_X64)
    if (g_backend == 1) {
        ggml_turboquant_vec_normalize_l2_avx(x, n);
        return;
    }
#endif
    ggml_turboquant_vec_normalize_l2_scalar(x, n);
}

void ggml_turboquant_encode_mse(
    const float * x, int d, int b, uint8_t * out_packed, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_init();
#if defined(__x86_64__) || defined(_M_X64)
    if (g_backend == 1) {
        ggml_turboquant_encode_mse_avx(x, d, b, out_packed, packed_cap, seed);
        return;
    }
#endif
    ggml_turboquant_encode_mse_scalar(x, d, b, out_packed, packed_cap, seed);
}

void ggml_turboquant_decode_mse(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_init();
#if defined(__x86_64__) || defined(_M_X64)
    if (g_backend == 1) {
        ggml_turboquant_decode_mse_avx(packed, d, b, out_x, seed);
        return;
    }
#endif
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    ggml_turboquant_init();
#if defined(__x86_64__) || defined(_M_X64)
    if (g_backend == 1) {
        return ggml_turboquant_ip_f32_mse_avx(q, packed, d, b, seed);
    }
#endif
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

float ggml_turboquant_kq_ip_mse(
    const float * q_row, const uint8_t * k_packed_row, int head_dim, int base_bits, uint64_t row_seed) {
    return ggml_turboquant_ip_f32_mse(q_row, k_packed_row, head_dim, base_bits, row_seed);
}

float ggml_turboquant_ip_f32_prod(
    const float * q, const uint8_t * packed_mse, int d, int b, uint64_t seed) {
    return ggml_turboquant_qjl_ip_f32(q, packed_mse, d, b, seed);
}
