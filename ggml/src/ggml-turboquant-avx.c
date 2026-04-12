/**
 * x86_64 AVX2 + FMA3 — hot loops for scaling / dequant (no AVX-512).
 * Stage A encode/decode delegate to scalar; vec_scale uses AVX2 mul for dequant-style scaling.
 */

#include "ggml-turboquant-internal.h"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

void ggml_turboquant_vec_scale_inplace_avx(float * x, int n, float scale) {
#if defined(__AVX2__) && defined(__FMA__)
    int i = 0;
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vs);
        _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= scale;
    }
#else
    for (int j = 0; j < n; j++) {
        x[j] *= scale;
    }
#endif
}

void ggml_turboquant_encode_mse_avx(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_avx(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    /* Reuse scalar reconstruction then AVX-scale the intermediate after FWHT is identical to scalar path */
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_avx(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

#else

void ggml_turboquant_encode_mse_avx(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_avx(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_avx(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

void ggml_turboquant_vec_scale_inplace_avx(float * x, int n, float scale) {
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

#endif
