/**
 * x86_64 AVX-512 TU (same 256-bit ops as AVX2; loops for dot / normalize / scale (no AVX-512).
 * FWHT + Lloyd–Max encode/decode remain scalar (Stage A); inner product uses FMA dot.
 */

#include "ggml-turboquant-internal.h"

#include "ggml-turboquant-alloca.h"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#include <math.h>

static float ggml_tq_hsum256_ps(__m256 x) {
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    const __m128 lo = _mm256_castps256_ps128(x);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/** <a,b> with FMA-friendly mul-add (256-bit; TU compiled with -mavx512f). */
static float ggml_turboquant_dot_f32_avx512(const float * a, const float * b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float s = ggml_tq_hsum256_ps(acc);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

void ggml_turboquant_vec_scale_inplace_avx512(float * x, int n, float scale) {
    int i = 0;
    const __m256 vs = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vs);
        _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= scale;
    }
}

void ggml_turboquant_vec_normalize_l2_avx512(float * x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    double sum = (double)ggml_tq_hsum256_ps(acc);
    for (; i < n; i++) {
        sum += (double)x[i] * (double)x[i];
    }
    if (sum <= 1e-20) {
        return;
    }
    float inv = (float)(1.0 / sqrt(sum));
    ggml_turboquant_vec_scale_inplace_avx512(x, n, inv);
}

void ggml_turboquant_encode_mse_avx512(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_avx512(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_avx512(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    float * xh = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    ggml_turboquant_decode_mse_scalar(packed, d, b, xh, seed);
    return ggml_turboquant_dot_f32_avx512(q, xh, d);
}

#else /* x86_64 without AVX2 at compile time */

void ggml_turboquant_vec_normalize_l2_avx512(float * x, int n) {
    ggml_turboquant_vec_normalize_l2_scalar(x, n);
}

void ggml_turboquant_vec_scale_inplace_avx512(float * x, int n, float scale) {
    for (int j = 0; j < n; j++) {
        x[j] *= scale;
    }
}

void ggml_turboquant_encode_mse_avx512(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_avx512(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_avx512(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

#endif /* AVX2+FMA for avx512 TU */

#else /* !x86_64 */

void ggml_turboquant_encode_mse_avx512(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_avx512(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_avx512(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

void ggml_turboquant_vec_scale_inplace_avx512(float * x, int n, float scale) {
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

void ggml_turboquant_vec_normalize_l2_avx512(float * x, int n) {
    ggml_turboquant_vec_normalize_l2_scalar(x, n);
}

#endif
