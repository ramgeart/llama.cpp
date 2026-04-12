/**
 * x86_64 SSE4.1 — float dot / normalize / scale (128-bit); no AVX2.
 */

#include "ggml-turboquant-internal.h"

#include "ggml-turboquant-alloca.h"

#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__SSE4_1__) && defined(__SSSE3__)
#include <immintrin.h>

static float ggml_tq_hsum128_ps(__m128 x) {
    __m128 t = _mm_add_ps(x, _mm_movehl_ps(x, x));
    t        = _mm_add_ss(t, _mm_shuffle_ps(t, t, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtss_f32(t);
}

static float ggml_turboquant_dot_f32_sse(const float * a, const float * b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i      = 0;
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc       = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float s = ggml_tq_hsum128_ps(acc);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

void ggml_turboquant_vec_scale_inplace_sse(float * x, int n, float scale) {
    int i           = 0;
    const __m128 vs = _mm_set1_ps(scale);
    for (; i + 3 < n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        vx        = _mm_mul_ps(vx, vs);
        _mm_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= scale;
    }
}

void ggml_turboquant_vec_normalize_l2_sse(float * x, int n) {
    __m128 acc = _mm_setzero_ps();
    int i        = 0;
    for (; i + 3 < n; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        acc       = _mm_add_ps(acc, _mm_mul_ps(vx, vx));
    }
    double sum = (double)ggml_tq_hsum128_ps(acc);
    for (; i < n; i++) {
        sum += (double)x[i] * (double)x[i];
    }
    if (sum <= 1e-20) {
        return;
    }
    float inv = (float)(1.0 / sqrt(sum));
    ggml_turboquant_vec_scale_inplace_sse(x, n, inv);
}

void ggml_turboquant_encode_mse_sse(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_sse(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_sse(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    float * xh = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    ggml_turboquant_decode_mse_scalar(packed, d, b, xh, seed);
    return ggml_turboquant_dot_f32_sse(q, xh, d);
}

#else /* no SSE4.1 at compile time */

void ggml_turboquant_vec_normalize_l2_sse(float * x, int n) {
    ggml_turboquant_vec_normalize_l2_scalar(x, n);
}

void ggml_turboquant_vec_scale_inplace_sse(float * x, int n, float scale) {
    for (int j = 0; j < n; j++) {
        x[j] *= scale;
    }
}

void ggml_turboquant_encode_mse_sse(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_sse(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_sse(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

#endif

#else /* !x86_64 */

void ggml_turboquant_encode_mse_sse(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_sse(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_sse(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    return ggml_turboquant_ip_f32_mse_scalar(q, packed, d, b, seed);
}

void ggml_turboquant_vec_scale_inplace_sse(float * x, int n, float scale) {
    for (int j = 0; j < n; j++) {
        x[j] *= scale;
    }
}

void ggml_turboquant_vec_normalize_l2_sse(float * x, int n) {
    ggml_turboquant_vec_normalize_l2_scalar(x, n);
}

#endif
