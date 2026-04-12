/**
 * ARM NEON — aarch64 and ARMv7-A: float dot / normalize / scale for TurboQuant VAL.
 * Encode/decode stay scalar (Stage A).
 */

#include "ggml-turboquant-internal.h"

#include "ggml-turboquant-alloca.h"

#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>

static float ggml_tq_dot_f32_neon(const float * a, const float * b, int n) {
    float32x4_t acc = vdupq_n_f32(0.f);
    int i           = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
#if defined(__aarch64__)
        acc = vfmaq_f32(acc, va, vb);
#else
        acc = vmlaq_f32(acc, va, vb);
#endif
    }
#if defined(__aarch64__)
    float s = vaddvq_f32(acc);
#else
    float32x2_t r2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float s          = vget_lane_f32(vpadd_f32(r2, r2), 0);
#endif
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

void ggml_turboquant_vec_scale_inplace_neon(float * x, int n, float scale) {
    float32x4_t vs = vdupq_n_f32(scale);
    int i          = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vx             = vmulq_f32(vx, vs);
        vst1q_f32(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= scale;
    }
}

void ggml_turboquant_vec_normalize_l2_neon(float * x, int n) {
    float32x4_t acc = vdupq_n_f32(0.f);
    int i           = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
#if defined(__aarch64__)
        acc = vfmaq_f32(acc, vx, vx);
#else
        acc = vmlaq_f32(acc, vx, vx);
#endif
    }
#if defined(__aarch64__)
    double sum = (double)vaddvq_f32(acc);
#else
    float32x2_t r2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    double sum     = (double)vget_lane_f32(vpadd_f32(r2, r2), 0);
#endif
    for (; i < n; i++) {
        sum += (double)x[i] * (double)x[i];
    }
    if (sum <= 1e-20) {
        return;
    }
    float inv = (float)(1.0 / sqrt(sum));
    ggml_turboquant_vec_scale_inplace_neon(x, n, inv);
}

void ggml_turboquant_encode_mse_neon(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    ggml_turboquant_encode_mse_scalar(x, d, b, out, packed_cap, seed);
}

void ggml_turboquant_decode_mse_neon(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    ggml_turboquant_decode_mse_scalar(packed, d, b, out_x, seed);
}

float ggml_turboquant_ip_f32_mse_neon(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    float * xh = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    ggml_turboquant_decode_mse_scalar(packed, d, b, xh, seed);
    return ggml_tq_dot_f32_neon(q, xh, d);
}

int ggml_turboquant_neon32_available(void) {
    return 1;
}

#else

int ggml_turboquant_neon32_available(void) {
    return 0;
}

#endif
