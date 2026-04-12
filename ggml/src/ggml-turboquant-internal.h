#ifndef GGML_TURBOQUANT_INTERNAL_H
#define GGML_TURBOQUANT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

void ggml_turboquant_vec_normalize_l2_scalar(float * x, int n);

void ggml_turboquant_encode_mse_scalar(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_scalar(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_scalar(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

#if defined(__x86_64__) || defined(_M_X64)
void ggml_turboquant_vec_normalize_l2_avx(float * x, int n);

void ggml_turboquant_encode_mse_avx(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_avx(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_avx(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

void ggml_turboquant_vec_scale_inplace_avx(float * x, int n, float scale);

void ggml_turboquant_vec_normalize_l2_avx512(float * x, int n);

void ggml_turboquant_encode_mse_avx512(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_avx512(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_avx512(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

void ggml_turboquant_vec_scale_inplace_avx512(float * x, int n, float scale);

void ggml_turboquant_vec_normalize_l2_sse(float * x, int n);

void ggml_turboquant_encode_mse_sse(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_sse(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_sse(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

void ggml_turboquant_vec_scale_inplace_sse(float * x, int n, float scale);
#endif

#if defined(__ARM_NEON)
void ggml_turboquant_vec_normalize_l2_neon(float * x, int n);

void ggml_turboquant_encode_mse_neon(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_neon(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_neon(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

void ggml_turboquant_vec_scale_inplace_neon(float * x, int n, float scale);
#endif

int ggml_turboquant_neon32_available(void);
int ggml_turboquant_gpu_available(void);

#endif
