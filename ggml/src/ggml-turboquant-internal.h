#ifndef GGML_TURBOQUANT_INTERNAL_H
#define GGML_TURBOQUANT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

void ggml_turboquant_encode_mse_scalar(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_scalar(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_scalar(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

#if defined(__x86_64__) || defined(_M_X64)
void ggml_turboquant_encode_mse_avx(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed);

void ggml_turboquant_decode_mse_avx(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed);

float ggml_turboquant_ip_f32_mse_avx(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed);

void ggml_turboquant_vec_scale_inplace_avx(float * x, int n, float scale);
#endif

#endif
