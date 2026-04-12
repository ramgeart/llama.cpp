#ifndef GGML_TURBOQUANT_QJL_H
#define GGML_TURBOQUANT_QJL_H

#include <stddef.h>
#include <stdint.h>

/**
 * Stage B (TurboQuant_prod + QJL). Disabled by default.
 * Policy: if dense S causes >50ms/token extra latency, keep disabled (see qjl.c).
 */
float ggml_turboquant_qjl_ip_f32(
    const float * q, const uint8_t * packed_mse, int d, int b, uint64_t seed);

#endif
