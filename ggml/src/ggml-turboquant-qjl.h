#ifndef GGML_TURBOQUANT_QJL_H
#define GGML_TURBOQUANT_QJL_H

#include <stdint.h>

/**
 * TurboQuant_prod (Algorithm 2) — QJL with dense Gaussian S.
 * Build-time default: off. Enable with -DGGML_TURBOQUANT_ENABLE_QJL=1 only after Stage A is validated.
 * Abort policy (reference x86_64): if QJL adds >50 ms/token latency vs MSE-only, disable in production and document.
 */
#ifndef GGML_TURBOQUANT_ENABLE_QJL
#define GGML_TURBOQUANT_ENABLE_QJL 0
#endif

float ggml_turboquant_qjl_ip_f32(
    const float * q, const uint8_t * packed_mse, int d, int b, uint64_t seed);

#endif
