/**
 * TurboQuant_prod / QJL (Stage B) — placeholder.
 * Default: disabled. Full dense-Gaussian S path is deferred; enable only after Stage A
 * validation and latency budget checks (50 ms/token abort policy).
 */

#include "ggml-turboquant-qjl.h"
#include "ggml-turboquant-internal.h"

float ggml_turboquant_qjl_ip_f32(
    const float * q, const uint8_t * packed_mse, int d, int b, uint64_t seed) {
    (void)q;
    (void)packed_mse;
    (void)d;
    (void)b;
    (void)seed;
    /* Fallback: unbiased IP not available without QJL; return MSE-based IP for debugging */
    return ggml_turboquant_ip_f32_mse_scalar(q, packed_mse, d, b, seed);
}
