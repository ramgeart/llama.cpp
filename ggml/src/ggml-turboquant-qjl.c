/**
 * TurboQuant_prod / QJL (Stage B). When GGML_TURBOQUANT_ENABLE_QJL=0 (default), same IP as Stage A.
 */

#include "ggml-turboquant-qjl.h"
#include "ggml-turboquant-internal.h"

float ggml_turboquant_qjl_ip_f32(
    const float * q, const uint8_t * packed_mse, int d, int b, uint64_t seed) {
#if GGML_TURBOQUANT_ENABLE_QJL
    /* TODO: unbiased TurboQuant_prod IP (dense S / QJL). Keep latency < 50 ms/token vs MSE baseline. */
    return ggml_turboquant_ip_f32_mse_scalar(q, packed_mse, d, b, seed);
#else
    return ggml_turboquant_ip_f32_mse_scalar(q, packed_mse, d, b, seed);
#endif
}
