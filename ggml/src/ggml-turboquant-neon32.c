/**
 * Hito 3 — ARM NEON (ARMv7-A / AArch32) TurboQuant kernels.
 * Deferred until Stage A math is locked on x86_64; VAL falls back to scalar when this returns 0.
 */

#include "ggml-turboquant-internal.h"

int ggml_turboquant_neon32_available(void) {
    return 0;
}
