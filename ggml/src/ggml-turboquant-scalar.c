/**
 * Scalar reference: FWHT + sign diagonal + Lloyd–Max codebooks (TurboQuant_mse, Stage A).
 * No intrinsics — safe fallback on all CPUs.
 */

#include "ggml-turboquant.h"
#include "ggml-turboquant-internal.h"
#include "ggml-turboquant-alloca.h"
#include "turboquant_codebooks.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

static uint64_t splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_signs(uint64_t seed, int n, int8_t * signs) {
    uint64_t st = seed ? seed : 1ULL;
    for (int i = 0; i < n; i++) {
        uint64_t r = splitmix64(&st);
        signs[i] = (r & 1) ? 1 : -1;
    }
}

void ggml_turboquant_vec_normalize_l2(float * x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += (double)x[i] * (double)x[i];
    }
    if (s <= 1e-20) {
        return;
    }
    float inv = (float)(1.0 / sqrt(s));
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

static void fwht_inplace(float * a, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = i; j < i + h; j++) {
                float u = a[j];
                float v = a[j + h];
                a[j]     = u + v;
                a[j + h] = u - v;
            }
        }
    }
}

static int tq_get_tables(int d, int b, const float ** centroids, int * nlev) {
    if (b < 1 || b > 4) {
        return -1;
    }
    if (d == 64) {
        switch (b) {
        case 1: *centroids = turboquant_centroids_d64_b1; *nlev = TURBOQUANT_NUM_LEVELS_d64_b1; return 0;
        case 2: *centroids = turboquant_centroids_d64_b2; *nlev = TURBOQUANT_NUM_LEVELS_d64_b2; return 0;
        case 3: *centroids = turboquant_centroids_d64_b3; *nlev = TURBOQUANT_NUM_LEVELS_d64_b3; return 0;
        case 4: *centroids = turboquant_centroids_d64_b4; *nlev = TURBOQUANT_NUM_LEVELS_d64_b4; return 0;
        default: return -1;
        }
    }
    if (d == 96) {
        switch (b) {
        case 1: *centroids = turboquant_centroids_d96_b1; *nlev = TURBOQUANT_NUM_LEVELS_d96_b1; return 0;
        case 2: *centroids = turboquant_centroids_d96_b2; *nlev = TURBOQUANT_NUM_LEVELS_d96_b2; return 0;
        case 3: *centroids = turboquant_centroids_d96_b3; *nlev = TURBOQUANT_NUM_LEVELS_d96_b3; return 0;
        case 4: *centroids = turboquant_centroids_d96_b4; *nlev = TURBOQUANT_NUM_LEVELS_d96_b4; return 0;
        default: return -1;
        }
    }
    if (d == 128) {
        switch (b) {
        case 1: *centroids = turboquant_centroids_d128_b1; *nlev = TURBOQUANT_NUM_LEVELS_d128_b1; return 0;
        case 2: *centroids = turboquant_centroids_d128_b2; *nlev = TURBOQUANT_NUM_LEVELS_d128_b2; return 0;
        case 3: *centroids = turboquant_centroids_d128_b3; *nlev = TURBOQUANT_NUM_LEVELS_d128_b3; return 0;
        case 4: *centroids = turboquant_centroids_d128_b4; *nlev = TURBOQUANT_NUM_LEVELS_d128_b4; return 0;
        default: return -1;
        }
    }
    if (d == 256) {
        switch (b) {
        case 1: *centroids = turboquant_centroids_d256_b1; *nlev = TURBOQUANT_NUM_LEVELS_d256_b1; return 0;
        case 2: *centroids = turboquant_centroids_d256_b2; *nlev = TURBOQUANT_NUM_LEVELS_d256_b2; return 0;
        case 3: *centroids = turboquant_centroids_d256_b3; *nlev = TURBOQUANT_NUM_LEVELS_d256_b3; return 0;
        case 4: *centroids = turboquant_centroids_d256_b4; *nlev = TURBOQUANT_NUM_LEVELS_d256_b4; return 0;
        default: return -1;
        }
    }
    return -1;
}

static int nearest_centroid(float y, const float * c, int n) {
    int best = 0;
    float bestd = fabsf(y - c[0]);
    for (int i = 1; i < n; i++) {
        float t = fabsf(y - c[i]);
        if (t < bestd) {
            bestd = t;
            best = i;
        }
    }
    return best;
}

static size_t packed_bytes(int d, int b) {
    return (size_t)((d * b + 7) / 8);
}

static void pack_indices(const int * idx, int d, int b, uint8_t * out) {
    memset(out, 0, packed_bytes(d, b));
    for (int j = 0; j < d; j++) {
        unsigned bit0 = (unsigned)(j * b);
        unsigned v = (unsigned)idx[j];
        for (unsigned k = 0; k < (unsigned)b; k++) {
            if (v & (1u << k)) {
                unsigned byte = (bit0 + k) >> 3;
                unsigned bit  = (bit0 + k) & 7u;
                out[byte] |= (uint8_t)(1u << bit);
            }
        }
    }
}

static void unpack_indices(const uint8_t * in, int d, int b, int * idx) {
    for (int j = 0; j < d; j++) {
        unsigned bit0 = (unsigned)(j * b);
        unsigned v = 0;
        for (unsigned k = 0; k < (unsigned)b; k++) {
            unsigned byte = (bit0 + k) >> 3;
            unsigned bit  = (bit0 + k) & 7u;
            if (in[byte] & (1u << bit)) {
                v |= (1u << k);
            }
        }
        idx[j] = (int)v;
    }
}

void ggml_turboquant_encode_mse_scalar(
    const float * x, int d, int b, uint8_t * out, size_t packed_cap, uint64_t seed) {
    const float * cent = NULL;
    int nlev = 0;
    if (tq_get_tables(d, b, &cent, &nlev) != 0) {
        memset(out, 0, packed_cap);
        return;
    }
    size_t need = packed_bytes(d, b);
    if (packed_cap < need) {
        return;
    }
    if ((d & (d - 1)) != 0) {
        return;
    }

    float * w = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    memcpy(w, x, (size_t)d * sizeof(float));
    ggml_turboquant_vec_normalize_l2(w, d);

    int8_t * signs = (int8_t *)ggml_tq_alloca((size_t)d);
    rng_signs(seed, d, signs);
    for (int i = 0; i < d; i++) {
        w[i] *= (float)signs[i];
    }

    fwht_inplace(w, d);
    float inv_sqrt_n = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        w[i] *= inv_sqrt_n;
    }

    int * idx = (int *)ggml_tq_alloca((size_t)d * sizeof(int));
    for (int j = 0; j < d; j++) {
        idx[j] = nearest_centroid(w[j], cent, nlev);
    }
    pack_indices(idx, d, b, out);
}

void ggml_turboquant_decode_mse_scalar(
    const uint8_t * packed, int d, int b, float * out_x, uint64_t seed) {
    const float * cent = NULL;
    int nlev = 0;
    if (tq_get_tables(d, b, &cent, &nlev) != 0) {
        memset(out_x, 0, (size_t)d * sizeof(float));
        return;
    }
    if ((d & (d - 1)) != 0) {
        memset(out_x, 0, (size_t)d * sizeof(float));
        return;
    }

    int * idx = (int *)ggml_tq_alloca((size_t)d * sizeof(int));
    unpack_indices(packed, d, b, idx);

    float * w = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    for (int j = 0; j < d; j++) {
        w[j] = cent[idx[j]];
    }

    fwht_inplace(w, d);
    float inv_sqrt_n = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) {
        w[i] *= inv_sqrt_n;
    }

    int8_t * signs = (int8_t *)ggml_tq_alloca((size_t)d);
    rng_signs(seed, d, signs);
    for (int i = 0; i < d; i++) {
        out_x[i] = w[i] * (float)signs[i];
    }
}

float ggml_turboquant_ip_f32_mse_scalar(
    const float * q, const uint8_t * packed, int d, int b, uint64_t seed) {
    float * xh = (float *)ggml_tq_alloca((size_t)d * sizeof(float));
    ggml_turboquant_decode_mse_scalar(packed, d, b, xh, seed);
    float s = 0.f;
    for (int i = 0; i < d; i++) {
        s += q[i] * xh[i];
    }
    return s;
}
