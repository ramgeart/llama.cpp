/**
 * Stack allocation helper — glibc, musl, and MSVC-friendly.
 * musl: avoids relying on <alloca.h> visibility; GCC/Clang use __builtin_alloca.
 */
#ifndef GGML_TURBOQUANT_ALLOCA_H
#define GGML_TURBOQUANT_ALLOCA_H

#include <stddef.h>

#if defined(_WIN32)
#include <malloc.h>
#ifndef alloca
#define alloca _alloca
#endif
#define ggml_tq_alloca(n) alloca(n)
#elif defined(__GNUC__) || defined(__clang__)
#define ggml_tq_alloca(n) __builtin_alloca(n)
#else
#include <alloca.h>
#define ggml_tq_alloca(n) alloca(n)
#endif

#endif /* GGML_TURBOQUANT_ALLOCA_H */
