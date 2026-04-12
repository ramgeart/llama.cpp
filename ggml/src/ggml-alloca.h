/**
 * Portable stack allocation for glibc, musl (Alpine), BSD, MSVC.
 * Prefer __builtin_alloca on GCC/Clang so <alloca.h> is not required on musl.
 */
#ifndef GGML_ALLOCA_H
#define GGML_ALLOCA_H

#include <stddef.h>

#if defined(_WIN32)
#include <malloc.h>
#ifndef alloca
#define alloca _alloca
#endif
#define ggml_alloca(n) alloca(n)
#elif defined(__GNUC__) || defined(__clang__)
#define ggml_alloca(n) __builtin_alloca(n)
#else
#include <alloca.h>
#define ggml_alloca(n) alloca(n)
#endif

#endif /* GGML_ALLOCA_H */
