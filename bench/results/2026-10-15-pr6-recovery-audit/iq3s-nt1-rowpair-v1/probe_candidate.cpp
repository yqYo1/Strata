// Standalone CPU probe using exactly the snippets proposed for iq_single_avx2.cpp.
// GGML-derived code: MIT, see third_party/ggml/LICENSE.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include <cmath>
#if defined(__clang__) && (defined(__znver3__) || defined(__znver3))
#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <string.h>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#define GGML_CPU_FP16_TO_FP32(h) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(h))))
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
namespace strata::kernels::cpu {
namespace {
#include "rowpair_kernel.inc"
}
#include "rowpair_public.inc"
// Probe-only access to raw gate/up dots; never part of the production API.
void iq3s_probe_raw(const uint8_t* row, size_t stride, const void* act, float* out, int count) {
    if (count == 1) iq3s_pair_dot<1>(row, stride, act, out);
    else iq3s_pair_dot<2>(row, stride, act, out);
}
}
#else
namespace strata::kernels::cpu {
bool iq256_iq3s_rowpair_gu_rows(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    return false;
}
void iq3s_probe_raw(const uint8_t*, size_t, const void*, float*, int) {}
}
#endif
