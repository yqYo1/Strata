// The upstream IQ4_NL down arithmetic, with read-only activation scales
// converted once per row group. No weight, product or FMA order is changed.
#include "strata/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cstdlib>
#include <cstring>
#include <immintrin.h>

namespace strata::kernels::cpu {
void iq4nl256_down_rows_icpx(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

template <int NT>
void down_rows(const uint8_t* w, const block_q8_0* const* y,
               float* const* out, int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    constexpr int nb = 640 / QK4_NL;
    // The native pack's 640-wide Q8_0 activation is immutable during down.
    // Only these 20 scales are needed; no conversion or length test per row.
    float scales[NT][nb];
    for (int t = 0; t < NT; ++t)
        for (int ib = 0; ib < nb; ++ib)
            scales[t][ib] = h2f(y[t][ib].d);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * (nb * sizeof(block_iq4_nl));
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            if (prefetch_ahead > 0) {
                _mm_prefetch((const char*) blk + prefetch_ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) blk + prefetch_ahead + 64, _MM_HINT_T0);
            }
            const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                      _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4);
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
                const float dy = scales[t][ib];
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * dy), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum8(accf[t]);
    }
}

}  // namespace

void iq4nl256_down_rows_scales(const uint8_t* w, size_t row_bytes, int n,
                              const void* const* hq, int nt, float* const* out, int r0, int r1) {
    if (n != 640 || row_bytes != 20 * sizeof(block_iq4_nl)) {
        iq4nl256_down_rows_icpx(w, row_bytes, n, hq, nt, out, r0, r1);
        return;
    }
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
        case 1: down_rows<1>(w, y, out, r0, r1); break;
        case 2: down_rows<2>(w, y, out, r0, r1); break;
        case 3: down_rows<3>(w, y, out, r0, r1); break;
        case 4: down_rows<4>(w, y, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            iq4nl256_down_rows_scales(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
        }
    }
}

}  // namespace strata::kernels::cpu
