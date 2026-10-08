// Keep upstream arithmetic intact; the optional GCC object supplies only the
// IQ2_S gate/up groups that measured faster on Ryzen 5 5600X.
#if defined(STRATA_IQ2S_GCC)
#define iq256_gu_rows iq256_gu_rows_icpx
#endif
#include "../../../../src/kernels/cpu/iq_avx2.cpp"

#if defined(STRATA_IQ2S_GCC)
#undef iq256_gu_rows
namespace strata::kernels::cpu {
void iq256_gu_rows_gcc(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                      const void* const* act, int nt, float* const* ff, int r0, int r1);

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                  const void* const* act, int nt, float* const* ff, int r0, int r1) {
    if (type == 22 && (nt == 2 || nt == 4))
        iq256_gu_rows_gcc(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
    else
        iq256_gu_rows_icpx(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}
}  // namespace strata::kernels::cpu
#endif
