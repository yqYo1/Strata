// Private prefill-only interface. No public kernel API or runtime policy here.
#pragma once
#include <cstdint>
#ifdef STRATA_SYCL_PREFILL_IQ4NL_DEQUANT
namespace strata::kernels {
// Nonaliasing, live device buffers: n positive, divisible by 256, type exactly 20.
void iq_dequant_f16_prefill_iq4nl(int type, const void* src, int64_t n,
                               uint16_t* dst, void* stream);
}
#endif
