// SYCL prompt recurrence variant for the existing Qwen GDN shapes.
#pragma once
#include <cstdint>

namespace strata::prefill {
// Original key-head arithmetic with SG16 and 256 GRFs. Returns false without
// launching for fewer than 256 tokens; the caller keeps the original recurrence.
bool gdn_recurrence_keyhead_variant(float *state, const float *h, const float *gate, const float *beta, const float *z,
                                    const float *gamma, float eps, float *y, uint16_t *y16, int64_t T, void *stream);
} // namespace strata::prefill
