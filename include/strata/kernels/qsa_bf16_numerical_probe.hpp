// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_bf16_attention.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
struct QsaBf16NumericalProbe {
  int64_t query = 0, head = 0;
  float *scores = nullptr, *probabilities = nullptr; // cap elements each
  uint16_t *rounded = nullptr;                      // cap BF16 elements
  float *states = nullptr; // ceil(cap/32) rows: maximum, alpha, norm, tile_sum
};
// Diagnostic-only late-flags kernel with additional stores for one query/head.
// Caller initializes captures and owns all spans through completion. Capture
// outputs are registered, correctly aligned and disjoint from every input,
// attention output and each other; all checks precede submission. Entries
// beyond the selected width remain unchanged. Instrumented output must be
// compared with the original kernel before attributing an original difference.
bool qsa_bf16_numerical_probe_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    const QsaBf16NumericalProbe &probe, void *stream);
} // namespace strata::kernels
#endif
