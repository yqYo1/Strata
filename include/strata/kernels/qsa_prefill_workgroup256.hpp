// Authored for Strata; repository LICENSE applies.
#pragma once
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include <array>
#include <cstdint>
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels::prefill {
struct QsaPrefillWorkgroup256Resources {
  std::uint64_t maximum, device_local, reported_private;
};
// Diagnostic only. Resolve IDs beside their kernel submissions: DPC++ cross-TU
// lookup fails in the installed runtime. No enqueue or timed-path work here.
std::array<QsaPrefillWorkgroup256Resources,2>
prefill_workgroup256_kernel_resources(void *stream);
// Component-only long-prefill candidate. The independent128-thread production
// export stays unchanged. Admission,rounding,32-cell tiles and ordered FP32
// state match it;SG16 ownership changes coherently with the256-thread workgroup.
bool qsa_bf16_prefix_workgroup256_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Numerical fixture only;additional state stores never enter speed batches.
bool qsa_bf16_prefix_workgroup256_state_probe_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch,
    const QsaPrefixStateProbe &probe, void *stream);
} // namespace strata::kernels::prefill
#endif
