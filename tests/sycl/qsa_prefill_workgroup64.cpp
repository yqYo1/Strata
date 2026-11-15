// Main-authored adapter; repository LICENSE applies.
// Use the existing independent oracle/guard/full-component cases unchanged.
// Include declarations before remapping only the fixture's tested entrypoint.
#include "strata/kernels/qsa_prefill_workgroup64.hpp"
namespace {
bool compare_prefill_state(const uint16_t *query, const strata::kernels::QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const strata::kernels::QsaShapes &shape,
    uint16_t *output, int64_t queries, const strata::kernels::QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const strata::kernels::QsaBf16PackedScratch &scratch,
    bool candidate, const strata::kernels::QsaPrefixStateProbe &probe, void *stream) {
  if (candidate)
    return strata::kernels::prefill::qsa_bf16_prefix_workgroup64_state_probe_batch(
        query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,probe,stream);
  return strata::kernels::qsa_bf16_prefix_state_probe_batch(
      query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,true,probe,stream);
}
}
#define qsa_bf16_prefix_batch strata::kernels::prefill::qsa_bf16_prefix_workgroup64_batch
#define qsa_bf16_prefix_state_probe_batch compare_prefill_state
#define STRATA_QSA_COMPARE_PREFIX33 1
#define STRATA_QSA_COMPARE_PREFIX_STATE 1
// This prefix diagnostic captures all16padded heads;the real-head flag only
// expands numerical coverage to all12real heads. Invalid coordinate remains16.
#define STRATA_QSA_PADDED_STATE_PROBE 1
#include "qsa_bf16_prefix.cpp"
