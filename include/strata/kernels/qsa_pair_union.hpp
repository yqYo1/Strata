// Main-authored Strata experiment; repository LICENSE (MIT) applies.
#pragma once
#include <cstdint>

namespace strata::kernels {
// Experimental GPU metadata producer; no inference caller/build target.
// Each input row has cap slots, of which steps[kStepWidth] are initialized.
// Per pair p, outputs use stride 2*cap. Only counts[p] entries are initialized.
// IDs are a sorted unique union; membership bits 0/1 identify the two queries.
// Steps use fixed four-cell index blocks: NBid == floor(NKv/4), even for an
// empty selection. Width zero is legal for positive NKv; no IDs are read.
// An absent odd final row has width zero. Invalid steps/IDs produce count=-1
// without publishing a payload. Output storage must not alias any input.
// Caller establishes input dependencies on this in-order queue and retains
// all allocations until completion. Other queues need explicit dependencies.
// The 262144 geometry limit does not prove that backing allocations fit.
// IDs/steps/unions/counts require4-byte alignment;membership needs only1.
void qsa_pair_union_prefill(const int32_t *ids, const int32_t *steps,
    int64_t queries, int64_t cap, int32_t *unions, uint8_t *membership,
    int32_t *counts, void *stream);
} // namespace strata::kernels
