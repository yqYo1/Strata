// Port-only prompt selector: the same integer radix threshold and ascending tie order.
#pragma once
#include "strata/kernels/qsa_select.hpp"
namespace strata::kernels {
// False means no work was launched; the caller keeps its original selector.
// The live bound must cover every query's n_bid + 1. Without it, capacity is used.
bool qsa_block_topk_prompt_variant(const float *scores, const int32_t *steps, int64_t nq, int64_t max_blocks,
                                   int64_t cap, const QsaShapes &s, int32_t *ids, void *stream,
                                   int64_t active_blocks = -1);
} // namespace strata::kernels
