// Independently authored for Strata. Repository LICENSE applies.
#pragma once

#include "strata/kernels/qsa_bf16_prefix.hpp"

namespace strata::kernels {
using PrefillVTileAttention = decltype(qsa_bf16_prefix_ordered_exp_batch);
using PrefillVTileState = bool(const uint16_t *, const QsaAttnPools &,
    const int32_t *, const int32_t *, int64_t, const QsaShapes &,
    uint16_t *, int64_t, const QsaPromptBf16Bounds &, int64_t,
    const QsaBf16PackedScratch &, const QsaPrefixStateProbe &, void *);

namespace separate_late_v {
PrefillVTileAttention qsa_bf16_prefix_ordered_exp_batch;
PrefillVTileState qsa_bf16_prefix_v_tile_state_batch;
}
namespace separate_early_v {
PrefillVTileAttention qsa_bf16_prefix_ordered_exp_batch;
PrefillVTileState qsa_bf16_prefix_v_tile_state_batch;
}
}
