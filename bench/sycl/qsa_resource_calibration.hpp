// Main-authored diagnostic; repository LICENSE applies. No engine entrypoint.
#pragma once
#include "strata/kernels/qsa_bf16_prefix.hpp"

namespace strata::kernels::resource_calibration {
enum class Arm { control, load, stage };
bool run(Arm arm, const uint16_t *query, const QsaAttnPools &pools,
         const int32_t *ids, const int32_t *steps, int64_t cap,
         const QsaShapes &shape, uint16_t *output, int64_t queries,
         const QsaPromptBf16Bounds &bounds, int64_t live_cells,
         const QsaBf16PackedScratch &scratch, void *stream, uint32_t *sink);
}
