// Strata long-prefill component. Each selected four-cell block contributes its first take cells.
#pragma once
#include <cstdint>

namespace strata::prefill::qsa_groups {

// At most one non-final block is cut by the threshold budget. The physical
// tail can also be short, but no selected block follows that tail.
constexpr uint32_t row_words(uint32_t cap) {
    return 1 + (cap == 0 ? 0 : (cap + 6) / 4);
}

// This component reserves the explicit-ID span in both arms. At cap >= 3
// the grouped rows fit in that span. The upper limit is this engine's context.
constexpr bool supports_capacity(int64_t cap) {
    return cap >= 3 && cap <= 262144;
}

// A work-item already knows its starting cell offset from the production
// ordered emission scan. Rounding up gives its descriptor offset even after a cut.
inline uint32_t emit(uint32_t* row, uint32_t offset, uint32_t block, uint32_t take, bool physical_tail) {
    row[1 + (offset + 3) / 4] = block;
    return take < 4 && !physical_tail ? ((offset + take) << 2) | (4 - take) : 0;
}

// Publish the OR of the producers' returned cut words in row[0] after emission.
// Only call with an active rank below the actual width. No expansion array.
inline uint32_t cell(const uint32_t* row, uint32_t rank) {
    const uint32_t cut = row[0];
    const uint32_t slot = rank + (rank >= (cut >> 2) ? cut & 3u : 0u);
    return row[1 + slot / 4] * 4 + slot % 4;
}

} // namespace strata::prefill::qsa_groups
