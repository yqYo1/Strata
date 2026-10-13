// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::prefill::workspace {
// Host-only production accounting: retain Alloc's extra 256 bytes for aligned payloads.
inline uint64_t take_bytes(uint64_t elements, uint64_t element_bytes) {
    constexpr uint64_t limit = std::numeric_limits<uint64_t>::max();
    if (!element_bytes || elements > (limit - 511) / element_bytes)
        throw std::overflow_error("prefill: aligned take overflow");
    return (elements * element_bytes + 511) & ~uint64_t{255};
}
inline size_t product(size_t rows, size_t stride) {
    if (stride && rows > std::numeric_limits<size_t>::max() / stride)
        throw std::overflow_error("prefill: layout product overflow");
    return rows * stride;
}
inline size_t extent_size(uint64_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error("prefill: region extent overflow");
    return static_cast<size_t>(bytes);
}
// Used by count-only, owned and bounded borrowed takes. Failure does not advance.
inline bool advance(uint64_t bytes, uint64_t& used, bool bounded, uint64_t capacity, uint64_t& offset) {
    if (bytes > std::numeric_limits<uint64_t>::max() - used)
        throw std::overflow_error("prefill: allocation count overflow");
    if (bounded && (used > capacity || bytes > capacity - used)) return false;
    offset = used;
    used += bytes;
    return true;
}
inline bool exact_one(const char* value) { return value && value[0] == '1' && value[1] == '\0'; }
struct RegionLayout {
    uint64_t block_max = 0, gate_extent = 0, region_extent = 0, allocated_region = 0;
    uint64_t saved_requested = 0;
    const char* block_winner = nullptr;
    const char* region_winner = nullptr;
};
inline RegionLayout region_layout(uint64_t gdn, uint64_t qsa, uint64_t moe,
                                  size_t gate_elements, bool alias) {
    RegionLayout result;
    result.block_max = std::max({gdn, qsa, moe});
    result.block_winner = gdn == result.block_max ? "gdn" : qsa == result.block_max ? "qsa" : "moe";
    result.gate_extent = take_bytes(gate_elements, sizeof(float));
    result.region_extent = alias ? std::max(result.block_max, result.gate_extent) : result.block_max;
    result.region_winner = alias && result.gate_extent > result.block_max ? "gated" : result.block_winner;
    extent_size(result.region_extent);
    result.allocated_region = take_bytes(result.region_extent, 1);
    const uint64_t old_region = take_bytes(result.block_max, 1);
    // old gate+region minus new region, without overflowing the old sum.
    if (alias) result.saved_requested = result.gate_extent - (result.allocated_region - old_region);
    return result;
}
inline float* gate_view(void* base, const RegionLayout& layout) {
    uint64_t used = 0, offset = 0;
    if (!base || !advance(layout.gate_extent, used, true, layout.region_extent, offset))
        throw std::runtime_error("prefill: gated view does not fit shared region");
    return static_cast<float*>(base);
}
} // namespace strata::prefill::workspace
