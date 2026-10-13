// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
namespace strata::core::layer_bulk {
constexpr size_t slots = 512;
using Offsets = std::array<uint64_t, slots + 1>;
// Immutable reserved offsets describe physical ownership; only this active view changes.
inline bool compact(const uint64_t* reserved, size_t count, uint64_t physical_bytes,
                    uint64_t stride, bool readers_drained, Offsets& active) {
    if (!readers_drained || !reserved || count != slots || !stride || stride % 256 ||
        stride > std::numeric_limits<uint64_t>::max() / count || reserved[0] != 0 ||
        reserved[count] != physical_bytes || stride * count > physical_bytes) return false;
    for (size_t e = 0; e < count; ++e)
        if (reserved[e + 1] < reserved[e] || stride > reserved[e + 1] - reserved[e]) return false;
    for (size_t e = 0; e < active.size(); ++e) active[e] = uint64_t(e) * stride;
    return true;
}
inline bool valid_span(const void* data, uint64_t bytes, uint64_t stride, int64_t experts,
                       const void* owner, const void* expected_owner, uint64_t expected_stride) {
    return bytes <= std::numeric_limits<size_t>::max() && data && owner && owner == expected_owner &&
           experts == 512 && stride && stride % 256 == 0 &&
           stride == expected_stride && stride <= std::numeric_limits<uint64_t>::max() / 512 && bytes == stride * 512;
}
// A failed checked copy cannot publish or enter a retry/fallback path.
template<class Copy, class Publish> bool complete_copy(Copy&& copy, Publish&& publish) {
    if (!copy()) return false;
    publish();
    return true;
}
}
