#pragma once
// New project code, repository LICENSE. No external implementation reference.
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace strata::core::detail {
// No field escapes until every allocation is nonnull and initialized. Allocation
// failure (null or exception) releases every preceding field exactly once.
template<size_t N, class Allocate, class Free>
std::array<void*, N> allocate_zeroed_fields(const std::array<size_t, N>& sizes,
                                          Allocate allocate, Free release) noexcept {
    std::array<void*, N> fields{};
    bool ok = true;
    try {
        for (size_t i = 0; i < N; ++i) {
            fields[i] = allocate(sizes[i]);
            if (!fields[i]) { ok = false; break; }
            std::memset(fields[i], 0, sizes[i]);
        }
    } catch (...) { ok = false; }
    if (!ok) {
        for (void*& field : fields) if (field) {
            try { release(field); }
            catch (...) { std::_Exit(88); } // cleanup failure cannot report success
            field = nullptr;
        }
    }
    return fields;
}
} // namespace strata::core::detail
