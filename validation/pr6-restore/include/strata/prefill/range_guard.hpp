#pragma once
// New project file, repository LICENSE. No external implementation reference.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {
// Check by subtraction before forming pos + n. Allocation may include reserved
// generation cells; the actual prefill prefix still stops at physical 262144.
inline bool pr6_position_span(int64_t n, int64_t pos, int64_t capacity) noexcept {
    if (n < 0 || pos < 0 || capacity < 0) return false;
    const int64_t end_limit = std::min<int64_t>(capacity, 262144);
    return pos <= end_limit && n <= end_limit - pos;
}
inline size_t pr6_product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("PR6 allocation/copy size overflow");
    return a * b;
}
} // namespace strata::prefill::detail
