#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

// Main-authored; repository LICENSE. Preserve the diagnostic hash's original
// [page][head][cell][value] byte order while coalescing complete pages. This
// only describes spans; the caller retains its checked device-copy semantics.
namespace strata::program::diagnostic {
template<class Consume>
bool kv_hash_spans(int64_t begin, int64_t end, int64_t page_cells,
                   int64_t heads, int64_t cell_bytes, Consume&& consume) {
    if (begin < 0 || end < begin || page_cells <= 0 || heads <= 0 || cell_bytes <= 0) return false;
    const uint64_t ps = uint64_t(page_cells), nh = uint64_t(heads), width = uint64_t(cell_bytes);
    const uint64_t pages = uint64_t(end) / ps + (uint64_t(end) % ps != 0);
    size_t extent = 1;
    for (const uint64_t n : {pages, nh, ps, width}) {
        if (n > std::numeric_limits<size_t>::max() ||
            (n && extent > std::numeric_limits<size_t>::max() / n)) return false;
        extent *= size_t(n);
    }
    (void) extent; // All emitted offsets and lengths are bounded by this product.
    uint64_t pos = uint64_t(begin), stop = uint64_t(end);
    auto partial = [&](uint64_t count) {
        const uint64_t pg = pos / ps, first = pos % ps;
        for (uint64_t hd = 0; hd < nh; ++hd) {
            const size_t offset = size_t(((pg * nh + hd) * ps + first) * width);
            if (!consume(offset, size_t(count * width))) return false;
        }
        pos += count;
        return true;
    };
    if (pos < stop && pos % ps && !partial(std::min(stop - pos, ps - pos % ps))) return false;
    const uint64_t full = (stop - pos) / ps;
    if (full) {
        if (!consume(size_t(pos * nh * width), size_t(full * ps * nh * width))) return false;
        pos += full * ps;
    }
    return pos == stop || partial(stop - pos);
}
} // namespace strata::program::diagnostic
