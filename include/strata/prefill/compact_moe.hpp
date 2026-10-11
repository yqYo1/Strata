// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::prefill {

struct CompactMoeGroup {
    size_t first = 0, last = 0;   // expert indices in the original logical window
    int32_t row0 = 0, rows = 0, max_rows = 0;
    std::vector<int32_t> bounds;  // host-only; Gemm::f16_groups consumes these synchronously
};

// Partition an already dequantized logical window, without moving its expert slots. Zero-count stream entries
// belong to the issuer's release path and must be absent from order. Validate the WHOLE window before dispatch.
inline std::vector<CompactMoeGroup> compact_moe_groups(const int32_t* bounds, const int32_t* order,
                                                       size_t order_size, const int32_t* counts, size_t n_expert,
                                                       size_t first, size_t last, int64_t cap, int64_t rows_gpu) {
    if (cap <= 0 || cap > std::numeric_limits<int32_t>::max() || rows_gpu < 0 ||
        rows_gpu > std::numeric_limits<int32_t>::max() || first > last || last > order_size)
        throw std::runtime_error("prefill compact MoE invalid window/capacity");
    if (first == last) return {};
    if (!bounds || !order || !counts || bounds[first] < 0 || bounds[last] > rows_gpu)
        throw std::runtime_error("prefill compact MoE route bounds outside GPU rows");
    for (size_t j = first; j < last; ++j) {
        const int32_t e = order[j];
        if (e < 0 || (size_t) e >= n_expert || counts[e] <= 0 || counts[e] > cap)
            throw std::runtime_error("prefill compact MoE expert count outside row capacity");
        const int64_t width = (int64_t) bounds[j + 1] - bounds[j];
        if (bounds[j] < 0 || bounds[j + 1] > rows_gpu || width != counts[e])
            throw std::runtime_error("prefill compact MoE inconsistent route prefix");
    }
    std::vector<CompactMoeGroup> groups;
    for (size_t a = first; a < last;) {
        CompactMoeGroup g;
        g.first = a; g.row0 = bounds[a]; g.bounds.push_back(0);
        int64_t sum = 0;
        size_t b = a;
        for (; b < last; ++b) {
            const int32_t cnt = counts[order[b]];
            // Subtraction admits only a checked sum <= cap, even for an arbitrary future group size.
            if (cnt > cap - sum) break;
            sum += cnt;
            g.max_rows = std::max(g.max_rows, cnt);
            g.bounds.push_back((int32_t) sum);
        }
        if (b == a || sum <= 0 || (int64_t) bounds[b] - g.row0 != sum || sum > rows_gpu - g.row0)
            throw std::runtime_error("prefill compact MoE invalid subgroup coverage");
        g.last = b; g.rows = (int32_t) sum;
        groups.push_back(std::move(g));
        a = b;
    }
    return groups;
}

}  // namespace strata::prefill
