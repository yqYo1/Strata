// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only fixtures for the production row partition. No GPU, model or SYCL headers required.
#include "strata/prefill/compact_moe.hpp"
#include "strata/prefill/publication.hpp"
#include <cassert>
#include <iostream>

using strata::prefill::compact_moe_groups;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

// Exercise route reconstruction through subgroup-local inputs and original q slots. This detects changed expert
// order, lost/duplicated rows, an incorrect weight rebase, or a Down destination that starts at local row zero.
size_t coverage(const std::vector<int32_t>& counts, const std::vector<int32_t>& order, int32_t row0,
                int32_t cap, size_t first, size_t last) {
    std::vector<int32_t> bounds{row0};
    for (const int32_t e : order) bounds.push_back(bounds.back() + counts[(size_t) e]);
    const auto groups = compact_moe_groups(bounds.data(), order.data(), order.size(), counts.data(), counts.size(),
                                           first, last, cap, bounds.back());
    std::vector<int64_t> expected((size_t) bounds.back(), -1), actual(expected.size(), -1);
    for (size_t j = first; j < last; ++j)
        for (int32_t r = bounds[j]; r < bounds[j + 1]; ++r)
            expected[(size_t) r] = 100000ll * order[j] + r;
    size_t next = first;
    for (const auto& g : groups) {
        assert(g.first == next && g.last > g.first && g.rows > 0 && g.rows <= cap);
        assert(g.row0 == bounds[g.first] && g.bounds.front() == 0 && g.bounds.back() == g.rows);
        assert(g.bounds.size() == g.last - g.first + 1);
        if (g.last < last) assert((int64_t) g.rows + counts[(size_t) order[g.last]] > cap);
        std::vector<int32_t> gathered((size_t) g.rows);
        for (int32_t r = 0; r < g.rows; ++r) gathered[(size_t) r] = g.row0 + r;
        int32_t max_rows = 0;
        for (size_t j = g.first; j < g.last; ++j) {
            const size_t local = j - g.first, q0 = g.first - first;
            const int32_t expert = order[first + q0 + local];
            max_rows = std::max(max_rows, counts[(size_t) expert]);
            for (int32_t r = g.bounds[local]; r < g.bounds[local + 1]; ++r) {
                const size_t dst = (size_t) g.row0 + (size_t) r;
                assert(actual[dst] == -1);
                actual[dst] = 100000ll * expert + gathered[(size_t) r];
            }
        }
        assert(g.max_rows == max_rows);
        next = g.last;
    }
    assert(next == last && actual == expected);
    return groups.size();
}

int main() try {
    assert(coverage({2, 3, 1}, {2, 0, 1}, 9, 6, 0, 3) == 1);  // unsplit, reordered experts, nonzero route base
    assert(coverage({4, 3, 2, 1}, {0, 1, 2, 3}, 7, 6, 0, 4) == 2);  // maximal [4], [3,2,1]
    assert(coverage({4, 4, 1, 3, 2}, {0, 1, 2, 3, 4}, 11, 4, 1, 5) == 3);  // q0 rebasing within tail window
    assert(coverage({1, 1, 1}, {0, 1, 2}, 0, 1, 0, 3) == 3);  // one-token tail
    assert(coverage({2047, 1, 2046}, {0, 1, 2}, 0, 2047, 0, 3) == 2);  // short chunk tail
    std::vector<int32_t> large_counts(65, 1), large_order;
    for (int32_t e = 0; e < 65; ++e) large_order.push_back(e);
    assert(coverage(large_counts, large_order, 13, 65, 0, 65) == 1);  // no hidden G16/G32 partition limit
    assert(coverage(large_counts, large_order, 13, 17, 0, 65) == 4);

    // Sweep sparse routing and tails. Zero-count entries release their original stream positions through the
    // production helper, and are omitted from the positive-row order passed to the partitioner.
    for (int pattern = 0; pattern < 4096; ++pattern) {
        int value = pattern;
        std::vector<int32_t> counts(6), order;
        size_t consumed = 0;
        for (int32_t e = 0; e < 6; ++e) {
            counts[(size_t) e] = value % 4; value /= 4;
            if (counts[(size_t) e] > 0) {
                order.push_back(e); ++consumed;
            } else {
                bool flushed = false, waited = false;
                strata::prefill::publication::skipped_entry([&] { flushed = true; },
                    [&] { assert(flushed); waited = true; }, [&] { assert(waited); ++consumed; });
            }
        }
        assert(consumed == counts.size());
        coverage(counts, order, 5, 3, 0, order.size());
    }

    int32_t bounds[] = {8, 10, 13}, order[] = {0, 1}, counts[] = {2, 3};
    auto partition = [&](int64_t cap = 3, int64_t rows = 13, size_t first = 0, size_t last = 2) {
        return compact_moe_groups(bounds, order, 2, counts, 2, first, last, cap, rows);
    };
    rejects([&] { partition(2); });  // invalid expert in the later subgroup still rejects the entire window
    rejects([&] { partition(0); });
    rejects([&] { partition(std::numeric_limits<int64_t>::max()); });
    rejects([&] { partition(3, std::numeric_limits<int64_t>::max()); });
    rejects([&] { partition(3, 12); });
    rejects([&] { partition(3, 13, 2, 1); });
    rejects([&] { partition(3, 13, 0, 3); });
    counts[1] = 0; rejects([&] { partition(); });
    counts[1] = -1; rejects([&] { partition(); });
    counts[1] = 4; rejects([&] { partition(); });
    counts[1] = 3; order[1] = 2; rejects([&] { partition(); });
    order[1] = -1; rejects([&] { partition(); });
    order[1] = 1; bounds[1] = 7; rejects([&] { partition(); });
    bounds[1] = 11; rejects([&] { partition(); });
    bounds[1] = 10; bounds[0] = -1; rejects([&] { partition(); });
    const int32_t max = std::numeric_limits<int32_t>::max();
    const int32_t big_bounds[] = {0, max - 1, max}, big_counts[] = {max - 1, 1}, big_order[] = {0, 1};
    const auto big = compact_moe_groups(big_bounds, big_order, 2, big_counts, 2, 0, 2, max, max);
    assert(big.size() == 1 && big[0].rows == max && big[0].bounds.back() == max);
    const auto big_split = compact_moe_groups(big_bounds, big_order, 2, big_counts, 2, 0, 2, max - 1, max);
    assert(big_split.size() == 2 && big_split[1].row0 == max - 1 && big_split[1].rows == 1);
    assert(compact_moe_groups(nullptr, nullptr, 0, nullptr, 0, 0, 0, 1, 0).empty());
    std::cout << "prefill compact MoE host fixtures passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
