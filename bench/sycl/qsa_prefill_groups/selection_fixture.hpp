// Test input, selector invocation, and combined selector/attention output checks.
#pragma once
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_prefill_groups.hpp"
#include "strata/prefill/qsa_group_ids.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::bench {
template<template<class> class Buffer>
class GroupSelectionFixture {
    using Shape = kernels::QsaShapes;
    static constexpr int guard = 16;
    static constexpr int32_t poison = -1234567;
    const int queries, cap, blocks;
    Buffer<float> scores;
    std::vector<int32_t> reference;
public:
    GroupSelectionFixture(int cells, int queries, int cap, std::vector<int32_t>& steps,
                          std::vector<int32_t>& initial_ids)
        : queries(queries), cap(cap), blocks(cells / 4 + 1), scores(size_t(queries) * blocks) {
        initial_ids.assign(size_t(queries) * cap + 2 * guard, poison);
        std::vector<float> host(size_t(queries) * blocks);
        for (int q = 0; q < queries; ++q) {
            int n = cells - (queries - 1 - q);
            if (queries <= 16 && q < 3) n = q == 2 ? cap : q; // empty and dense rows
            const int width = queries <= 16 && q == 3 ? 0 : std::min(n, cap);
            int32_t *st = steps.data() + q * kernels::kStepCount;
            st[kernels::kStepPos] = n - 1;
            st[kernels::kStepWidth] = width;
            st[kernels::kStepNKv] = n;
            st[kernels::kStepNBid] = n / 4;
            float *row = host.data() + size_t(q) * blocks;
            // Repeated scores exercise partial ties. Vary tail priority as well.
            for (int b = 0; b < blocks; ++b)
                row[b] = float((uint32_t(b) * 2654435761u + uint32_t(q) * 17u) >> 27);
            row[n / 4] = q % 8 < 4 ? 40.f : -1.f;
        }
        scores.put(host);
    }
    int32_t* data(Buffer<int32_t>& ids) const { return ids.data() + guard; }
    void submit(bool grouped, int32_t *ids, const int32_t *steps, const Shape& shape, void *stream) {
        auto select = grouped ? kernels::qsa_prefill_select_groups : kernels::qsa_block_topk_prefill;
        select(scores.data(), steps, queries, blocks, cap, shape, ids, stream, blocks);
    }
    void check(bool grouped, const std::vector<int32_t>& result, const std::vector<int32_t>& steps) {
        namespace fmt = prefill::qsa_groups;
        const int stride = grouped ? fmt::row_words(cap) : cap;
        auto padding = [&](size_t begin, size_t end) {
            if (end > result.size() || begin > end ||
                !std::all_of(result.begin() + begin, result.begin() + end, [](int32_t v){ return v == poison; }))
                throw std::runtime_error("selected-row padding or guard changed");
        };
        padding(0, guard);
        padding(guard + size_t(queries) * stride, result.size());
        const bool first = reference.empty();
        if (first && grouped) throw std::runtime_error("explicit selector must establish the reference");
        if (first) reference = result;
        for (int q = 0; q < queries; ++q) {
            const auto *st = steps.data() + q * kernels::kStepCount;
            const int width = st[kernels::kStepWidth], n = st[kernels::kStepNKv];
            const size_t base = guard + size_t(q) * stride;
            const int32_t *row = result.data() + base;
            const int32_t *expected = reference.data() + guard + size_t(q) * cap;
            if (first && queries <= 16) {
                // Independent stable, descending cell ranking with ascending-ID ties.
                const float *sc = scores.upload.data() + size_t(q) * blocks;
                std::vector<int> oracle(n);
                std::iota(oracle.begin(), oracle.end(), 0);
                std::stable_sort(oracle.begin(), oracle.end(), [&](int a, int b){ return sc[a/4] > sc[b/4]; });
                oracle.resize(width);
                std::sort(oracle.begin(), oracle.end());
                if (!std::equal(oracle.begin(), oracle.end(), expected))
                    throw std::runtime_error("explicit selector and stable cell sort disagree");
            }
            int unique = 0, previous = -1;
            for (int i = 0; i < width; ++i) {
                const int cell = grouped ? int(fmt::cell(reinterpret_cast<const uint32_t*>(row), i)) : row[i];
                if (cell < 0 || cell >= n || cell <= previous || cell != expected[i])
                    throw std::runtime_error("selected cell identity or order differs");
                unique += i == 0 || cell / 4 != previous / 4;
                previous = cell;
            }
            padding(base + (grouped ? unique + (width != 0) : width), base + stride);
        }
    }
    void check_scores() {
        if (scores.get() != scores.upload) throw std::runtime_error("selector scores changed");
    }
};
} // namespace strata::bench
