// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/prefill/route_union.hpp"
#include <array>
#include <cstdlib>
#include <limits>
#include <initializer_list>
#include <vector>

static void check(bool value) { if (!value) std::abort(); }
int main() {
    using strata::prefill::RouteUnion;
    // Chunk 1 and chunk 2 each have ten distinct routes and overlap at 0/511.
    const std::array<int32_t, 10> a{0, 511, 1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<int32_t, 10> b{0, 511, 9, 10, 11, 12, 13, 14, 15, 16};
    RouteUnion c;
    c.reset(2, 512, 2, 2, 10);
    for (int l = 0; l < 2; ++l) {
        c.observe(l, 0, 1, a.data(), a.size(), 100);
        c.observe(l, 1, 1, b.data(), b.size(), 100);
    }
    check(c.finish(true));
    check(c.layers[0].visits == 20 && c.layers[0].unique == 18);
    check((c.layers[0].bits[0] & 1) && (c.layers[0].bits[7] & (uint64_t(1) << 63)));
    check(!c.finish(false) && !c.completed); // cancellation/error despite full routes
    c.reset(1, 512, 1, 1, 10); // second request must discard the first union
    c.observe(0, 0, 1, b.data(), b.size(), 100);
    check(c.finish(true) && c.layers[0].unique == 10 && c.layers[0].visits == 10);
    check(!(c.layers[0].bits[0] & (uint64_t(1) << 1)));
    for (int bad : {-1, 512}) {
        auto ids = a; ids[4] = bad;
        c.reset(1, 512, 1, 1, 10);
        c.observe(0, 0, 1, ids.data(), ids.size(), 100);
        check(c.invalid && !c.finish(true));
    }
    for (int bad : {-1, 1, 64}) {
        c.reset(1, 512, 1, 1, 10);
        c.observe(bad, 0, 1, a.data(), a.size(), 100);
        check(c.invalid && !c.finish(true));
    }
    c.reset(2, 512, 1, 1, 10);
    c.observe(1, 0, 1, a.data(), a.size(), 100);
    check(c.invalid); // layer order
    c.reset(1, 512, 2, 2, 10);
    c.observe(0, 1, 1, a.data(), a.size(), 100);
    check(c.invalid); // row gap
    c.reset(1, 512, 2, 2, 10);
    c.observe(0, 0, 1, a.data(), a.size(), 100);
    check(!c.finish(true) && c.invalid); // partial coverage cannot complete
    c.reset(1, 512, 2, 2, 10);
    c.observe(0, 0, 1, a.data(), a.size(), 100);
    c.observe(0, 0, 1, b.data(), b.size(), 100);
    check(c.invalid); // repeated chunk
    c.reset(1, 512, 2, 2, 10);
    c.observe(0, 0, 1, a.data(), a.size(), 100);
    c.observe(0, 1, 1, b.data(), b.size(), 101);
    check(c.invalid); // blob extent changed within a layer
    c.reset(1, 512, 1, 1, 10);
    c.observe(0, 0, 1, a.data(), 9, 100);
    check(c.invalid); // incomplete route row
    c.reset(1, 512, std::numeric_limits<uint64_t>::max(), 1, 10);
    check(c.invalid); // expected route extent overflow
    c.reset(1, 512, 1, 1, 10);
    c.layers[0].visits = std::numeric_limits<uint64_t>::max();
    c.observe(0, 0, 1, a.data(), a.size(), 100);
    check(c.invalid); // active visit counter overflow
    c.reset(1, 512, 1, 1, 10);
    c.observe(0, 0, 1, a.data(), a.size(), std::numeric_limits<uint64_t>::max());
    check(!c.finish(true) && c.invalid); // logical extent overflow
    c.reset(64, 512, 1, 1, 10);
    for (int l = 0; l < 64; ++l)
        c.observe(l, 0, 1, a.data(), a.size(), std::numeric_limits<uint64_t>::max() / 512);
    check(!c.finish(true) && c.invalid); // aggregate extent overflow
    for (int experts : {0, 64, 511, 513}) {
        c.reset(48, experts, 1, 1, 10);
        check(!c.eligible && !c.finish(true)); // geometry only; no hardware model name
    }
    c.reset(65, 512, 1, 1, 10); check(!c.eligible);
    c.reset(0, 512, 1, 1, 10); check(!c.eligible);
    c.reset(48, 512, 1, 1, 8); check(!c.eligible);
    c.reset(48, 512, 0, 0, 10); check(!c.eligible);
    // The final full-context chunk and the adjacent one-row-short boundary.
    // These check host receipt arithmetic, not physical GPU KV stores.
    std::vector<int32_t> routes(4096 * 10);
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = int32_t(i % 512);
    for (uint64_t rows : {uint64_t(262143), uint64_t(262144)}) {
        c.reset(1, 512, rows, 64, 10);
        for (uint64_t chunk = 0; chunk < 64; ++chunk) {
            const uint64_t offset = chunk * 4096;
            const uint64_t tokens = rows - offset < 4096 ? rows - offset : 4096;
            c.observe(0, offset, tokens, routes.data(), tokens * 10, 100);
        }
        check(c.finish(true) && c.layers[0].rows == rows && c.layers[0].chunks == 64);
        check(c.layers[0].unique == 512 && c.layers[0].visits == 64 * 512);
    }
    return 0;
}
