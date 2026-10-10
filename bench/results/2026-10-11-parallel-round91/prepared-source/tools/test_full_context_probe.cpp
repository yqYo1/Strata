// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only test of the production diagnostic's bounds and coverage helpers.
#include "strata/program/full_context_probe.hpp"
#include <cstdio>

int main() {
    namespace probe = strata::program::full_context_probe;
    int failures = 0;
    auto check = [&](bool ok) { if (!ok) ++failures; };
    check(probe::exact_extent(262144, 262144));
    check(!probe::exact_extent(262143, 262144));
    check(!probe::exact_extent(262144, 262145));
    check(probe::valid_token(0, 10) && probe::valid_token(9, 10));
    check(!probe::valid_token(-1, 10) && !probe::valid_token(10, 10));
    probe::Coverage coverage;
    check(!coverage.complete(1, 4096) && !coverage.complete(0, 0));
    check(!coverage.complete(-1, 1) && !coverage.complete(0, INT64_MAX));
    for (int64_t begin = 0; begin < probe::cells; begin += 4096) check(coverage.complete(begin, 4096));
    check(coverage.full() && coverage.chunks == 64 && coverage.last_begin == 258048 && coverage.last_count == 4096);
    check(!coverage.complete(262144, 1) && !coverage.complete(258048, 4096));
    probe::Coverage uneven;
    check(uneven.complete(0, 262143) && !uneven.full());
    check(!uneven.complete(262143, 2) && uneven.end == 262143);
    check(uneven.complete(262143, 1) && uneven.full());
    std::vector<int64_t> tokens(static_cast<size_t>(probe::cells));
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = int64_t(i % 101);
    std::vector<int32_t> next;
    check(probe::successors(tokens, 0, 4096, 7, next));
    check(next.size() == 4096);
    for (size_t i = 0; i < next.size(); ++i) check(next[i] == tokens[i + 1]);
    check(probe::successors(tokens, 258048, 4096, 7, next));
    check(next.size() == 4096);
    for (size_t i = 0; i + 1 < next.size(); ++i) check(next[i] == tokens[258049 + i]);
    check(!next.empty() && next.back() == 7);
    check(probe::successors(tokens, 262143, 1, 9, next) && next[0] == 9);
    check(!probe::successors(tokens, 262144, 1, 9, next));
    check(!probe::successors(tokens, -1, 1, 9, next));
    check(!probe::successors(tokens, 262143, 2, 9, next));
    check(!probe::successors(tokens, 0, 1, -1, next));
    tokens[1] = INT64_MAX;
    check(!probe::successors(tokens, 0, 1, 9, next));
    tokens.pop_back();
    check(!probe::successors(tokens, 0, 1, 9, next));
    uint64_t bytes = 5;
    check(probe::add_bytes(bytes, 4, 2) && bytes == 13);
    const uint64_t limit = UINT64_MAX;
    check(!probe::add_bytes(bytes, limit) && bytes == 13);
    check(!probe::add_bytes(bytes, limit, 2) && bytes == 13);
    if (failures) std::fprintf(stderr, "full-context probe: %d failures\n", failures);
    return failures ? 1 : 0;
}
