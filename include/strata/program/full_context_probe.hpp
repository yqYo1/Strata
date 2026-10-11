// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::program::full_context_probe {

// This is an exact physical-boundary diagnostic, never a generation admission policy.
inline constexpr int64_t cells = 262144;
inline bool exact_extent(size_t tokens, int64_t capacity) {
    return tokens == size_t(cells) && capacity == cells;
}
inline bool valid_token(int64_t token, int64_t vocabulary) {
    return token >= 0 && token < vocabulary && token <= INT32_MAX;
}

// Advance only after the target chunk and all its draft rows have completed.
struct Coverage {
    int64_t end = 0, chunks = 0, last_begin = -1, last_count = 0;
    int64_t real_next_rows = 0, probe_rows = 0;
    bool accepts(int64_t begin, int64_t count) const {
        return begin == end && begin >= 0 && count > 0 && begin < cells && count <= cells - begin;
    }
    bool complete(int64_t begin, int64_t count) {
        if (!accepts(begin, count)) return false;
        last_begin = begin; last_count = count; end += count; ++chunks;
        const int64_t probe = end == cells ? 1 : 0;
        real_next_rows += count - probe; probe_rows += probe;
        return true;
    }
    bool full() const { return end == cells && real_next_rows == cells - 1 && probe_rows == 1; }
};

// Validates the whole range before indexing. The last residual has no semantic
// successor: exactly that row receives the explicit, valid storage probe token.
inline bool successors(const std::vector<int64_t>& tokens, int64_t begin, int64_t count,
                       int32_t terminal, std::vector<int32_t>& output) {
    if (tokens.size() != size_t(cells) || begin < 0 || count <= 0 || begin >= cells || count > cells - begin ||
        terminal < 0) return false;
    output.resize(size_t(count));
    for (int64_t row = 0; row < count; ++row) {
        const int64_t cell = begin + row;
        const int64_t token = cell == cells - 1 ? terminal : tokens[size_t(cell + 1)];
        if (token < 0 || token > INT32_MAX) return false;
        output[size_t(row)] = int32_t(token);
    }
    return true;
}
inline bool add_bytes(uint64_t& total, uint64_t bytes, uint64_t count = 1) {
    if (count && bytes > std::numeric_limits<uint64_t>::max() / count) return false;
    const uint64_t extra = bytes * count;
    if (extra > std::numeric_limits<uint64_t>::max() - total) return false;
    total += extra;
    return true;
}
} // namespace strata::program::full_context_probe
