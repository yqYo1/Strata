// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace strata::core::context_bounds {

// Device step records include cell+1 as int32. Resident ring slots do not
// extend this logical capacity. Subtract only after checking the operands.
inline bool capacity(int64_t cells) {
    return cells > 0 && cells <= std::numeric_limits<int32_t>::max();
}
inline bool span(int64_t cells, int64_t pos, int64_t count) {
    return capacity(cells) && pos >= 0 && pos <= cells && count >= 0 && count <= cells - pos;
}
inline bool admission(int64_t cells, int64_t prompt, int64_t outputs) {
    return outputs > 0 && span(cells, prompt, outputs);
}

// A T-row verifier emits at most T outputs: T-1 candidates plus one bonus.
// Zero means no legal row/output, never permission to enqueue at pos==cells.
inline int verifier_window(int64_t cells, int64_t pos, int requested, int64_t outputs) {
    if (!span(cells, pos, 0) || requested < 1 || outputs < 1) return 0;
    return (int) std::min<int64_t>(requested, std::min(cells - pos, outputs));
}

struct MtpWindow {
    int catchup = 0;
    int candidates = 0;
    int64_t first_cell = 0;
};

// Validate the ORIGINAL verified span even when rejected rows are not caught up.
// A proposal at pos+accepted+j feeds row j+1 of the NEXT verifier window.
// Reserve that next window's row 0 and its bonus output before proposing.
inline bool mtp_window(int64_t cells, int64_t pos, int rows, int accepted,
                       int max_rows, int max_candidates, int64_t outputs,
                       bool catchup_all, MtpWindow& out) {
    out = {};
    if (rows < 1 || max_rows < 1 || rows > max_rows || accepted < 0 || accepted >= rows ||
        max_candidates < 0 || outputs < 0 || !span(cells, pos, rows)) return false;
    out.catchup = catchup_all ? rows : accepted + 1;
    out.first_cell = pos + accepted;   // proved <= cells-1 by span and accepted<rows
    const int next_rows = verifier_window(cells, out.first_cell + 1, max_rows, outputs);
    out.candidates = std::min(max_candidates, next_rows > 0 ? next_rows - 1 : 0);
    return true;
}
} // namespace strata::core::context_bounds
