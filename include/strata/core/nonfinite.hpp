// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>

namespace strata::core {
// Consume completed device-written flags. Caller must first establish queue/event completion.
// Unused rows are deliberately ignored: smaller windows do not need to clear the previous tail.
inline uint32_t nonfinite_mask(const uint32_t* flags, int rows, int blocks) {
    if (flags == nullptr || rows < 1 || rows > 32 || blocks < 1) return 0;
    uint32_t mask = 0;
    for (int t = 0; t < rows; ++t)
        for (int b = 0; b < blocks; ++b)
            if (flags[t * blocks + b] != 0) { mask |= uint32_t{1} << t; break; }
    return mask;
}
}  // namespace strata::core
