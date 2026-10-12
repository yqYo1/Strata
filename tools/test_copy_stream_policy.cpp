// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/copy_stream_policy.hpp"
#include <cstdio>
#include <initializer_list>
int main() {
    using strata::core::copy_engine_opt_in;
    if (copy_engine_opt_in(nullptr) || !copy_engine_opt_in("1")) return 1;
    for (const char* value : {"", "0", "01", "1 ", " 1", "+1", "-1", "true", "2", "1x"})
        if (copy_engine_opt_in(value)) return 1;
    const bool original = strata::core::copy_engine_requested();
    if (setenv("STRATA_COPY_ENGINE", original ? "0" : "1", 1) != 0) return 1;
    if (strata::core::copy_engine_requested() != original) return 1;
    std::puts("PASS exact opt-in and frozen policy");
    return 0;
}
