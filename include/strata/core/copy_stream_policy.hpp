// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdlib>
#include <cstring>
namespace strata::core {
inline bool copy_engine_opt_in(const char* value) {
    return value && std::strcmp(value, "1") == 0;
}
inline bool copy_engine_requested() {
    static const bool requested = copy_engine_opt_in(std::getenv("STRATA_COPY_ENGINE"));
    return requested;
}
}  // namespace strata::core
