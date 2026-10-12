// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Only unreachable GPU/remote plan references in the ACTUAL CPU dispatcher need these symbols.
// Trap immediately if fixture eligibility fails; no GPU runtime/driver implementation is linked.
#include "strata/core/remote_experts.hpp"
#include <cstdio>
#include <cstdlib>
namespace {
[[noreturn]] void forbidden() {
    std::fputs("FAIL CPU histogram fixture: forbidden GPU/cache/remote branch\n", stderr);
    std::abort();
}
}
namespace strata::core {
int32_t ExpertCache::slot_of(int64_t, int64_t) const { forbidden(); }
bool RemoteExperts::begin(int64_t, const float*, const int32_t*, int64_t, int64_t,
                          const int32_t*, const int32_t*, std::string&) { forbidden(); }
bool RemoteExperts::finish(float*, std::string&) { forbidden(); }
}
