// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace strata::core::native_copy {
// Shared with Runtime's fatal path. No C++ unwinding, static destruction or
// atexit handlers: borrowed native handles and pending USM remain until exit.
[[noreturn]] inline void fail_stop(const char* message) noexcept {
    std::fprintf(stderr, "Xe runtime: %s; terminating without releasing in-flight allocations\n", message);
    std::fflush(stderr);
    std::_Exit(1);
}

// The caller retains the native list/loader outside this function. Enter only
// after native creation and input preparation, immediately before make_queue.
// Import failure is ambiguous: SYCL may already have created a UR borrower.
// Catch here, before caller ownership unwinds. Never retry/fallback afterward.
// On insertion failure catch inside the live local queue's scope, so even its
// destructor cannot run before fail-stop. Production makes no submissions here.
template <class Import, class Insert>
auto import_and_store(Import&& import, Insert&& insert) {
    try {
        auto queue = std::forward<Import>(import)();
        try {
            return std::forward<Insert>(insert)(std::move(queue));
        } catch (...) {
            fail_stop("native copy queue insertion failed after import; native resources retained");
        }
    } catch (...) {
        fail_stop("native copy import attempted with uncertain UR ownership; native resources retained");
    }
}

// Used by actual list/loader destruction and checked native-stream drain.
// Both API error results and C++ exceptions must stop before subsequent owners
// or allocations release. The callback's true result means checked success.
template <class Operation>
void checked_release(Operation&& operation, const char* message) noexcept {
    try {
        if (std::forward<Operation>(operation)()) return;
    } catch (...) { fail_stop(message); }
    fail_stop(message);
}
}  // namespace strata::core::native_copy
