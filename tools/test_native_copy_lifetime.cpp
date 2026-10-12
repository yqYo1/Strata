// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU only. Run each mode in a fresh supervised process. Fatal cases must exit
// 1, with the expected boundary message and without UNWIND or RELEASE markers.
// These are fake native resources feeding the production helper, not SYCL/UR.
#include "strata/core/native_copy_lifetime.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
using strata::core::native_copy::checked_release;
using strata::core::native_copy::import_and_store;
void mark(const char* message) {
    std::puts(message);
    std::fflush(stdout);
}
struct Borrower {
    ~Borrower() { mark("RELEASE borrower"); }
};
bool list_result(const char* mode) {
    if (std::strcmp(mode, "list-throw") == 0) throw 7;
    return std::strcmp(mode, "list-error") != 0;
}
struct Owner {
    const char* mode;
    ~Owner() {
        checked_release([&] { return list_result(mode); }, "fixture list teardown failure");
        mark("RELEASE native");
        checked_release([&] { return std::strcmp(mode, "loader-error") != 0; },
                        "fixture loader teardown failure");
        mark("RELEASE loader");
    }
};
struct Unwind {
    ~Unwind() { mark("UNWIND outer"); }
};
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const char* mode = argv[1];
    const char* modes[] = {"success", "pre-import", "import-before", "import-after",
                          "insertion", "drain-error", "drain-throw", "list-error",
                          "list-throw", "loader-error"};
    bool known = false;
    for (const char* value : modes) known = known || std::strcmp(mode, value) == 0;
    if (!known) return 2;
    Unwind outer;
    Owner owner{mode};
    mark("CREATED native");
    if (std::strcmp(mode, "pre-import") == 0) {
        // Production input preparation may throw before any UR borrower exists.
        try { throw std::runtime_error("input preparation"); }
        catch (...) { mark("FALLBACK pre-import only"); }
        return 0;
    }
    std::unique_ptr<Borrower> stored;
    auto* result = import_and_store([&] {
        mark("IMPORT attempted");
        if (std::strcmp(mode, "import-before") == 0) throw 7;
        // Represents a raw UR queue created before queue_impl::create throws.
        // It deliberately has no local RAII owner, matching the audited gap.
        auto* raw = new Borrower;
        mark("CREATED borrower");
        if (std::strcmp(mode, "import-after") == 0) throw std::runtime_error("queue_impl");
        return std::unique_ptr<Borrower>(raw);
    }, [&](std::unique_ptr<Borrower>&& queue) {
        mark("INSERT attempted");
        if (std::strcmp(mode, "insertion") == 0) throw std::bad_alloc();
        stored = std::move(queue);
        return stored.get();
    });
    if (!result || !stored) return 3;
    checked_release([&] {
        mark("DRAIN attempted");
        if (std::strcmp(mode, "drain-throw") == 0) throw 7;
        return std::strcmp(mode, "drain-error") != 0;
    }, "fixture drain failure");
    mark("DRAIN checked");
    stored.reset();
    mark("SUCCESS before native teardown");
    return 0;
}
