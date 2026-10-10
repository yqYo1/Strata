// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only fault cutpoints through MTP's real production ownership helper.
#include "strata/core/mtp_completion.hpp"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace {
void expect(bool value, const char* what) {
    if (!value) throw std::runtime_error(what);
}
}

int main() try {
    using strata::core::MtpCompletion;
    std::string err;
    const auto failure = [] { return std::string("injected queue failure"); };
    {
        MtpCompletion state;
        expect(state.host_ready(), "initial host ownership");
        expect(state.submit(err) && state.pending() && !state.host_ready(), "ownership precedes submission");
        expect(state.submit(err) && state.submissions() == 2, "a later writer shares the pending allocation");
        int waits = 0;
        expect(state.complete([&] { ++waits; return true; }, failure, err), "checked whole-queue completion");
        expect(state.host_ready() && !state.pending() && waits == 1, "only completion permits host reads");
        expect(state.complete([&] { ++waits; return true; }, failure, err) && waits == 2,
               "idle still checks queue for external stream users");
    }
    for (const char* fault : {"rejected partial graph", "rejected input upload", "rejected event record"}) {
        MtpCompletion state;
        expect(state.submit(err), "pending before a rejected operation");
        state.fail(fault);
        expect(!state.host_ready(), "failed state is never valid for host use");
        expect(!state.complete([] { return true; }, failure, err), "a successful drain cannot report valid failed state");
        expect(!state.pending() && err == fault, "drain retires ownership and preserves the operation failure");
        expect(!state.submit(err), "sticky failure refuses another enqueue");
        expect(!state.complete([] { return true; }, failure, err) && err == fault,
               "later idle cannot turn failed state into success");
    }
    {
        MtpCompletion state;
        expect(state.submit(err), "start a failed wait");
        expect(!state.complete([] { return false; }, failure, err), "failed completion");
        expect(state.pending() && !state.host_ready() && err == "injected queue failure",
               "failed wait retains storage ownership");
        state.fail("later fault");
        expect(!state.complete([] { return true; }, failure, err), "retry drains but cannot erase failure");
        expect(!state.pending() && err == "injected queue failure", "first error survives retry");
    }
    {
        MtpCompletion state;
        // A wait can fail for work submitted by an external stream user even if
        // this helper did not mark that submission itself.
        expect(!state.complete([] { return false; }, failure, err), "external queue failure is checked");
        expect(state.pending() && !state.host_ready(), "external uncertainty prevents free/rewrite");
    }
    {
        MtpCompletion state;
        expect(state.submit(err), "start exceptional completion");
        bool caught = false;
        try {
            (void) state.complete([]() -> bool { throw std::runtime_error("injected wait exception"); }, failure, err);
        } catch (const std::runtime_error&) { caught = true; }
        expect(caught && state.pending() && !state.host_ready(), "exception keeps ownership");
        expect(!state.complete([] { return true; }, failure, err) && !state.pending(),
               "exceptional state can drain but cannot be reused");
    }
    std::puts("MTP completion fault cutpoints passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
