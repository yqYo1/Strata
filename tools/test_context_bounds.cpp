// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only source: exercise the actual MTP/generation bounds helper.
#include "strata/core/context_bounds.hpp"
#include "strata/core/mtp_completion.hpp"
#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
namespace bounds = strata::core::context_bounds;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check_plan(int64_t cells, int64_t pos, int rows, int accepted, int64_t remaining, bool all) {
    bounds::MtpWindow plan;
    expect(bounds::mtp_window(cells, pos, rows, accepted, 4, 3, remaining, all, plan), "legal catch-up span");
    expect(plan.catchup == (all ? rows : accepted + 1), "catch-up policy preserves the accepted prefix");
    expect(bounds::span(cells, pos, plan.catchup), "catch-up never reaches a cell outside capacity");
    expect(plan.candidates >= 0 && plan.candidates <= 3, "spec4 draft depth");
    for (int j = 0; j < plan.candidates; ++j) {
        const int64_t cell = plan.first_cell + j;
        expect(cell >= 0 && cell < cells, "every retained MTP position is in range");
        expect(cell + 1 <= std::numeric_limits<int32_t>::max(), "exclusive step is representable");
    }
    if (remaining > 0 && plan.first_cell + 1 < cells) {
        const int next = bounds::verifier_window(cells, plan.first_cell + 1, 4, remaining);
        expect(plan.candidates + 1 == next, "next window reserves the bonus row and output");
    } else expect(plan.candidates == 0, "no legal next row produces no candidates");
    if (remaining == 1) expect(plan.candidates == 0, "one remaining output uses the single-row verifier");
}

// Model only the real control-plane transitions: no model/SYCL execution is claimed.
void decode_budget(int64_t cells, int64_t prompt, int64_t outputs, bool native, bool reject, bool all) {
    expect(bounds::admission(cells, prompt, outputs), "exact-capacity request admitted");
    int64_t remaining = outputs, pos = prompt - 1, emitted = 0;
    int candidates = 0;
    bool first = true;
    if (!native) {   // the token loop has already emitted the first output
        --remaining; ++emitted; ++pos; first = false;
        bounds::MtpWindow boot;
        const int repeated = bounds::verifier_window(cells, prompt - 1, 4, std::numeric_limits<int64_t>::max());
        expect(bounds::mtp_window(cells, prompt - 1, repeated, 0, 4, 3, remaining, all, boot), "bounded bootstrap");
        candidates = boot.candidates;
    }
    int rounds = 0;
    while (remaining > 0) {
        const int requested = first ? 1 : 1 + candidates;
        const int rows = bounds::verifier_window(cells, pos, requested, remaining);
        expect(rows > 0 && bounds::span(cells, pos, rows), "legal shortened verifier span");
        if (remaining == 1) expect(rows == 1, "final remaining output is a single verifier row");
        const int accepted = reject ? (rows > 2 ? 1 : 0) : rows - 1;
        const int take = accepted + 1;   // accepted candidates plus bonus token
        expect(take <= remaining, "bonus output never exceeds remaining budget");
        remaining -= take; emitted += take;
        check_plan(cells, pos, rows, accepted, remaining, all);
        bounds::MtpWindow next;
        expect(bounds::mtp_window(cells, pos, rows, accepted, 4, 3, remaining, all, next), "post-verdict bound");
        candidates = next.candidates;
        pos += take;
        first = false;
        expect(++rounds <= outputs, "bounded progress through rejected windows");
    }
    expect(emitted == outputs && pos == prompt - 1 + outputs, "all requested outputs emitted without extra KV cell");
    expect(pos <= cells - 1, "last output remains unconsumed inside logical capacity");
}
} // namespace

int main() try {
    constexpr int64_t big = std::numeric_limits<int64_t>::max();
    constexpr int64_t i32 = std::numeric_limits<int32_t>::max();
    expect(bounds::span(4, 0, 4) && bounds::span(7, 3, 4), "p+T==C is legal");
    expect(!bounds::span(4, -1, 1) && !bounds::span(4, 0, -1), "negative positions/counts rejected");
    expect(!bounds::span(4, big, 1) && !bounds::span(4, 1, big), "overflowing spans rejected by subtraction");
    expect(!bounds::span(big, 0, 1) && !bounds::span(i32 + 1, 0, 1), "unrepresentable device capacity rejected");
    expect(bounds::span(i32, i32 - 1, 1), "largest representable exclusive step");
    expect(!bounds::admission(4, 3, 2) && bounds::admission(4, 3, 1), "exact admission without hidden slack");
    expect(!bounds::admission(4, big, big) && !bounds::admission(4, 3, -1), "invalid admission cannot overflow");
    expect(bounds::verifier_window(4, 3, 4, 1) == 1, "last physical row can produce its bonus");
    expect(bounds::verifier_window(4, 4, 4, 1) == 0, "one-past has no verifier row");
    expect(bounds::verifier_window(7, 5, 4, big) == 2, "unaligned tail shrinks the window");
    expect(bounds::verifier_window(262144, 262143, 4, 1) == 1, "262144 physical last row");
    bounds::MtpWindow plan;
    expect(!bounds::mtp_window(4, 4, 1, 0, 4, 3, 1, false, plan), "p+a==C rejected before staging");
    expect(!bounds::mtp_window(4, 0, 4, 4, 4, 3, 1, false, plan), "accepted row outside original span rejected");
    expect(!bounds::mtp_window(7, 5, 4, 0, 4, 3, 3, false, plan), "invalid original span rejected even without catchup-all");
    expect(!bounds::mtp_window(4, big, 1, 0, 4, 3, 1, true, plan), "MTP position overflow rejected");
    expect(!bounds::mtp_window(4, -1, 1, 0, 4, 3, 1, false, plan), "negative MTP position rejected");
    expect(!bounds::mtp_window(4, 0, 1, -1, 4, 3, 1, false, plan), "negative accepted row rejected");
    expect(!bounds::mtp_window(4, 0, std::numeric_limits<int>::max(), 0, 4, 3, 1, false, plan),
           "huge input row count rejected before integer record arithmetic");
    expect(!bounds::mtp_window(4, 0, 1, 0, 4, -1, 1, false, plan), "negative draft depth rejected");
    expect(!bounds::mtp_window(4, 0, 1, 0, 4, 3, -1, false, plan), "negative remaining budget rejected");
    expect(bounds::mtp_window(4, 3, 1, 0, 4, 3, 1, true, plan) && plan.candidates == 0,
           "last legal catch-up row makes no speculative proposal");
    expect(bounds::mtp_window(4, 0, 4, 0, 4, 3, 4, false, plan) && plan.candidates == 2,
           "candidate depth also reserves next verifier's physical rows");
    expect(bounds::mtp_window(7, 0, 4, 0, 4, 1, 7, false, plan) && plan.candidates == 1,
           "batch maximum of one proposal is preserved");
    expect(bounds::mtp_window(7, 0, 4, 0, 4, 0, 7, false, plan) && plan.candidates == 0,
           "zero configured candidates use the single-row path");
    expect(bounds::mtp_window(7, 0, 4, 0, 4, 3, 0, true, plan) && plan.candidates == 0,
           "exhausted output budget needs no candidate");
    for (bool all : {false, true}) {
        check_plan(11, 7, 4, 0, 4, all);
        check_plan(11, 7, 4, 3, 1, all);
        check_plan(262144, 262140, 4, 1, 2, all);
        check_plan(i32, i32 - 4, 4, 2, big, all);
        for (int64_t cells : {4LL, 7LL, 262144LL})
            for (bool native : {false, true})
                for (bool reject : {false, true}) {
                    decode_budget(cells, cells - 1, 1, native, reject, all);
                    decode_budget(cells, cells - 3, 3, native, reject, all);
                    const int64_t last = std::min<int64_t>(4, cells - 1);
                    decode_budget(cells, cells - last, last, native, reject, all);
                }
        // Enumerate short, nonaligned tails and every partial acceptance.
        for (int cells = 1; cells <= 11; ++cells)
            for (int pos = 0; pos < cells; ++pos)
                for (int rows = 1; rows <= 4 && rows <= cells - pos; ++rows)
                    for (int accepted = 0; accepted < rows; ++accepted)
                        for (int remaining = 0; remaining <= 5; ++remaining)
                            check_plan(cells, pos, rows, accepted, remaining, all);
    }
    // A rejected bound has no permission to submit, and does not erase a sticky queue fault.
    strata::core::MtpCompletion ownership;
    std::string err;
    if (bounds::mtp_window(4, 4, 1, 0, 4, 3, 1, false, plan)) (void) ownership.submit(err);
    expect(ownership.submissions() == 0 && ownership.host_ready(), "invalid bounds precede any new writer");
    ownership.fail("injected earlier queue fault");
    expect(!ownership.complete([] { return true; }, [] { return std::string("wait failed"); }, err) &&
           err == "injected earlier queue fault", "bounds cannot clear completion's sticky failure");
    std::puts("Context-window bounds passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
