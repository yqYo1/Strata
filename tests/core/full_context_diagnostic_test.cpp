#include "strata/program/full_context_diagnostic.hpp"

#include <cstdio>
#include <initializer_list>
#include <limits>

int main() {
    namespace d = strata::program::full_context_diagnostic;
    int cases = 0;
    auto check = [&](bool ok) { ++cases; return ok; };
    for (const auto context : {int64_t(-1), int64_t(0), int64_t(262143), int64_t(262144),
                               int64_t(262145), std::numeric_limits<int64_t>::max()}) {
        for (const size_t prompt : {size_t(0), size_t(262143), size_t(262144), size_t(262145),
                                    std::numeric_limits<size_t>::max()}) {
            for (const auto output : {int64_t(-1), int64_t(0), int64_t(1),
                                      std::numeric_limits<int64_t>::max()}) {
                const bool expected = context == 262144 && prompt == 262144 && output == 0;
                if (!check(d::request(context, prompt, output) == expected)) return 1;
            }
        }
    }
    if (!check(d::last_window(true, 262144, 262143, 1, 42, 42, 0))) return 1;
    if (!check(!d::last_window(false, 262144, 262143, 1, 42, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262144, 262143, 1, 41, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262144, 262143, 1, 42, 42, 1))) return 1;
    for (int64_t position : {int64_t(-1), int64_t(262142), int64_t(262144),
                             std::numeric_limits<int64_t>::max()})
        if (!check(!d::last_window(true, 262144, position, 1, 42, 42, 0))) return 1;
    for (int width : {-1, 0, 2, 4})
        if (!check(!d::last_window(true, 262144, 262143, width, 42, 42, 0))) return 1;
    if (!check(!d::last_window(true, 262143, 262143, 1, 42, 42, 0))) return 1;
    // First-window eligibility is captured before the loop clears its flag.
    bool first = true;
    const bool terminal = d::last_window(first, 262144, 262143, 1, 42, 42, 0);
    first = false;
    if (!check(terminal && !d::last_window(first, 262144, 262143, 1, 42, 42, 0))) return 1;
    std::printf("cases=%d full_context_request_and_terminal_bounds=PASS\n", cases);
}
