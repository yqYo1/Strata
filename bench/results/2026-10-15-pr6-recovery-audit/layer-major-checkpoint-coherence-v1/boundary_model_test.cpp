// PROPOSAL ONLY. Not executed by the implementation agent.
// Standalone CPU planner/oracle; no model, GPU, or tensor correctness claim.
#include "strata/program/prompt_checkpoint.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using strata::program::checkpoint_next;
using strata::program::checkpoint_segment_end;
void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }

// Independent oracle: walk tokens and observe chunk-end events. No division,
// remainder, round-up formula, or calls to the candidate planner.
int64_t oracle_end(int64_t a, int64_t b, int64_t next, int64_t chunk) {
    int64_t in_chunk = 0;
    while (a < b) {
        ++a;
        ++in_chunk;
        if (a == b || in_chunk == chunk) {
            if (a >= next || a == b) return a;
            in_chunk = 0;
        }
    }
    throw std::runtime_error("oracle empty segment");
}

int64_t oracle_next(int64_t a, int64_t interval) {
    // Only small intervals in this walking oracle; large cases have explicit
    // expected constants below, so it cannot accidentally run INT64_MAX steps.
    if (interval == 0) return INT64_MAX;
    for (int64_t i = 0; i < interval; ++i) {
        if (a == INT64_MAX) return a;
        ++a;
    }
    return a;
}

std::vector<int64_t> oracle_schedule(int64_t a, const std::vector<int64_t>& ends,
                                    int64_t interval, int64_t chunk,
                                    int64_t short_read) {
    std::vector<int64_t> saved;
    auto next = oracle_next(a, interval);
    for (auto b : ends) {
        if (b <= a) continue;
        const bool windows = b - a <= short_read;
        if (windows) { a = b; continue; } // existing windows do not make periodic saves
        int64_t in_chunk = 0;
        while (a < b) {
            ++a; ++in_chunk;
            if (a == b || in_chunk == chunk) {
                if (interval > 0 && a >= next) {
                    saved.push_back(a);
                    next = oracle_next(a, interval);
                }
                in_chunk = 0;
            }
        }
    }
    return saved;
}

std::vector<int64_t> candidate_schedule(int64_t a, const std::vector<int64_t>& ends,
                                       int64_t interval, int64_t chunk,
                                       int64_t short_read, int fail_run = -1,
                                       bool fail_refill = false) {
    std::vector<int64_t> saved;
    auto next = checkpoint_next(a, interval);
    int runs = 0;
    std::array<int64_t, 36> gdn_at{};
    for (auto b : ends) {
        if (b <= a) continue;
        if (b - a <= short_read) { a = b; continue; }
        while (a < b) {
            auto end = interval > 0 ? checkpoint_segment_end(a, b, next, chunk) : b;
            require(end > a && end <= b, "nonprogress/out-of-range segment");
            // Layer-major earlier layers reach the segment end before final
            // callbacks. A failed/cancelled run publishes no checkpoint.
            ++runs;
            if (runs == fail_run) return saved;
            gdn_at.fill(end);
            bool mtp_complete = true; // completed callbacks, then wrapper restore
            bool refill_complete = !fail_refill;
            if (!refill_complete) return saved;
            if (interval > 0 && end >= next) {
                require(mtp_complete && refill_complete, "save before retirement");
                for (auto p : gdn_at) require(p == end, "GDN/prefix boundary mismatch");
                // QSA tail/dead/block position, PLE history/previous IDs and
                // target/draft KV need actual-engine evidence, not these tags.
                saved.push_back(end);
                next = checkpoint_next(end, interval);
            }
            a = end;
        }
    }
    return saved;
}
}

int main() try {
    for (int64_t a = 0; a < 9; ++a)
        for (int64_t len = 1; len <= 35; ++len)
            for (int64_t chunk = 1; chunk <= 12; ++chunk)
                for (int64_t next = 0; next <= a + len + 4; ++next)
                    require(checkpoint_segment_end(a, a + len, next, chunk) ==
                            oracle_end(a, a + len, next, chunk), "planner differs from event oracle");
    for (auto interval : {int64_t(0), int64_t(1), int64_t(3), int64_t(16), int64_t(17)})
        for (auto chunk : {int64_t(1), int64_t(4), int64_t(8), int64_t(12)})
            for (auto short_read : {int64_t(0), int64_t(3), int64_t(8)})
                for (auto start : {int64_t(0), int64_t(5)}) {
                    const std::vector<int64_t> outer = {-1, 7, 21, 47};
                    require(candidate_schedule(start, outer, interval, chunk, short_read) ==
                            oracle_schedule(start, outer, interval, chunk, short_read),
                            "root/turn/window schedule differs");
                }
    require(checkpoint_segment_end(0, 34602, 16384, 8192) == 16384, "16K default boundary");
    require(checkpoint_segment_end(0, 34602, 16385, 8192) == 24576, "interval rounds to next chunk");
    require(checkpoint_segment_end(100, 103, 200, 8192) == 103, "outer partial tail");
    require(checkpoint_segment_end(100, 200, 99, 8) == 108, "overdue after windows uses next chunk");
    require(checkpoint_segment_end(INT64_MAX - 20, INT64_MAX, INT64_MAX - 1, 8) == INT64_MAX,
            "near-max final tail");
    require(checkpoint_segment_end(INT64_MAX - 20, INT64_MAX, INT64_MAX - 18, 8) == INT64_MAX - 12,
            "near-max rounded boundary");
    require(checkpoint_segment_end(0, INT64_MAX, 1, INT64_MAX) == INT64_MAX, "max chunk no overflow");
    require(checkpoint_next(INT64_MAX - 3, 4) == INT64_MAX, "next saturation");
    require(checkpoint_next(1, INT64_MAX) == INT64_MAX, "max interval saturation");
    require(checkpoint_next(0, INT64_MAX) == INT64_MAX, "max exact interval");
    require(checkpoint_next(INT64_MAX, 0) == INT64_MAX, "zero interval disabled");
    require(candidate_schedule(0, {40}, 16, 8, 0, 1).empty(), "failed first run saved state");
    require(candidate_schedule(0, {40}, 16, 8, 0, 2) == std::vector<int64_t>{16},
            "cancel/error after one completed segment lost/added checkpoint");
    require(candidate_schedule(0, {40}, 16, 8, 0, -1, true).empty(), "failed refill saved state");
    // Explicit counterexample: original full-segment layer-major save at 16
    // copies earlier GDN layers already at 40. Equal byte sizes cannot fix it.
    std::array<int64_t, 36> unsafe_gdn_at{};
    unsafe_gdn_at.fill(40);
    require(unsafe_gdn_at.front() != 16, "unsafe callback counterexample missing");
    for (const auto args : {std::array<int64_t, 4>{-1, 1, 0, 1}, {0, 0, 0, 1},
                            {0, 1, -1, 1}, {0, 1, 0, 0}}) {
        bool rejected = false;
        try { (void) checkpoint_segment_end(args[0], args[1], args[2], args[3]); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid boundary accepted");
    }
    std::cout << "PASS CPU boundary oracle only; no engine/256K qualification\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
}
