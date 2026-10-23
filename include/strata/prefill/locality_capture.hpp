#pragma once
// Main-authored bounded diagnostic; repository LICENSE applies.
#include "strata/prefill/selection_dump.hpp"
#include <algorithm>

namespace strata::prefill::detail {

// Diagnostic only. Both source spans belong to one still-live prefill chunk.
// The outer guard also drains previously submitted indexer work on any refusal
// or exception, before the caller can release its device scratch.
template<class Copy, class Drain, class Fatal>
bool locality_capture(const char* base, int64_t layer, int64_t first,
                      int64_t rows, int64_t capacity_rows, int64_t cap,
                      int64_t step_count, const int32_t* ids, const int32_t* steps,
                      Copy copy, Drain drain, Fatal fatal, std::string& error) {
    auto stop = [&]() noexcept {
        try { fatal(); } catch (...) {}
        std::abort();
    };
    DraftUploadGuard inherited(drain, stop);
    inherited.arm();
    const auto fits = [](int64_t a, int64_t b) {
        return a > 0 && b > 0 && uint64_t(a) <=
            std::numeric_limits<size_t>::max() / sizeof(int32_t) / uint64_t(b);
    };
    if (!base || !*base || !ids || !steps || layer < 0 || layer > INT32_MAX ||
        first < 0 || first > INT32_MAX || rows <= 0 || rows > capacity_rows ||
        capacity_rows > INT32_MAX || rows > INT32_MAX - first ||
        cap <= 0 || cap > INT32_MAX || step_count <= 0 || step_count > 64 ||
        !fits(capacity_rows, cap) || !fits(capacity_rows, step_count)) {
        error = "QSA locality capture: invalid bounded chunk geometry";
        return false;
    }
    try {
        const int64_t take = std::min<int64_t>(512, rows), offset = rows - take;
        const std::string id_path = std::string(base) + ".ids";
        const std::string step_path = std::string(base) + ".steps";
        if (!selection_dump(id_path.c_str(), layer, first + offset, take, cap,
                ids + size_t(offset) * size_t(cap), copy, drain, fatal, error) ||
            !selection_dump(step_path.c_str(), layer, first + offset, take, step_count,
                steps + size_t(offset) * size_t(step_count), copy, drain, fatal, error))
            return false;
        // The successful second helper has drained this same stream, including
        // both copies and all prior indexer operations. It owns no pending host DMA.
        inherited.retire();
        return true;
    } catch (const std::exception& exc) {
        error = std::string("QSA locality capture: ") + exc.what();
        return false;
    }
}
} // namespace strata::prefill::detail
