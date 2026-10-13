// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace strata::core {
constexpr bool copy_cohort_adjacent(int64_t layer, int64_t expert, uintptr_t source,
                                    int64_t next_layer, int64_t next_expert, uintptr_t next_source,
                                    size_t row, size_t width) {
    if (!width || width > UINT32_MAX || row > 15 || layer != next_layer || expert < 0 ||
        expert > INT64_MAX - static_cast<int64_t>(row) ||
        next_expert != expert + static_cast<int64_t>(row) || source > UINTPTR_MAX - row * width) return false;
    return next_source == source + row * width;
}
constexpr bool copy_cohort_fits(size_t width, size_t height, size_t dp, size_t sp,
                               size_t dst_available, size_t src_available,
                               size_t slot, size_t ring, size_t credit) {
    constexpr size_t u32 = std::numeric_limits<uint32_t>::max();
    if (!width || height < 2 || height > 16 || width > dp || width > sp ||
        width > u32 || dp > u32 || sp > u32 || height > credit || slot >= ring || height > ring - slot)
        return false;
    const auto fits = [=](size_t pitch, size_t available) {
        return available >= width && height - 1 <= (available - width) / pitch;
    };
    return fits(dp, dst_available) && fits(sp, src_available);
}
class CopyCohortCounts {
public:
    CopyCohortCounts() {
        const char* flag = std::getenv("STRATA_PREFILL_COPY_COHORT_COUNTS");
        enabled_ = flag && std::strcmp(flag, "1") == 0;
    }
    void add(bool cohort, size_t rows, size_t width) {
        if (!enabled_) return;
        auto sum = [this](uint64_t& value, uint64_t delta) {
            if (delta > UINT64_MAX - value) overflow_ = true;
            else value += delta;
        };
        if (width && rows > UINT64_MAX / width) { overflow_ = true; return; }
        sum(cohort ? copies_ : fallback_, 1);
        sum(cohort ? rows_ : fallback_rows_, rows);
        sum(cohort ? bytes_ : fallback_bytes_, rows * width);
    }
    void complete() { complete_ = true; }
    ~CopyCohortCounts() {
        if (enabled_) std::fprintf(stderr, "strata prefill copy cohorts: status=%s copies=%llu rows=%llu bytes=%llu "
            "fallback_copies=%llu fallback_rows=%llu fallback_bytes=%llu max_height=16 overflow=%d\n",
            complete_ && !overflow_ ? "complete" : "incomplete", (unsigned long long) copies_,
            (unsigned long long) rows_, (unsigned long long) bytes_, (unsigned long long) fallback_,
            (unsigned long long) fallback_rows_, (unsigned long long) fallback_bytes_, (int) overflow_);
    }
private:
    bool enabled_ = false, overflow_ = false, complete_ = false;
    uint64_t copies_ = 0, rows_ = 0, bytes_ = 0, fallback_ = 0, fallback_rows_ = 0, fallback_bytes_ = 0;
};
}  // namespace strata::core
