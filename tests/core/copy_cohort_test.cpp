// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/copy_cohort.hpp"
#include <cassert>
int main() {
    using strata::core::copy_cohort_fits;
    using strata::core::copy_cohort_adjacent;
    static_assert(copy_cohort_adjacent(2, 10, 1000, 2, 12, 1200, 2, 100));
    static_assert(!copy_cohort_adjacent(2, 10, 1000, 2, 13, 1300, 2, 100)); // resident expert gap
    static_assert(!copy_cohort_adjacent(2, 10, 1000, 3, 12, 1200, 2, 100)); // layer
    static_assert(!copy_cohort_adjacent(2, 10, UINTPTR_MAX - 1, 2, 12, 100, 2, 100));
    static_assert(!copy_cohort_adjacent(2, 10, 1000, 2, 12, 1201, 2, 100)); // source adjacency
    static_assert(copy_cohort_fits(37, 4, 64, 48, 256, 192, 0, 16, 4));
    static_assert(!copy_cohort_fits(37, 4, 32, 48, 256, 192, 0, 16, 4));
    static_assert(!copy_cohort_fits(37, 4, 64, 48, 228, 180, 0, 16, 3)); // credits
    static_assert(!copy_cohort_fits(37, 4, 64, 48, 256, 192, 14, 16, 4)); // wrap
    static_assert(!copy_cohort_fits(37, 4, 64, 48, 228, 180, 0, 16, 4)); // short allocation
    static_assert(!copy_cohort_fits(37, 4, size_t(UINT32_MAX) + 1, 48, SIZE_MAX, 192, 0, 16, 4));
    static_assert(!copy_cohort_fits(0, 4, 64, 48, 256, 192, 0, 16, 4));
    static_assert(!copy_cohort_fits(37, 17, 64, 48, 4096, 4096, 0, 32, 17));
    assert(copy_cohort_fits(37, 16, 64, 48, 1024, 768, 0, 16, 16));
}
