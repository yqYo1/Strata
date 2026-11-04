// Main-authored row-uniform copy component; repository LICENSE.
// Unselected prototype for the named mapped-row copy comparison. The existing
// production copy_rows_from_mapped entry remains unchanged.
#pragma once

#include "strata/sycl/launch.hpp"
#include <cstdint>

namespace strata::kernels::component {

// Caller contract, matching the existing copy's controlled producer:
// - dst/src/count/hit_rows are live, same-context USM visible to the queue;
// - 0 <= *count <= rows, and hit_rows[0..*count) contain valid row IDs;
// - input stores complete before submission/replay, and no source is changed
//   until completion. The destination is disjoint from every input.
// No model dispatch, environment selector, extra queue or host readback here.
template <int Workgroup, int Elements>
void mapped_row_copy(float* dst, const float* src, int64_t rows, int64_t width,
                     const int32_t* hit_rows, const int32_t* count, void* stream) {
    static_assert(Workgroup == 64 || Workgroup == 128 || Workgroup == 256 ||
                  Workgroup == 512 || Workgroup == 1024);
    static_assert(Elements == 1 || Elements == 4);
    using namespace strata::sycl_backend;
    const size_t n = checked_count(rows, width);
    if (!n) return;
    if (n > SIZE_MAX / sizeof(float) || size_t(rows) > SIZE_MAX / Workgroup)
        throw std::invalid_argument("mapped row copy extent overflow");
    validate_spans({{dst, n * sizeof(float)}}, {{src, n * sizeof(float)},
        {hit_rows, size_t(rows) * sizeof(int32_t)}, {count, sizeof(int32_t)}});
    auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>(size_t(rows) * Workgroup, Workgroup),
        [=](sycl::nd_item<1> item) {
            const size_t row = item.get_group_linear_id();
            const size_t lane = item.get_local_linear_id();
            int hit = 0;
            if (lane == 0) {
                const int c = *count;
                for (int j = 0; j < c; ++j)
                    hit |= hit_rows[j] == int64_t(row);
            }
            // Every work-item participates before any row-dependent branch.
            hit = sycl::group_broadcast(item.get_group(), hit, 0);
            const size_t base = row * size_t(width);
            const size_t bulk = size_t(width) / Elements * Elements;
            if (hit) {
                for (size_t c = lane * Elements; c < bulk;
                     c += Workgroup * Elements) {
#pragma unroll
                    for (int j = 0; j < Elements; ++j) dst[base + c + j] = 0.f;
                }
                for (size_t c = bulk + lane; c < size_t(width); c += Workgroup)
                    dst[base + c] = 0.f;
            } else {
                for (size_t c = lane * Elements; c < bulk;
                     c += Workgroup * Elements) {
                    // Scalar float accesses also permit four-byte-aligned rows.
                    // Elements4 is a source grouping, not an emitted-ISA claim.
#pragma unroll
                    for (int j = 0; j < Elements; ++j)
                        dst[base + c + j] = src[base + c + j];
                }
                for (size_t c = bulk + lane; c < size_t(width); c += Workgroup)
                    dst[base + c] = src[base + c];
            }
        });
    finish(stream, event);
}
} // namespace strata::kernels::component
