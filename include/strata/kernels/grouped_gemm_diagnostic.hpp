// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <atomic>
#include <cstddef>
namespace strata::kernels {
struct GroupedGemmSubmission {
    sycl::event event;  // Exact selected kernel command; never a barrier or timing tag.
    const char* backend;
    int64_t max_rows, n, k;
    int groups, tile_m, tile_n, pipeline;  // XMX only; zero means fallback leaf owns tile selection.
};
// Created only after a checked D2H of actual device bounds. Copying a validation does not extend buffer lifetime.
class GroupedBoundsValidation {
public:
    const int32_t* device_bounds() const { return bounds_; }
    const sycl::queue* queue() const { return queue_; }
    int64_t rows() const { return rows_; }
    int64_t max_rows() const { return max_rows_; }
private:
    GroupedBoundsValidation(const int32_t* bounds, const sycl::queue* queue, int64_t rows, int64_t max_rows)
        : bounds_(bounds), queue_(queue), rows_(rows), max_rows_(max_rows) {}
    const int32_t* bounds_;
    const sycl::queue* queue_;
    int64_t rows_, max_rows_;
    friend GroupedBoundsValidation validate_grouped_device_bounds(sycl::queue&, const int32_t*,
        int32_t*, size_t, int64_t, int64_t, const std::atomic<bool>&);
};
// Caller owns fixed scratch[17] outside catch and keeps it alive through successful drain on failure.
// Device bounds remain immutable from initialization through validation and selected kernel completion.
GroupedBoundsValidation validate_grouped_device_bounds(sycl::queue& q, const int32_t* bounds,
    int32_t* scratch, size_t scratch_count, int64_t rows, int64_t max_rows, const std::atomic<bool>& async_error);
// Caller validates X/W/Y extents/alignment and retains all operands until successful queue drain.
// Diagnostic queue is profiling/in-order, Runtime's exact device/context. No queue mutation.
GroupedGemmSubmission grouped_gemm_diagnostic(sycl::queue& q, const uint16_t* X, const uint16_t* W,
    int64_t w_stride, float* Y, const GroupedBoundsValidation& bounds, int G,
    int64_t max_rows, int64_t N, int64_t K);
}  // namespace strata::kernels
