// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/runtime.hpp"
#include "strata/kernels/grouped_gemm_diagnostic.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // One cell per owned process: phase gu/down and maximum rows 1/8/16/128/129/256/384/512.
    if (argc != 3) return 2;
    const std::string phase = argv[1], count = argv[2];
    int nt = 0;
    for (int allowed : {1, 8, 16, 128, 129, 256, 384, 512})
        if (count == std::to_string(allowed)) nt = allowed;
    if (!nt || (phase != "gu" && phase != "down")) return 2;
    using Clock = std::chrono::steady_clock;
    const auto cell_start = Clock::now();
    const int n = phase == "gu" ? 1280 : 2560, k = phase == "gu" ? 2560 : 640;
    auto& runtime = strata::core::Runtime::get();
    if (!runtime.device().has(sycl::aspect::queue_profiling) ||
        !runtime.device().has(sycl::aspect::usm_device_allocations)) return 77;
    std::atomic<bool> async_error{false};
    sycl::queue q(runtime.context(), runtime.device(), [&](const sycl::exception_list& errors) {
        for (const auto& error : errors) {
            async_error.store(true);
            try { std::rethrow_exception(error); }
            catch (const std::exception& e) { std::fprintf(stderr, "async: %.512s\n", e.what()); }
        }
    }, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    int32_t* bounds = nullptr;
    sycl::half *x = nullptr, *w = nullptr;
    float* y = nullptr;
    // Host owners outlive the catch/drain: a throwing copy must not unwind their source/destination.
    std::vector<sycl::half> host_x, host_w;
    std::vector<float> host_y;
    int32_t host_bounds[17]{};
    int32_t validation_scratch[17]{};  // lives through catch/drain on D2H failure
    const auto mul = [](size_t lhs, size_t rhs) {
        if (rhs && lhs > std::numeric_limits<size_t>::max() / rhs)
            throw std::runtime_error("extent multiplication overflow");
        return lhs * rhs;
    };
    const auto add = [](size_t lhs, size_t rhs) {
        if (lhs > std::numeric_limits<size_t>::max() - rhs)
            throw std::runtime_error("extent addition overflow");
        return lhs + rhs;
    };
    const auto checked_event = [&](sycl::event event) {
        event.wait_and_throw();
        q.wait_and_throw();
        if (async_error.load() || event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete)
            throw std::runtime_error("copy completion failure");
    };
    try {
        // Empty and ragged experts, exact max NT, final positive group and non-tile tails.
        for (int g = 0; g < 16; ++g) {
            const int rows = g == 0 ? nt : g % 5 == 0 ? 0 : std::max(1, nt - g % 7);
            host_bounds[g + 1] = host_bounds[g] + rows;
        }
        const size_t rows = static_cast<size_t>(host_bounds[16]);
        const size_t xc = mul(rows, static_cast<size_t>(k));
        const size_t wc = mul(mul(16, static_cast<size_t>(n)), static_cast<size_t>(k));
        const size_t yc = add(mul(rows, static_cast<size_t>(n)), 16);
        const size_t xb = mul(xc, sizeof(sycl::half)), wb = mul(wc, sizeof(sycl::half));
        const size_t yb = mul(yc, sizeof(float));
        const size_t gpu_bytes = add(add(add(xb, wb), yb), sizeof(host_bounds));
        const size_t host_bytes = add(add(add(xb, wb), yb), sizeof(host_bounds) * 2);
        constexpr size_t budget = 192ULL * 1024 * 1024;
        const auto limit = runtime.device().get_info<sycl::info::device::max_mem_alloc_size>();
        if (xb > limit || wb > limit || yb > limit || sizeof(host_bounds) > limit ||
            gpu_bytes > budget || host_bytes > budget) {
            std::fprintf(stderr, "CELL status=UNAVAILABLE reason=allocation_budget\n");
            return 77;
        }
        host_x.resize(xc); host_w.resize(wc); host_y.resize(yc);
        const size_t host_capacity_bytes = add(add(mul(host_x.capacity(), sizeof(sycl::half)),
            mul(host_w.capacity(), sizeof(sycl::half))), add(mul(host_y.capacity(), sizeof(float)), sizeof(host_bounds) * 2));
        if (host_capacity_bytes > budget) {
            std::fprintf(stderr, "CELL status=UNAVAILABLE reason=host_capacity_budget\n");
            return 77;
        }
        // Bounds are device-USM as in production, immutable through validation and kernel completion.
        bounds = sycl::aligned_alloc_device<int32_t>(64, 17, q);
        x = sycl::aligned_alloc_device<sycl::half>(64, xc, q);
        w = sycl::aligned_alloc_device<sycl::half>(64, wc, q);
        y = sycl::aligned_alloc_device<float>(64, yc, q);
        if (!bounds || !x || !w || !y) throw std::runtime_error("USM allocation failed");
        for (size_t r = 0; r < rows; ++r)
            for (int j = 0; j < k; ++j)
                host_x[r * k + j] = sycl::half(static_cast<float>(static_cast<int>((r + j) % 3) - 1) / 16.f);
        for (int g = 0; g < 16; ++g)
            for (int c = 0; c < n; ++c)
                for (int j = 0; j < k; ++j)
                    host_w[(static_cast<size_t>(g) * n + c) * k + j] = sycl::half(
                        static_cast<float>((static_cast<int>((j + g) % 3) - 1) * (c % 3 - 1)) / 16.f);
        for (size_t i = 0; i < yc; ++i) host_y[i] = 12345.f;
        std::printf("BOUNDS");
        for (int g = 0; g <= 16; ++g) std::printf(" %d", host_bounds[g]);
        std::printf("\n");
        const auto init_start = Clock::now();
        const auto init_x = q.memcpy(x, host_x.data(), xb);
        const auto init_w = q.memcpy(w, host_w.data(), wb);
        const auto init_y = q.memcpy(y, host_y.data(), yb);
        const auto init_bounds = q.memcpy(bounds, host_bounds, sizeof(host_bounds));
        checked_event(init_x); checked_event(init_w); checked_event(init_y); checked_event(init_bounds);
        const auto init_end = Clock::now();
        // Initialization is completed/checked before the exact kernel command timing interval.
        const auto bounds_validation_start = Clock::now();
        const auto validated_bounds = strata::kernels::validate_grouped_device_bounds(q, bounds,
            validation_scratch, 17, static_cast<int64_t>(rows), nt, async_error);
        const auto bounds_validation_end = Clock::now();
        if (std::memcmp(host_bounds, validation_scratch, sizeof(host_bounds)) != 0)
            throw std::runtime_error("actual bounds differ from initialized geometry");
        const auto before = Clock::now();
        auto submitted = strata::kernels::grouped_gemm_diagnostic(q,
            reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const uint16_t*>(w),
            static_cast<int64_t>(n) * k, y, validated_bounds, 16, nt, n, k);
        const auto after_submit = Clock::now();
        submitted.event.wait_and_throw();
        q.wait_and_throw();
        const auto after_complete = Clock::now();
        if (async_error.load() || submitted.event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) throw std::runtime_error("completion failure");
        const auto start = submitted.event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end = submitted.event.get_profiling_info<sycl::info::event_profiling::command_end>();
        if (start == 0 || end <= start) throw std::runtime_error("invalid event timestamps");
        const auto copyback_start = Clock::now();
        const auto copyback = q.memcpy(host_y.data(), y, yb);
        checked_event(copyback);
        const auto copyback_end = Clock::now();
        // Independent host dot of actual FP16 operands. W's column factor allows checking every output
        // without billions of redundant reference products; no device arithmetic is mirrored.
        for (int g = 0; g < 16; ++g) {
            const int ref_column = 2;  // column factor +1
            for (int r = validation_scratch[g]; r < validation_scratch[g + 1]; ++r) {
                double dot = 0;
                for (int j = 0; j < k; ++j)
                    dot += static_cast<float>(host_x[static_cast<size_t>(r) * k + j]) *
                           static_cast<float>(host_w[(static_cast<size_t>(g) * n + ref_column) * k + j]);
                for (int c = 0; c < n; ++c) {
                    const double expected = dot * (c % 3 - 1);
                    const float actual = host_y[static_cast<size_t>(r) * n + c];
                    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5 * (1 + std::abs(expected)))
                        throw std::runtime_error("full-output reference mismatch");
                }
            }
        }
        for (size_t i = rows * n; i < rows * n + 16; ++i)
            if (host_y[i] != 12345.f) throw std::runtime_error("output tail canary changed");
        q.wait_and_throw();
        if (async_error.load()) throw std::runtime_error("late async failure");
        sycl::free(bounds, q); sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
        bounds = nullptr; x = nullptr; w = nullptr; y = nullptr;
        const double bounds_validation_us = std::chrono::duration<double, std::micro>(
            bounds_validation_end - bounds_validation_start).count();
        const double h2d_us = std::chrono::duration<double, std::micro>(init_end - init_start).count();
        const double d2h_us = std::chrono::duration<double, std::micro>(copyback_end - copyback_start).count();
        const double wholecell_us = std::chrono::duration<double, std::micro>(Clock::now() - cell_start).count();
        const double submit_us = std::chrono::duration<double, std::micro>(after_submit - before).count();
        const double completion_us = std::chrono::duration<double, std::micro>(after_complete - before).count();
        std::printf("CELL status=PASS scope=synthetic_profiling_queue phase=%s backend=%s groups=16 "
                    "max_rows=%d rows=%zu N=%d K=%d tile_m=%d tile_n=%d pipeline=%d "
                    "start_ns=%llu end_ns=%llu event_ns=%llu host_submit_us=%.3f host_completion_us=%.3f "
                    "h2d_host_us=%.3f d2h_host_us=%.3f wholecell_host_us=%.3f bounds_validation_host_us=%.3f "
                    "gpu_requested_bytes=%zu host_capacity_bytes=%zu operands=device_usm bounds=device_usm\n",
                    phase.c_str(), submitted.backend, nt, rows, n, k, submitted.tile_m, submitted.tile_n,
                    submitted.pipeline, static_cast<unsigned long long>(start), static_cast<unsigned long long>(end),
                    static_cast<unsigned long long>(end - start), submit_us, completion_us,
                    h2d_us, d2h_us, wholecell_us, bounds_validation_us, gpu_bytes, host_capacity_bytes);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CELL status=FAILED reason=%.512s\n", e.what());
        // Never unwind/free memory while failed queue work might still reference it.
        try { q.wait_and_throw(); }
        catch (...) { std::fprintf(stderr, "drain failed; fail-stop without freeing GPU state\n"); std::_Exit(1); }
        if (async_error.load()) {
            std::fprintf(stderr, "asynchronous failure; fail-stop without freeing uncertain GPU state\n");
            std::_Exit(1);
        }
        if (bounds) sycl::free(bounds, q);
        if (x) sycl::free(x, q);
        if (w) sycl::free(w, q);
        if (y) sycl::free(y, q);
        return 1;
    }
}
