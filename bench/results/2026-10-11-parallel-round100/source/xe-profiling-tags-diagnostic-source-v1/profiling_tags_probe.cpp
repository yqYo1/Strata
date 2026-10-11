// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Standalone integration diagnostic: link the actual production gpu.cpp/device.cpp.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/prefill/publication.hpp"

#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#ifndef STRATA_PROBE_MODE_FREE
#error Set STRATA_PROBE_MODE_FREE to 1 (free) or 0 (contrib-llvm) explicitly.
#endif
#if STRATA_PROBE_MODE_FREE != 0 && STRATA_PROBE_MODE_FREE != 1
#error Invalid STRATA_PROBE_MODE_FREE.
#endif
#if STRATA_PROBE_MODE_FREE && !defined(STRATA_NO_BLAS)
#error The free recipe must define STRATA_NO_BLAS.
#endif

namespace {
namespace gpu = strata::gpu;
namespace publication = strata::prefill::publication;
constexpr size_t kBytes = size_t{16} << 20;
constexpr size_t kWords = kBytes / sizeof(uint32_t);
constexpr unsigned kSamples = 8;
constexpr float kSentinel = -12345.25F;
// A deliberately loose sanity ceiling, not a timer-domain calibration or accuracy claim.
constexpr double kWallMultiplier = 1.25;
constexpr double kWallAllowanceMs = 10.0;
#ifdef STRATA_NO_BLAS
constexpr int kNoBlas = 1;
#else
constexpr int kNoBlas = 0;
#endif

struct Counts {
    size_t api_calls = 0;
    size_t records = 0;
    size_t copies = 0;
    size_t compute_waits = 0;
    size_t cleanup_waits = 0;
    size_t elapsed_queries = 0;
    size_t words_checked = 0;
    unsigned samples_completed = 0;
    unsigned positive_reversals = 0;
};

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Call> bool api(Counts& counts, const char* operation, Call call) {
    ++counts.api_calls;
    try {
        const bool ok = call();
        if (!ok) {
            std::fprintf(stderr, "API operation=%s result=0 error=%s\n", operation, gpu::last_error());
            std::fflush(stderr);
        }
        return ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "API operation=%s exception=%s\n", operation, e.what());
        std::fflush(stderr);
        throw;
    } catch (...) {
        std::fprintf(stderr, "API operation=%s exception=unknown\n", operation);
        std::fflush(stderr);
        throw;
    }
}

template<class Call> void require_api(Counts& counts, const char* operation, Call call) {
    if (!api(counts, operation, call))
        throw std::runtime_error(std::string(operation) + ": " + gpu::last_error());
}

// A failed/throwing completion call cannot authorize another wait or stack release.
template<class Call> void completion(Counts& counts, const char* operation, Call call) noexcept {
    ++counts.api_calls;
    try {
        const bool ok = call();
        if (!ok) std::fflush(stdout);
        publication::cleanup_require(ok, operation, gpu::last_error);
    } catch (const std::exception& e) {
        std::fflush(stdout);
        publication::cleanup_require(false, operation, [&] { return e.what(); });
    } catch (...) {
        std::fflush(stdout);
        publication::cleanup_require(false, operation, [] { return "unknown exception"; });
    }
}

struct Owned {
    Counts& counts;
    strata::core::Runtime& runtime;
    gpu::Stream stream;
    uint32_t* source = nullptr;
    uint32_t* roundtrip = nullptr;
    uint32_t* device = nullptr;
    gpu::Event* begin = nullptr;
    gpu::Event* end = nullptr;
    gpu::Event* unrecorded = nullptr;
    gpu::Event* untimed_begin = nullptr;
    gpu::Event* untimed_end = nullptr;
    bool open = true;

    Owned(Counts& c, strata::core::Runtime& r)
        : counts(c), runtime(r), stream(static_cast<gpu::Stream>(&r.compute())) {}
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned() { close(); }

    void close() noexcept {
        if (!open) return;
        // All probe submissions use compute. Also drain production's transfer/other
        // queues before release; partial construction follows this same path.
        ++counts.cleanup_waits;
        completion(counts, "profiling probe cleanup device_sync", gpu::device_sync);
        try {
            gpu::event_destroy(begin);
            gpu::event_destroy(end);
            gpu::event_destroy(unrecorded);
            gpu::event_destroy(untimed_begin);
            gpu::event_destroy(untimed_end);
            // Probe-owned real USM: free in its actual Runtime context only after
            // the checked terminal drain, without introducing another wait here.
            if (source) sycl::free(source, runtime.context());
            if (roundtrip) sycl::free(roundtrip, runtime.context());
            if (device) sycl::free(device, runtime.context());
            open = false;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "profiling probe: release failed after checked completion: %s; exiting before further release\n", e.what());
            std::fflush(stderr);
            std::_Exit(74);
        } catch (...) {
            std::fprintf(stderr, "profiling probe: release failed after checked completion: unknown exception; exiting before further release\n");
            std::fflush(stderr);
            std::_Exit(74);
        }
    }

    void wait() noexcept {
        ++counts.compute_waits;
        completion(counts, "profiling probe compute stream_sync", [&] { return gpu::stream_sync(stream); });
    }
};

// Parse only the guard's diagnostic. Timestamp subtraction/conversion remains
// exclusively in production event_elapsed_ms, not in this probe.
bool reversed_endpoints(std::string_view text, uint64_t& later, uint64_t& earlier) {
    constexpr std::string_view prefix = "event_elapsed_ms: end < start (start=";
    if (!text.starts_with(prefix)) return false;
    text.remove_prefix(prefix.size());
    const char* const finish = text.data() + text.size();
    auto start = std::from_chars(text.data(), finish, later);
    if (start.ec != std::errc{}) return false;
    text = std::string_view(start.ptr, static_cast<size_t>(finish - start.ptr));
    constexpr std::string_view middle = ", end=";
    if (!text.starts_with(middle)) return false;
    text.remove_prefix(middle.size());
    auto end = std::from_chars(text.data(), finish, earlier);
    return end.ec == std::errc{} && end.ptr != finish && finish - end.ptr == 1 && *end.ptr == ')';
}

bool unchanged(float value) {
    return std::bit_cast<uint32_t>(value) == std::bit_cast<uint32_t>(kSentinel);
}

void record(Owned& owned, gpu::Event* event, const char* operation) {
    ++owned.counts.records;
    require_api(owned.counts, operation, [&] { return gpu::event_record(event, owned.stream); });
}

bool elapsed(Counts& counts, float& value, gpu::Event* from, gpu::Event* to, const char* operation) {
    ++counts.elapsed_queries;
    return api(counts, operation, [&] { return gpu::event_elapsed_ms(&value, from, to); });
}

void expected_refusal(Owned& owned, const char* name, gpu::Event* from, gpu::Event* to,
                      const char* expected_error) {
    float value = kSentinel;
    const bool ok = elapsed(owned.counts, value, from, to, name);
    const std::string error = gpu::last_error();
    std::printf("CONTROL name=%s result=%d sentinel_unchanged=%d error=%s\n",
                name, static_cast<int>(ok), static_cast<int>(unchanged(value)), error.c_str());
    std::fflush(stdout);
    check(!ok && unchanged(value) && error == expected_error, "negative control refusal contract failed");
}

void sample(Owned& owned, unsigned index) {
    Counts& counts = owned.counts;
    const size_t api_before = counts.api_calls;
    std::printf("SAMPLE_BEGIN index=%u planned_copies=2 planned_transfer_bytes=%zu\n", index, 2 * kBytes);
    std::fflush(stdout);
    // Previous sample/controls already completed. Never modify host storage while
    // a submitted command may still own it; poison differs from every source word.
    for (size_t i = 0; i < kWords; ++i) {
        owned.source[i] = static_cast<uint32_t>(i) * UINT32_C(2654435761) ^
                          (UINT32_C(0x6a09e667) + index * UINT32_C(0x9e3779b9));
        owned.roundtrip[i] = ~owned.source[i];
    }
    const auto wall_begin = std::chrono::steady_clock::now();
    record(owned, owned.begin, "sample begin event_record");
    ++counts.copies;
    require_api(counts, "sample host-to-device copy_async", [&] {
        return gpu::copy_async(owned.device, owned.source, kBytes, owned.stream);
    });
    ++counts.copies;
    require_api(counts, "sample device-to-host copy_async", [&] {
        return gpu::copy_async(owned.roundtrip, owned.device, kBytes, owned.stream);
    });
    record(owned, owned.end, "sample end event_record");
    owned.wait();
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    for (size_t i = 0; i < kWords; ++i) {
        ++counts.words_checked;
        if (owned.roundtrip[i] != owned.source[i]) {
            std::fprintf(stderr, "DATA sample=%u word=%zu expected=%08" PRIx32 " actual=%08" PRIx32 "\n",
                         index, i, owned.source[i], owned.roundtrip[i]);
            throw std::runtime_error("roundtrip data mismatch");
        }
    }

    float ms = kSentinel;
    const bool normal_ok = elapsed(counts, ms, owned.begin, owned.end, "sample normal event_elapsed_ms");
    const std::string normal_error = normal_ok ? "none" : gpu::last_error();
    float reversed_ms = kSentinel;
    const bool reverse_ok = elapsed(counts, reversed_ms, owned.end, owned.begin, "sample reversed event_elapsed_ms");
    const std::string reverse_error = reverse_ok ? "none" : gpu::last_error();
    uint64_t raw_later = 0, raw_earlier = 0;
    const bool endpoints = !reverse_ok && reversed_endpoints(reverse_error, raw_later, raw_earlier);
    float same_ms = kSentinel;
    const bool same_ok = elapsed(counts, same_ms, owned.end, owned.end, "sample same-event event_elapsed_ms");
    const std::string same_error = same_ok ? "none" : gpu::last_error();
    const bool plausible = normal_ok && std::isfinite(ms) && ms > 0.0F &&
                           std::isfinite(wall_ms) && wall_ms > 0.0 &&
                           static_cast<double>(ms) <= wall_ms * kWallMultiplier + kWallAllowanceMs;
    std::printf("SAMPLE index=%u copies=2 transfer_bytes=%zu records=2 compute_waits=1 "
                "elapsed_queries=3 api_calls=%zu checked_words=%zu normal_result=%d elapsed_ms=%.9g "
                "normal_error=%s host_submit_to_completion_wall_ms=%.9g reversed_result=%d "
                "sentinel_unchanged=%d raw_endpoints_available=%d raw_start_ns=%" PRIu64 " raw_end_ns=%" PRIu64
                " reversed_error=%s same_result=%d same_ms=%.9g same_error=%s plausible=%d\n",
                index, 2 * kBytes, counts.api_calls - api_before, kWords, static_cast<int>(normal_ok),
                static_cast<double>(ms), normal_error.c_str(), wall_ms, static_cast<int>(reverse_ok),
                static_cast<int>(unchanged(reversed_ms)), static_cast<int>(endpoints), raw_earlier, raw_later,
                reverse_error.c_str(), static_cast<int>(same_ok), static_cast<double>(same_ms), same_error.c_str(),
                static_cast<int>(plausible));
    std::fflush(stdout);
    check(normal_ok, "normal interval query failed");
    check(endpoints && raw_earlier > 0 && raw_later > raw_earlier && unchanged(reversed_ms),
          "positive distinct reversed completed-pair refusal not established (zero resolution is failure)");
    check(same_ok && same_ms == 0.0F, "same completed event did not yield zero");
    check(plausible, "interval failed conservative host-wall plausibility check");
    ++counts.positive_reversals;
    ++counts.samples_completed;
}

void run(Counts& counts) {
    // Runtime owns real selection, queue creation, mark policy and watchdog.
    // No explicit device-info query, kernel, preload, new stream, graph or polling.
    auto& runtime = strata::core::Runtime::get();
    Owned owned(counts, runtime);
    check(runtime.compute().has_property<sycl::property::queue::in_order>(), "Runtime compute queue is not in-order");
    owned.source = sycl::malloc_host<uint32_t>(kWords, runtime.context());
    check(owned.source != nullptr, "source host USM allocation failed");
    owned.roundtrip = sycl::malloc_host<uint32_t>(kWords, runtime.context());
    check(owned.roundtrip != nullptr, "roundtrip host USM allocation failed");
    owned.device = sycl::malloc_device<uint32_t>(kWords, runtime.device(), runtime.context());
    check(owned.device != nullptr, "device USM allocation failed");
    require_api(counts, "begin timing event_create", [&] { return gpu::event_create(&owned.begin, true); });
    require_api(counts, "end timing event_create", [&] { return gpu::event_create(&owned.end, true); });
    require_api(counts, "unrecorded timing event_create", [&] { return gpu::event_create(&owned.unrecorded, true); });
    require_api(counts, "untimed begin event_create", [&] { return gpu::event_create(&owned.untimed_begin, false); });
    require_api(counts, "untimed end event_create", [&] { return gpu::event_create(&owned.untimed_end, false); });

    expected_refusal(owned, "unrecorded pair", owned.unrecorded, owned.begin,
                     "event_elapsed_ms: an event was not recorded");
    record(owned, owned.untimed_begin, "untimed begin event_record");
    record(owned, owned.untimed_end, "untimed end event_record");
    owned.wait();
    expected_refusal(owned, "completed non-timing pair", owned.untimed_begin, owned.untimed_end,
                     "event_elapsed_ms: an event was not created for timing");
    for (unsigned i = 0; i < kSamples; ++i) sample(owned, i);
    check(counts.samples_completed == kSamples && counts.positive_reversals > 0,
          "fixed sample/count contract incomplete");
    owned.close();
}

void totals(const Counts& counts, const char* status) {
    std::printf("RESULT status=%s samples=%u planned_samples=%u positive_reversals=%u api_calls=%zu "
                "records=%zu copies=%zu compute_waits=%zu cleanup_waits=%zu elapsed_queries=%zu checked_words=%zu\n",
                status, counts.samples_completed, kSamples, counts.positive_reversals, counts.api_calls,
                counts.records, counts.copies, counts.compute_waits, counts.cleanup_waits,
                counts.elapsed_queries, counts.words_checked);
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::fprintf(stderr, "usage: profiling_tags_probe (no arguments)\n");
        return 2;
    }
    Counts counts;
    std::printf("PROBE mode=%s STRATA_NO_BLAS=%d samples=%u device_bytes=%zu host_usm_bytes=%zu per_sample_transfer_bytes=%zu "
                "queue=Runtime.compute(in_order) wall_limit=wall*%.3g+%.3gms "
                "scope=queue_timeline_not_exclusive_kernel_service\n",
                STRATA_PROBE_MODE_FREE ? "free" : "contrib-llvm", kNoBlas, kSamples, kBytes, 2 * kBytes,
                2 * kBytes, kWallMultiplier, kWallAllowanceMs);
    std::fflush(stdout);
    try {
        run(counts);
        // Main must also require normal exit and its fault/completion receipt.
        totals(counts, "PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL profiling_tags_probe: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "FAIL profiling_tags_probe: unknown exception\n");
    }
    std::fflush(stderr);
    totals(counts, "FAIL");
    return 1;
}
