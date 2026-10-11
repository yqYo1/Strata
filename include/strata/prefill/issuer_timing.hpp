// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only observer. No GPU commands or synchronization.
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
namespace strata::prefill {
template<class Clock = std::chrono::steady_clock> class BasicIssuerTiming {
    struct Metric { uint64_t count = 0, sum = 0, min = 0, max = 0; };
    struct Bucket { uint64_t entries = 0, bytes = 0; Metric wait, submit; };
    struct Layer { Bucket kind[3]; };
    bool enabled_, clock_valid_ = Clock::is_steady, accounting_valid_ = true, counter_overflow_ = false;
    std::vector<Layer> layers_;
    bool add(uint64_t& v, uint64_t n) {
        if (n > std::numeric_limits<uint64_t>::max() - v) { accounting_valid_ = false; counter_overflow_ = true; return false; }
        v += n;
        return true;
    }
    void sample(Metric& m, typename Clock::time_point begin, typename Clock::time_point end) {
        if (end < begin) { clock_valid_ = false; return; }
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        if (ns < 0) { clock_valid_ = false; return; }
        const uint64_t n = (uint64_t) ns;
        if (!add(m.count, 1) || !add(m.sum, n)) return;
        if (m.count == 1 || n < m.min) m.min = n;
        if (n > m.max) m.max = n;
    }
public:
    enum Kind { PinnedDirect, UnpinnedStaged, TransientStaged };
    static bool enabled_value(const char* v) { return v && std::strcmp(v, "1") == 0; }
    static bool requested() {
        static const bool enabled = enabled_value(std::getenv("STRATA_PREFILL_ISSUER_TIMING"));
        return enabled;  // process policy, cached once; no repeated per-chunk environment lookup
    }
    BasicIssuerTiming(bool enabled, std::size_t layers) : enabled_(enabled) {
        if (enabled_) layers_.resize(layers);
    }
    // Exclusive issuer writer; emit only after existing join/drain. Disabled
    // mode calls f directly, with no clock read or access to bucket storage.
    template<class F> auto measure(std::size_t layer, Kind kind, bool wait, F&& f) -> decltype(f()) {
        if (!enabled_) return f();
        const auto begin = Clock::now();
        auto result = f();
        const auto end = Clock::now();
        auto& b = layers_[layer].kind[kind];
        sample(wait ? b.wait : b.submit, begin, end);
        return result;
    }
    void entry(std::size_t layer, Kind kind, std::size_t bytes) {
        if (!enabled_) return;
        auto& b = layers_[layer].kind[kind];
        add(b.entries, 1);
        add(b.bytes, (uint64_t) bytes);
    }
    // Reports successful issuer scope, not DMA service or calibrated overhead.
    void emit(int device, int64_t lb, int64_t le, int64_t offset, int64_t pos,
              int64_t tokens, bool stream_all, bool threaded, int ring, uint64_t expected, FILE* sink = stderr) {
        if (!enabled_) return;
        uint64_t entries = 0;
        for (const auto& l : layers_) for (int k = 0; k < 3; ++k) {
            const auto& b = l.kind[k];
            add(entries, b.entries);
            if (b.submit.count != b.entries || b.wait.count != (k == PinnedDirect ? 0 : b.entries)) accounting_valid_ = false;
        }
        const bool covered = accounting_valid_ && entries == expected;
        const char* names[] = {"pinned_direct", "unpinned_staged", "transient_staged"};
        for (std::size_t l = 0; l < layers_.size(); ++l) for (int k = 0; k < 3; ++k) {
            const auto& b = layers_[l].kind[k];
            std::fprintf(sink,
                "[prefill-issuer-host-v1] device=%d lb=%lld le=%lld chunk_offset=%lld pos=%lld tokens=%lld layer=%lld stream_all=%d threaded_issue=%d ring=%d dataset_kind=%s submit_op=%s entries=%llu expert_bytes=%llu wait_count=%llu wait_sum_ns=%llu wait_min_ns=%llu wait_max_ns=%llu submit_count=%llu submit_sum_ns=%llu submit_min_ns=%llu submit_max_ns=%llu clock_valid=%d accounting_valid=%d counter_overflow=%d coverage_complete=%d host_observation_eligible=%d chunk_expected=%llu chunk_observed=%llu scope=nonresident_expert_issuers added_gpu_events=0 overhead_calibrated=0 performance_eligible=0\n",
                device, (long long) lb, (long long) le, (long long) offset, (long long) pos,
                (long long) tokens, (long long) lb + (long long) l, (int) stream_all, (int) threaded, ring,
                names[k], stream_all ? "copy_async_after" : "copy_async",
                (unsigned long long) b.entries, (unsigned long long) b.bytes,
                (unsigned long long) b.wait.count, (unsigned long long) b.wait.sum,
                (unsigned long long) b.wait.min, (unsigned long long) b.wait.max,
                (unsigned long long) b.submit.count, (unsigned long long) b.submit.sum,
                (unsigned long long) b.submit.min, (unsigned long long) b.submit.max,
                (int) clock_valid_, (int) accounting_valid_, (int) counter_overflow_, (int) covered, (int) (clock_valid_ && covered && entries != 0),
                (unsigned long long) expected, (unsigned long long) entries);
        }
    }
};
using IssuerTiming = BasicIssuerTiming<>;
} // namespace strata::prefill
