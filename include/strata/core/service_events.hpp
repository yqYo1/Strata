// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {
inline bool service_events_requested() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_PREFILL_SERVICE_EVENTS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
using ServiceClock = std::chrono::steady_clock;
inline ServiceClock::time_point service_now() {
    return service_events_requested() ? ServiceClock::now() : ServiceClock::time_point{};
}
struct ServiceMeta {
    const char* kind = "unknown";
    const char* backend = "none";
    const char* src = "none";
    const char* dst = "none";
    uint64_t bytes = 0;
    int64_t groups = 0, rows = 0, n = 0, k = 0, type = 0, first = 0;
    int tile_m = 0, tile_n = 0, pipeline = 0;
};
// Only event handles/diagnostic metadata are owned here, never operand storage or queue drains.
class ServiceEvents {
public:
    static constexpr size_t cap = 200000, category_cap = 4096;
    bool begin() {
        if (!service_events_requested()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_.load()) throw std::runtime_error("prefill service events: overlapping Runtime scopes");
        records_.clear(); failed_ = false; overflow_ = false; excluded_graph_ = 0;
        active_.store(true);
        return true;
    }
    bool active() const { return active_.load(); }
    void exclude_graph() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_.load()) {
                if (excluded_graph_ == std::numeric_limits<uint64_t>::max()) failed_ = true;
                else ++excluded_graph_;
            }
        }
        catch (...) { failure(); }
    }
    void failure() noexcept {
        // A diagnostic allocation/lock exception must remain visible without attempting that lock again.
        if (active_.load()) failed_.store(true);
    }
    void record(const sycl::event& event, ServiceMeta meta, ServiceClock::time_point before, ServiceClock::time_point after) noexcept {
        if (!active_.load()) return;
        try {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_.load()) return;
            if (ns < 0) failed_ = true;
            if (records_.size() >= cap) { overflow_ = true; return; }
            if (records_.size() == records_.capacity())
                records_.reserve(std::min(cap, std::max(size_t(4096), records_.size() * 2)));
            records_.push_back({event, meta, static_cast<uint64_t>(ns < 0 ? 0 : ns)});
        } catch (...) { failure(); }
    }
    void finish(bool checked_boundary) noexcept {
        if (!service_events_requested()) return;
        try {
            const auto query_before = ServiceClock::now();
            std::vector<Record> records;
            bool failed, overflow;
            uint64_t excluded_graph;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!active_.load()) return;
                active_.store(false); records.swap(records_);
                failed = failed_; overflow = overflow_; excluded_graph = excluded_graph_;
            }
            uint64_t pending = 0, invalid = 0, query_errors = 0;
            struct Aggregate { uint64_t count = 0, valid = 0, bytes = 0, host = 0, sum = 0, min = 0, max = 0; };
            std::map<std::string, Aggregate> categories;
            const auto add = [&failed](uint64_t& dst, uint64_t value) {
                if (value > std::numeric_limits<uint64_t>::max() - dst) { failed = true; return; }
                dst += value;
            };
            if (checked_boundary) for (const auto& record : records) {
                const auto& m = record.meta;
                char key[320];
                const int key_size = std::snprintf(key, sizeof key, "kind=%s backend=%s src=%s dst=%s G=%lld rows=%lld N=%lld K=%lld "
                              "type=%lld first=%lld tile_m=%d tile_n=%d pipeline=%d",
                              m.kind, m.backend, m.src, m.dst, (long long) m.groups, (long long) m.rows,
                              (long long) m.n, (long long) m.k, (long long) m.type, (long long) m.first,
                              m.tile_m, m.tile_n, m.pipeline);
                if (key_size < 0 || static_cast<size_t>(key_size) >= sizeof key) { failed = true; continue; }
                if (!categories.count(key) && categories.size() >= category_cap) { failed = true; continue; }
                auto& a = categories[key];
                add(a.count, 1); add(a.bytes, m.bytes); add(a.host, record.host_ns);
                try {
                    if (record.event.get_info<sycl::info::event::command_execution_status>() !=
                        sycl::info::event_command_status::complete) { ++pending; continue; }
                    const auto start = record.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto end = record.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    if (!start || end <= start) { ++invalid; continue; }
                    const uint64_t duration = end - start;
                    add(a.sum, duration); add(a.valid, 1);
                    a.min = a.valid == 1 ? duration : std::min(a.min, duration);
                    a.max = std::max(a.max, duration);
                } catch (...) { ++query_errors; }
            }
            const bool complete = checked_boundary && !failed && !overflow && !pending && !invalid && !query_errors;
            const auto query_host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ServiceClock::now() - query_before).count();
            std::fprintf(stderr, "strata prefill service events: status=%s captured=%zu cap=%zu overflow=%d "
                         "pending=%llu invalid_timestamp=%llu query_errors=%llu checked_boundary=%d excluded_graph=%llu "
                         "bookkeeping_failed=%d query_aggregate_host_ns=%lld "
                         "scope=selected_leaf_commands sums_may_overlap=1 gaps_not_idle=1\n",
                         complete ? "complete" : "incomplete", records.size(), cap, (int) overflow,
                         (unsigned long long) pending, (unsigned long long) invalid,
                         (unsigned long long) query_errors, (int) checked_boundary, (unsigned long long) excluded_graph, (int) failed, (long long) query_host_ns);
            for (const auto& [key, a] : categories)
                std::fprintf(stderr, "strata prefill service category: %s count=%llu valid=%llu bytes=%llu "
                             "host_submit_ns=%llu service_sum_ns=%llu service_min_ns=%llu service_max_ns=%llu\n",
                             key.c_str(), (unsigned long long) a.count, (unsigned long long) a.valid,
                             (unsigned long long) a.bytes, (unsigned long long) a.host,
                             (unsigned long long) a.sum, (unsigned long long) a.min, (unsigned long long) a.max);
        } catch (...) {
            active_.store(false);
            std::fprintf(stderr, "strata prefill service events: status=incomplete reason=bookkeeping_error\n");
        }
    }
private:
    struct Record { sycl::event event; ServiceMeta meta; uint64_t host_ns; };
    std::mutex mutex_;
    std::atomic<bool> active_{false};
    std::vector<Record> records_;
    std::atomic<bool> failed_{false};
    bool overflow_ = false;
    uint64_t excluded_graph_ = 0;
};
class ServiceScope {
public:
    explicit ServiceScope(ServiceEvents& events) : events_(events), owned_(events.begin()) {}
    ~ServiceScope() { if (owned_) events_.finish(false); }
    void complete() { if (owned_) { events_.finish(true); owned_ = false; } }
    ServiceScope(const ServiceScope&) = delete;
    ServiceScope& operator=(const ServiceScope&) = delete;
private:
    ServiceEvents& events_;
    bool owned_;
};
}  // namespace strata::core
