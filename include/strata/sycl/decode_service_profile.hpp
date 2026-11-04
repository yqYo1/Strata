#pragma once
// Independently authored for Strata; repository LICENSE applies.
// Host clocks only. No queue changes, allocations, device work or per-call logs.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <utility>
#if defined(__linux__)
#include <time.h>
#endif

namespace strata::sycl_backend::decode_service_profile {
enum Region { capture_setup, staging, layer_completion, cpu_pool, expert_retirement,
    graph_submission, final_compute, final_copies, head_sampling, region_count };
struct Tick { uint64_t wall = 0, cpu = 0; bool valid = false; };
inline Tick clock_tick() noexcept {
    const int saved = errno;
    Tick t;
#if defined(__linux__)
    timespec w{}, c{};
    if (::clock_gettime(CLOCK_MONOTONIC, &w) == 0 &&
        ::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c) == 0 && w.tv_sec >= 0 && c.tv_sec >= 0) {
        t.wall = uint64_t(w.tv_sec) * 1000000000ULL + uint64_t(w.tv_nsec);
        t.cpu = uint64_t(c.tv_sec) * 1000000000ULL + uint64_t(c.tv_nsec);
        t.valid = true;
    }
#endif
    errno = saved;
    return t;
}
using Clock = Tick (*)() noexcept;
struct Counter { uint64_t calls = 0, throws = 0, wall = 0, cpu = 0; };
struct State {
    Counter parts[region_count]{}, waits[region_count + 1]{};
    Clock clock = clock_tick;
    unsigned part_depth = 0, wait_depth = 0;
    int region = region_count;
    bool invalid = false;
};
inline thread_local State* active = nullptr;
inline void add(State& s, uint64_t& dst, uint64_t n) noexcept {
    if (n > std::numeric_limits<uint64_t>::max() - dst) s.invalid = true;
    else dst += n;
}
inline void elapsed(State& s, Counter& dst, Tick a, Tick b) noexcept {
    if (!a.valid || !b.valid || b.wall < a.wall || b.cpu < a.cpu) {
        s.invalid = true; return;
    }
    add(s, dst.wall, b.wall - a.wall);
    add(s, dst.cpu, b.cpu - a.cpu);
}
// Disjoint caller regions contain the complete public wait wrappers. NativeWait
// separately records their nested blocking calls; these are NOT additive costs.
class Part {
    State* s_ = active;
    Region region_;
    Tick begin_;
    int exceptions_ = 0;
    bool outer_ = false;
public:
    explicit Part(Region r) noexcept : region_(r) {
        if (!s_) return;
        outer_ = s_->part_depth++ == 0;
        if (!outer_ || s_->wait_depth || r < 0 || r >= region_count) {
            s_->invalid = true; outer_ = false; return;
        }
        s_->region = r;
        add(*s_, s_->parts[r].calls, 1);
        exceptions_ = std::uncaught_exceptions();
        begin_ = s_->clock();
    }
    void finish() noexcept {
        if (!s_) return;
        State* s = std::exchange(s_, nullptr);
        --s->part_depth;
        if (!outer_) return;
        const auto end = s->clock();
        if (s->wait_depth) s->invalid = true;
        elapsed(*s, s->parts[region_], begin_, end);
        if (std::uncaught_exceptions() > exceptions_) add(*s, s->parts[region_].throws, 1);
        s->region = region_count;
    }
    ~Part() noexcept { finish(); }
    Part(const Part&) = delete;
    Part& operator=(const Part&) = delete;
};
template<class F> decltype(auto) measure(Region r, F&& f) {
    Part part(r);
    return std::forward<F>(f)();
}
class NativeWait {
    State* s_ = active;
    Tick begin_;
    int region_ = region_count, exceptions_ = 0;
    bool outer_ = false;
public:
    NativeWait() noexcept {
        if (!s_) return;
        outer_ = s_->wait_depth++ == 0;
        if (!outer_) return;
        region_ = s_->region;
        add(*s_, s_->waits[region_].calls, 1);
        exceptions_ = std::uncaught_exceptions();
        begin_ = s_->clock();
    }
    ~NativeWait() noexcept {
        if (!s_) return;
        --s_->wait_depth;
        if (!outer_) return;
        elapsed(*s_, s_->waits[region_], begin_, s_->clock());
        if (std::uncaught_exceptions() > exceptions_) add(*s_, s_->waits[region_].throws, 1);
    }
    NativeWait(const NativeWait&) = delete;
    NativeWait& operator=(const NativeWait&) = delete;
};
class Scope {
    State state_;
    State* previous_ = nullptr;
    Tick begin_;
    uint64_t request_;
    std::FILE* output_;
    bool enabled_, nested_ = false;
public:
    Scope(bool enabled, uint64_t request, std::FILE* output = stderr,
          Clock clock = clock_tick) noexcept : request_(request), output_(output), enabled_(enabled) {
        if (!enabled_) return;
        previous_ = active;
        if (previous_) { previous_->invalid = true; nested_ = true; return; }
        state_.clock = clock;
        begin_ = clock();
        active = &state_;
    }
    void finish(bool complete) noexcept {
        if (!std::exchange(enabled_, false) || nested_) return;
        const auto end = state_.clock();
        active = previous_; // No observer state is active during formatting.
        Counter total{}, parts{}, waits{};
        elapsed(state_, total, begin_, end);
        for (int k = 0; k < region_count; ++k) {
            add(state_, parts.cpu, state_.parts[k].cpu);
            add(state_, parts.wall, state_.parts[k].wall);
            if (state_.waits[k].cpu > state_.parts[k].cpu ||
                state_.waits[k].wall > state_.parts[k].wall) state_.invalid = true;
        }
        for (const auto& w : state_.waits) {
            add(state_, waits.cpu, w.cpu); add(state_, waits.wall, w.wall);
        }
        if (parts.cpu > total.cpu || parts.wall > total.wall ||
            waits.cpu > total.cpu || waits.wall > total.wall ||
            (parts.cpu <= total.cpu && state_.waits[region_count].cpu > total.cpu - parts.cpu) ||
            (parts.wall <= total.wall && state_.waits[region_count].wall > total.wall - parts.wall))
            state_.invalid = true;
        const bool valid = !state_.invalid && !state_.part_depth && !state_.wait_depth;
        // Integer nanoseconds make conservation exact. One bounded line/request;
        // unclassified includes draft/commit and other uninstrumented work.
        std::fprintf(output_, "strata decode service: version=1 request=%llu available=%d complete=%d "
            "open_parts=%u open_waits=%u wall_ns=%llu cpu_ns=%llu "
            "unclassified_wall_ns=%llu unclassified_cpu_ns=%llu",
            (unsigned long long)request_, int(valid), int(complete), state_.part_depth, state_.wait_depth,
            (unsigned long long)total.wall, (unsigned long long)total.cpu,
            (unsigned long long)(parts.wall <= total.wall ? total.wall - parts.wall : 0),
            (unsigned long long)(parts.cpu <= total.cpu ? total.cpu - parts.cpu : 0));
        const char* names[region_count + 1] = {"capture", "stage", "layer_wait", "pool", "copy_retire",
            "submit", "final_compute", "final_copy", "sample", "unclassified"};
        for (int k = 0; k <= region_count; ++k) {
            if (k < region_count) {
                const auto& p = state_.parts[k];
                std::fprintf(output_, " %s_calls=%llu %s_throws=%llu %s_wall_ns=%llu %s_cpu_ns=%llu",
                    names[k], (unsigned long long)p.calls, names[k], (unsigned long long)p.throws,
                    names[k], (unsigned long long)p.wall, names[k], (unsigned long long)p.cpu);
            }
            const auto& w = state_.waits[k];
            std::fprintf(output_, " %s_wait_calls=%llu %s_wait_throws=%llu %s_wait_wall_ns=%llu %s_wait_cpu_ns=%llu",
                names[k], (unsigned long long)w.calls, names[k], (unsigned long long)w.throws,
                names[k], (unsigned long long)w.wall, names[k], (unsigned long long)w.cpu);
        }
        std::fputc('\n', output_);
    }
    ~Scope() noexcept { finish(false); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
} // namespace strata::sycl_backend::decode_service_profile
