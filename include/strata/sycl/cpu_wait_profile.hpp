#pragma once
// Authored for Strata; repository LICENSE (MIT) applies.
// Host-only diagnostic. No SYCL objects, allocations, or synchronization.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <exception>
#if defined(__linux__)
#include <time.h>
#endif

namespace strata::sycl_backend::cpu_wait_profile {
enum Kind { compute_queue, transfer_queue, event, compat_queue, direct_queue, kinds };
struct Stamp { double wall = 0, cpu = 0; bool valid = false; };
inline Stamp stamp() noexcept {
    const int saved_errno = errno;
    Stamp s;
    s.wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#if defined(__linux__)
    timespec t{};
    if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0) {
        s.cpu = double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
        s.valid = true;
    }
#endif
    errno = saved_errno;
    return s;
}
struct State {
    double cpu[kinds]{}, wall[kinds]{};
    unsigned long long count[kinds]{}, throws[kinds]{};
    unsigned depth = 0;
    bool invalid = false;
};
inline thread_local State* active = nullptr;
class Wait {
    State* s_ = active;
    Kind kind_;
    Stamp begin_;
    int exceptions_ = 0;
    bool outer_ = false;
public:
    explicit Wait(Kind k) noexcept : kind_(k) {
        if (!s_) return;
        outer_ = s_->depth++ == 0;
        if (!outer_) return;
        ++s_->count[kind_];
        exceptions_ = std::uncaught_exceptions();
        begin_ = stamp();
    }
    ~Wait() noexcept {
        if (!s_) return;
        --s_->depth;
        if (!outer_) return;
        const auto end = stamp();
        if (std::uncaught_exceptions() > exceptions_) ++s_->throws[kind_];
        if (!begin_.valid || !end.valid || end.cpu < begin_.cpu || end.wall < begin_.wall) {
            s_->invalid = true;
            return;
        }
        s_->cpu[kind_] += end.cpu - begin_.cpu;
        s_->wall[kind_] += end.wall - begin_.wall;
    }
    Wait(const Wait&) = delete;
    Wait& operator=(const Wait&) = delete;
};
// Wrap exactly the blocking call, not argument lookup, submission or error check.
template<class F> inline void blocking(Kind k, F&& f) {
    Wait observation(k);
    f();
}
class Scope {
    State state_;
    State* previous_ = nullptr;
    Stamp begin_;
    bool enabled_ = false, nested_ = false, complete_ = false;
public:
    explicit Scope(bool enabled) noexcept : enabled_(enabled) {
        if (!enabled_) return;
        previous_ = active;
        nested_ = previous_ != nullptr;
        if (nested_) { previous_->invalid = true; return; }
        begin_ = stamp();
        active = &state_;
    }
    void complete(bool value) noexcept { complete_ = value; }
    ~Scope() noexcept {
        if (!enabled_) return;
        if (nested_) {
            std::fputs("strata prefill cpu wait: version=1 available=0 complete=0 nested_scope=1\n", stderr);
            return;
        }
        const auto end = stamp();
        active = previous_; // retire TLS before formatting/output
        double wc = 0, ww = 0;
        unsigned long long count = 0, errors = 0;
        for (int k = 0; k < kinds; ++k) {
            wc += state_.cpu[k]; ww += state_.wall[k];
            count += state_.count[k]; errors += state_.throws[k];
        }
        const double cpu = end.cpu - begin_.cpu, wall = end.wall - begin_.wall;
        const bool available = begin_.valid && end.valid && !state_.invalid &&
            !state_.depth && cpu >= wc && wall >= ww;
        std::fprintf(stderr, "strata prefill cpu wait: version=1 available=%d complete=%d "
            "clock_or_accounting_error=%d open_waits=%u wait_throws=%llu wall_s=%.9f main_cpu_s=%.9f "
            "wait_cpu_s=%.9f wait_wall_s=%.9f outside_wait_cpu_s=%.9f wait_count=%llu "
            "compute_count=%llu transfer_count=%llu event_count=%llu compat_count=%llu direct_count=%llu "
            "compute_cpu_s=%.9f transfer_cpu_s=%.9f event_cpu_s=%.9f compat_cpu_s=%.9f direct_cpu_s=%.9f "
            "compute_wall_s=%.9f transfer_wall_s=%.9f event_wall_s=%.9f compat_wall_s=%.9f direct_wall_s=%.9f\n",
            int(available), int(complete_), int(!available), state_.depth, errors,
            available ? wall : -1., available ? cpu : -1., available ? wc : -1., available ? ww : -1.,
            available ? cpu - wc : -1., count,
            state_.count[0], state_.count[1], state_.count[2], state_.count[3], state_.count[4],
            available ? state_.cpu[0] : -1., available ? state_.cpu[1] : -1., available ? state_.cpu[2] : -1.,
            available ? state_.cpu[3] : -1., available ? state_.cpu[4] : -1.,
            available ? state_.wall[0] : -1., available ? state_.wall[1] : -1., available ? state_.wall[2] : -1.,
            available ? state_.wall[3] : -1., available ? state_.wall[4] : -1.);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
} // namespace strata::sycl_backend::cpu_wait_profile
