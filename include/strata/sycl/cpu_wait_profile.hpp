#pragma once
// Authored for Strata; repository LICENSE (MIT) applies.
// Host-only diagnostic. No SYCL objects, allocations, or synchronization.
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <exception>
#include <cmath>
#if defined(__linux__)
#include <time.h>
#endif

namespace strata::sycl_backend::cpu_wait_profile {
enum HostKind { source_copy, host_group, gemm_submit, host_kinds };
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
inline bool duration(double x) noexcept { return std::isfinite(x) && x >= 0; }
struct State {
    double cpu[kinds]{}, wall[kinds]{};
    unsigned long long count[kinds]{}, throws[kinds]{};
    double host_cpu[host_kinds]{}, host_wall[host_kinds]{};
    double host_wait_cpu[host_kinds]{}, host_wait_wall[host_kinds]{};
    unsigned long long host_count[host_kinds]{}, host_throws[host_kinds]{};
    unsigned host_depth = 0;
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
        if (!begin_.valid || !end.valid || !duration(end.cpu - begin_.cpu) || !duration(end.wall - begin_.wall)) {
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
// Regions must not overlap or begin inside an explicit wait. Invalid nesting
// poisons the enclosing scope instead of reporting a double-counted partition.
class Host {
    State* s_ = active;
    HostKind kind_;
    Stamp begin_;
    double wc_ = 0, ww_ = 0;
    int exceptions_ = 0;
    bool open_ = false, measured_ = false;
public:
    explicit Host(HostKind k) noexcept : kind_(k) {
        if (!s_) return;
        open_ = true;
        ++s_->host_count[kind_];
        if (s_->host_depth++ || s_->depth) { s_->invalid = true; return; }
        measured_ = true;
        for (int i = 0; i < kinds; ++i) { wc_ += s_->cpu[i]; ww_ += s_->wall[i]; }
        exceptions_ = std::uncaught_exceptions();
        begin_ = stamp();
    }
    void finish() noexcept {
        if (!open_) return;
        open_ = false;
        --s_->host_depth;
        if (!measured_) return;
        const auto end = stamp();
        if (std::uncaught_exceptions() > exceptions_) ++s_->host_throws[kind_];
        double wc = 0, ww = 0;
        for (int i = 0; i < kinds; ++i) { wc += s_->cpu[i]; ww += s_->wall[i]; }
        const double cpu = end.cpu - begin_.cpu, wall = end.wall - begin_.wall;
        wc -= wc_; ww -= ww_;
        if (!begin_.valid || !end.valid || s_->depth || s_->host_depth ||
            !duration(cpu) || !duration(wall) || !duration(wc) || !duration(ww) ||
            !duration(cpu - wc) || !duration(wall - ww)) { s_->invalid = true; return; }
        s_->host_cpu[kind_] += cpu; s_->host_wall[kind_] += wall;
        s_->host_wait_cpu[kind_] += wc; s_->host_wait_wall[kind_] += ww;
    }
    ~Host() noexcept { finish(); }
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;
};
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
            std::fputs("strata prefill cpu wait: version=2 available=0 complete=0 nested_scope=1\n", stderr);
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
        double hc = 0, hw = 0;
        bool host_valid = !state_.host_depth;
        for (int k = 0; k < host_kinds; ++k) {
            const double c = state_.host_cpu[k] - state_.host_wait_cpu[k];
            const double w = state_.host_wall[k] - state_.host_wait_wall[k];
            host_valid = host_valid && duration(state_.host_cpu[k]) && duration(state_.host_wall[k]) &&
                duration(state_.host_wait_cpu[k]) && duration(state_.host_wait_wall[k]) && duration(c) && duration(w);
            hc += c; hw += w;
        }
        const bool available = begin_.valid && end.valid && !state_.invalid &&
            !state_.depth && host_valid && duration(cpu) && duration(wall) &&
            duration(wc) && duration(ww) && duration(cpu - wc - hc) && duration(wall - ww - hw);
        std::fprintf(stderr, "strata prefill cpu wait: version=2 available=%d complete=%d "
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
        // One bounded extra line; no per-call output or dynamic allocation.
        std::fprintf(stderr, "strata prefill cpu host: version=2 available=%d open_regions=%u "
            "unclassified_cpu_s=%.9f unclassified_wall_s=%.9f",
            int(available), state_.host_depth, available ? cpu - wc - hc : -1., available ? wall - ww - hw : -1.);
        const char* names[host_kinds] = {"source_copy", "host_group", "gemm_submit"};
        for (int k = 0; k < host_kinds; ++k)
            std::fprintf(stderr, " %s_count=%llu %s_throws=%llu %s_cpu_s=%.9f %s_wall_s=%.9f "
                "%s_outside_wait_cpu_s=%.9f %s_outside_wait_wall_s=%.9f",
                names[k], state_.host_count[k], names[k], state_.host_throws[k],
                names[k], available ? state_.host_cpu[k] : -1., names[k], available ? state_.host_wall[k] : -1.,
                names[k], available ? state_.host_cpu[k] - state_.host_wait_cpu[k] : -1.,
                names[k], available ? state_.host_wall[k] - state_.host_wait_wall[k] : -1.);
        std::fputc('\n', stderr);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
} // namespace strata::sycl_backend::cpu_wait_profile
