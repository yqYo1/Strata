#pragma once

#include "strata/sycl/cpu_wait_profile.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace strata::prefill::detail {
namespace wait_profile = sycl_backend::cpu_wait_profile;

// Sample the existing disjoint host ledger at layer-major boundaries. This
// observer reads only calling-thread state and never queues device work.
class ColdHostPhases {
public:
    enum class Boundary { allocated_plan = 1, provisioned, layer_passes_finished };

    explicit ColdHostPhases(bool enabled) noexcept : enabled_(enabled) {
        if (!enabled_) return;
        previous_ = active_;
        state_ = wait_profile::active;
        active_ = this;
        valid_ = !previous_ && capture(points_[0]);
    }

    static void mark(Boundary boundary) noexcept {
        if (!active_ || !active_->valid_) return;
        const auto index = static_cast<unsigned>(boundary);
        active_->valid_ = index == active_->next_ && index < 4 && active_->capture(active_->points_[index]);
        if (active_->valid_) ++active_->next_;
    }

    void finish(bool complete) noexcept {
        if (!enabled_) return;
        enabled_ = false;
        active_ = previous_;
        const int saved_errno = errno;
        const bool available = complete && valid_ && next_ == 4 && capture(points_[4]) && intervals_valid();
        std::fprintf(stderr, "strata prefill cold host phases: version=1 available=%d phases=%d", int(available),
                     available ? 4 : 0);
        if (available) {
            constexpr const char* names[] = {"admission", "provision", "layers", "retire"};
            for (unsigned i = 0; i < 4; ++i) write_phase(names[i], points_[i], points_[i + 1]);
        }
        std::fputc('\n', stderr);
        errno = saved_errno;
    }

    ~ColdHostPhases() noexcept { finish(false); }
    ColdHostPhases(const ColdHostPhases&) = delete;
    ColdHostPhases& operator=(const ColdHostPhases&) = delete;

private:
    struct Point {
        wait_profile::Stamp stamp;
        double wait_cpu = 0, wait_wall = 0;
        double host_cpu = 0, host_wall = 0;
        long minor_faults = 0, major_faults = 0;
    };

    bool capture(Point& point) const noexcept {
        if (!state_ || wait_profile::active != state_ || state_->invalid || state_->depth || state_->host_depth ||
            state_->status_depth || state_->status_part_depth) return false;
#if defined(__linux__)
        const int saved_errno = errno;
        rusage usage{};
        const bool have_faults = ::getrusage(RUSAGE_THREAD, &usage) == 0;
        errno = saved_errno;
        if (!have_faults) return false;
        point.minor_faults = usage.ru_minflt;
        point.major_faults = usage.ru_majflt;
        point.stamp = wait_profile::stamp();
        for (int k = 0; k < wait_profile::kinds; ++k) {
            point.wait_cpu += state_->cpu[k];
            point.wait_wall += state_->wall[k];
        }
        for (int k = 0; k < wait_profile::host_kinds; ++k) {
            point.host_cpu += state_->host_cpu[k] - state_->host_wait_cpu[k];
            point.host_wall += state_->host_wall[k] - state_->host_wait_wall[k];
        }
        return point.stamp.valid && point.minor_faults >= 0 && point.major_faults >= 0;
#else
        (void) point;
        return false;
#endif
    }

    bool intervals_valid() const noexcept {
        for (unsigned i = 0; i < 4; ++i) {
            const Point& a = points_[i];
            const Point& b = points_[i + 1];
            if (!interval_valid(b.stamp.wall - a.stamp.wall, b.wait_wall - a.wait_wall, b.host_wall - a.host_wall) ||
                !interval_valid(b.stamp.cpu - a.stamp.cpu, b.wait_cpu - a.wait_cpu, b.host_cpu - a.host_cpu) ||
                b.minor_faults < a.minor_faults || b.major_faults < a.major_faults) return false;
        }
        return true;
    }

    static bool interval_valid(double total, double wait, double host) noexcept {
        return wait_profile::duration(total) && wait_profile::duration(wait) && wait_profile::duration(host) &&
               total + 1e-7 >= wait + host;
    }

    static void write_phase(const char* name, const Point& a, const Point& b) noexcept {
        const double wall = b.stamp.wall - a.stamp.wall, cpu = b.stamp.cpu - a.stamp.cpu;
        const double wait_wall = b.wait_wall - a.wait_wall, wait_cpu = b.wait_cpu - a.wait_cpu;
        const double known_wall = b.host_wall - a.host_wall, known_cpu = b.host_cpu - a.host_cpu;
        std::fprintf(stderr, " %s_wall_s=%.9f %s_cpu_s=%.9f %s_wait_wall_s=%.9f %s_wait_cpu_s=%.9f"
            " %s_host_wall_s=%.9f %s_host_cpu_s=%.9f %s_unclassified_wall_s=%.9f %s_unclassified_cpu_s=%.9f"
            " %s_minor_faults=%ld %s_major_faults=%ld",
            name, wall, name, cpu, name, wait_wall, name, wait_cpu, name, known_wall, name, known_cpu,
            name, std::max(0.0, wall - wait_wall - known_wall), name, std::max(0.0, cpu - wait_cpu - known_cpu),
            name, b.minor_faults - a.minor_faults, name, b.major_faults - a.major_faults);
    }

    inline static thread_local ColdHostPhases* active_ = nullptr;
    ColdHostPhases* previous_ = nullptr;
    const wait_profile::State* state_ = nullptr;
    std::array<Point, 5> points_{};
    unsigned next_ = 1;
    bool enabled_ = false, valid_ = false;
};

} // namespace strata::prefill::detail
