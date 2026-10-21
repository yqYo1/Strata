#pragma once
// Authored for Strata; repository LICENSE (MIT) applies.
// Prefill handoff observation only. No GPU work or synchronization is added.
#include "strata/sycl/cpu_wait_profile.hpp"
#include <cstdint>
#include <limits>

namespace strata::sycl_backend::prefill_transition_profile {
namespace hp = cpu_wait_profile;
enum Kind { preflight, retire, allocate, submit, drain, publish, release,
    rollback, canonical, graph_resume, kinds };
struct Cell {
    double wall = 0, cpu = 0, wait_wall = 0, wait_cpu = 0;
    uint64_t calls = 0, items = 0, logical_bytes = 0, throws = 0;
};
struct Summary {
    bool available = false, complete = false;
    double wall = 0, cpu = 0, wait_wall = 0, wait_cpu = 0;
    double other_wall = 0, other_cpu = 0;
    unsigned open_parts = 0;
    uint64_t wait_errors = 0;
    Cell cells[kinds]{};
};
class Part;
class Scope;
inline thread_local Scope* active = nullptr;
class Scope {
    friend class Part;
    hp::State waits_{};
    hp::State* previous_waits_ = nullptr;
    Scope* previous_ = nullptr;
    hp::Stamp begin_{};
    Summary result_{};
    const char* label_;
    bool on_, installed_ = false, closed_ = false, invalid_ = false;
    unsigned depth_ = 0;
    int exceptions_ = 0;
    void wait_totals(double& cpu, double& wall) const noexcept {
        cpu = wall = 0;
        for (int i = 0; i < hp::kinds; ++i) { cpu += waits_.cpu[i]; wall += waits_.wall[i]; }
    }
public:
    explicit Scope(bool enabled, const char* label) noexcept : label_(label), on_(enabled) {
        if (!on_) return;
        previous_ = active;
        previous_waits_ = hp::active;
        if (previous_ || previous_waits_) {
            invalid_ = true;
            if (previous_) previous_->invalid_ = true;
            if (previous_waits_) previous_waits_->invalid = true;
            return;
        }
        begin_ = hp::stamp();
        exceptions_ = std::uncaught_exceptions();
        active = this; hp::active = &waits_; installed_ = true;
    }
    const Summary& finish(bool complete) noexcept {
        if (closed_) return result_;
        closed_ = true;
        if (!on_ || !installed_) return result_;
        const auto end = hp::stamp();
        invalid_ = invalid_ || active != this || hp::active != &waits_;
        active = previous_; hp::active = previous_waits_; // retire before output
        result_.wall = end.wall - begin_.wall; result_.cpu = end.cpu - begin_.cpu;
        wait_totals(result_.wait_cpu, result_.wait_wall);
        for (auto n : waits_.throws) result_.wait_errors += n;
        result_.other_wall = result_.wall; result_.other_cpu = result_.cpu;
        double part_wait_cpu = 0, part_wait_wall = 0;
        uint64_t errors = 0;
        for (const auto& cell : result_.cells) {
            result_.other_wall -= cell.wall; result_.other_cpu -= cell.cpu;
            part_wait_cpu += cell.wait_cpu; part_wait_wall += cell.wait_wall;
            errors += cell.throws;
        }
        result_.open_parts = depth_;
        result_.available = begin_.valid && end.valid && !invalid_ && !waits_.invalid &&
            !depth_ && !waits_.depth && hp::duration(result_.wall) && hp::duration(result_.cpu) &&
            hp::duration(result_.other_wall) && hp::duration(result_.other_cpu) &&
            hp::duration(result_.wait_wall - part_wait_wall) &&
            hp::duration(result_.wait_cpu - part_wait_cpu) &&
            hp::duration(result_.wall - result_.wait_wall) &&
            hp::duration(result_.cpu - result_.wait_cpu);
        result_.complete = complete && result_.available && !errors && !result_.wait_errors &&
            std::uncaught_exceptions() == exceptions_;
        return result_;
    }
    void emit(bool complete) noexcept {
        if (!on_ || closed_) return;
        const auto& s = finish(complete);
        std::fprintf(stderr, "strata prefill transition: v=1 scope=%s available=%d complete=%d "
            "wall_s=%.9f thread_cpu_s=%.9f wait_wall_s=%.9f wait_cpu_s=%.9f "
            "other_wall_s=%.9f other_cpu_s=%.9f open_parts=%u wait_errors=%llu parts_disjoint=1 waits_nested=1 units=charged_wrapper_intervals logical_bytes_not_bus_traffic=1\n",
            label_, int(s.available), int(s.complete), s.wall, s.cpu, s.wait_wall, s.wait_cpu,
            s.other_wall, s.other_cpu, s.open_parts, (unsigned long long)s.wait_errors);
        static const char* names[kinds] = {"preflight", "retire", "allocate", "submit", "drain",
            "publish", "release", "rollback", "canonical", "graph_resume"};
        for (int i = 0; i < kinds; ++i) {
            const auto& c = s.cells[i];
            if (!c.calls) continue;
            std::fprintf(stderr, "strata prefill transition part: v=1 scope=%s kind=%s calls=%llu "
                "items=%llu logical_bytes=%llu throws=%llu wall_s=%.9f thread_cpu_s=%.9f "
                "wait_wall_s=%.9f wait_cpu_s=%.9f outside_wait_wall_s=%.9f outside_wait_cpu_s=%.9f\n",
                label_, names[i], (unsigned long long)c.calls, (unsigned long long)c.items,
                (unsigned long long)c.logical_bytes, (unsigned long long)c.throws, c.wall, c.cpu,
                c.wait_wall, c.wait_cpu, c.wall-c.wait_wall, c.cpu-c.wait_cpu);
        }
    }
    ~Scope() noexcept { emit(false); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
class Part {
    Scope* scope_ = active;
    Kind kind_;
    hp::Stamp begin_{};
    double wait_cpu_ = 0, wait_wall_ = 0;
    int exceptions_ = 0;
    bool measured_ = false;
public:
    explicit Part(Kind kind, uint64_t items = 0, uint64_t bytes = 0) noexcept : kind_(kind) {
        if (!scope_) return;
        if (scope_->depth_++ || scope_->waits_.depth || kind < 0 || kind >= kinds ||
            hp::active != &scope_->waits_) { scope_->invalid_ = true; return; }
        auto& c = scope_->result_.cells[kind];
        if (c.calls == UINT64_MAX || items > UINT64_MAX-c.items || bytes > UINT64_MAX-c.logical_bytes) {
            scope_->invalid_ = true; return;
        }
        ++c.calls; c.items += items; c.logical_bytes += bytes;
        scope_->wait_totals(wait_cpu_, wait_wall_);
        exceptions_ = std::uncaught_exceptions(); begin_ = hp::stamp(); measured_ = true;
    }
    ~Part() noexcept {
        if (!scope_) return;
        --scope_->depth_;
        if (!measured_) return;
        if (scope_->closed_) { scope_->invalid_ = true; return; }
        const auto end = hp::stamp();
        double wc = 0, ww = 0; scope_->wait_totals(wc, ww);
        wc -= wait_cpu_; ww -= wait_wall_;
        const double cpu = end.cpu-begin_.cpu, wall = end.wall-begin_.wall;
        if (!begin_.valid || !end.valid || scope_->depth_ || scope_->waits_.depth ||
            !hp::duration(cpu) || !hp::duration(wall) || !hp::duration(wc) || !hp::duration(ww) ||
            !hp::duration(cpu-wc) || !hp::duration(wall-ww)) { scope_->invalid_ = true; return; }
        auto& c = scope_->result_.cells[kind_];
        c.cpu += cpu; c.wall += wall; c.wait_cpu += wc; c.wait_wall += ww;
        if (std::uncaught_exceptions() > exceptions_) ++c.throws;
    }
    Part(const Part&) = delete;
    Part& operator=(const Part&) = delete;
};
} // namespace strata::sycl_backend::prefill_transition_profile
