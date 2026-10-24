#pragma once
// Authored for Strata; repository LICENSE (MIT) applies.
// Diagnostic only. Graph clocks describe an envelope, not arithmetic service.
#include "strata/sycl/cpu_wait_profile.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace strata::sycl_backend {
class MtpGraphObserver {
    using Stamp = cpu_wait_profile::Stamp;
    static Stamp stamp() noexcept { return cpu_wait_profile::stamp(); }
    static constexpr unsigned max_windows = 64, max_raw = 32;
public:
    enum Phase { capture, submit, drain, query, phases };
private:
    struct Interval { double wall = 0, cpu = 0; std::uint64_t calls = 0, throws = 0; };
    struct Raw {
        compat::GraphEnvelopeTimestamps t{};
        std::uint64_t window = 0;
        int step = 0, status = 0;
        int T = 0, accepted_row = 0;
        std::int64_t position = 0;
        bool coupled = false;
        bool valid = false, queue_delay_valid = false;
    };
    bool enabled_ = false, profiled_ = false, open_ = false, consumed_ = false, invalid_ = false;
    bool request_override_ = false;
    std::uint64_t request_ = 5, skip_ = 8, limit_ = max_windows, seen_ = 0, windows_ = 0;
    std::uint64_t completed_ = 0, failed_ = 0, replayed_ = 0, valid_clocks_ = 0, queue_clocks_ = 0;
    std::uint64_t envelope_ns_ = 0, last_end_ = 0, raw_count_ = 0;
    std::uint64_t rows_ = 0, T_mask_ = 0;
    unsigned active_windows_ = 0;
    std::uint64_t kind_envelope_ns_[2]{}, kind_clocks_[2]{};
    double wall_ = 0, cpu_ = 0;
    std::array<std::array<Interval, phases>, 2> intervals_{};
    std::array<Raw, max_raw> raw_{};
    static bool number(const char* name, std::uint64_t& value, std::uint64_t lo, std::uint64_t hi) noexcept {
        const char* s = std::getenv(name);
        if (!s) return true;
        if (!*s) return false;
        std::uint64_t n = 0;
        for (; *s; ++s) {
            if (*s < '0' || *s > '9' || n > (hi - unsigned(*s - '0')) / 10) return false;
            n = n * 10 + unsigned(*s - '0');
        }
        if (n < lo || n > hi) return false;
        value = n; return true;
    }
    bool add_interval(Interval& out, Stamp a, Stamp b, bool threw) noexcept {
        ++out.calls; out.throws += threw;
        if (!a.valid || !b.valid || !cpu_wait_profile::duration(b.wall - a.wall) ||
            !cpu_wait_profile::duration(b.cpu - a.cpu)) { invalid_ = true; return false; }
        out.wall += b.wall - a.wall; out.cpu += b.cpu - a.cpu; return true;
    }
    bool begin(std::uint64_t request, bool single_request) noexcept {
        if (!enabled_) return false;
        if (open_) { invalid_ = true; return false; }
        if (single_request && !request_override_ && !consumed_) request_ = 1;
        if (request != request_ || consumed_) return false;
        consumed_ = true; open_ = true;
        return true;
    }
    void end(bool complete) noexcept {
        if (!open_) return;
        if (active_windows_) invalid_ = true;
        open_ = false;
        double classified_wall = 0, classified_cpu = 0;
        for (const auto& kind : intervals_)
            for (const auto& i : kind) { classified_wall += i.wall; classified_cpu += i.cpu; }
        const double other_wall = wall_ - classified_wall, other_cpu = cpu_ - classified_cpu;
        if (!cpu_wait_profile::duration(other_wall) || !cpu_wait_profile::duration(other_cpu)) invalid_ = true;
        std::fprintf(stderr, "strata MTP_OBSERVER v=1 request=%llu profile=%d request_complete=%d sample_complete=%d "
            "accounting_valid=%d skip=%llu limit=%llu seen=%llu windows=%llu completed=%llu failed=%llu replays=%llu rows=%llu T_mask=%llu "
            "wall_s=%.9f thread_cpu_s=%.9f other_wall_s=%.9f other_cpu_s=%.9f valid_graph_clocks=%llu "
            "valid_queue_clocks=%llu graph_envelope_ns=%llu round_clocks=%llu round_envelope_ns=%llu "
            "chain_clocks=%llu chain_envelope_ns=%llu raw_retained=%llu raw_omitted=%llu "
            "scope=draft_entry_through_return clocks=HOST_DISJOINT_DEVICE_NONADDITIVE "
            "arithmetic_service=UNKNOWN hardware_saturation=UNKNOWN observer_charged=1\n",
            (unsigned long long)request_, int(profiled_), int(complete),
            int(complete && windows_ == limit_ && completed_ == windows_ && !invalid_), int(!invalid_),
            (unsigned long long)skip_, (unsigned long long)limit_, (unsigned long long)seen_,
            (unsigned long long)windows_, (unsigned long long)completed_, (unsigned long long)failed_,
            (unsigned long long)replayed_, (unsigned long long)rows_, (unsigned long long)T_mask_, wall_, cpu_, other_wall, other_cpu,
            (unsigned long long)valid_clocks_, (unsigned long long)queue_clocks_,
            (unsigned long long)envelope_ns_,
            (unsigned long long)kind_clocks_[0], (unsigned long long)kind_envelope_ns_[0],
            (unsigned long long)kind_clocks_[1], (unsigned long long)kind_envelope_ns_[1], (unsigned long long)raw_count_,
            (unsigned long long)(profiled_ ? replayed_ - raw_count_ : 0));
        const char* names[phases] = {"capture", "submit", "drain", "query"};
        for (unsigned kind = 0; kind < 2; ++kind) for (unsigned p = 0; p < phases; ++p) {
            const auto& i = intervals_[kind][p];
            std::fprintf(stderr, "strata MTP_OBSERVER_PHASE v=1 request=%llu kind=%s phase=%s calls=%llu throws=%llu "
                "wall_s=%.9f thread_cpu_s=%.9f\n", (unsigned long long)request_, kind ? "chain" : "round", names[p],
                (unsigned long long)i.calls, (unsigned long long)i.throws, i.wall, i.cpu);
        }
        for (std::uint64_t j = 0; j < raw_count_; ++j) {
            const auto& r = raw_[j];
            std::fprintf(stderr, "strata MTP_GRAPH_CLOCK v=1 request=%llu window=%llu T=%d position=%lld accepted_row=%d coupled=%d step=%d status=%d "
                "available=%u submit=%llu start=%llu end=%llu resolution_ns=%llu envelope_valid=%d queue_delay_valid=%d\n",
                (unsigned long long)request_, (unsigned long long)r.window, r.T, (long long)r.position, r.accepted_row,
                int(r.coupled), r.step, r.status, r.t.available_fields,
                (unsigned long long)r.t.submit, (unsigned long long)r.t.start, (unsigned long long)r.t.end,
                (unsigned long long)r.t.timer_resolution, int(r.valid), int(r.queue_delay_valid));
        }
    }
public:
    // Called once during load, before graph capture. Never mutate process env.
    bool configure(std::string& err) {
        const char* mode = std::getenv("STRATA_SYCL_MTP_OBSERVER");
        if (!mode || std::strcmp(mode, "0") == 0) return true;
        if (std::strcmp(mode, "host") != 0 && std::strcmp(mode, "graph") != 0) {
            err = "mtp observer: mode must be host or graph"; return false;
        }
#if !defined(__linux__)
        err = "mtp observer: calling-thread CPU attribution requires Linux"; return false;
#endif
        if (!number("STRATA_SYCL_MTP_OBSERVER_REQUEST", request_, 1, 1000000) ||
            !number("STRATA_SYCL_MTP_OBSERVER_SKIP", skip_, 0, 1000000) ||
            !number("STRATA_SYCL_MTP_OBSERVER_WINDOWS", limit_, 1, max_windows)) {
            err = "mtp observer: invalid request/skip/windows budget"; return false;
        }
        request_override_ = std::getenv("STRATA_SYCL_MTP_OBSERVER_REQUEST") != nullptr;
        enabled_ = true; profiled_ = std::strcmp(mode, "graph") == 0; return true;
    }
    bool enabled() const noexcept { return enabled_; }
    bool profiled() const noexcept { return profiled_; }
    class Request {
        MtpGraphObserver* owner_;
    public:
        Request(MtpGraphObserver& o, std::uint64_t id, bool single_request) noexcept
            : owner_(o.begin(id, single_request) ? &o : nullptr) {}
        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;
        void finish(bool complete) noexcept { if (owner_) { owner_->end(complete); owner_ = nullptr; } }
        ~Request() noexcept { finish(false); }
    };
    Request request(std::uint64_t id, bool single_request = false) noexcept { return Request(*this, id, single_request); }
    class Window {
        MtpGraphObserver* o_ = nullptr;
        Stamp begin_{};
        bool complete_ = false;
        std::uint64_t index_ = 0;
        int T_ = 0, accepted_row_ = 0;
        std::int64_t position_ = 0;
        bool coupled_ = false;
        unsigned depth_ = 0;
    public:
        Window(MtpGraphObserver& o, int T, std::int64_t position, int accepted_row, bool coupled) noexcept
            : T_(T), accepted_row_(accepted_row), position_(position), coupled_(coupled) {
            if (!o.open_) return;
            if (o.active_windows_) { o.invalid_ = true; return; }
            if (o.seen_ == std::numeric_limits<std::uint64_t>::max()) { o.invalid_ = true; return; }
            const auto i = o.seen_++;
            if (i < o.skip_ || o.windows_ >= o.limit_) return;
            o_ = &o; index_ = i; ++o.windows_; begin_ = stamp();
            ++o.active_windows_;
            if (T > 0 && T <= 8) { o.rows_ += unsigned(T); o.T_mask_ |= std::uint64_t(1) << T; }
            else o.invalid_ = true;
        }
        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;
        bool enabled() const noexcept { return o_ != nullptr; }
        template<class F> auto measure(Phase phase, F&& f, int step = 0) -> decltype(f()) {
            if (!o_) return f();
            if (depth_++) o_->invalid_ = true;
            auto& interval = o_->intervals_[step > 0 ? 1 : 0][phase];
            const auto a = stamp();
            try { auto result = f(); o_->add_interval(interval, a, stamp(), false); --depth_; return result; }
            catch (...) { o_->add_interval(interval, a, stamp(), true); --depth_; throw; }
        }
        // Caller has just checked its existing stream drain. No extra wait/marker.
        bool graph_clock(cudaEvent_t done, int step, std::string& err) {
            if (!o_) return true;
            ++o_->replayed_;
            if (!o_->profiled_) return true;
            Raw r; r.window = index_; r.step = step; r.T = T_; r.accepted_row = accepted_row_;
            r.position = position_; r.coupled = coupled_;
            const unsigned kind = step > 0 ? 1 : 0;
            const double query_before = o_->intervals_[kind][query].wall;
            const auto status = measure(query, [&] { return compat::graph_completion_timestamps(done, &r.t); }, step);
            r.status = int(status);
            const auto& t = r.t;
            r.valid = status == cudaSuccess && t.available_fields == 7 && t.timer_resolution > 0 &&
                t.start > 0 && t.end > t.start && t.end - t.start >= t.timer_resolution &&
                (!o_->last_end_ || t.start >= o_->last_end_) &&
                o_->intervals_[kind][query].wall - query_before <= 1.0;
            r.queue_delay_valid = r.valid && t.start >= t.submit;
            if (r.valid && t.end - t.start <= std::numeric_limits<std::uint64_t>::max() - o_->envelope_ns_) {
                ++o_->valid_clocks_; o_->queue_clocks_ += r.queue_delay_valid;
                o_->envelope_ns_ += t.end - t.start; o_->last_end_ = t.end;
                ++o_->kind_clocks_[kind]; o_->kind_envelope_ns_[kind] += t.end - t.start;
            } else { r.valid = false; r.queue_delay_valid = false; o_->invalid_ = true; }
            if (o_->raw_count_ < max_raw) o_->raw_[o_->raw_count_++] = r;
            if (!r.valid) { err = "mtp observer: graph timestamps unavailable/invalid; diagnostic incomplete"; return false; }
            return true;
        }
        void complete() noexcept { complete_ = true; }
        ~Window() noexcept {
            if (!o_) return;
            --o_->active_windows_;
            if (depth_) o_->invalid_ = true;
            Interval total;
            o_->add_interval(total, begin_, stamp(), false);
            o_->wall_ += total.wall; o_->cpu_ += total.cpu;
            if (complete_) ++o_->completed_; else ++o_->failed_;
        }
    };
    Window window(int T, std::int64_t position, int accepted_row, bool coupled) noexcept {
        return Window(*this, T, position, accepted_row, coupled);
    }
};
} // namespace strata::sycl_backend
