// Optional CPU-time observation. These clocks do not measure GPU service or memory traffic.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#if defined(__linux__)
#include <time.h>
#endif

namespace strata::core {

inline bool decode_cpu_profile_enabled() noexcept {
    // Read once; the launcher must set the environment before native exec.
    static const bool enabled = std::getenv("STRATA_DECODE_CPU_PROFILE") != nullptr;
    return enabled;
}

struct CpuTimeReading {
    uint64_t ns = 0;
    bool valid = false;
};

#if defined(__linux__)
inline CpuTimeReading read_cpu_clock(clockid_t id) noexcept {
    timespec value{};
    if (clock_gettime(id, &value) != 0 || value.tv_sec < 0 || value.tv_nsec < 0 ||
        value.tv_nsec >= 1000000000L) return {};
    const auto seconds = static_cast<uint64_t>(value.tv_sec);
    const auto nanos = static_cast<uint64_t>(value.tv_nsec);
    if (seconds > (std::numeric_limits<uint64_t>::max() - nanos) / 1000000000ULL) return {};
    return {seconds * 1000000000ULL + nanos, true};
}
#endif

inline CpuTimeReading thread_cpu_time() noexcept {
#if defined(__linux__)
    return read_cpu_clock(CLOCK_THREAD_CPUTIME_ID);
#else
    return {}; // Unsupported clocks invalidate observation, never inference.
#endif
}

inline CpuTimeReading process_cpu_time() noexcept {
#if defined(__linux__)
    return read_cpu_clock(CLOCK_PROCESS_CPUTIME_ID);
#else
    return {};
#endif
}

struct CpuTimeCounter {
    uint64_t ns = 0, calls = 0, errors = 0, unwound = 0;

    void invalidate() noexcept {
        if (errors != std::numeric_limits<uint64_t>::max()) ++errors;
    }

    void add(CpuTimeReading begin, CpuTimeReading end, bool exception = false) noexcept {
        if (calls == std::numeric_limits<uint64_t>::max() ||
            (exception && unwound == std::numeric_limits<uint64_t>::max())) {
            invalidate(); return;
        }
        ++calls;
        unwound += exception ? 1 : 0;
        if (!begin.valid || !end.valid || end.ns < begin.ns ||
            end.ns - begin.ns > std::numeric_limits<uint64_t>::max() - ns) {
            invalidate();
            return;
        }
        ns += end.ns - begin.ns;
    }

    void merge(const CpuTimeCounter& other) noexcept {
        // Every observation has a finite job deadline. Check nevertheless so a
        // failed accounting operation cannot masquerade as a valid short time.
        if (other.ns > std::numeric_limits<uint64_t>::max() - ns ||
            other.calls > std::numeric_limits<uint64_t>::max() - calls ||
            other.errors > std::numeric_limits<uint64_t>::max() - errors ||
            other.unwound > std::numeric_limits<uint64_t>::max() - unwound) {
            invalidate();
            return;
        }
        ns += other.ns; calls += other.calls; errors += other.errors; unwound += other.unwound;
    }
};

class CpuTimeRegion {
public:
    explicit CpuTimeRegion(CpuTimeCounter* counter) noexcept : counter_(counter) {
        if (counter_) { begin_ = thread_cpu_time(); exceptions_ = std::uncaught_exceptions(); }
    }
    ~CpuTimeRegion() { close(); }
    CpuTimeRegion(const CpuTimeRegion&) = delete;
    CpuTimeRegion& operator=(const CpuTimeRegion&) = delete;

    void close() noexcept {
        if (!counter_) return;
        const auto end = thread_cpu_time();
        counter_->add(begin_, end, std::uncaught_exceptions() > exceptions_);
        counter_ = nullptr;
    }

private:
    CpuTimeCounter* counter_ = nullptr;
    CpuTimeReading begin_{};
    int exceptions_ = 0;
};

template<class Fn>
inline auto observe_cpu_call(CpuTimeCounter* counter, Fn&& call) -> decltype(call()) {
    CpuTimeRegion region(counter);
    return call();
}

inline CpuTimeReading cpu_time_delta(CpuTimeReading begin, CpuTimeReading end) noexcept {
    if (!begin.valid || !end.valid || end.ns < begin.ns) return {};
    return {end.ns - begin.ns, true};
}

inline CpuTimeCounter cpu_counter_delta(const CpuTimeCounter& begin, const CpuTimeCounter& end) noexcept {
    CpuTimeCounter out;
    if (end.ns < begin.ns || end.calls < begin.calls || end.errors < begin.errors || end.unwound < begin.unwound) {
        out.invalidate(); return out;
    }
    out.ns = end.ns - begin.ns;
    out.calls = end.calls - begin.calls;
    out.errors = end.errors - begin.errors;
    out.unwound = end.unwound - begin.unwound;
    // An earlier bad clock is a process-level observation failure, even if the
    // counter did not change during this particular request.
    if (begin.errors || end.errors || begin.unwound || end.unwound) out.invalidate();
    return out;
}

} // namespace strata::core
