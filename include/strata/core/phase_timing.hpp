// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
namespace strata::core {
// Host monotonic spans can nest: their sum is not wall time.
struct PhaseTiming { uint64_t calls = 0, succeeded = 0, bytes = 0, successful_bytes = 0, nanoseconds = 0; };
class PhaseTimedCall {
public:
    explicit PhaseTimedCall(PhaseTiming& m, uint64_t bytes = 0)
        : metric_(m), bytes_(bytes), start_(std::chrono::steady_clock::now()) {}
    bool result(bool success) { success_ = success; return success; }
    ~PhaseTimedCall() {
        ++metric_.calls; metric_.bytes += bytes_;
        metric_.succeeded += success_ ? 1 : 0;
        metric_.successful_bytes += success_ ? bytes_ : 0;
        metric_.nanoseconds += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_).count());
    }
private:
    PhaseTiming& metric_; uint64_t bytes_; std::chrono::steady_clock::time_point start_; bool success_ = false;
};
inline void phase_timing_delta(const char* stage, const char* operation, const PhaseTiming& a, const PhaseTiming& b) {
    std::fprintf(stderr, "strata phase-cache: stage=%s operation=%s calls=%llu success=%llu failed=%llu attempt_bytes=%llu success_bytes=%llu host_elapsed_ms=%.3f nested=1\n",
        stage, operation, static_cast<unsigned long long>(a.calls - b.calls),
        static_cast<unsigned long long>(a.succeeded - b.succeeded),
        static_cast<unsigned long long>((a.calls - b.calls)-(a.succeeded - b.succeeded)),
        static_cast<unsigned long long>(a.bytes - b.bytes), static_cast<unsigned long long>(a.successful_bytes - b.successful_bytes),
        static_cast<double>(a.nanoseconds - b.nanoseconds) / 1e6);
}
// A cycle operation emits one actual completed-call record, including exceptions.
class PhaseLoggedCall {
public:
    explicit PhaseLoggedCall(const char* operation) : record_{{}, operation}, timer_(record_.value) {}
    bool result(bool success) { return timer_.result(success); }
    ~PhaseLoggedCall() = default;
private:
    struct Record {
        PhaseTiming value;
        const char* operation;
        ~Record() { phase_timing_delta("cycle", operation, value, {}); }
    };
    // Record is destroyed after the timer records its result.
    Record record_;
    PhaseTimedCall timer_;
};
} // namespace strata::core
