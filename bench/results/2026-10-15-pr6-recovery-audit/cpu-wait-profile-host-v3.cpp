// SPDX-License-Identifier: MIT
// Exercise production v3 partition accounting, including errors and observer cost; no GPU.
#include "strata/sycl/cpu_wait_profile.hpp"
#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>

namespace p = strata::sycl_backend::cpu_wait_profile;
thread_local int cpu_reads = 0;
thread_local int bad_read = 0;
thread_local bool backward = false;
extern "C" int __real_clock_gettime(clockid_t, timespec*);
extern "C" int __wrap_clock_gettime(clockid_t clock, timespec* value) {
    if (clock == CLOCK_THREAD_CPUTIME_ID && ++cpu_reads == bad_read) {
        if (!backward) { errno = EIO; return -1; }
        const int result = __real_clock_gettime(clock, value);
        value->tv_sec -= 1;
        return result;
    }
    return __real_clock_gettime(clock, value);
}
void busy() {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    volatile unsigned long value = 1;
    while (std::chrono::steady_clock::now() < end) value = value * 1664525 + 1013904223;
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    if (mode == "off") {
        p::Scope scope(false);
        p::blocking(p::event, [] {});
        scope.complete(true);
        assert(cpu_reads == 0);
        return 0;
    }
    if (mode == "overhead" || mode == "overhead-off") {
        constexpr int iterations = 100000;
        timespec a{}, b{};
        __real_clock_gettime(CLOCK_THREAD_CPUTIME_ID, &a);
        {
            p::Scope scope(mode == "overhead");
            for (int i = 0; i < iterations; ++i) { p::Host h(p::gemm_submit); }
            scope.complete(true);
        }
        __real_clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
        if (mode == "overhead-off") assert(cpu_reads == 0);
        const double seconds = double(b.tv_sec-a.tv_sec) + double(b.tv_nsec-a.tv_nsec)*1e-9;
        std::printf("overhead mode=%s iterations=%d cpu_s=%.9f cpu_ns_per_guard=%.6f clock_reads=%d\n",mode.c_str(),iterations,seconds,seconds*1e9/iterations,cpu_reads);
        return 0;
    }
    if (mode == "all-kinds" || mode == "all-kinds-off") {
        p::Scope scope(mode == "all-kinds");
        for (int k = 0; k < p::host_kinds; ++k) {
            p::Host h(static_cast<p::HostKind>(k));
            busy();
            if (k == p::route_exchange)
                p::blocking(p::compat_queue, [] { busy(); });
            h.finish(); h.finish();
        }
        scope.complete(true);
        if (mode == "all-kinds-off") assert(cpu_reads == 0);
        return 0;
    }
    if (mode.rfind("host-", 0) == 0) {
        p::Scope scope(true);
        if (mode == "host-nan" || mode == "host-inf") {
            p::active->host_cpu[p::source_copy] = mode == "host-nan" ? NAN : INFINITY;
        } else if (mode == "host-nested") {
            p::Host outer(p::source_copy); p::Host inner(p::gemm_submit);
        } else if (mode == "host-inwait") {
            p::blocking(p::event, [] { p::Host invalid(p::host_group); });
        } else if (mode == "host-throw") {
            try { p::Host h(p::gemm_submit); throw std::runtime_error("original"); }
            catch (const std::runtime_error& e) { assert(std::string(e.what())=="original"); }
        } else if (mode == "host-fail") {
            bad_read=2; p::Host h(p::source_copy); busy();
        } else {
            assert(mode == "host-normal");
            { p::Host h(p::source_copy); busy();
              p::blocking(p::event, [] { busy(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }); }
            { p::Host h(p::host_group); busy(); h.finish(); h.finish(); }
            { p::Host h(p::gemm_submit); busy(); }
            busy();
        }
        scope.complete(true);
        return 0;
    }
    if (mode == "fail-begin") bad_read = 1;
    if (mode == "fail-end" || mode == "backward") bad_read = 3;
    backward = mode == "backward";
    if (mode == "nested-scope") {
        p::Scope outer(true);
        { p::Scope inner(true); inner.complete(true); }
        outer.complete(true);
        return 0;
    }
    if (mode == "unwind") {
        try {
            p::Scope scope(true);
            p::blocking(p::event, [] { throw std::runtime_error("original"); });
        } catch (const std::runtime_error& e) {
            assert(std::string(e.what()) == "original");
            return 0;
        }
        return 1;
    }
    assert(mode == "normal" || mode == "false" || mode == "fail-begin" ||
           mode == "fail-end" || mode == "backward");
    for (int iteration = 0; iteration < (mode == "normal" ? 2 : 1); ++iteration) {
        p::Scope scope(true);
        errno = EDOM;
        p::blocking(p::event, [] { busy(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); });
        assert(errno == EDOM);
        p::blocking(p::compat_queue, [] {
            p::blocking(p::compute_queue, [] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        });
        try {
            p::blocking(p::event, [] { throw std::runtime_error("original"); });
        } catch (const std::runtime_error& e) { assert(std::string(e.what()) == "original"); }
        std::thread worker([] { p::blocking(p::transfer_queue, [] { busy(); }); });
        worker.join();
        busy();
        scope.complete(mode != "false");
    }
}
