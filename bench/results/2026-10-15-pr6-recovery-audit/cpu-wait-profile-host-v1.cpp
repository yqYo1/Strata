// SPDX-License-Identifier: MIT
// Exercise the production header directly; no SYCL runtime or GPU.
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
