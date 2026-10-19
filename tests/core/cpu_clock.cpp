// Failed CPU clocks/counters must not look like valid short times.
#include "strata/core/cpu_clock.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace strata::core;

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "on") && std::strcmp(argv[1], "off"))) return 2;
    int failures = 0;
    const auto require = [&](bool condition, const char* what) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    };
    require(decode_cpu_profile_enabled() == !std::strcmp(argv[1], "on"), "launch-time observer selection");
    CpuTimeCounter good;
    good.add({100, true}, {145, true});
    require(good.ns == 45 && good.calls == 1 && good.errors == 0, "valid reading delta");
    good.add({}, {180, true});
    require(good.ns == 45 && good.calls == 2 && good.errors == 1, "failed clock preserves prior evidence");
    CpuTimeCounter reversed;
    reversed.add({200, true}, {100, true});
    require(reversed.errors == 1 && reversed.ns == 0, "reversed clock rejected");
    const auto max = std::numeric_limits<uint64_t>::max();
    CpuTimeCounter overflow;
    overflow.ns = max - 4;
    overflow.add({0, true}, {5, true});
    require(overflow.errors == 1 && overflow.ns == max - 4, "duration overflow rejected");
    overflow.errors = max;
    overflow.add({}, {});
    require(overflow.errors == max, "error counter saturates");
    CpuTimeCounter merge_overflow;
    merge_overflow.calls = max;
    merge_overflow.merge({1, 1, 0, 0});
    require(merge_overflow.errors == 1 && merge_overflow.calls == max, "aggregation overflow rejected");
    const CpuTimeCounter before{100, 5, 0, 0}, after{180, 7, 0, 0};
    const auto delta = cpu_counter_delta(before, after);
    require(delta.ns == 80 && delta.calls == 2 && !delta.errors, "snapshot counter delta");
    require(cpu_counter_delta(after, before).errors != 0, "snapshot regression rejected");
    require(cpu_counter_delta(good, good).errors != 0, "earlier clock failure cannot disappear");
    require(!cpu_time_delta({10, true}, {9, true}).valid &&
            !cpu_time_delta({}, {10, true}).valid, "invalid lifetime readings rejected");
#if defined(__linux__)
    const auto thread0 = thread_cpu_time(), process0 = process_cpu_time();
    volatile uint64_t value = 19;
    for (int i = 0; i < 1000000; ++i) value = value * 6364136223846793005ULL + 1;
    const auto thread1 = thread_cpu_time(), process1 = process_cpu_time();
    require(cpu_time_delta(thread0, thread1).valid && thread1.ns > thread0.ns, "live calling-thread CPU clock");
    require(cpu_time_delta(process0, process1).valid && process1.ns > process0.ns, "live process CPU clock");
    CpuTimeCounter region;
    {
        CpuTimeRegion scope(&region);
        for (int i = 0; i < 10000; ++i) value = value * 3 + 1;
        scope.close(); scope.close();
    }
    require(region.calls == 1 && !region.errors && !region.unwound, "region closes exactly once");
    const auto returned = observe_cpu_call(&region, [] { return 17; });
    require(returned == 17 && region.calls == 2, "observed call preserves return value");
    try { observe_cpu_call(&region, []() -> int { throw std::runtime_error("fixture"); }); }
    catch (const std::runtime_error&) {}
    require(region.calls == 3 && region.unwound == 1 && !region.errors, "unwinding is visible");
    { CpuTimeRegion disabled(nullptr); }
    require(region.calls == 3, "disabled region changes no counter");
#else
    require(!thread_cpu_time().valid && !process_cpu_time().valid, "unsupported platform stays invalid");
#endif
    std::printf("cpu_clock: %d failures\n", failures);
    return failures ? 1 : 0;
}
