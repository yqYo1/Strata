// Independently authored for Strata; repository LICENSE applies.
// Production clock backend on the same host core. Loop/check overhead remains
// included; this is not the cost of a whole observer or a hardware ceiling.
#include "strata/sycl/decode_service_profile.hpp"
#include <sched.h>
#include <cstdio>
#include <cstdint>

namespace profile = strata::sycl_backend::decode_service_profile;
int main() {
    cpu_set_t allowed{}, selected{};
    if (::sched_getaffinity(0, sizeof(allowed), &allowed) != 0 || !CPU_ISSET(0, &allowed)) {
        std::fputs("host core0 unavailable\n", stderr); return 1;
    }
    CPU_SET(0, &selected);
    if (::sched_setaffinity(0, sizeof(selected), &selected) != 0) {
        std::perror("host core0 affinity"); return 1;
    }
    timespec wall_resolution{}, cpu_resolution{};
    if (::clock_getres(CLOCK_MONOTONIC, &wall_resolution) != 0 ||
        ::clock_getres(CLOCK_THREAD_CPUTIME_ID, &cpu_resolution) != 0 ||
        wall_resolution.tv_sec != 0 || cpu_resolution.tv_sec != 0 ||
        wall_resolution.tv_nsec <= 0 || cpu_resolution.tv_nsec <= 0) {
        std::fputs("clock resolution unavailable\n", stderr); return 1;
    }
    constexpr unsigned bands = 5, reads = 50000;
    std::printf("host_clock_cost version=1 cpu=0 bands=%u reads=%u wall_resolution_ns=%ld cpu_resolution_ns=%ld\n",
        bands, reads, wall_resolution.tv_nsec, cpu_resolution.tv_nsec);
    for (unsigned band = 0; band < bands; ++band) {
        const auto begin = profile::clock_tick();
        auto previous = begin;
        uint64_t checksum = 0;
        for (unsigned i = 0; i < reads; ++i) {
            const auto tick = profile::clock_tick();
            if (!tick.valid || tick.wall < previous.wall || tick.cpu < previous.cpu) {
                std::fputs("nonmonotonic production clock\n", stderr); return 1;
            }
            checksum ^= tick.wall + tick.cpu;
            previous = tick;
        }
        const auto end = profile::clock_tick();
        if (!begin.valid || !end.valid || end.wall <= begin.wall || end.cpu <= begin.cpu) {
            std::fputs("invalid production clock interval\n", stderr); return 1;
        }
        std::printf("host_clock_band index=%u reads=%u wall_ns=%llu cpu_ns=%llu checksum=%llu\n",
            band, reads, (unsigned long long)(end.wall - begin.wall),
            (unsigned long long)(end.cpu - begin.cpu), (unsigned long long)checksum);
    }
    return std::fflush(stdout) != 0 || std::ferror(stdout) ? 1 : 0;
}
