// Main-authored CPU clock calibration; repository LICENSE. No GPU/runtime calls.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <link.h>
#include <time.h>

namespace {
int provider(dl_phdr_info* info, size_t, void*) {
    if (std::strstr(info->dlpi_name, "libstdc++.so.6"))
        std::printf("clock_provider=%s\n", info->dlpi_name);
    return 0;
}
}

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr size_t count = 65536;
    std::array<int64_t, count> ns{};
    timespec resolution{};
    if (clock_getres(CLOCK_MONOTONIC, &resolution)) return 2;
    for (auto& value : ns) {
        const auto begin = Clock::now();
        const auto end = Clock::now();
        value = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    }
    std::sort(ns.begin(), ns.end());
    dl_iterate_phdr(provider, nullptr);
    std::printf("clock_samples=%zu steady=%d period_num=%lld period_den=%lld "
                "resolution_ns=%lld min_ns=%lld p99_ns=%lld max_ns=%lld\n",
        count, int(Clock::is_steady), (long long)Clock::period::num,
        (long long)Clock::period::den, (long long)(resolution.tv_sec*1000000000ll+resolution.tv_nsec),
        (long long)ns.front(), (long long)ns[count*99/100], (long long)ns.back());
}
