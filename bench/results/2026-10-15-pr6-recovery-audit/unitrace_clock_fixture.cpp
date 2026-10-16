// CPU-only source-clock reproduction. No SYCL, Level Zero, or GPU runtime.
#include <ctime>
#include <iostream>
#include <stdexcept>
#include "unitimer.h"

static int64_t ns(clockid_t id) {
  timespec t{};
  if (clock_gettime(id, &t)) throw std::runtime_error("clock_gettime failed");
  return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

int main() {
  UniTimer::StartUniTimer();
  for (int i = 0; i < 32; ++i) {
    const int64_t real_before = ns(CLOCK_REALTIME);
    const int64_t boot_before = ns(CLOCK_BOOTTIME);
    const int64_t raw = UniTimer::GetHostTimestamp();
    const int64_t reported = UniTimer::GetEpochTime(raw);
    const int64_t boot_after = ns(CLOCK_BOOTTIME);
    const int64_t real_after = ns(CLOCK_REALTIME);
    // boot-before/after bracket the raw read; no historical trace is shifted.
    std::cout << i << ' ' << real_before << ' ' << boot_before << ' ' << raw
              << ' ' << reported << ' ' << boot_after << ' ' << real_after << '\n';
  }
}
