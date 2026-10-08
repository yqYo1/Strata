#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace strata {
// A host-only deadline: no driver call is made while testing readiness.
// Lost DMA completion leaves queued buffer accesses of unknown lifetime.
// End the failing process without running GPU destructors or returning a
// buffer to its producer. This cannot unblock a kernel-side D-state close.
inline constexpr auto kHostWaitBudget = std::chrono::minutes(5);
template<class Ready, class Stop>
bool wait_host_ready(Ready ready, Stop stop, const char* what,
                     std::chrono::steady_clock::duration budget = kHostWaitBudget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    unsigned polls = 0;
    for (;;) {
        if (stop()) return false;
        if (ready()) return true;
        if ((++polls & 255u) == 0 && std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "strata: host wait timed out: %s; ending the process without GPU cleanup\n", what);
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::yield();
    }
}
template<class Ready>
void wait_host_ready(Ready ready, const char* what,
                     std::chrono::steady_clock::duration budget = kHostWaitBudget) {
    (void) wait_host_ready(ready, [] { return false; }, what, budget);
}
} // namespace strata
