#pragma once
#include <cstdlib>
#include <string>

namespace strata {
// The legacy device waits return after their spin bound without establishing
// that CPU plans/results are ready. Keep those modes out of graph capture.
inline bool require_sycl_host_boundaries(std::string& error) {
    const char* boundary = std::getenv("STRATA_SYCL_HOST_BOUNDARY");
    if (std::getenv("STRATA_VERIFY_NO_HOST") != nullptr ||
        (boundary != nullptr && std::atoi(boundary) == 0)) {
        error = "SYCL device-spin verification is disabled; unset STRATA_VERIFY_NO_HOST "
                "and use STRATA_SYCL_HOST_BOUNDARY=1 (the default)";
        return false;
    }
    return true;
}

// The interactive VRAM path can destroy physical cache allocations while
// verifier graphs retain their residency pointers. Its partial-failure path
// also needs to restore the residency table before another request can run.
inline bool require_sycl_cache_policy(bool elastic, std::string& error) {
    if (elastic) {
        error = "--vram-elastic is temporarily unavailable on SYCL; "
                "choose --expert-cache at startup";
        return false;
    }
    return true;
}
} // namespace strata
