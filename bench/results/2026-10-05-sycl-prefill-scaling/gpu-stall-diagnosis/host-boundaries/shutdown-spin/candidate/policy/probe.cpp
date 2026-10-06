
#include "strata/sycl_execution_policy.hpp"
#include <cassert>
#include <cstdio>
int main() {
    std::string error;
    unsetenv("STRATA_VERIFY_NO_HOST"); unsetenv("STRATA_SYCL_HOST_BOUNDARY");
    assert(strata::require_sycl_host_boundaries(error));
    for (const char* value : {"1", "2"}) {
        setenv("STRATA_SYCL_HOST_BOUNDARY",value,1);
        assert(strata::require_sycl_host_boundaries(error));
    }
    for (const char* value : {"0", "", "bad"}) {
        setenv("STRATA_SYCL_HOST_BOUNDARY",value,1);
        assert(!strata::require_sycl_host_boundaries(error) && !error.empty());
    }
    unsetenv("STRATA_SYCL_HOST_BOUNDARY");
    for (const char* value : {"0", "1", ""}) {
        setenv("STRATA_VERIFY_NO_HOST",value,1);
        assert(!strata::require_sycl_host_boundaries(error));
    }
    std::puts("PASS default/explicit boundaries and all legacy selectors");
}
