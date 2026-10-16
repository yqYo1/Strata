// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include <sycl/sycl.hpp>
namespace strata::prefill {
// Read once per process, before submissions. Does not mutate the environment.
// Throws for malformed configuration or a requested backend absent at build.
void native16_backend_validate_config(sycl::queue &queue);
}
