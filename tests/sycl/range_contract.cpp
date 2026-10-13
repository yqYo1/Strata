// Host-only regression for the compiler's global-range admission contract.
// This deliberately does not launch billions of GPU work-items or claim a
// physical 256K model run. Use the actual SYCL header validator used at launch.
#include <sycl/sycl.hpp>
#include <sycl/detail/id_queries_fit_in_int.hpp>

#include <cstdint>
#include <iostream>
#include <limits>

#if __SYCL_ID_QUERIES_FIT_IN_INT__ || __SYCL_ID_QUERIES_FIT_IN_UINT__
#error "Strata SYCL consumers must use the size_t global-range contract"
#endif

int main() {
  static_assert(sizeof(size_t) == sizeof(uint64_t));
  try {
#ifndef __SYCL_DEVICE_ONLY__
    const size_t signed_boundary = size_t(std::numeric_limits<int>::max()) + 1;
    sycl::detail::checkValueRange(sycl::range<1>(signed_boundary));
    sycl::detail::checkValueRange(sycl::range<2>(65536, 32768));
    sycl::detail::checkValueRange(
        sycl::nd_range<2>({262144, 32768}, {1, 256}));
#endif
    std::cout << "host launch validation: 1D/2D signed-int boundary and "
                 "8 Gi-item global range admitted; no GPU execution: PASS\n";
    return 0;
  } catch (const sycl::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
