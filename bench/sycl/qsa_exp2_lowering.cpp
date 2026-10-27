// Authored for Strata; repository LICENSE applies.
// Compile-only seam for the configured SYCL/IGC toolchain. No host entrypoint.
// This does not change the engine or qualify accuracy, FTZ, or performance.
#include <sycl/sycl.hpp>

namespace strata::diagnostic {
class RegularExp2;
class NativeExp2;
class BoundedNativeExp2;

void exp2_lowering(sycl::queue &queue, const float *input, float *output) {
  const sycl::nd_range<1> range(sycl::range<1>(128), sycl::range<1>(128));
  queue.parallel_for<RegularExp2>(range,
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const auto i = item.get_global_linear_id();
        output[i] = sycl::exp2(input[i]);
      });
  queue.parallel_for<NativeExp2>(range,
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const auto i = item.get_global_linear_id();
        output[i] = sycl::native::exp2(input[i]);
      });
  // A possible future negative-domain/FTZ seam, not selected inference math.
  // Its domain and x=-126 boundary still require source and device evidence.
  queue.parallel_for<BoundedNativeExp2>(range,
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const auto i = item.get_global_linear_id();
        const float x = input[i];
        output[i] = x < -126.f ? 0.f : sycl::native::exp2(x);
      });
}
} // namespace strata::diagnostic
