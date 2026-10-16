// Mixed SYCL/ESIMD probe for the compiler review in docs/SYCL_IMPLEMENTATION.md.
// Set ONEAPI_DEVICE_SELECTOR=level_zero:gpu when running it.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <iostream>

int main() {
  try {
    sycl::queue q;
    auto *p = sycl::malloc_shared<float>(16, q);
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
       p[it.get_global_linear_id()] = 1;
     }).wait_and_throw();
    q.parallel_for(
         sycl::nd_range<1>(1, 1),
         sycl::ext::oneapi::experimental::properties{
             sycl::ext::intel::experimental::grf_size<256>},
         [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
           sycl::ext::intel::esimd::simd<float, 16> v;
           v.copy_from(p);
           v += 2;
           v.copy_to(p);
         }).wait_and_throw();
    bool valid = true;
    for (int i = 0; i < 16; ++i)
      valid &= p[i] == 3;
    sycl::free(p, q);
    if (!valid)
      return 2;
    std::cout << "PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
