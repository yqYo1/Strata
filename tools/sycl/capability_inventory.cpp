// Query only: no debug overrides, allocations, kernel submission, or resets.
// Build: icpx -std=c++20 -fsycl tools/sycl/capability_inventory.cpp -o
// /tmp/sycl-caps
#include <iostream>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

template <typename T> void values(const char *name, const T &xs) {
  std::cout << name << '=';
  for (auto x : xs)
    std::cout << static_cast<int>(x) << ',';
  std::cout << '\n';
}
void fp_values(const char *name, const std::vector<sycl::info::fp_config> &xs) {
  std::cout << name << '=';
  for (auto x : xs) {
    switch (x) {
#define FP_CASE(NAME)                                                          \
  case sycl::info::fp_config::NAME:                                            \
    std::cout << #NAME << ',';                                                 \
    break
      FP_CASE(denorm);
      FP_CASE(inf_nan);
      FP_CASE(round_to_nearest);
      FP_CASE(round_to_zero);
      FP_CASE(round_to_inf);
      FP_CASE(fma);
      FP_CASE(correctly_rounded_divide_sqrt);
      FP_CASE(soft_float);
#undef FP_CASE
    }
  }
  std::cout << '\n';
}
int main() {
  try {
    sycl::device d{sycl::gpu_selector_v};
    std::cout << "device=" << d.get_info<sycl::info::device::name>() << '\n'
              << "driver=" << d.get_info<sycl::info::device::driver_version>()
              << '\n';
#define __SYCL_ASPECT(NAME, ID)                                                \
  try {                                                                        \
    std::cout << #NAME << '=' << d.has(sycl::aspect::NAME) << '\n';            \
  } catch (const sycl::exception &e) {                                         \
    std::cout << "query_error=" << e.what() << '\n';                           \
  }
#include <sycl/info/aspects.def>
#undef __SYCL_ASPECT
#define QUERY(NAME)                                                            \
  std::cout << #NAME << '=' << d.get_info<sycl::info::device::NAME>() << '\n'
    QUERY(max_compute_units);
    QUERY(max_work_group_size);
    QUERY(local_mem_size);
    QUERY(global_mem_size);
    QUERY(max_mem_alloc_size);
    QUERY(mem_base_addr_align);
    QUERY(max_parameter_size);
    QUERY(image2d_max_width);
    QUERY(image2d_max_height);
    QUERY(image3d_max_width);
    QUERY(image3d_max_height);
    QUERY(image3d_max_depth);
#undef QUERY
    values("sub_group_sizes",
           d.get_info<sycl::info::device::sub_group_sizes>());
    fp_values("half_fp_config",
              d.get_info<sycl::info::device::half_fp_config>());
    fp_values("single_fp_config",
              d.get_info<sycl::info::device::single_fp_config>());
    fp_values("double_fp_config",
              d.get_info<sycl::info::device::double_fp_config>());
    values("atomic_memory_order_capabilities",
           d.get_info<sycl::info::device::atomic_memory_order_capabilities>());
    values("atomic_fence_order_capabilities",
           d.get_info<sycl::info::device::atomic_fence_order_capabilities>());
    values("atomic_memory_scope_capabilities",
           d.get_info<sycl::info::device::atomic_memory_scope_capabilities>());
    values("atomic_fence_scope_capabilities",
           d.get_info<sycl::info::device::atomic_fence_scope_capabilities>());
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    for (auto c : d.get_info<sycl::ext::oneapi::experimental::info::device::
                                 matrix_combinations>())
      std::cout << "matrix=" << int(c.atype) << ',' << int(c.btype) << ','
                << int(c.ctype) << ',' << int(c.dtype) << " max=" << c.max_msize
                << ',' << c.max_nsize << ',' << c.max_ksize
                << " exact=" << c.msize << ',' << c.nsize << ',' << c.ksize
                << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
