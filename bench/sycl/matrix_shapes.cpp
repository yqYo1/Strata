// Authored for Strata; repository LICENSE applies.
// Device properties only: no allocation, queue, kernel, or runtime teardown test.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static void gate(const char *phase) {
  std::cout << "MATRIX_GATE " << phase << std::endl;
  if (std::cin.get() != 'X') throw std::runtime_error("Main gate token");
}

int main() {
  try {
    gate("before-runtime");
    const sycl::device device{sycl::gpu_selector_v};
    if (device.get_backend() != sycl::backend::ext_oneapi_level_zero ||
        !device.is_gpu() || device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        device.get_info<sycl::info::device::name>() != "Intel(R) Arc(TM) B570 Graphics")
      throw std::runtime_error("qualified B570 Level Zero device required");
    gate("runtime-ready");
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    const auto fits = [](size_t fixed, size_t maximum, size_t needed) {
      return fixed == needed || (!fixed && maximum >= needed);
    };
    bool m4 = false, m8 = false, sg16 = false;
    const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "MATRIX_PROPERTIES {\"workgroup\":"
              << device.get_info<sycl::info::device::max_work_group_size>()
              << ",\"workitem1d\":"
              << device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]
              << ",\"local_bytes\":" << device.get_info<sycl::info::device::local_mem_size>()
              << ",\"subgroups\":[";
    for (size_t i = 0; i < subgroups.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << subgroups[i]; sg16 |= subgroups[i] == 16;
    }
    std::cout << "],\"matrix_tuples\":[";
    bool first = true;
    for (const auto &c : device.get_info<
             sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << "{\"types\":[" << int(c.atype) << ',' << int(c.btype) << ','
                << int(c.ctype) << ',' << int(c.dtype) << "],\"exact\":["
                << c.msize << ',' << c.nsize << ',' << c.ksize << "],\"maximum\":["
                << c.max_msize << ',' << c.max_nsize << ',' << c.max_ksize << "]}";
      if (c.atype == mx::matrix_type::bf16 && c.btype == mx::matrix_type::bf16 &&
          c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
          fits(c.nsize, c.max_nsize, 16) && fits(c.ksize, c.max_ksize, 16)) {
        m4 |= fits(c.msize, c.max_msize, 4);
        m8 |= fits(c.msize, c.max_msize, 8);
      }
    }
    std::cout << "],\"SG16\":" << sg16 << ",\"bf16_M4N16K16\":" << m4
              << ",\"bf16_M8N16K16\":" << m8 << "}" << std::endl;
    gate("before-exit");
    std::cout.flush(); std::cerr.flush();
    // The loader/preload reference remains held; ordinary teardown is unqualified.
    std::_Exit(0);
  } catch (const std::exception &error) {
    std::cerr << "MATRIX_ERROR " << error.what() << std::endl;
    return 1;
  }
}
