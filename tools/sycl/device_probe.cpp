// Standalone research probe; see docs/SYCL_RESEARCH.md for build instructions
// and limits.
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto d = q.get_device();
    std::cout << d.get_info<sycl::info::device::name>()
              << "\nVRAM=" << d.get_info<sycl::info::device::global_mem_size>()
              << " max_alloc="
              << d.get_info<sycl::info::device::max_mem_alloc_size>()
              << " local=" << d.get_info<sycl::info::device::local_mem_size>()
              << " max_wg="
              << d.get_info<sycl::info::device::max_work_group_size>()
              << "\nsubgroups=";
    for (auto n : d.get_info<sycl::info::device::sub_group_sizes>())
      std::cout << n << ",";
    std::cout << "\nfp16=" << d.has(sycl::aspect::fp16)
              << " fp64=" << d.has(sycl::aspect::fp64)
              << " host_usm=" << d.has(sycl::aspect::usm_host_allocations)
              << " shared_usm=" << d.has(sycl::aspect::usm_shared_allocations)
              << " atomic_host="
              << d.has(sycl::aspect::usm_atomic_host_allocations)
              << " atomic_shared="
              << d.has(sycl::aspect::usm_atomic_shared_allocations)
              << " graph=" << d.has(sycl::aspect::ext_oneapi_graph) << "\n";
    const size_t n = 64 * 1024 * 1024;
    auto h = sycl::malloc_host<unsigned char>(n, q);
    auto p = sycl::malloc_device<unsigned char>(n, q);
    auto p2 = sycl::malloc_device<unsigned char>(n, q);
    if (!h || !p || !p2)
      throw std::runtime_error("USM allocation failed");
    std::memset(h, 17, n);
    for (int kind = 0; kind < 3; ++kind) {
      auto dst = kind == 0 ? p : kind == 1 ? h : p2;
      auto src = kind == 0 ? h : p;
      q.memcpy(dst, src, n).wait_and_throw();
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 32; ++i)
        q.memcpy(dst, src, n);
      q.wait_and_throw();
      double t = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count();
      std::cout << (kind == 0   ? "H2D"
                    : kind == 1 ? "D2H"
                                : "D2D")
                << " GB/s=" << n * 32 / t / 1e9 << "\n";
    }
    auto val = sycl::malloc_shared<int>(1, q);
    if (!val)
      throw std::runtime_error("USM allocation failed");
    *val = 0;
    namespace ex = sycl::ext::oneapi::experimental;
    ex::command_graph graph{q.get_context(), d};
    graph.begin_recording(q);
    q.single_task([=] { *val += 1; });
    graph.end_recording(q);
    auto exec = graph.finalize();
    for (int i = 0; i < 100; ++i)
      q.ext_oneapi_graph(exec);
    q.wait_and_throw();
    std::cout << "graph replay=" << *val << " expected=100\n";
    if (*val != 100)
      throw std::runtime_error("graph replay mismatch");
    sycl::free(val, q);
    sycl::free(h, q);
    sycl::free(p, q);
    sycl::free(p2, q);
  } catch (std::exception const &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
