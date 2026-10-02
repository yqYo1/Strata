// Research probe, not an engine backend. See docs/SYCL_USM_ATOMICS.md.
#include <chrono>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
struct Mailbox {
  uint32_t gpu_payload[16], cpu_payload[16];
};
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto p = sycl::malloc_host<Mailbox>(1, q);
    auto errors = sycl::malloc_device<uint32_t>(1, q);
    if (!p || !errors)
      return 1;
    constexpr uint32_t rounds = 16;
    for (int repeat = 0; repeat < 3; ++repeat) {
      std::memset(p, 0, sizeof(*p));
      q.memset(errors, 0, 4).wait_and_throw();
      uint32_t host_errors = 0;
      auto start = std::chrono::steady_clock::now();
      for (uint32_t i = 1; i <= rounds + 1; ++i) {
        q.single_task([=] {
           if (i > 1)
             for (uint32_t j = 0; j < 16; ++j)
               *errors += p->cpu_payload[j] != ((i - 1) * 53 + j);
           if (i <= rounds)
             for (uint32_t j = 0; j < 16; ++j)
               p->gpu_payload[j] = i * 37 + j;
         }).wait_and_throw();
        if (i <= rounds) {
          for (uint32_t j = 0; j < 16; ++j)
            host_errors += p->gpu_payload[j] != (i * 37 + j);
          for (uint32_t j = 0; j < 16; ++j)
            p->cpu_payload[j] = i * 53 + j;
        }
      }
      uint32_t gpu_errors;
      q.memcpy(&gpu_errors, errors, 4).wait_and_throw();
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
      std::cout << "repeat=" << repeat << " rounds=" << rounds
                << " host_errors=" << host_errors
                << " gpu_errors=" << gpu_errors << " elapsed_ms=" << ms
                << std::endl;
      if (host_errors || gpu_errors)
        return 1;
    }
    sycl::free(p, q);
    sycl::free(errors, q);
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
