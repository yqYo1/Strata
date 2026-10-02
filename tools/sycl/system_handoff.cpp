// Research probe, not an engine backend. See docs/SYCL_USM_ATOMICS.md.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <thread>
struct alignas(4096) Mailbox {
  uint32_t seq, ack;
  uint32_t gpu_payload[16], cpu_payload[16];
};
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v};
    auto d = q.get_device();
    if (q.get_backend() != sycl::backend::ext_oneapi_level_zero ||
        !d.has(sycl::aspect::usm_system_allocations))
      return 77;
    ze_device_memory_access_properties_t caps{};
    caps.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
    if (zeDeviceGetMemoryAccessProperties(
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d), &caps) !=
            ZE_RESULT_SUCCESS ||
        !(caps.sharedSystemAllocCapabilities &
          ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC))
      return 77;
    auto p = static_cast<Mailbox *>(std::aligned_alloc(4096, sizeof(Mailbox)));
    if (!p)
      return 1;
    auto result = sycl::malloc_device<uint32_t>(2, q);
    if (!result)
      return 1;
    using A = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::system,
                               sycl::access::address_space::global_space>;
    constexpr uint32_t rounds = 16, limit = 1000000;
    for (int repeat = 0; repeat < 3; ++repeat) {
      std::memset(p, 0, sizeof(*p));
      q.memset(result, 0, 8).wait_and_throw();
      auto start = std::chrono::steady_clock::now();
      auto e = q.single_task([=] {
        A seq(p->seq), ack(p->ack);
        uint32_t done = 0, errors = 0;
        for (uint32_t i = 1; i <= rounds; ++i) {
          for (uint32_t j = 0; j < 16; ++j)
            p->gpu_payload[j] = i * 37 + j;
          seq.store(i, sycl::memory_order::release);
          uint32_t spins = 0;
          while (ack.load(sycl::memory_order::acquire) != i) {
            if (++spins == limit) {
              result[0] = done;
              result[1] = 0x80000000u | errors;
              return;
            }
          }
          for (uint32_t j = 0; j < 16; ++j)
            errors += p->cpu_payload[j] != (i * 53 + j);
          done = i;
        }
        result[0] = done;
        result[1] = errors;
      });
      uint32_t host_done = 0, host_errors = 0;
      for (uint32_t i = 1; i <= rounds; ++i) {
        while (std::atomic_ref<uint32_t>(p->seq).load(
                   std::memory_order_acquire) != i) {
          if (std::chrono::steady_clock::now() - start >
              std::chrono::seconds(5))
            break;
          std::this_thread::yield();
        }
        if (std::atomic_ref<uint32_t>(p->seq).load(std::memory_order_acquire) !=
            i)
          break;
        for (uint32_t j = 0; j < 16; ++j)
          host_errors += p->gpu_payload[j] != (i * 37 + j);
        for (uint32_t j = 0; j < 16; ++j)
          p->cpu_payload[j] = i * 53 + j;
        std::atomic_ref<uint32_t>(p->ack).store(i, std::memory_order_release);
        host_done = i;
      }
      e.wait_and_throw();
      uint32_t r[2];
      q.memcpy(r, result, 8).wait_and_throw();
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
      std::cout << "repeat=" << repeat << " host_done=" << host_done
                << " gpu_done=" << r[0] << " host_errors=" << host_errors
                << " gpu_errors=" << r[1] << " elapsed_ms=" << ms << std::endl;
      if (host_done != rounds || r[0] != rounds || host_errors || r[1])
        return 1;
    }
    sycl::free(result, q);
    std::free(p);
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
