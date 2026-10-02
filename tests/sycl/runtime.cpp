#include "strata/sycl/runtime.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

using namespace strata::sycl_backend;

int main(int argc, char **argv) {
  try {
    auto runtime = std::make_shared<Runtime>();
    std::cout << runtime->device().get_info<sycl::info::device::name>() << '\n';
    if (argc == 2 && std::string_view(argv[1]) == "--large") {
      constexpr size_t boundary = size_t{1} << 32;
      const auto free_bytes =
          runtime->device()
              .get_info<sycl::ext::intel::info::device::free_memory>();
      if (free_bytes < boundary + (size_t{1} << 30))
        throw std::runtime_error(
            "large-address test needs at least 5 GiB free GPU memory");
      Allocation large(runtime, boundary + 65536, MemoryKind::Device);
      auto *first = large.as<uint64_t>();
      auto *last =
          static_cast<uint64_t *>(large.at(boundary + 8, sizeof(uint64_t)));
      auto written = runtime->compute().single_task([=] {
        *first = 0x123456789abcdef0ull;
        *last = 0xfedcba9876543210ull;
      });
      uint64_t values[2]{};
      runtime->wait(runtime->copy(values, first, sizeof(uint64_t), {written}));
      runtime->wait(
          runtime->copy(values + 1, last, sizeof(uint64_t), {written}));
      if (values[0] != 0x123456789abcdef0ull ||
          values[1] != 0xfedcba9876543210ull)
        throw std::runtime_error("64-bit device addressing mismatch");
      std::cout << "distinct writes below and above 4 GiB: PASS\n";
      return 0;
    }
    if (argc != 1)
      throw std::invalid_argument("usage: sycl_runtime_test [--large]");
    constexpr size_t count = 4097;
    Allocation host(runtime, count * sizeof(int), MemoryKind::Host);
    Allocation gpu(runtime, host.size(), MemoryKind::Device);
    auto *h = host.as<int>();
    auto *d = gpu.as<int>();
    for (int iteration = 0; iteration < 64; ++iteration) {
      for (size_t i = 0; i < count; ++i)
        h[i] = static_cast<int>(i) - iteration * 7;
      auto uploaded = runtime->copy(d, h, host.size());
      auto computed = runtime->compute().submit([&](sycl::handler &handler) {
        handler.depends_on(uploaded);
        handler.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
          d[i] = 3 * d[i] + iteration;
        });
      });
      runtime->wait(runtime->copy(h, d, host.size(), {computed}));
      for (size_t i = 0; i < count; ++i)
        if (h[i] != 3 * (static_cast<int>(i) - iteration * 7) + iteration)
          throw std::runtime_error("cross-queue handoff mismatch");
    }
    bool rejected = false;
    try {
      gpu.at(std::numeric_limits<size_t>::max(), 8);
    } catch (const std::out_of_range &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("overflowing subrange accepted");
    rejected = false;
    try {
      Allocation impossible(runtime, std::numeric_limits<size_t>::max(),
                            MemoryKind::Device);
    } catch (const std::length_error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("impossible allocation accepted");
    Allocation moved(std::move(gpu));
    if (gpu.data() || moved.data() != d)
      throw std::runtime_error("allocation ownership move failed");
    Allocation empty(runtime, 0, MemoryKind::Device);
    runtime->wait(runtime->copy(nullptr, nullptr, 0));
    runtime->wait();
    auto completed = runtime->compute().single_task([] {});
    if (!runtime->wait_for(completed, std::chrono::seconds(5)))
      throw std::runtime_error("bounded completion wait timed out");
    auto failing = std::make_shared<Runtime>();
    failing->compute().submit([](sycl::handler &handler) {
      handler.host_task([] {
        throw std::runtime_error("intentional SYCL host-task failure");
      });
    });
    rejected = false;
    try {
      failing->wait();
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("asynchronous error not propagated");
    rejected = false;
    try {
      failing->check();
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("asynchronous error not retained");
    std::cout << "64 changing cross-queue handoffs, bounds, allocation limits "
                 "and ownership: PASS\n";
    std::cout
        << "bounded completion wait and sticky asynchronous errors: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
