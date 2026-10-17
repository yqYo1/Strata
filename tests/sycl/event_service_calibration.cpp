// Authored for Strata; repository LICENSE (MIT) applies.
// API/timestamp calibration only, not GPU throughput or model performance.
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace strata::sycl_backend;
std::shared_ptr<Runtime> runtime;
void need(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void drain() noexcept {
  checked_cleanup([] { if (runtime) runtime->wait(); }, [] {}, "event calibration drain");
}
void gate(bool enabled, const char* phase) {
  if (!enabled) return;
  std::cout << "EVENT_PROFILE_READY " << phase << '\n';
  need(std::cin.get() == 'X', "Main gate token");
}
uint32_t transform(uint32_t value) {
  for (int i = 0; i < 16; ++i) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
  }
  return value;
}
void run() {
  constexpr size_t N = 1 << 18, Guard = 16, Count = N + 2 * Guard;
  constexpr size_t Repeats = 6, EventCount = 1 + 4 * Repeats;
  constexpr uint32_t Sentinel = 0x1942ab57;
  auto& q = runtime->compute();
  need(q.has_property<sycl::property::queue::in_order>() &&
       q.has_property<sycl::property::queue::enable_profiling>(), "profiled in-order compute queue required");
  need(q.get_backend() == sycl::backend::ext_oneapi_level_zero &&
       q.get_device().is_gpu(), "native Level Zero GPU required");
  need(!runtime->transfer().has_property<sycl::property::queue::enable_profiling>(),
       "calibration must not silently substitute the unprofiled transfer queue");
  const auto resolution = q.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
  need(resolution > 0, "invalid profiling timer resolution");
  std::cout << "queue=in_order_profiling backend=level_zero timer_resolution_ns=" << resolution
            << " repeats=" << Repeats << " max_retained_events=" << EventCount << '\n';
  Allocation input(runtime, Count * 4, MemoryKind::Host);
  Allocation output(runtime, Repeats * Count * 4, MemoryKind::Host);
  Allocation device(runtime, Count * 4, MemoryKind::Device);
  auto* host = input.as<uint32_t>();
  auto* result = output.as<uint32_t>();
  auto* base = device.as<uint32_t>();
  auto* data = base + Guard;
  for (size_t i = 0; i < Count; ++i)
    host[i] = i < Guard || i >= N + Guard ? Sentinel : uint32_t(i * 1664525U + 1013904223U);
  std::array<std::optional<sycl::event>, EventCount> events;
  std::array<const char*, EventCount> names{};
  size_t used = 0;
  bool drained = false;
  struct Pending {
    bool& done;
    ~Pending() noexcept { if (!done) drain(); }
  } pending{drained}; // A failed submit cannot release any host/device owner first.
  auto keep = [&](const char* name, sycl::event event) {
    need(used < EventCount, "fixed event capacity");
    names[used] = name;
    events[used++].emplace(std::move(event));
  };
  keep("initialize", q.memcpy(base, host, Count * 4));
  for (size_t rep = 0; rep < Repeats; ++rep) {
    keep("H2D", q.memcpy(data, host + Guard, N * 4));
    keep("short_kernel", q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> item) {
      data[item.get_global_linear_id()] ^= 0x49e721abU ^ uint32_t(rep);
    }));
    keep("long_kernel", q.parallel_for(sycl::nd_range<1>(N, 256), [=](sycl::nd_item<1> item) {
      const size_t i = item.get_global_linear_id();
      uint32_t value = data[i];
      for (int j = 0; j < 16; ++j) {
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
      }
      data[i] = value;
    }));
    keep("D2H", q.memcpy(result + rep * Count, base, Count * 4));
  }
  need(used == EventCount, "complete planned event sequence");
  runtime->wait(); // Sole hot-sequence drain; timestamps are never queried before it.
  drained = true;
  uint64_t previous_end = 0, max_query_wall_ns = 0, minimum_kernel_ns = UINT64_MAX;
  for (size_t i = 0; i < used; ++i) {
    const auto before = std::chrono::steady_clock::now();
    const auto& event = *events[i];
    need(event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete,
         "event not complete after checked drain");
    const uint64_t submitted = event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    const uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    const uint64_t query_wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - before).count();
    need(submitted <= start && start < end, "profiling timestamp order or zero service interval");
    need(!i || previous_end <= start, "in-order dependent command timestamps overlap");
    need(query_wall < 1000000000ULL, "post-drain profiling query exceeded one-second diagnostic budget");
    previous_end = end;
    max_query_wall_ns = std::max(max_query_wall_ns, query_wall);
    if (std::string(names[i]).find("kernel") != std::string::npos)
      minimum_kernel_ns = std::min(minimum_kernel_ns, end - start);
    std::cout << "sample=" << i << " kind=" << names[i] << " submit=" << submitted
              << " start=" << start << " end=" << end << " service_ns=" << end - start
              << " queue_delay_ns=" << start - submitted << " query_wall_ns=" << query_wall << '\n';
    events[i].reset(); // Completed handles only; allocation owners remain live.
  }
  need(minimum_kernel_ns >= resolution, "kernel service below reported timer granularity");
  for (size_t rep = 0; rep < Repeats; ++rep) {
    for (size_t i = 0; i < Count; ++i) {
      uint32_t expected = Sentinel;
      if (i >= Guard && i < N + Guard) {
        expected = host[i];
        if (i - Guard < 256) expected ^= 0x49e721abU ^ uint32_t(rep);
        expected = transform(expected);
      }
      need(result[rep * Count + i] == expected, "full per-repeat result or allocation guard mismatch");
      const uint32_t original = i < Guard || i >= N + Guard ? Sentinel : uint32_t(i * 1664525U + 1013904223U);
      need(host[i] == original, "host input changed");
    }
  }
  std::cout << "case status=PASS guards=PASS inputs=PASS all_repeats=PASS timestamps=PASS"
            << " events=" << used << " min_kernel_service_ns=" << minimum_kernel_ns
            << " max_query_wall_ns=" << max_query_wall_ns << " hardware_busy=UNMEASURED model_PP=UNRUN\n";
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    need(argc == 2 && std::string(argv[1]) == "--calibrate", "--calibrate required");
    const char* value = std::getenv("STRATA_TEST_MAIN_GATES");
    const bool gates = value && std::string(value) == "1";
    gate(gates, "before-runtime");
    runtime = strata::sycl_backend::runtime_for();
    std::cout << "device=" << runtime->device().get_info<sycl::info::device::name>()
              << " driver=" << runtime->device().get_info<sycl::info::device::driver_version>() << '\n';
    gate(gates, "runtime-ready");
    run();
    runtime->wait();
    runtime.reset();
    gate(gates, "after-cleanup");
    std::cout << "fixture_status=PASS model_PP=UNRUN faults=EXTERNAL_REQUIRED\n";
    return 0;
  } catch (const std::exception& error) {
    drain();
    std::cerr << "event calibration failed: " << error.what() << '\n';
    return 1;
  }
}
