// Main-authored; repository LICENSE applies. Independent transfer calibration.
// No third-party application code. Does not change any inference path.
// Effective payload rates only: not physical PCIe/GDDR traffic or model speed.
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <time.h>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
constexpr size_t Guard = 256, Ring = 4;
constexpr uint8_t Sentinel = 0xa5;
std::shared_ptr<sb::Runtime> runtime;
using Clock = std::chrono::steady_clock;
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
double cpu() {
  timespec value{};
  need(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0, "thread CPU clock");
  return double(value.tv_sec) + double(value.tv_nsec) / 1e9;
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void drain() noexcept { try { runtime->wait(); } catch (...) { std::_Exit(86); } }
void wait(sycl::event event) {
  try { event.wait_and_throw(); runtime->check(); } catch (...) { drain(); throw; }
}
template<class Submit> void checked(Submit submit) {
  try { wait(submit()); } catch (...) { drain(); throw; }
}
struct Mapping {
  uint8_t* base;
  size_t bytes;
  explicit Mapping(size_t count) : base(nullptr), bytes((count + 4095) / 4096 * 4096) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    need(p != MAP_FAILED, "ordinary mapping allocation"); base = static_cast<uint8_t*>(p);
  }
  ~Mapping() { if (base) munmap(base, bytes); }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  void readonly() { need(mprotect(base, bytes, PROT_READ) == 0, "read-only source mapping"); }
};

struct Case {
  size_t blob, segments, stride, span, allocation;
  Mapping mapped;
  sb::Allocation pinned, staging, source, destination;
  sycl::queue& queue;
  Case(size_t bytes, size_t count) : blob(bytes), segments(count), stride((bytes + 4095) / 4096 * 4096),
      span(stride * count), allocation(span + 2 * Guard), mapped(allocation),
      pinned(runtime, allocation, sb::MemoryKind::Host),
      staging(runtime, Ring * (stride + 4096), sb::MemoryKind::Host),
      source(runtime, allocation, sb::MemoryKind::Device),
      destination(runtime, allocation, sb::MemoryKind::Device), queue(runtime->compute()) {
    need(blob > 0 && blob <= 2662400 && segments >= 4 && segments <= 64 && stride >= blob,
         "fixed transfer fixture extent");
    try {
      for (size_t i = 0; i < allocation; ++i) pinned.as<uint8_t>()[i] = expected(i);
      std::memcpy(mapped.base, pinned.data(), allocation); mapped.readonly();
      checked([&] { return queue.memcpy(source.data(), pinned.data(), allocation); });
    } catch (...) { drain(); throw; }
  }
  ~Case() { drain(); }
  uint8_t expected(size_t index) const {
    if (index < Guard || index >= Guard + span || (index - Guard) % stride >= blob) return Sentinel;
    return uint8_t((index * size_t(131)) ^ (index >> 8) ^ size_t(0x5a));
  }
  void validate(bool ring, size_t copies) {
    std::vector<uint8_t> readback(allocation);
    checked([&] { return queue.memcpy(readback.data(), destination.data(), allocation); });
    for (size_t i = 0; i < allocation; ++i) {
      const auto value = expected(i);
      need(readback[i] == value, "device destination/guard mismatch");
      need(mapped.base[i] == value && pinned.as<uint8_t>()[i] == value, "source/guard changed");
    }
    checked([&] { return queue.memcpy(readback.data(), source.data(), allocation); });
    for (size_t i = 0; i < allocation; ++i) need(readback[i] == expected(i), "device source changed");
    for (size_t slot = 0; slot < Ring; ++slot) {
      const size_t last = copies - 1 - (copies - 1 - slot) % Ring;
      for (size_t i = 0; i < stride + 4096; ++i) {
        const uint8_t value = ring && i >= Guard && i < Guard + blob
            ? expected(Guard + (last % segments) * stride + i - Guard) : Sentinel;
        need(staging.as<uint8_t>()[slot * (stride + 4096) + i] == value, "staging payload/guard changed");
      }
    }
  }
  void batch(const std::string& arm, int process, int round, bool smoke) {
    const bool ring = arm == "mapped_via_host_usm", device = arm == "device_to_device";
    need(arm == "ordinary_mapping" || arm == "host_usm" || ring || device, "unknown transfer arm");
    const size_t copies = smoke ? segments : device ? 1024 : 128;
    need(copies >= segments && copies % segments == 0 && copies <= 1024, "copy batch extent");
    checked([&] { return queue.memset(destination.data(), Sentinel, allocation); });
    std::memset(staging.data(), Sentinel, staging.size());
    std::array<std::optional<sycl::event>, Ring> pending;
    std::vector<sycl::event> events; events.reserve(copies);
    double stage_wall = 0, stage_cpu = 0, reuse_wait_wall = 0, reuse_wait_cpu = 0;
    double enqueue_wall = 0, enqueue_cpu = 0;
    const auto begin = Clock::now(); const double begin_cpu = cpu();
    try {
      for (size_t i = 0; i < copies; ++i) {
        const size_t slot = i % Ring, offset = Guard + (i % segments) * stride;
        const void* input = device ? source.as<uint8_t>() + offset :
            arm == "host_usm" ? pinned.as<uint8_t>() + offset : mapped.base + offset;
        if (ring) {
          if (pending[slot]) {
            const auto t = Clock::now(); const double c = cpu();
            wait(*pending[slot]); reuse_wait_cpu += cpu() - c; reuse_wait_wall += seconds(t);
          }
          const auto t = Clock::now(); const double c = cpu();
          std::memcpy(staging.as<uint8_t>() + slot * (stride + 4096) + Guard, input, blob);
          stage_cpu += cpu() - c; stage_wall += seconds(t);
          input = staging.as<uint8_t>() + slot * (stride + 4096) + Guard;
        }
        const auto t = Clock::now(); const double c = cpu();
        auto event = queue.memcpy(destination.as<uint8_t>() + offset, input, blob);
        enqueue_cpu += cpu() - c; enqueue_wall += seconds(t);
        events.push_back(event); if (ring) pending[slot] = event;
      }
      const auto final_begin = Clock::now(); const double final_cpu_begin = cpu();
      wait(events.back());
      const double final_cpu = cpu() - final_cpu_begin, final_wall = seconds(final_begin);
      const double wall = seconds(begin), scheduled = cpu() - begin_cpu;
      uint64_t service = 0, first = 0, last = 0;
      std::vector<std::pair<uint64_t, size_t>> durations;
      for (size_t i = 0; i < events.size(); ++i) {
        const auto a = events[i].get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto b = events[i].get_profiling_info<sycl::info::event_profiling::command_end>();
        need(b > a && (!i || a >= last), "invalid/overlapping in-order GPU timestamps");
        if (!i) first = a;
        last = b; service += b - a;
        durations.emplace_back(b - a, i);
      }
      std::sort(durations.begin(), durations.end());
      const size_t n = durations.size();
      const double median = n % 2 ? double(durations[n / 2].first) :
          (double(durations[n / 2 - 1].first) + double(durations[n / 2].first)) / 2;
      const size_t slow = size_t(std::count_if(durations.begin(), durations.end(),
          [&](const auto& value) { return double(value.first) > 4 * median; }));
      const double gpu_span_ms = double(last - first) / 1e6;
      // A failed floor remains ineligible; do not silently change repetitions.
      const bool eligible = smoke || (wall >= .01 && gpu_span_ms >= 10);
      validate(ring, copies);
      const size_t payload = copies * blob;
      std::cout << std::setprecision(17) << "{\"kind\":\"sample\",\"arm\":\"" << arm
          << "\",\"process\":" << process << ",\"round\":" << round << ",\"blob_bytes\":" << blob
          << ",\"segments\":" << segments << ",\"source_span_bytes\":" << span
          << ",\"copies\":" << copies << ",\"payload_bytes\":" << payload
          << ",\"wall_ms\":" << wall * 1e3 << ",\"thread_CPU_ms\":" << scheduled * 1e3
          << ",\"stage_wall_ms\":" << stage_wall * 1e3 << ",\"stage_CPU_ms\":" << stage_cpu * 1e3
          << ",\"reuse_wait_wall_ms\":" << reuse_wait_wall * 1e3 << ",\"reuse_wait_CPU_ms\":" << reuse_wait_cpu * 1e3
          << ",\"enqueue_wall_ms\":" << enqueue_wall * 1e3 << ",\"enqueue_CPU_ms\":" << enqueue_cpu * 1e3
          << ",\"final_wait_wall_ms\":" << final_wall * 1e3 << ",\"final_wait_CPU_ms\":" << final_cpu * 1e3
          << ",\"GPU_service_ns\":" << service << ",\"GPU_span_ms\":" << gpu_span_ms
          << ",\"copy_min_ns\":" << durations.front().first << ",\"copy_median_ns\":" << median
          << ",\"copy_p95_ns\":" << durations[(n * 95 + 99) / 100 - 1].first
          << ",\"copy_p99_ns\":" << durations[(n * 99 + 99) / 100 - 1].first
          << ",\"copy_max_ns\":" << durations.back().first << ",\"copy_gt_4x_median\":" << slow
          << ",\"slowest_copy_indices\":[" << durations[n - 1].second << ',' << durations[n - 2].second << ']'
          << ",\"payload_GB_s_wall\":" << double(payload) / wall / 1e9
          << ",\"payload_GB_s_GPU_service\":" << double(payload) / double(service)
          << ",\"eligible\":" << (eligible ? "true" : "false")
          << ",\"bytewise_outputs_inputs_guards\":true,\"physical_traffic_measured\":false}\n";
    } catch (...) { drain(); throw; }
  }
};

// A finite independent work kernel tests whether an ordinary-host 4-byte read
// waits inside enqueue. Completion then explicit wait are timed together too:
// moving a wait between APIs alone is not a performance gain.
uint32_t mix(uint32_t value, size_t iterations) {
  for (size_t step = 0; step < iterations; ++step)
    value = (value ^ (value >> 13)) * uint32_t(1664525) + uint32_t(1013904223);
  return value;
}
void status_batch(int process, int round, int mode, bool smoke) {
  const bool queued = mode >= 2, pinned = mode % 2;
  const size_t words = smoke ? 4096 : 262144, iterations = smoke ? 4 : 256;
  sb::Allocation work(runtime, (words + 2) * sizeof(uint32_t), sb::MemoryKind::Device);
  sb::Allocation status(runtime, 3 * sizeof(uint32_t), sb::MemoryKind::Device);
  sb::Allocation host(runtime, 3 * sizeof(uint32_t), sb::MemoryKind::Host);
  auto& q = runtime->compute();
  std::array<uint32_t, 3> ordinary{0xdeadbeefU, 0U, 0xfeedfaceU};
  std::array<uint32_t, 3> seed{0xdeadbeefU, 0x1234abcdU, 0xfeedfaceU};
  checked([&] { return q.memcpy(status.data(), seed.data(), sizeof(seed)); });
  checked([&] { return q.memset(work.data(), Sentinel, work.size()); });
  std::memcpy(host.data(), ordinary.data(), sizeof(ordinary));
  auto* output = pinned ? host.as<uint32_t>() : ordinary.data();
  std::optional<sycl::event> producer;
  auto* data = work.as<uint32_t>();
  const auto total_start = Clock::now(); const double total_cpu = cpu();
  try {
    const auto submit_start = Clock::now(); const double submit_cpu = cpu();
    if (queued) producer = q.parallel_for(sycl::range<1>(words), [=](sycl::id<1> id) {
      const size_t i = id[0];
      uint32_t value = uint32_t(i) + 1;
      for (size_t step = 0; step < iterations; ++step)
        value = (value ^ (value >> 13)) * uint32_t(1664525) + uint32_t(1013904223);
      data[i + 1] = value;
    });
    const double producer_submit_cpu = cpu() - submit_cpu, producer_submit_wall = seconds(submit_start);
    const auto copy_start = Clock::now(); const double copy_cpu = cpu();
    auto copied = q.memcpy(output + 1, status.as<uint32_t>() + 1, sizeof(uint32_t));
    const double enqueue_cpu = cpu() - copy_cpu, enqueue_wall = seconds(copy_start);
    const auto wait_start = Clock::now(); const double wait_cpu = cpu();
    wait(copied);
    const double drain_cpu = cpu() - wait_cpu, drain_wall = seconds(wait_start);
    const double total_wall = seconds(total_start), scheduled = cpu() - total_cpu;
    need(output[0] == seed[0] && output[1] == seed[1] && output[2] == seed[2], "status value/guards");
    std::vector<uint32_t> values(words + 2);
    checked([&] { return q.memcpy(values.data(), work.data(), work.size()); });
    need(values.front() == 0xa5a5a5a5U && values.back() == 0xa5a5a5a5U, "producer guards");
    for (size_t i = 0; i < words; ++i)
      need(values[i + 1] == (queued ? mix(uint32_t(i) + 1, iterations) : 0xa5a5a5a5U), "producer output");
    std::array<uint32_t, 3> source_check{};
    checked([&] { return q.memcpy(source_check.data(), status.data(), sizeof(source_check)); });
    need(source_check == seed, "status source changed");
    uint64_t work_ns = 0;
    if (producer) {
      const auto begin = producer->get_profiling_info<sycl::info::event_profiling::command_start>();
      const auto end = producer->get_profiling_info<sycl::info::event_profiling::command_end>();
      need(end > begin, "producer profiling"); work_ns = end - begin;
    }
    // Host-pointer readback may complete without a profiled GPU-copy packet.
    // Its API/wall clocks remain useful, but do not invent a device duration.
    std::optional<uint64_t> copy_ns;
    try {
      const auto a = copied.get_profiling_info<sycl::info::event_profiling::command_start>();
      const auto b = copied.get_profiling_info<sycl::info::event_profiling::command_end>();
      if (b > a) copy_ns = b - a;
    } catch (const sycl::exception& e) {
      if (e.code() != sycl::errc::profiling) throw;
    }
    std::cout << std::setprecision(17) << "{\"kind\":\"status\",\"process\":" << process
      << ",\"round\":" << round << ",\"mode\":" << mode
      << ",\"host_usm\":" << (pinned ? "true" : "false")
      << ",\"queued_work\":" << (queued ? "true" : "false")
      << ",\"work_words\":" << words << ",\"iterations\":" << iterations
      << ",\"wall_ms\":" << total_wall * 1e3 << ",\"thread_CPU_ms\":" << scheduled * 1e3
      << ",\"work_submit_wall_ms\":" << producer_submit_wall * 1e3
      << ",\"work_submit_CPU_ms\":" << producer_submit_cpu * 1e3
      << ",\"enqueue_wall_ms\":" << enqueue_wall * 1e3 << ",\"enqueue_CPU_ms\":" << enqueue_cpu * 1e3
      << ",\"final_wait_wall_ms\":" << drain_wall * 1e3 << ",\"final_wait_CPU_ms\":" << drain_cpu * 1e3
      << ",\"GPU_work_ns\":" << work_ns << ",\"GPU_copy_ns\":";
    if (copy_ns) std::cout << *copy_ns; else std::cout << "null";
    std::cout << ",\"bytewise_outputs_inputs_guards\":true,\"physical_traffic_measured\":false"
      << ",\"eligible\":" << ((!queued || work_ns >= 1000000) ? "true" : "false") << "}\n";
  } catch (...) { drain(); throw; }
}
} // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  try {
    need(argc >= 2, "use --smoke or --bench PROCESS");
    const bool smoke = std::string(argv[1]) == "--smoke";
    need((smoke && argc == 2) || (!smoke && argc == 3 && std::string(argv[1]) == "--bench"), "fixture mode");
    const int process = smoke ? 0 : std::stoi(argv[2]); need(process >= 0 && process < 3, "process ordinal");
    runtime = sb::runtime_for(); auto& q = runtime->compute();
    need(q.get_backend() == sycl::backend::ext_oneapi_level_zero && q.get_device().is_gpu() &&
        q.has_property<sycl::property::queue::in_order>() &&
        q.has_property<sycl::property::queue::enable_profiling>(), "profiled in-order LevelZero GPU required");
    constexpr const char* arms[] = {"ordinary_mapping", "host_usm", "mapped_via_host_usm", "device_to_device"};
    for (size_t blob : {size_t(1971200), size_t(2662400)}) {
      Case c(blob, smoke ? 4 : 64);
      for (int round = -1; round < (smoke ? 0 : 5); ++round)
        for (int order = 0; order < 4; ++order)
          c.batch(arms[(process + round + 5 + order) % 4], process, round, smoke);
    }
    for (int round = -1; round < (smoke ? 0 : 5); ++round)
      for (int order = 0; order < 4; ++order)
        status_batch(process, round, (process + round + 5 + order) % 4, smoke);
    drain(); std::cout << "fixture_status=PASS no_model_or_physical_traffic_qualification=1\n";
    return 0;
  } catch (const std::exception& error) {
    if (runtime) drain();
    std::cerr << "transfer fixture: " << error.what() << '\n'; return 1;
  }
}
