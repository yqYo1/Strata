// Main-authored comparison harness; repository LICENSE (MIT) applies.
// Reuse the maintained independent weighted-cell oracle, all boundary/NaN/tie
// cases and immutable-input/canary/drain checks; do not duplicate that fixture.
// One arm per process, selected before runtime initialization. No env writes.
#include "strata/kernels/qsa_radix_batched.hpp"
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <time.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#if defined(STRATA_QSA_EVENT_CAPTURE)
#include "qsa_event_capture.hpp"
#endif

namespace {
using Selector = decltype(&strata::kernels::qsa_block_topk_prefill);
Selector qsa_fixture_select = nullptr;
}
namespace strata::kernels {
void qsa_block_topk_selected_fixture(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  if (!qsa_fixture_select) throw std::logic_error("fixture arm not selected");
  qsa_fixture_select(scores, steps, nq, blocks, cap, shape, ids, stream, active);
}
}

// The public API header is already included above. Only the dedicated-prefill
// fixture call is redirected; public original/ref checks stay unmodified.
#define qsa_block_topk_prefill qsa_block_topk_selected_fixture
#define main qsa_original_radix_fixture_main
#include "qsa_select_radix.cpp"
#undef main
#undef qsa_block_topk_prefill

namespace {
int property_fixture(int argc, char **argv) {
  check(argc == 6 && argv[2][0] == '/',
        "usage: --properties ABSOLUTE_ZE_LOADER ORIGINAL_KERNEL CONTROL_KERNEL BATCHED_KERNEL");
  // Same already-loaded provider discipline as qsa_k_loop_admission.hpp. This
  // helper instantiates executable kernels and queries properties, not queues
  // of work. Its identities come from the committed build's actual device IR.
  struct Loader { void *handle; ~Loader() { if (handle) dlclose(handle); } };
  Loader loader{dlopen(argv[2], RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)};
  check(loader.handle != nullptr, "qualified ZE loader is not already loaded");
  const auto device_properties = reinterpret_cast<decltype(&zeDeviceGetComputeProperties)>(
      dlsym(loader.handle, "zeDeviceGetComputeProperties"));
  const auto kernel_properties = reinterpret_cast<decltype(&zeKernelGetProperties)>(
      dlsym(loader.handle, "zeKernelGetProperties"));
  check(device_properties && kernel_properties, "property API symbols missing");
  Dl_info origin{};
  check(dladdr(reinterpret_cast<void *>(kernel_properties), &origin) && origin.dli_fname,
        "property provider identity missing");
  std::cout << "qsa_property_loader=" << origin.dli_fname << '\n';
  const auto &device = runtime->device();
  check(device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == 0x8086,
        "property device is not Intel GPU");
  ze_device_compute_properties_t limits{};
  limits.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  check(device_properties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device), &limits) ==
            ZE_RESULT_SUCCESS && limits.maxTotalGroupSize >= 256 && limits.maxGroupSizeX >= 256,
        "WG256 unavailable");
  const auto ids = sycl::get_kernel_ids();
  for (int arm = 0; arm < 3; ++arm) {
    const std::string name = argv[3 + arm];
    for (int previous = 0; previous < arm; ++previous)
      check(name != argv[3 + previous], "duplicate property identity");
    check(std::count_if(ids.begin(), ids.end(), [&](const auto &id) {
      return name == id.get_name();
    }) == 1, "property kernel identity missing or duplicated");
    const auto id = *std::find_if(ids.begin(), ids.end(), [&](const auto &value) {
      return name == value.get_name();
    });
    const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        runtime->context(), {device}, {id});
    const auto kernel = bundle.get_kernel(id);
    ze_kernel_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    check(kernel_properties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel), &props) ==
              ZE_RESULT_SUCCESS, "native property query failed");
    const auto maximum = kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(device);
    const auto subgroup = kernel.get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(device);
    check(maximum >= 256 && (!props.requiredGroupSizeX || props.requiredGroupSizeX == 256) &&
          (!props.requiredGroupSizeY || props.requiredGroupSizeY == 1) &&
          (!props.requiredGroupSizeZ || props.requiredGroupSizeZ == 1), "native group contract");
    if (arm) check(subgroup == 16 && props.requiredSubgroupSize == 16, "native SG16 contract");
    // Local argument payload is not yet bound. Add the complete source-declared
    // candidate payload plus a conservative collective reserve, not occupancy.
    const uint64_t budget = uint64_t(props.localMemSize) + (arm == 2 ? 1088 : 0) + 4096;
    check(budget <= limits.maxSharedLocalMemory, "native local-memory budget");
    std::cout << "qsa_property arm=" << arm << " kernel=" << name << " maximum_wg=" << maximum
              << " subgroup=" << subgroup << " arguments=" << props.numKernelArgs
              << " required_subgroup=" << props.requiredSubgroupSize
              << " reported_local=" << props.localMemSize << " spill_descriptor=" << props.spillMemSize
              << " conservative_local_budget=" << budget
              << " device_local_max=" << limits.maxSharedLocalMemory << '\n';
  }
  std::cout << "PASS properties_no_kernel_enqueue\n";
  return 0;
}

struct Clock {
  double wall, cpu, process_cpu;
};
Clock clock_now() {
  timespec cpu{}, process{};
  check(::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu) == 0 &&
        ::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process) == 0, "CPU clocks");
  return {std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(),
          double(cpu.tv_sec) + double(cpu.tv_nsec) * 1e-9,
          double(process.tv_sec) + double(process.tv_nsec) * 1e-9};
}
int timing_fixture(int argc, char **argv) {
  check(argc == 6 && std::string(argv[4]) == "--dump",
        "usage: --timing REPEATS clean|events --dump PATH");
  char *end = nullptr;
  const long repeats = std::strtol(argv[2], &end, 10);
  check(end != argv[2] && !*end && repeats >= 1 && repeats <= 4096,
        "timing repetitions must be 1..4096");
  const std::string mode = argv[3];
  check(mode == "clean" || mode == "events", "timing mode clean|events");
  const bool profiling = mode == "events";
#if !defined(STRATA_QSA_EVENT_CAPTURE)
  check(!profiling, "event timing needs the separate captured-event fixture");
#else
  check(profiling, "use the unchanged production-source fixture for clean timing");
#endif
  const bool smoke = std::string(argv[1]) == "--event-smoke";
  check(!smoke || (repeats == 2 && profiling), "event smoke requires2repeats/events");
  double maximum_clock_pair = 0;
  for (int i = 0; i < 128; ++i) {
    const auto b = clock_now(), e = clock_now();
    maximum_clock_pair = std::max(maximum_clock_pair, e.wall - b.wall);
  }
  std::cout << std::setprecision(12) << "qsa_clock maximum_pair_ns="
            << maximum_clock_pair * 1e9 << '\n';
  dump.open(argv[5], std::ios::binary | std::ios::trunc);
  check(bool(dump), "cannot open timing dump");
  runtime = sycl_backend::runtime_for();
  auto &queue = runtime->compute();
  check(queue.is_in_order() && queue.has_property<sycl::property::queue::enable_profiling>() &&
        queue.get_backend() == sycl::backend::ext_oneapi_level_zero, "qualified timing queue");
  const auto resolution = runtime->device().get_info<sycl::info::device::profiling_timer_resolution>();
  std::cout << std::setprecision(12) << "qsa_timing_device="
            << runtime->device().get_info<sycl::info::device::name>() << '\n';
  // Synthetic selector-only work. Same score row replicated over256queries;
  // one independent weighted-cell oracle is sufficient for every row. Neither
  // scoring, transfers, sorting the oracle nor checking results is timed.
  const int64_t nq = smoke ? 3 : 256, cap = smoke ? 7 : 2051;
  const std::vector<int64_t> lengths = smoke ? std::vector<int64_t>{20} :
      std::vector<int64_t>{32768, 78453, 117489, 262144};
  const std::vector<int> patterns = smoke ? std::vector<int>{0} : std::vector<int>{0,1};
  for (int64_t n : lengths) {
    for (int pattern : patterns) {
      const int64_t blocks = n / 4 + 3, active = n / 4 + 1;
      std::vector<float> scores(size_t(nq * blocks) + 2 * guard,
                                std::numeric_limits<float>::infinity());
      std::vector<int32_t> steps(size_t(nq) * kStepCount);
      for (int64_t q = 0; q < nq; ++q) {
        for (int64_t b = 0; b <= n / 4; ++b)
          scores[size_t(guard + q * blocks + b)] = score(pattern, b, 0);
        if (!(n % 4)) scores[size_t(guard + q * blocks + n / 4)] =
            std::numeric_limits<float>::infinity();
        auto *st = steps.data() + q * kStepCount;
        st[kStepPos] = int32_t(n - 1); st[kStepNKv] = int32_t(n);
        st[kStepNBid] = int32_t(n / 4); st[kStepWidth] = int32_t(cap);
      }
      const auto expected = oracle(scores.data() + guard, int32_t(n), int32_t(cap));
      Buffer<float> ds(scores.size());
      Buffer<int32_t> dt(steps.size()), out(size_t(nq * cap) + 2 * guard);
      Drain drain; // Always drains before allocation owners on exceptions.
      ds.put(scores); dt.put(steps);
      out.put(std::vector<int32_t>(size_t(nq * cap) + 2 * guard, sentinel));
      const auto invoke = [&] {
        qsa_fixture_select(ds.data() + guard, dt.data(), nq, blocks, cap,
                           qsa_real_shapes(), out.data() + guard, &queue, active);
      };
      const auto validate_output = [&] {
        const auto values = out.get();
        for (int j = 0; j < guard; ++j)
          check(values[size_t(j)] == sentinel && values[values.size() - 1 - size_t(j)] == sentinel,
                "timing output canary");
        for (int64_t q = 0; q < nq; ++q)
          check(std::equal(expected.begin(), expected.end(), values.begin() + guard + q * cap),
                "timing weighted-cell oracle");
        return values;
      };
      const auto first = clock_now(); invoke(); runtime->wait();
      const auto first_end = clock_now();
      std::cout << "qsa_first n=" << n << " pattern=" << pattern
                << " wall_ms=" << 1e3 * (first_end.wall - first.wall)
                << " cpu_ms=" << 1e3 * (first_end.cpu - first.cpu) << '\n';
      write_dump(validate_output());
      for (int i = 0; i < 4; ++i) invoke();
      runtime->wait();
      for (int round = 0; round < (smoke ? 1 : 3); ++round) {
        std::vector<sycl::event> events;
        if (profiling) events.reserve(size_t(repeats));
        double invoke_wall = 0, invoke_cpu = 0, retention_wall = 0, retention_cpu = 0;
        const auto begin = clock_now();
        for (int i = 0; i < repeats; ++i) {
          if (profiling) {
#if defined(STRATA_QSA_EVENT_CAPTURE)
            strata::qsa_fixture::Capture capture;
#endif
            const auto b = clock_now(); invoke(); const auto e = clock_now();
            invoke_wall += e.wall - b.wall; invoke_cpu += e.cpu - b.cpu;
            // Use the actual parallel_for event retained by observed_finish.
            // The V1 last-event proxy failed identity qualification; it is not
            // a measurement input or a fallback in this generation.
            std::optional<sycl::event> event;
#if defined(STRATA_QSA_EVENT_CAPTURE)
            event.emplace(capture.take());
#endif
            check(event.has_value(), "missing actual selector event");
            check(events.empty() || *event != events.back(), "reused selector event");
            events.push_back(*event);
            const auto l = clock_now();
            retention_wall += l.wall - e.wall; retention_cpu += l.cpu - e.cpu;
          } else invoke();
        }
        const auto before_drain = clock_now(); runtime->wait(); const auto after = clock_now();
        uint64_t service = 0, first_start = 0, last_end = 0, previous_end = 0;
        for (const auto &event : events) {
          check(event.get_backend() == sycl::backend::ext_oneapi_level_zero &&
                event.get_info<sycl::info::event::command_execution_status>() ==
                    sycl::info::event_command_status::complete, "incomplete selector event");
          const auto s = event.get_profiling_info<sycl::info::event_profiling::command_start>();
          const auto e = event.get_profiling_info<sycl::info::event_profiling::command_end>();
          check(e > s && s >= previous_end, "invalid in-order selector timestamp");
          if (!first_start) first_start = s;
          last_end = previous_end = e; service += e - s;
          // Check after returned APIs; a blocked provider call is bounded by
          // the owned child's wall deadline. All GPU work is already drained.
          check(clock_now().wall - after.wall <= 1.0, "profiling query-work cap (1second)");
        }
        const auto query_end = clock_now();
        std::cout << "qsa_timing n=" << n << " pattern=" << pattern << " nq=" << nq
                  << " cap=" << cap << " repeats=" << repeats << " round=" << round
                  << " events=" << events.size() << " mode=" << mode
                  << " total_ms=" << 1e3 * (after.wall - begin.wall)
                  << " cpu_ms=" << 1e3 * (after.cpu - begin.cpu)
                  << " process_cpu_ms=" << 1e3 * (after.process_cpu - begin.process_cpu)
                  << " service_ns=" << service << " extent_ns=" << (last_end - first_start)
                  << " timer_resolution_ns=" << resolution
                  << " invoke_ms=" << 1e3 * invoke_wall << " invoke_cpu_ms=" << 1e3 * invoke_cpu
                  << " retention_ms=" << 1e3 * retention_wall << " retention_cpu_ms=" << 1e3 * retention_cpu
                  << " drain_ms=" << 1e3 * (after.wall - before_drain.wall)
                  << " drain_cpu_ms=" << 1e3 * (after.cpu - before_drain.cpu)
                  << " query_ms=" << 1e3 * (query_end.wall - after.wall)
                  << " query_cpu_ms=" << 1e3 * (query_end.cpu - after.cpu)
                  << " query_process_cpu_ms=" << 1e3 * (query_end.process_cpu - after.process_cpu) << '\n';
        validate_output(); // Also checks all rows after every timed batch.
      }
      const auto unchanged = ds.get();
      check(std::memcmp(unchanged.data(), scores.data(), scores.size() * sizeof(float)) == 0 &&
            dt.get() == steps, "timing inputs changed");
    }
  }
  runtime->wait(); dump.flush(); check(bool(dump), "timing dump flush");
  dump.close(); check(!dump.fail(), "timing dump close");
  std::cout << (smoke ? "PASS event_smoke_cases=1\n" : "PASS timing_cases=8\n");
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 3 || std::string(argv[1]) != "--arm") {
    std::cerr << "usage: fixture --arm original|control|batched [original fixture options]\n";
    return 2;
  }
  const std::string arm = argv[2];
  if (arm == "original") qsa_fixture_select = strata::kernels::qsa_block_topk_prefill;
  else if (arm == "control") qsa_fixture_select = strata::kernels::qsa_block_topk_sg16_control;
  else if (arm == "batched") qsa_fixture_select = strata::kernels::qsa_block_topk_batched_prefill;
  else { std::cerr << "unknown fixture arm\n"; return 2; }
  const char *radix = std::getenv("STRATA_SYCL_SELECT_RADIX4");
  const char *cyclic = std::getenv("STRATA_SYCL_SELECT_CYCLIC");
  if (!radix || std::string(radix) != "1" ||
      (cyclic && std::string(cyclic) != "0")) {
    std::cerr << "fixture requires pre-exec radix4=1 and cyclic unset/0\n";
    return 2;
  }
  std::cout << "qsa_histogram_fixture_arm=" << arm << '\n';
  try {
    const char *hold = std::getenv("STRATA_QSA_FIXTURE_HOLD");
    const bool admitted = hold && std::string(hold) == "1";
    runtime = sycl_backend::runtime_for();
    if (admitted) {
      std::cout << "QSA_READY name=" << runtime->device().get_info<sycl::info::device::name>()
                << std::endl;
      std::string line;
      check(bool(std::getline(std::cin, line)) && line == "GO", "initial ownership admission");
    }
    int result;
    if (argc > 3 && (std::string(argv[3]) == "--timing" || std::string(argv[3]) == "--event-smoke"))
      result = timing_fixture(argc - 2, argv + 2);
    else if (argc > 3 && std::string(argv[3]) == "--properties")
      result = property_fixture(argc - 2, argv + 2);
    else
      // argv[2] becomes the inherited fixture's unused program name.
      result = qsa_original_radix_fixture_main(argc - 2, argv + 2);
    if (!result && admitted) {
      runtime->wait();
      std::cout << "QSA_DONE" << std::endl;
      std::string line;
      check(bool(std::getline(std::cin, line)) && line == "DONE", "final ownership admission");
    }
    return result;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n'; return 1;
  }
}
