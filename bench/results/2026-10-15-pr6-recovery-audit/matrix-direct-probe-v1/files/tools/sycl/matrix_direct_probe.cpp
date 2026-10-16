// Synthetic component fixture. Uses the unchanged product matrix entrypoint;
// declined calls are never filled by a fallback. No model or source capture.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata;
using namespace strata::kernels;
using sycl_backend::Allocation;
using sycl_backend::MemoryKind;
using sycl_backend::Runtime;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 16;
constexpr int Heads = 24, KvHeads = 2, Dim = 256, PageSize = 4;
constexpr int LogicalCells = 262144, LogicalPages = LogicalCells / PageSize;
constexpr size_t LiveBudget = size_t(512) << 20;
constexpr double OracleTolerance = 1e-5;
constexpr float OutputSentinel = 12345.25f;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::invalid_argument(message);
}
size_t multiply(uint64_t a, uint64_t b) {
  const uint64_t maximum = std::min<uint64_t>(SIZE_MAX, std::numeric_limits<std::ptrdiff_t>::max());
  require(!b || a <= maximum / b, "geometry byte/element multiplication overflow");
  return size_t(a * b);
}
size_t add(size_t a, size_t b) {
  require(b <= SIZE_MAX - a, "geometry byte addition overflow");
  return a + b;
}
struct Geometry {
  int64_t nq, cap, physical_pages;
  size_t query_elements, selection_elements, step_elements, pool_rows, live_bound;
  Geometry(int64_t queries, int64_t capacity, int64_t physical)
      : nq(queries), cap(capacity), physical_pages(physical) {
    require(nq > 0 && nq <= 65535 && cap > 0 && cap <= INT32_MAX &&
            physical_pages > 0 && physical_pages <= LogicalPages, "unsupported fixture geometry");
    query_elements = multiply(multiply(nq, Heads), Dim);
    selection_elements = multiply(nq, cap);
    step_elements = multiply(nq, kStepCount);
    pool_rows = multiply(physical_pages, PageSize * KvHeads);
    // Conservative combined host/device/reference bound. Includes guarded
    // arrays and all format representations even though only one is populated.
    live_bound = add(multiply(query_elements, 32), multiply(selection_elements, 16));
    live_bound = add(live_bound, multiply(step_elements, 16));
    live_bound = add(live_bound, multiply(LogicalPages, 16));
    live_bound = add(live_bound, multiply(multiply(pool_rows, Dim), 48));
    live_bound = add(live_bound, size_t(1) << 20);
    require(live_bound <= LiveBudget, "fixture exceeds conservative 512 MiB live-byte budget");
  }
};
enum class Mode { Validate, Benchmark, Host, Help };
struct Options {
  Mode mode = Mode::Validate;
  int samples = 40, repeats = 4, queries = 0;
};
int decimal(const std::string& text) {
  unsigned value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  require(!text.empty() && result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
          value <= INT32_MAX, "invalid decimal CLI value");
  return int(value);
}
Options parse(const std::vector<std::string>& args) {
  Options result;
  if (args.empty()) return result;
  if (args == std::vector<std::string>{"--host-only"}) { result.mode = Mode::Host; return result; }
  if (args == std::vector<std::string>{"--help"}) { result.mode = Mode::Help; return result; }
  if (args == std::vector<std::string>{"--validate"}) return result;
  require(args[0] == "--benchmark", "usage: sycl_matrix_direct_probe [--help|--host-only|--validate|--benchmark [--samples N] [--repeats N] [--queries 32|256]]");
  result.mode = Mode::Benchmark;
  bool samples = false, repeats = false, queries = false;
  for (size_t i = 1; i < args.size(); i += 2) {
    require(i + 1 < args.size(), "CLI option needs a value");
    const int value = decimal(args[i + 1]);
    if (args[i] == "--samples" && !samples) { result.samples = value; samples = true; }
    else if (args[i] == "--repeats" && !repeats) { result.repeats = value; repeats = true; }
    else if (args[i] == "--queries" && !queries) { result.queries = value; queries = true; }
    else throw std::invalid_argument("unknown/duplicate benchmark option");
  }
  require(result.samples >= 40 && result.samples <= 400 && result.samples % 4 == 0,
          "samples must be 40..400 and divisible by 4 (at least 20/arm)");
  require(result.repeats >= 1 && result.repeats <= 32, "repeats must be 1..32");
  require(result.queries == 0 || result.queries == 32 || result.queries == 256,
          "benchmark queries must be 32 or 256");
  return result;
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
double decode(uint16_t x) { return double(std::bit_cast<_Float16>(x)); }
template<class T> uint64_t hash(const std::vector<T>& values) {
  uint64_t result = 14695981039346656037ull;
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  for (size_t i = 0; i < values.size() * sizeof(T); ++i) { result ^= bytes[i]; result *= 1099511628211ull; }
  return result;
}

int64_t capacity(int width) {
  require(width >= 0 && width <= 2051, "fixture width must be 0..2051");
  return std::max(1, width);
}
struct Input {
  Geometry g;
  int format, width;
  bool dense, nonresident, zero_scales;
  std::vector<float> query;
  std::vector<int32_t> table, ids, steps;
  std::vector<uint16_t> k16, v16, ks, vs;
  std::vector<int8_t> k8, v8;
  std::vector<uint8_t> k4, v4;
  Input(int mode, int64_t nq, int count, bool hot = false, bool missing = false, bool zero = false)
      : g(nq, capacity(count), hot ? (capacity(count) + 3) / 4 + 1 : 5),
        format(mode), width(count), dense(hot), nonresident(missing), zero_scales(zero),
        query(g.query_elements), table(LogicalPages, -1), ids(g.selection_elements, -1), steps(g.step_elements, 0) {
    require(format >= 0 && format <= 3 && width >= 0 && width <= g.cap,
            "invalid input format/width");
    if (!nonresident) {
      if (dense) {
        const int pages = int(g.physical_pages) - 1;
        for (int i = 0; i < pages; ++i) table[size_t(i)] = (i + 7) % pages;
        table.back() = pages;
      } else {
        table[0] = 3; table[1] = 1; table[17] = 4;
        table[65534] = 0; table[65535] = 2;
      }
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(std::sin(double(i) * .019));
    for (int64_t b = 0; b < g.nq; ++b) {
      steps[size_t(b) * kStepCount + kStepWidth] = width;
      steps[size_t(b) * kStepCount + kStepPos] = LogicalCells - 1;
      steps[size_t(b) * kStepCount + kStepNKv] = LogicalCells;
      steps[size_t(b) * kStepCount + kStepNBid] = LogicalPages;
      for (int i = 0; i < width; ++i) {
        int cell = i == width - 1 ? LogicalCells - 1 : dense ? i : i * 3 + int(b % 2);
        if (!dense && i % 11 == 9 && i != width - 1) cell = -7;
        ids[size_t(b * g.cap + i)] = cell;
      }
    }
    const size_t elements = multiply(g.pool_rows, Dim), scales = multiply(g.pool_rows, 4);
    if (format == 0) { k16.resize(elements); v16.resize(elements); }
    if (format == 1 || format == 3) { k8.resize(elements); ks.resize(scales); }
    if (format == 1) { v8.resize(elements); vs.resize(scales); }
    if (format == 2) k4.resize(multiply(g.pool_rows, 144));
    if (format == 2 || format == 3) v4.resize(multiply(g.pool_rows, 144));
    for (size_t i = 0; i < elements; ++i) {
      if (!k16.empty()) { k16[i] = half(zero_scales ? 0.f : float(std::sin(double(i) * .073))); v16[i] = half(zero_scales ? 0.f : float(std::cos(double(i) * .017))); }
      if (!k8.empty()) k8[i] = int8_t(int(i * 7 % 255) - 127);
      if (!v8.empty()) v8[i] = int8_t(int(i * 11 % 255) - 127);
    }
    for (size_t i = 0; i < scales; ++i) {
      if (!ks.empty()) ks[i] = half(zero_scales || i % 17 == 5 ? 0.f : .007f + float(i % 7) * .0001f);
      if (!vs.empty()) vs[i] = half(zero_scales || i % 19 == 7 ? 0.f : .008f + float(i % 11) * .0002f);
    }
    for (size_t i = 0; i < g.pool_rows * 8; ++i) {
      if (!k4.empty()) {
        const uint16_t scale = half(zero_scales ? 0.f : .11f + float(i % 7) * .002f);
        std::memcpy(k4.data() + i * 18, &scale, 2);
        for (int j = 0; j < 16; ++j) k4[i * 18 + 2 + size_t(j)] = uint8_t((i * 7 + size_t(j) * 13) % 256);
      }
      if (!v4.empty()) {
        const uint16_t scale = half(zero_scales ? 0.f : -.09f - float(i % 9) * .003f);
        std::memcpy(v4.data() + i * 18, &scale, 2);
        for (int j = 0; j < 16; ++j) v4[i * 18 + 2 + size_t(j)] = uint8_t((i * 11 + size_t(j) * 19) % 256);
      }
    }
    // Full host address validation precedes runtime creation for a caller
    // constructing inputs before obtaining its runtime.
    for (const int cell : ids) {
      require(cell < LogicalCells, "selected logical cell exceeds page table");
      if (cell < 0) continue;
      const int page = table[size_t(cell / PageSize)];
      require(page < g.physical_pages, "physical page exceeds small pool");
    }
  }
  double load(bool value, size_t row, int dimension) const {
    if (format == 0) return decode((value ? v16 : k16)[row * Dim + size_t(dimension)]);
    if (format == 1 || (format == 3 && !value))
      return double((value ? v8 : k8)[row * Dim + size_t(dimension)]) *
             decode((value ? vs : ks)[row * 4 + size_t(dimension / 64)]);
    const auto& data = value ? v4 : k4;
    const size_t block = row * 8 + size_t(dimension / 32);
    uint16_t scale;
    std::memcpy(&scale, data.data() + block * 18, 2);
    const uint8_t bits = data[block * 18 + 2 + size_t(dimension % 16)];
    return double(int(dimension % 32 < 16 ? bits & 15 : bits >> 4) - 8) * decode(scale);
  }
  int64_t row(int64_t batch, int head, int column) const {
    const int cell = ids[size_t(batch * g.cap + column)];
    if (cell < 0) return -1;
    const int page = table[size_t(cell / PageSize)];
    return page < 0 ? -1 : (int64_t(page) * KvHeads + head / 12) * PageSize + cell % PageSize;
  }
};

std::vector<double> oracle(const Input& input) {
  std::vector<double> result(input.g.query_elements, 0.);
  for (int64_t b = 0; b < input.g.nq; ++b)
    for (int h = 0; h < Heads; ++h) {
      const int raw_width = input.steps[size_t(b) * kStepCount + kStepWidth];
      const int count = raw_width >= 0 && raw_width <= input.g.cap ? raw_width : 0;
      std::vector<double> weights(size_t(count), -INFINITY);
      double maximum = -INFINITY, denominator = 0.;
      for (int j = 0; j < count; ++j) {
        const int64_t row = input.row(b, h, j);
        if (row < 0) continue;
        double dot = 0.;
        for (int d = 0; d < Dim; ++d)
          dot += double(input.query[size_t((b * Heads + h) * Dim + d)]) * input.load(false, size_t(row), d);
        weights[size_t(j)] = dot / 16.;
        maximum = std::max(maximum, weights[size_t(j)]);
      }
      for (double& value : weights) { value = std::isfinite(value) ? std::exp(value - maximum) : 0.; denominator += value; }
      if (denominator == 0.) continue;
      for (int d = 0; d < Dim; ++d) {
        double numerator = 0.;
        for (int j = 0; j < count; ++j) {
          const int64_t row = input.row(b, h, j);
          if (row >= 0) numerator += weights[size_t(j)] * input.load(true, size_t(row), d);
        }
        result[size_t((b * Heads + h) * Dim + d)] = numerator / denominator;
      }
    }
  return result;
}

[[noreturn]] void fail_stop(const char* context) noexcept {
  std::fprintf(stderr, "matrix-direct probe: cannot prove completion (%s); aborting with owners live\n", context);
  std::fflush(stderr); sycl_backend::cleanup_fail_stop(context);
}
void drain(const std::shared_ptr<Runtime>& runtime) noexcept {
  try { runtime->wait(); } catch (...) { fail_stop("checked runtime drain"); }
}
// Declared after all input/device owners, before the first upload/submission.
struct Lease {
  std::shared_ptr<Runtime> runtime;
  ~Lease() noexcept { drain(runtime); }
};
template<class T> struct Buffer {
  std::shared_ptr<Runtime> runtime;
  std::vector<T> initial;
  Allocation memory;
  Buffer(std::shared_ptr<Runtime> rt, const std::vector<T>& values)
      : runtime(std::move(rt)), initial(add(values.size(), 2 * Guard), T(91)),
        memory(runtime, multiply(initial.size(), sizeof(T)), MemoryKind::Device) {
    std::copy(values.begin(), values.end(), initial.begin() + Guard);
  }
  T* base() { return memory.as<T>(); }
  T* data() { return base() + Guard; }
  void upload() {
    try { runtime->wait(runtime->compute().memcpy(base(), initial.data(), memory.size())); }
    catch (...) { fail_stop("upload submission/wait"); }
  }
  std::vector<T> read() {
    std::vector<T> result(initial.size());
    try { runtime->wait(runtime->compute().memcpy(result.data(), base(), memory.size())); }
    catch (...) { fail_stop("readback submission/wait"); }
    return result;
  }
  void immutable() {
    const auto current = read();
    require(std::memcmp(current.data(), initial.data(), memory.size()) == 0, "input payload/guard modified");
  }
  void guards(const std::vector<T>& values) const {
    require(values.size() == initial.size(), "output readback size");
    require(std::memcmp(values.data(), initial.data(), Guard * sizeof(T)) == 0 &&
            std::memcmp(values.data() + values.size() - Guard,
                        initial.data() + initial.size() - Guard, Guard * sizeof(T)) == 0,
            "output prefix/suffix guard modified");
  }
};
struct DeviceInput {
  Buffer<float> query, output;
  Buffer<int32_t> table, ids, steps;
  Buffer<uint16_t> k16, v16, ks, vs;
  Buffer<int8_t> k8, v8;
  Buffer<uint8_t> k4, v4;
  DeviceInput(std::shared_ptr<Runtime> rt, const Input& input)
      : query(rt, input.query), output(rt, std::vector<float>(input.g.query_elements, OutputSentinel)),
        table(rt, input.table), ids(rt, input.ids), steps(rt, input.steps),
        k16(rt, input.k16), v16(rt, input.v16), ks(rt, input.ks), vs(rt, input.vs),
        k8(rt, input.k8), v8(rt, input.v8), k4(rt, input.k4), v4(rt, input.v4) {}
  void upload() {
    query.upload(); output.upload(); table.upload(); ids.upload(); steps.upload();
    k16.upload(); v16.upload(); ks.upload(); vs.upload(); k8.upload(); v8.upload(); k4.upload(); v4.upload();
  }
  QsaAttnPools pools(int format) {
    QsaAttnPools p;
    p.page_table = table.data();
    if (format == 0) { p.k_pool = k16.data(); p.v_pool = v16.data(); }
    if (format == 1 || format == 3) { p.k_q = k8.data(); p.k_scale = ks.data(); }
    if (format == 1) { p.v_q = v8.data(); p.v_scale = vs.data(); }
    if (format == 2) p.k_q4 = k4.data();
    if (format == 2 || format == 3) p.v_q4 = v4.data();
    return p;
  }
  void immutable() {
    query.immutable(); table.immutable(); ids.immutable(); steps.immutable();
    k16.immutable(); v16.immutable(); ks.immutable(); vs.immutable();
    k8.immutable(); v8.immutable(); k4.immutable(); v4.immutable();
  }
};
void set_arm(int arm) {
  require(arm == 0 || arm == 1, "invalid Direct arm");
  require(setenv("STRATA_SYCL_PROMPT_DIRECT", arm ? "1" : "0", 1) == 0,
          "cannot set Direct arm in single host issuer");
}
bool attention(std::shared_ptr<Runtime> runtime, const Input& input, DeviceInput& device) {
  auto shapes = qsa_real_shapes();
  return qsa_prompt_attn_batch(device.query.data(), device.pools(input.format), device.ids.data(),
      device.steps.data(), input.g.cap, shapes, device.output.data(), input.g.nq, &runtime->compute());
}
void describe(const Input& input, const char* type) {
  std::cout << "{\"type\":\"" << type << "\",\"format\":" << input.format << ",\"nq\":" << input.g.nq
            << ",\"cap\":" << input.g.cap << ",\"width\":" << input.width
            << ",\"dense_hot\":" << input.dense << ",\"all_nonresident\":" << input.nonresident
            << ",\"zero_scales\":" << input.zero_scales << ",\"physical_pages\":" << input.g.physical_pages
            << ",\"logical_cells\":262144,\"last_logical_cell\":262143,\"live_byte_bound\":" << input.g.live_bound
            << ",\"q_fnv64\":" << hash(input.query) << ",\"table_fnv64\":" << hash(input.table)
            << ",\"ids_fnv64\":" << hash(input.ids) << ",\"steps_fnv64\":" << hash(input.steps)
            << ",\"k16_fnv64\":" << hash(input.k16) << ",\"v16_fnv64\":" << hash(input.v16)
            << ",\"k8_fnv64\":" << hash(input.k8) << ",\"v8_fnv64\":" << hash(input.v8)
            << ",\"ks_fnv64\":" << hash(input.ks) << ",\"vs_fnv64\":" << hash(input.vs)
            << ",\"k4_fnv64\":" << hash(input.k4) << ",\"v4_fnv64\":" << hash(input.v4) << "}" << std::endl;
}
void validate(std::shared_ptr<Runtime> runtime, const Input& input, const char* label) {
  describe(input, label);
  const auto reference = oracle(input); // independent double dot/exp/sum, no kernel helper reuse
  DeviceInput device(runtime, input);
  Lease lease{runtime};
  device.upload();
  std::array<bool, 2> admitted{};
  std::array<std::vector<float>, 2> outputs;
  std::array<double, 2> worst{};
  std::array<size_t, 2> nonfinite{};
  std::array<bool, 2> valid{true, true};
  for (int arm : {0, 1}) {
    set_arm(arm);
    device.output.upload();
    admitted[size_t(arm)] = attention(runtime, input, device);
    std::cout << "{\"type\":\"admission\",\"case\":\"" << label << "\",\"format\":" << input.format
              << ",\"width\":" << input.width << ",\"direct\":" << arm
              << ",\"admitted\":" << admitted[size_t(arm)] << ",\"fallback_calls\":0}" << std::endl;
    drain(runtime);
    outputs[size_t(arm)] = device.output.read();
    device.output.guards(outputs[size_t(arm)]);
    if (!admitted[size_t(arm)]) {
      require(std::memcmp(outputs[size_t(arm)].data(), device.output.initial.data(), device.output.memory.size()) == 0,
              "declined matrix call changed output");
    } else {
      for (size_t i = 0; i < reference.size(); ++i) {
        const float actual = outputs[size_t(arm)][Guard + i];
        const double error = std::abs(double(actual) - reference[i]);
        worst[size_t(arm)] = std::max(worst[size_t(arm)], error);
        if (!std::isfinite(actual)) ++nonfinite[size_t(arm)];
        if (!std::isfinite(actual) || error > OracleTolerance * (1. + std::abs(reference[i]))) valid[size_t(arm)] = false;
      }
    }
    device.immutable();
    std::cout << "{\"type\":\"correctness_arm\",\"case\":\"" << label << "\",\"format\":" << input.format
              << ",\"width\":" << input.width << ",\"direct\":" << arm << ",\"admitted\":" << admitted[size_t(arm)]
              << ",\"oracle_compared\":" << admitted[size_t(arm)] << ",\"all_output_values\":" << reference.size()
              << ",\"oracle_tolerance\":1e-5,\"max_abs\":" << worst[size_t(arm)]
              << ",\"nonfinite_outputs\":" << nonfinite[size_t(arm)]
              << ",\"oracle_pass\":" << (admitted[size_t(arm)] && valid[size_t(arm)])
              << ",\"input_immutable_and_guards_pass\":true,\"output_guards_pass\":true}" << std::endl;
  }
  const bool comparable = admitted[0] && admitted[1];
  const bool bitwise = comparable && std::memcmp(outputs[0].data(), outputs[1].data(), outputs[0].size() * sizeof(float)) == 0;
  std::cout << "{\"type\":\"correctness_pair\",\"case\":\"" << label << "\",\"format\":" << input.format
            << ",\"width\":" << input.width << ",\"comparable\":" << comparable
            << ",\"bitwise\":" << bitwise << ",\"fallback_calls\":0,\"status\":\""
            << (!comparable ? "UNMEASURED_DECLINED" : bitwise && valid[0] && valid[1] ? "PASS" : "FAIL") << "\"}" << std::endl;
  require(valid[0] && valid[1] && (!comparable || bitwise), "matrix Direct pair or double oracle failed");
}

void benchmark(std::shared_ptr<Runtime> runtime, int nq, const Options& options) {
  Input input(1, nq, 2051, true);
  describe(input, "benchmark_input");
  DeviceInput device(runtime, input);
  Lease lease{runtime};
  device.upload();
  bool ready = true;
  // Admission and both JIT variants are prewarmed outside every timed sample.
  for (int arm : {0, 1}) {
    set_arm(arm);
    int admitted = 0;
    for (int warm = 0; warm < 4; ++warm) admitted += attention(runtime, input, device);
    drain(runtime);
    ready = ready && admitted == 4;
    std::cout << "{\"type\":\"benchmark_admission\",\"nq\":" << nq << ",\"direct\":" << arm
              << ",\"warm_calls\":4,\"admitted\":" << (admitted == 4)
              << ",\"admitted_calls\":" << admitted << ",\"fallback_calls\":0}" << std::endl;
  }
  if (!ready) {
    std::cout << "{\"type\":\"benchmark_status\",\"nq\":" << nq << ",\"status\":\"UNMEASURED_DECLINED\"}" << std::endl;
    return;
  }
  std::array<int, 2> sample_counts{};
  std::array<std::vector<double>, 2> wall_per_call, envelope_per_call;
  for (int sample = 0; sample < options.samples; ++sample) {
    const int arm = sample % 4 == 0 || sample % 4 == 3 ? 0 : 1; // ABBA
    set_arm(arm);
    drain(runtime);
    auto& queue = runtime->compute();
    int admitted = 0;
    const auto start = Clock::now();
    const auto begin = queue.single_task([] {});
    for (int repeat = 0; repeat < options.repeats; ++repeat) admitted += attention(runtime, input, device);
    const auto end = queue.single_task([] {});
    // One whole-queue drain after the batch; no per-call wait/readback/oracle.
    drain(runtime);
    const double wall_ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
    const auto first_end = begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto last_start = end.get_profiling_info<sycl::info::event_profiling::command_start>();
    require(last_start >= first_end, "nonmonotonic profiling envelope");
    const uint64_t envelope_ns = last_start - first_end;
    const bool measured = admitted == options.repeats;
    std::cout << "{\"type\":\"sample\",\"nq\":" << nq << ",\"width\":2051,\"sample\":" << sample
              << ",\"direct\":" << arm << ",\"calls\":" << options.repeats << ",\"admitted_calls\":" << admitted
              << ",\"measured\":" << measured << ",\"submit_plus_whole_drain_wall_ns\":" << wall_ns
              << ",\"marker_device_envelope_ns\":" << envelope_ns
              << ",\"wall_ns_per_call\":" << wall_ns / options.repeats
              << ",\"envelope_ns_per_call\":" << double(envelope_ns) / options.repeats
              << ",\"exclusive_kernel_time\":false,\"fallback_calls\":0}" << std::endl;
    require(measured, "admission changed during benchmark; sample is unmeasured");
    ++sample_counts[size_t(arm)];
    wall_per_call[size_t(arm)].push_back(wall_ns / options.repeats);
    envelope_per_call[size_t(arm)].push_back(double(envelope_ns) / options.repeats);
  }
  auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * .5;
  };
  for (int arm : {0, 1}) {
    require(sample_counts[size_t(arm)] >= 20, "fewer than 20 measured samples for an arm");
    std::cout << "{\"type\":\"benchmark_summary\",\"nq\":" << nq << ",\"direct\":" << arm
              << ",\"samples\":" << sample_counts[size_t(arm)] << ",\"median_wall_ns_per_call\":" << median(wall_per_call[size_t(arm)])
              << ",\"median_device_envelope_ns_per_call\":" << median(envelope_per_call[size_t(arm)])
              << ",\"correctness_checked_in_benchmark\":false,\"status\":\"MEASURED\"}" << std::endl;
  }
}

void host_contracts() {
  auto reject = [](auto function) {
    bool caught = false;
    try { function(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "host invalid-argument refusal failed");
  };
  require(parse({}).mode == Mode::Validate && parse({"--host-only"}).mode == Mode::Host &&
          parse({"--help"}).mode == Mode::Help && parse({"--benchmark"}).samples == 40, "CLI defaults");
  for (auto args : {std::vector<std::string>{"--bad"}, {"--validate", "--benchmark"},
                   {"--benchmark", "--samples", "39"}, {"--benchmark", "--samples", "42"},
                   {"--benchmark", "--repeats", "0"}, {"--benchmark", "--queries", "4096"},
                   {"--benchmark", "--samples", "18446744073709551616"},
                   {"--benchmark", "--samples", "+40"}, {"--benchmark", "--samples"},
                   {"--benchmark", "--samples", "40", "--samples", "40"}})
    reject([&] { (void)parse(args); });
  for (auto dimensions : {std::array<int64_t, 3>{0, 1, 5}, {-1, 1, 5}, {65536, 1, 5},
                         {1, 0, 5}, {1, INT64_MAX, 5}, {1, 1, 0}, {1, 1, 65537},
                         {65535, INT32_MAX, 65536}, {1, 1, 65536}})
    reject([&] { Geometry g(dimensions[0], dimensions[1], dimensions[2]); (void)g; });
  reject([&] { (void)multiply(UINT64_MAX, 2); });
  reject([&] { (void)add(SIZE_MAX, 1); });
  reject([&] { Input input(1, 1, INT32_MAX, true); (void)input; });
  reject([&] { Input input(1, 1, -1); (void)input; });
  for (int nq : {32, 256}) {
    Geometry g(nq, 2051, 514);
    require(g.query_elements == size_t(nq) * 24 * 256 && g.pool_rows == 4112, "dense geometry accounting");
  }
  for (int width : {0, 31, 32, 33, 63, 64, 65, 2051}) {
    Input input(1, 3, width);
    require(input.g.physical_pages == 5 && input.table.back() == 2, "small physical/nonidentity logical map");
    if (width) require(input.ids[size_t(width - 1)] == 262143, "last logical cell is selected");
  }
  // Actual product early geometry refusals must precede queue lookup. The
  // deliberately invalid stream is never dereferenced on these host paths.
  auto shapes = qsa_real_shapes();
  void* invalid_stream = reinterpret_cast<void*>(uintptr_t(1));
  require(qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 0, shapes, nullptr, 0, invalid_stream),
          "empty product matrix request returns before runtime initialization");
  for (int64_t nq : {int64_t(-1), int64_t(65536)})
    require(!qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 1, shapes, nullptr, nq, invalid_stream),
            "product query-count gate precedes queue lookup");
  for (int64_t cap : {int64_t(0), int64_t(-1), int64_t(INT32_MAX) + 1})
    require(!qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, cap, shapes, nullptr, 1, invalid_stream),
            "product capacity gate precedes queue lookup");
  shapes.head_dim = 128;
  require(!qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 1, shapes, nullptr, 1, invalid_stream),
          "product head-dimension gate precedes queue lookup");
  Input zero(1, 1, 65, false, true);
  const auto reference = oracle(zero);
  require(std::all_of(reference.begin(), reference.end(), [](double x) { return x == 0.; }), "all-nonresident oracle zero");
  std::cout << "{\"type\":\"host_contracts\",\"status\":\"PASS\",\"runtime_created\":false}" << std::endl;
}
void benchmark_environment() {
  // Targeted checks only, never an environment dump. Main must also use its
  // known clean launch whitelist: injected collectors are outside this list.
  for (const char* name : {"STRATA_TRACE", "UR_ENABLE_LAYERS", "SYCL_PI_TRACE", "UR_TRACE",
                          "UR_LOG_LOADER", "UR_LAYER_FULL_VALIDATION", "UR_LAYER_PARAMETER_VALIDATION",
                          "ZE_ENABLE_VALIDATION_LAYER", "ZE_ENABLE_TRACING_LAYER", "ZE_ENABLE_LOADER_DEBUG_TRACE"}) {
    const char* value = std::getenv(name);
    require(!value || !*value || std::strcmp(value, "0") == 0,
            std::string("benchmark requires tracing/validation disabled: ") + name);
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    const auto options = parse(args); // help/CLI/geometry gates precede runtime_for
    std::cout << std::boolalpha << std::setprecision(17);
    if (options.mode == Mode::Help) {
      std::cout << "sycl_matrix_direct_probe [--host-only|--validate|--benchmark [--samples 40..400, multiple of 4] [--repeats 1..32] [--queries 32|256]|--help]\n"
                   "Default: all-output double-oracle correctness; matrix declines are UNMEASURED. Benchmark is dense-hot INT8 width2051, nq32 and256, ABBA, >=20 samples/arm. Native FP16 KV optimization is disabled.\n";
      return 0;
    }
    if (options.mode != Mode::Benchmark) host_contracts();
    if (options.mode == Mode::Host) return 0;
    if (options.mode == Mode::Benchmark) benchmark_environment();
    // Arithmetic and planned shape admission before any SYCL runtime/device init.
    for (int nq : {32, 256}) { Geometry geometry(nq, 2051, 514); (void)geometry; }
    require(setenv("STRATA_SYCL_PROMPT_NATIVE_KV_MIN", "0", 1) == 0, "cannot disable independent native-KV option");
    auto runtime = sycl_backend::runtime_for();
    std::cout << "device=" << runtime->device().get_info<sycl::info::device::name>()
              << " driver=" << runtime->device().get_info<sycl::info::device::driver_version>()
              << " local_mem_bytes=" << runtime->device().get_info<sycl::info::device::local_mem_size>()
              << " max_work_group=" << runtime->device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    const char* q4_option = std::getenv("STRATA_PROMPT_ATTN_Q4");
    std::cout << "{\"type\":\"existing_q4_gate\",\"disabled_by_environment\":"
              << (q4_option && q4_option[0] == '0') << ",\"option_changed_by_fixture\":false}" << std::endl;
    std::cout << "{\"type\":\"policy\",\"native_kv_min\":0,\"product_matrix_gate_unchanged\":true,"
                 "\"fallback_calls\":0,\"pair_bitwise_required_when_both_admitted\":true,\"oracle_abs_rel\":1e-5,"
                 "\"benchmark_clock_definition\":\"wall includes markers+submissions+whole queue drain; marker device envelope includes kernel work and submission gaps, not exclusive kernel time\"}" << std::endl;
    if (options.mode == Mode::Benchmark) {
      for (int nq : {32, 256}) if (!options.queries || options.queries == nq) benchmark(runtime, nq, options);
    } else {
      for (int format = 0; format < 4; ++format) {
        for (int width : {0, 31, 32, 33, 63, 64, 65, 2051})
          validate(runtime, Input(format, 3, width), "sparse_boundary");
        validate(runtime, Input(format, 1, 65, false, true), "all_nonresident");
        validate(runtime, Input(format, 3, 65, true, false, true), "zero_scales");
      }
      validate(runtime, Input(0, 3, 65, true), "mode0_dense_native_disabled");
      validate(runtime, Input(1, 3, 2051, true), "mode1_dense_hot");
      Input invalid(1, 3, 65);
      for (int value : {-1, 66}) {
        for (int b = 0; b < 3; ++b) invalid.steps[size_t(b) * kStepCount + kStepWidth] = value;
        validate(runtime, invalid, value < 0 ? "negative_width" : "width_exceeds_cap");
      }
    }
    drain(runtime);
    std::cout << "{\"type\":\"completed\",\"status\":\"CLOSED\",\"success_exit\":true,\"declines_are_not_passes\":true}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL matrix_direct_probe: " << error.what() << std::endl;
    return 1;
  }
}
