// UNBUILT / UNTESTED head-split component fixture proposal.
// Derived from matrix-direct-probe-v1; original proposal remains unchanged.
// Product entrypoint, independent FP64 oracle, guards, immutable owners.
// No model capture; no fallback. Admission alone does not prove dispatch.
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
#include <utility>

namespace {
using namespace strata;
using namespace strata::kernels;
using sycl_backend::Allocation;
using sycl_backend::MemoryKind;
using sycl_backend::Runtime;
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
  Input(int mode, int64_t nq, int count, bool hot = false, bool missing = false, bool zero = false, int64_t explicit_cap = 0)
      : g(nq, explicit_cap ? explicit_cap : capacity(count), hot ? (capacity(count) + 3) / 4 + 1 : 5),
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
struct Arm { const char* direct; const char* split; const char* expected; };
constexpr std::array<Arm, 3> Arms{{{"1", nullptr, "legacy_direct"},
                                {"1", "32", "head_split_32"},
                                {"1", "64", "head_split_64"}}};
void set_arm(const Arm& arm) {
  require(setenv("STRATA_SYCL_PROMPT_DIRECT", arm.direct, 1) == 0, "Direct env");
  require((arm.split ? setenv("STRATA_SYCL_PROMPT_HEAD_SPLIT", arm.split, 1)
                     : unsetenv("STRATA_SYCL_PROMPT_HEAD_SPLIT")) == 0, "split env");
}
bool attention(const std::shared_ptr<Runtime>& rt, const Input& in, DeviceInput& dev) {
  const auto shapes = qsa_real_shapes();
  return qsa_prompt_attn_batch(dev.query.data(), dev.pools(in.format), dev.ids.data(), dev.steps.data(),
                               in.g.cap, shapes, dev.output.data(), in.g.nq, &rt->compute());
}
void describe(const Input& in, const char* label) {
  std::cout << "{\"type\":\"input\",\"case\":\"" << label << "\",\"format\":" << in.format
            << ",\"nq\":" << in.g.nq << ",\"cap\":" << in.g.cap << ",\"width\":" << in.width
            << ",\"step_width0\":" << in.steps[kStepWidth]
            << ",\"physical_pages\":" << in.g.physical_pages << ",\"logical_cells\":262144"
            << ",\"live_byte_bound\":" << in.g.live_bound
            << ",\"q_fnv64\":" << hash(in.query) << ",\"table_fnv64\":" << hash(in.table)
            << ",\"ids_fnv64\":" << hash(in.ids) << ",\"steps_fnv64\":" << hash(in.steps)
            << ",\"k16_fnv64\":" << hash(in.k16) << ",\"v16_fnv64\":" << hash(in.v16)
            << ",\"k8_fnv64\":" << hash(in.k8) << ",\"v8_fnv64\":" << hash(in.v8)
            << ",\"ks_fnv64\":" << hash(in.ks) << ",\"vs_fnv64\":" << hash(in.vs)
            << ",\"k4_fnv64\":" << hash(in.k4) << ",\"v4_fnv64\":" << hash(in.v4) << "}" << std::endl;
}
// Returns false for any decline, which the controller must count as unmeasured.
bool validate(const std::shared_ptr<Runtime>& rt, const Input& in, const char* label,
              bool include64) {
  describe(in, label);
  const auto ref = oracle(in);
  DeviceInput dev(rt, in);
  Lease lease{rt}; // declared after owners, before submission
  dev.upload();
  std::array<std::vector<float>, 3> outputs;
  std::array<bool, 3> admitted{}, good{true, true, true};
  const int arms = include64 ? 3 : 2;
  bool measured = true;
  for (int a=0; a<arms; ++a) {
    set_arm(Arms[size_t(a)]);
    dev.output.upload();
    admitted[size_t(a)] = attention(rt, in, dev);
    drain(rt);
    outputs[size_t(a)] = dev.output.read();
    dev.output.guards(outputs[size_t(a)]);
    dev.immutable();
    double maximum = 0.;
    size_t nonfinite = 0, untouched = 0;
    measured = measured && admitted[size_t(a)];
    if (!admitted[size_t(a)]) {
      require(std::memcmp(outputs[size_t(a)].data(), dev.output.initial.data(), dev.output.memory.size()) == 0,
              "declined call modified output");
    } else {
      for (size_t i=0; i<ref.size(); ++i) {
        const float value = outputs[size_t(a)][Guard+i];
        const double error = std::abs(double(value)-ref[i]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(value)) ++nonfinite;
        if (value == OutputSentinel) ++untouched;
        if (!std::isfinite(value) || error > OracleTolerance*(1.+std::abs(ref[i]))) good[size_t(a)] = false;
      }
    }
    std::cout << "{\"type\":\"arm\",\"case\":\"" << label << "\",\"arm\":" << a
              << ",\"expected_route\":\"" << Arms[size_t(a)].expected << "\",\"route_observed\":false"
              << ",\"admitted\":" << admitted[size_t(a)] << ",\"fallback_calls\":0"
              << ",\"outputs_checked\":" << (admitted[size_t(a)] ? ref.size() : 0)
              << ",\"max_abs\":" << maximum << ",\"nonfinite\":" << nonfinite << ",\"untouched\":" << untouched
              << ",\"output_fnv64\":" << hash(outputs[size_t(a)]) << ",\"oracle_abs_rel\":1e-5"
              << ",\"oracle_pass\":" << (admitted[size_t(a)] && good[size_t(a)])
              << ",\"input_immutable\":true,\"output_guards\":true}" << std::endl;
  }
  bool pairs_good = true;
  for (int a=1; a<arms; ++a) {
    const bool comparable = admitted[0] && admitted[size_t(a)];
    const bool bitwise = comparable && std::memcmp(outputs[0].data(), outputs[size_t(a)].data(),
                                                    outputs[0].size()*sizeof(float)) == 0;
    double maximum = 0.; bool close = true;
    if (comparable) for (size_t i=0; i<ref.size(); ++i) {
      const double base = outputs[0][Guard+i], value = outputs[size_t(a)][Guard+i];
      maximum = std::max(maximum, std::abs(value-base));
      if (!std::isfinite(value) || !std::isfinite(base) ||
          std::abs(value-base) > OracleTolerance*(1.+std::abs(base))) close = false;
    }
    const bool pair_pass = comparable && good[0] && good[size_t(a)] && (a==1 ? bitwise : close);
    pairs_good = pairs_good && (!comparable || pair_pass);
    std::cout << "{\"type\":\"pair\",\"case\":\"" << label << "\",\"candidate\":" << a
              << ",\"comparable\":" << comparable << ",\"bitwise\":" << bitwise
              << ",\"bitwise_required\":" << (a==1) << ",\"max_abs\":" << maximum
              << ",\"status\":\"" << (!comparable ? "UNMEASURED_DECLINED" : pair_pass ? "PASS" : "FAIL") << "\"}" << std::endl;
  }
  require(good[0] && good[1] && (!include64 || good[2]) && pairs_good, "oracle or pair gate failed");
  return measured;
}
// Dispatch isolation checks numerical invariance. Observed kernel names remain
// a separate Main-owned diagnostic; equal outputs alone cannot identify a route.
bool isolation(const std::shared_ptr<Runtime>& rt, const Input& in, const Arm& option) {
  DeviceInput dev(rt, in); Lease lease{rt}; dev.upload();
  std::vector<float> baseline;
  bool measured=true, same=true;
  for (int i=0; i<2; ++i) {
    set_arm(i ? option : Arm{option.direct, nullptr, "legacy"});
    dev.output.upload();
    const bool admitted=attention(rt, in, dev);
    drain(rt);
    auto out=dev.output.read(); dev.output.guards(out); dev.immutable();
    measured=measured && admitted;
    std::cout << "{\"type\":\"isolation_arm\",\"format\":" << in.format
              << ",\"arm\":" << i << ",\"admitted\":" << admitted
              << ",\"fallback_calls\":0,\"output_fnv64\":" << hash(out) << "}" << std::endl;
    if (!admitted) require(std::memcmp(out.data(), dev.output.initial.data(), dev.output.memory.size())==0,
                           "isolation declined output changed");
    if (!i) baseline=std::move(out);
    else same=std::memcmp(baseline.data(), out.data(), baseline.size()*sizeof(float))==0;
  }
  std::cout << "{\"type\":\"isolation\",\"format\":" << in.format << ",\"direct\":\"" << option.direct
            << "\",\"split\":\"" << (option.split ? option.split : "<unset>")
            << "\",\"route_observed\":false,\"bitwise\":" << same << ",\"status\":\""
            << (!measured ? "UNMEASURED_DECLINED" : same ? "PASS" : "FAIL") << "\"}" << std::endl;
  require(!measured || same, "off/invalid/other-mode output changed");
  return measured;
}
void host_contracts() {
  // Prove proposed ownership enumeration; this does not instrument GPU stores.
  for (size_t nq : {size_t(1), size_t(3), size_t(65535)}) {
    // For largest nq, enumerate heads/bands for first and last batches only.
    for (size_t batch : {size_t(0), nq-1}) {
      std::array<unsigned char, Heads*Dim> owners{};
      for (size_t kv=0; kv<KvHeads; ++kv) for (size_t tile=0; tile<2; ++tile) {
        const size_t group=(batch*KvHeads+kv)*2+tile;
        require(group%2==tile && (group/2)%KvHeads==kv && group/(2*KvHeads)==batch, "group decode");
        for (size_t r=0; r<(tile ? 4u : 8u); ++r)
          for (size_t sg=0; sg<4; ++sg) for (size_t j=0; j<4; ++j) for (size_t c=0; c<16; ++c)
            ++owners[(kv*12+tile*8+r)*Dim+(sg*4+j)*16+c];
      }
      require(std::all_of(owners.begin(), owners.end(), [](auto n){return n==1;}), "24-head coverage");
    }
  }
  const auto zero=oracle(Input(1, 1, 65, false, true));
  require(std::all_of(zero.begin(), zero.end(), [](double n){return n==0.;}), "all-missing oracle");
  Input high(1,3,65,false,false,false,LogicalCells);
  require(high.g.cap==262144 && high.ids[64]==262143 && high.ids[size_t(high.g.cap)+64]==262143,
          "high logical ID and per-query capacity stride");
  for (int width : {0,1,31,32,33,63,64,65,2051}) { Input in(1,3,width); (void)in; }
  // Existing early product gates only, before queue lookup.
  auto shapes=qsa_real_shapes(); void* invalid=reinterpret_cast<void*>(uintptr_t(1));
  require(qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,0,shapes,nullptr,0,invalid), "nq0 early return");
  for (int64_t cap : {int64_t(0),int64_t(-1),int64_t(INT32_MAX)+1})
    require(!qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,cap,shapes,nullptr,1,invalid), "capacity early refusal");
  std::cout << "{\"type\":\"host_contracts\",\"status\":\"PASS\",\"runtime_created\":false}" << std::endl;
}
// Common max must come from both tiles in turn, independently for each KV head.
void extreme_query(Input& in, int large_tile) {
  for (int64_t b=0; b<in.g.nq; ++b) for (int h=0; h<Heads; ++h) for (int d=0; d<Dim; ++d) {
    const bool large=(h%12 < 8 ? 0 : 1)==large_tile;
    const float mantissa=float(1.+.125*std::sin(double((b*Heads+h)*Dim+d)*.013));
    in.query[size_t((b*Heads+h)*Dim+d)]=std::ldexp(mantissa, large ? 20 : -20);
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    const std::string option=argc==2 ? argv[1] : "--validate";
    require(argc<=2 && (option=="--validate" || option=="--validate32" || option=="--host-only" || option=="--help"), "invalid CLI");
    if (option=="--help") {
      std::cout << "sycl_int8_prompt_head_split_probe [--host-only|--validate32|--validate|--help]\n"
                   "--validate32 excludes CH64; --validate adds CH64 tolerance gate. No benchmark/fallback.\n"; return 0;
    }
    std::cout << std::boolalpha << std::setprecision(17);
    host_contracts();
    if (option=="--host-only") return 0;
    require(setenv("STRATA_SYCL_PROMPT_NATIVE_KV_MIN","0",1)==0,"native independent option off");
    auto rt=sycl_backend::runtime_for();
    std::cout << "device=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << std::endl;
    const bool include64=option=="--validate"; bool measured=true;
    for (int width : {0,1,31,32,33,63,64,65,2051}) {
      measured=validate(rt,Input(1,3,width),"sparse_tail",include64) && measured;
      measured=validate(rt,Input(1,1,width,true),"dense_tail",include64) && measured;
    }
    measured=validate(rt,Input(1,1,65,false,true),"all_missing",include64) && measured;
    measured=validate(rt,Input(1,3,65,true,false,true),"zero_scales",include64) && measured;
    measured=validate(rt,Input(1,3,65,false,false,false,LogicalCells),"cap262144_logical262143",include64) && measured;
    for (int tile : {0,1}) {
      Input in(1,3,65,true); extreme_query(in,tile);
      measured=validate(rt,in,tile ? "large_query_tile1" : "large_query_tile0",include64) && measured;
    }
    Input subq(1,3,65,true);
    for (size_t i=0;i<subq.query.size();++i)
      subq.query[i]=std::numeric_limits<float>::denorm_min()*float(int(i%15)-7);
    measured=validate(rt,subq,"subnormal_query",include64) && measured;
    Input subscale(1,3,65,true);
    for (size_t i=0;i<subscale.ks.size();++i) subscale.ks[i]=uint16_t(1+i%3);
    for (size_t i=0;i<subscale.vs.size();++i) subscale.vs[i]=uint16_t(1+i%3);
    measured=validate(rt,subscale,"fp16_subnormal_scales",include64) && measured;
    Input qzero(1,3,65,true); std::fill(qzero.query.begin(),qzero.query.end(),0.f);
    measured=validate(rt,qzero,"zero_query",include64) && measured;
    Input invalid(1,3,65);
    for (int width : {-1,66}) {
      for (int b=0;b<3;++b) invalid.steps[size_t(b)*kStepCount+kStepWidth]=width;
      measured=validate(rt,invalid,width<0 ? "negative_width" : "width_gt_cap",include64) && measured;
    }
    // Direct strings other than exact "1" and split values other than exact
    // "32"/"64" remain on the old dispatch. Preserve legacy Direct prefix rule.
    for (const char* split : {"", "0", "off", "032", "+32", "32x", "64 ", "64x", "128"})
      measured=isolation(rt,Input(1,1,65),Arm{"1",split,"legacy"}) && measured;
    for (const char* direct : {"0", "", "10", "1x"})
      for (const char* split : {"32","64"})
        measured=isolation(rt,Input(1,1,65),Arm{direct,split,"legacy"}) && measured;
    for (int mode : {0,2,3}) for (const char* split : {"32","64"})
      measured=isolation(rt,Input(mode,1,65),Arm{"1",split,"legacy_other_mode"}) && measured;
    drain(rt);
    std::cout << "{\"type\":\"completed\",\"all_measured\":" << measured
              << ",\"status\":\"" << (measured ? "PASS" : "UNMEASURED_DECLINED") << "\"}" << std::endl;
    return measured ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL int8_prompt_head_split_probe: " << error.what() << std::endl; return 1;
  }
}
