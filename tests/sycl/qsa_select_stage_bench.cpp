// Main-authored component timing for the existing SYCL QSA selector.
// Uses this repository's Runtime/Allocation and existing public kernel APIs.
// This is synthetic component evidence, not model throughput or task quality.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
constexpr int guard = 16;
constexpr float score_canary = -12345.5f;
constexpr int32_t id_canary = -123456789;
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
struct Drain {
  ~Drain() noexcept { try { rt->wait(); } catch (...) { std::abort(); } }
};
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  explicit Buffer(size_t count)
      : mem(rt, count * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &value) {
    require(value.size() * sizeof(T) == mem.size(), "upload extent");
    rt->wait(rt->compute().memcpy(data(), value.data(), mem.size()));
  }
  std::vector<T> get() {
    std::vector<T> value(mem.size() / sizeof(T));
    rt->wait(rt->compute().memcpy(value.data(), data(), mem.size()));
    return value;
  }
};
void gate(bool enabled, const char *phase) {
  if (!enabled) return;
  std::cout << "QSA_STAGE_READY " << phase << std::endl;
  require(std::cin.get() == 'X', "Main gate token");
}
double clock_ms(clockid_t kind) {
  timespec ts{};
  require(clock_gettime(kind, &ts) == 0, "clock_gettime");
  return double(ts.tv_sec) * 1000.0 + double(ts.tv_nsec) / 1e6;
}
template <class T> uint64_t fingerprint(const std::vector<T> &v) {
  uint64_t h = 14695981039346656037ull;
  const auto *p = reinterpret_cast<const unsigned char *>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(T); ++i) {
    h ^= p[i]; h *= 1099511628211ull;
  }
  return h;
}
template <class T> bool bit_equal(const std::vector<T> &a, const std::vector<T> &b) {
  return a.size() == b.size() && std::equal(
      reinterpret_cast<const unsigned char *>(a.data()),
      reinterpret_cast<const unsigned char *>(a.data()) + a.size() * sizeof(T),
      reinterpret_cast<const unsigned char *>(b.data()));
}
float value(uint32_t &state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return float(int32_t(state & 65535u) - 32768) / 32768.f;
}
uint32_t order(float x) {
  if (std::isnan(x)) return 0;
  if (x == 0.f) x = 0.f;
  const auto b = std::bit_cast<uint32_t>(x);
  return std::signbit(x) ? ~b : (b | 0x80000000u);
}
std::vector<int32_t> oracle_ids(const float *scores, int n, int width) {
  // Independent rank over individual cells (including the weighted tail).
  struct Cell { uint32_t key; int32_t id; };
  std::vector<Cell> cells;
  cells.reserve(size_t(n));
  for (int id = 0; id < n; ++id) cells.push_back({order(scores[id / 4]), id});
  auto less = [](Cell a, Cell b) {
    return a.key != b.key ? a.key > b.key : a.id < b.id;
  };
  if (width < n) std::nth_element(cells.begin(), cells.begin() + width, cells.end(), less);
  std::vector<int32_t> ids;
  ids.reserve(size_t(width));
  for (int i = 0; i < width; ++i) ids.push_back(cells[size_t(i)].id);
  std::sort(ids.begin(), ids.end());
  return ids;
}
void run(int context, int capacity, int queries, int repetitions) {
  require(context >= queries && capacity >= context && capacity <= 262144,
          "context/capacity bounds");
  require(queries > 0 && queries <= 256 && repetitions > 0 && repetitions <= 20,
          "query/repetition bounds");
  const auto shape = qsa_real_shapes();
  const int blocks = capacity / 4 + 2, active = context / 4 + 1, cap = 2051;
  std::vector<float> pooled(size_t(blocks) * 128 + 2 * guard, score_canary);
  std::vector<float> dead(128 + 2 * guard, score_canary);
  std::vector<float> query(size_t(queries) * 512 + 2 * guard, score_canary);
  std::vector<int32_t> steps(size_t(queries) * kStepCount + 2 * guard, id_canary);
  uint32_t seed = 0x64d5a763u;
  for (size_t i = guard; i < pooled.size() - guard; ++i) pooled[i] = value(seed);
  for (int d = 0; d < 128; ++d) dead[size_t(guard + d)] = value(seed);
  for (size_t i = guard; i < query.size() - guard; ++i) query[i] = value(seed);
  for (int q = 0; q < queries; ++q) {
    const int n = context - queries + q + 1;
    auto *s = steps.data() + guard + q * kStepCount;
    s[kStepPos] = n - 1; s[kStepNKv] = n; s[kStepNBid] = n / 4;
    s[kStepWidth] = int32_t(qsa_selection_width(n, shape));
  }
  std::vector<float> initial_scores(size_t(queries) * blocks + 2 * guard, score_canary);
  std::vector<int32_t> initial_ids(size_t(queries) * cap + 2 * guard, id_canary);
  Buffer<float> pk(pooled.size()), dk(dead.size()), qq(query.size()), sc(initial_scores.size());
  Buffer<int32_t> st(steps.size()), ids(initial_ids.size());
  // Retire every asynchronous reader before either host or device owners die.
  Drain drain;
  pk.put(pooled); dk.put(dead); qq.put(query); st.put(steps);
  sc.put(initial_scores); ids.put(initial_ids);
  auto score = [&] {
    qsa_block_scores(pk.data() + guard, dk.data() + guard, qq.data() + guard,
                     st.data() + guard, queries, blocks, shape, sc.data() + guard,
                     &rt->compute(), active);
  };
  auto topk = [&] {
    qsa_block_topk_prefill(sc.data() + guard, st.data() + guard, queries, blocks, cap,
                         shape, ids.data() + guard, &rt->compute(), active);
  };
  // One qualification pair and two warmup pairs. All CPU oracles/copies are
  // outside timed regions; each timed region starts from a checked drained queue.
  score(); topk(); rt->wait();
  const auto first_scores = sc.get();
  const auto first_ids = ids.get();
  size_t score_checks = 0;
  double max_error = 0, max_budget_ratio = 0;
  for (int q : std::array<int, 5>{0, queries / 4, queries / 2, 3 * queries / 4, queries - 1}) {
    const int n = steps[size_t(guard + q * kStepCount + kStepNKv)], nb = n / 4;
    for (int i = 0; i < 67; ++i) {
      const int b = i == 64 ? nb : i == 65 ? 0 : i == 66 ? nb / 2 :
                        int(uint64_t(i) * uint64_t(nb + 1) / 64);
      const float *key = b == nb ? dead.data() + guard : pooled.data() + guard + b * 128;
      double ref = 0, magnitude = 0;
      for (int h = 0; h < 4; ++h) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) {
          const double product = double(key[d]) * query[size_t(guard + q * 512 + h * 128 + d)];
          dot += product; magnitude += std::abs(product);
        }
        ref += std::max(0., dot);
      }
      if (b == nb && n % 4) ref += 1e9;
      const double got = first_scores[size_t(guard + q * blocks + b)];
      // Fixed prospective FP32 roundoff bound for the unchanged scorer.
      // Local four-term sums, five subgroup steps, ReLU and four-head sum.
      const double budget = 16. * std::numeric_limits<float>::epsilon() *
                            (magnitude + std::abs(ref)) + 1e-7;
      const double err = std::abs(got - ref);
      require(std::isfinite(got) && err <= budget, "FP32 scorer sampled oracle");
      max_error = std::max(max_error, err); max_budget_ratio = std::max(max_budget_ratio, err / budget);
      ++score_checks;
    }
  }
  for (int q = 0; q < queries; ++q) {
    const auto *s = steps.data() + guard + q * kStepCount;
    const auto expected = oracle_ids(first_scores.data() + guard + q * blocks,
                                     s[kStepNKv], s[kStepWidth]);
    require(std::equal(expected.begin(), expected.end(), first_ids.begin() + guard + q * cap),
            "independent full-row cell top-k oracle");
    for (int i = s[kStepWidth]; i < cap; ++i)
      require(first_ids[size_t(guard + q * cap + i)] == id_canary, "ID row tail");
    for (int b = s[kStepNBid] + 1; b < blocks; ++b)
      require(first_scores[size_t(guard + q * blocks + b)] == score_canary,
              "score row inactive tail");
  }
  for (int g = 0; g < guard; ++g) {
    require(first_scores[size_t(g)] == score_canary &&
            first_scores[first_scores.size() - 1 - size_t(g)] == score_canary, "score outer guards");
    require(first_ids[size_t(g)] == id_canary &&
            first_ids[first_ids.size() - 1 - size_t(g)] == id_canary, "ID outer guards");
  }
  for (float x : first_scores) require(std::isfinite(x), "all scores finite");
  for (int warm = 0; warm < 2; ++warm) { score(); topk(); rt->wait(); }
  std::cout << "frame context=" << context << " capacity=" << capacity << " queries=" << queries
            << " blocks=" << blocks << " active=" << active << " cap=" << cap
            << " repeats=" << repetitions << " warmup_pairs=2" << std::endl;
  for (int rep = 0; rep < repetitions; ++rep) {
    for (int slot = 0; slot < 3; ++slot) {
      const int stage = (slot + rep) % 3;
      const char *name = stage == 0 ? "score" : stage == 1 ? "topk" : "pair";
      rt->wait();
      const double w0 = clock_ms(CLOCK_MONOTONIC), c0 = clock_ms(CLOCK_THREAD_CPUTIME_ID);
      if (stage != 1) score();
      if (stage != 0) topk();
      const double c1 = clock_ms(CLOCK_THREAD_CPUTIME_ID), w1 = clock_ms(CLOCK_MONOTONIC);
      rt->wait();
      const double w2 = clock_ms(CLOCK_MONOTONIC), c2 = clock_ms(CLOCK_THREAD_CPUTIME_ID);
      std::cout << "sample=" << rep << " slot=" << slot << " stage=" << name
                << " submit_wall_ms=" << w1 - w0 << " submit_thread_cpu_ms=" << c1 - c0
                << " wait_wall_ms=" << w2 - w1 << " wait_thread_cpu_ms=" << c2 - c1
                << " total_wall_ms=" << w2 - w0 << " total_thread_cpu_ms=" << c2 - c0 << std::endl;
    }
  }
  const auto final_scores = sc.get();
  const auto final_ids = ids.get();
  require(bit_equal(first_scores, final_scores) && bit_equal(first_ids, final_ids), "repeat bit identity");
  require(bit_equal(pk.get(), pooled) && bit_equal(dk.get(), dead) &&
          bit_equal(qq.get(), query) && bit_equal(st.get(), steps), "input immutability and guards");
  std::cout << "inputs pooled_fnv=" << fingerprint(pooled) << " dead_fnv=" << fingerprint(dead)
            << " query_fnv=" << fingerprint(query) << " steps_fnv=" << fingerprint(steps) << std::endl;
  std::cout << "case score_checks=" << score_checks << " score_max_abs_error=" << max_error
            << " score_max_budget_ratio=" << max_budget_ratio
            << " topk_full_rows=" << queries << " guards=PASS inputs=PASS repeat_bits=PASS"
            << " score_fnv=" << fingerprint(first_scores) << " ids_fnv=" << fingerprint(first_ids)
            << " status=PASS" << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  std::cout << std::setprecision(12);
  bool gates = false;
  bool ok = false;
  try {
    int context = 2113, capacity = 32768, queries = 3, repetitions = 1;
    for (int i = 1; i < argc; i += 2) {
      require(i + 1 < argc, "missing argument value");
      const std::string key(argv[i]); const int v = std::stoi(argv[i + 1]);
      if (key == "--context") context = v;
      else if (key == "--capacity") capacity = v;
      else if (key == "--queries") queries = v;
      else if (key == "--repeats") repetitions = v;
      else throw std::invalid_argument("unknown argument");
    }
    const char *g = std::getenv("STRATA_TEST_MAIN_GATES"); gates = g && std::string(g) == "1";
    require(std::string(std::getenv("STRATA_SYCL_SELECT_RADIX4") ? std::getenv("STRATA_SYCL_SELECT_RADIX4") : "") == "1",
            "use the measured engine radix4 mapping");
    require(std::string(std::getenv("STRATA_SYCL_SELECT_CYCLIC") ? std::getenv("STRATA_SYCL_SELECT_CYCLIC") : "") == "0",
            "cyclic mapping must be disabled");
    gate(gates, "before-runtime"); rt = sycl_backend::runtime_for();
    std::cout << "device=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << std::endl;
    gate(gates, "runtime-ready");
    run(context, capacity, queries, repetitions);
    rt->wait(); rt.reset(); ok = true;
  } catch (const std::exception &e) {
    std::cerr << "fixture_error=" << e.what() << std::endl;
    try { if (rt) rt->wait(); } catch (...) { std::abort(); }
    rt.reset();
  }
  try { gate(gates, "after-cleanup"); } catch (...) { return 2; }
  std::cout << "fixture_status=" << (ok ? "PASS" : "FAIL") << " closed=1" << std::endl;
  return ok ? 0 : 1;
}
