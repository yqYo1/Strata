// Independent manual selector qualification. No process-environment writes.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
constexpr int guard = 16;
constexpr int32_t sentinel = -123456789;
std::ofstream dump;
int cases = 0;
void check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error(what);
}
// Declare after all allocation owners in each scope, so it drains before them
// on both success and exception. A failed checked drain is fail-stop.
struct Drain {
  ~Drain() { try { runtime->wait(); } catch (...) { std::abort(); } }
};
template <typename T> struct Buffer {
  sycl_backend::Allocation mem;
  explicit Buffer(size_t n)
      : mem(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    check(v.size() * sizeof(T) == mem.size(), "upload size mismatch");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), mem.size()));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
struct Row {
  int32_t n, width;
  int pattern;
  int invalid = 0;
};
// Host oracle follows the ordering specification and expands every block into
// individual weighted cells. Rank order and returned ID order are distinct.
uint32_t oracle_key(float value) {
  if (std::isnan(value)) return 0;
  if (value == 0.f) value = 0.f;
  const auto bits = std::bit_cast<uint32_t>(value);
  return std::signbit(value) ? ~bits : (bits | 0x80000000u);
}
std::vector<int32_t> oracle(const float *scores, int32_t n, int32_t width) {
  struct Cell { uint32_t key; int32_t id; };
  std::vector<Cell> cells;
  cells.reserve(size_t(n));
  for (int32_t id = 0; id < n; ++id)
    cells.push_back({oracle_key(scores[id / 4]), id});
  std::sort(cells.begin(), cells.end(), [](Cell a, Cell b) {
    return a.key != b.key ? a.key > b.key : a.id < b.id;
  });
  std::vector<int32_t> chosen;
  chosen.reserve(size_t(width));
  for (int32_t i = 0; i < width; ++i) chosen.push_back(cells[i].id);
  std::sort(chosen.begin(), chosen.end());
  return chosen;
}
float score(int pattern, int64_t b, int row) {
  switch (pattern) {
  case 0: { // Finite, varied sign/exponent/mantissa, reproducible per row.
    uint32_t h = uint32_t(b) * 1664525u + uint32_t(row) * 1013904223u;
    h ^= h >> 13;
    return std::bit_cast<float>((h & 0x807fffffu) |
                                ((uint32_t(113) + h % 28u) << 23));
  }
  case 1: return 1.f;
  case 2: {
    constexpr uint32_t bits[] = {0xff800000u, 0x7f800000u, 0x80000000u,
      0x00000000u, 0x7fc00001u, 0xffc12345u, 0xbf800000u, 0x3f800000u,
      0xff7fffffu, 0x7f7fffffu, 0x00000001u, 0x80000001u};
    return std::bit_cast<float>(bits[(b * 7 + row) % 12]);
  }
  case 3: return std::bit_cast<float>(uint32_t((b & 1) ? 0xffc00000u :
                                                             0x7fc00000u) |
                                               uint32_t(b % 65535 + 1));
  case 4: return std::bit_cast<float>(uint32_t(b & 1) << 31);
  case 5: {
    constexpr uint32_t bits[] = {0x3f7fffffu, 0x3f800000u, 0x3f800001u,
      0xbf7fffffu, 0xbf800000u, 0xbf800001u, 0x00000001u, 0x80000001u};
    return std::bit_cast<float>(bits[(b + row) % 8]);
  }
  case 6: // Equal groups cross partitions and consume a partial block budget.
    return b < 2 ? 3.f : (b % 13 == 0 ? -1.f : 2.f);
  case 7: return float((b * 8191 + row * 17) % 65537) - 32768.f;
  default: throw std::invalid_argument("unknown score pattern");
  }
}
void write_dump(const std::vector<int32_t> &v) {
  if (!dump.is_open()) return;
  // Every ID and guard, with a count before each case, for exact arm comparison.
  const uint64_t count = v.size();
  dump.write(reinterpret_cast<const char *>(&count), sizeof(count));
  dump.write(reinterpret_cast<const char *>(v.data()),
             std::streamsize(v.size() * sizeof(int32_t)));
  check(bool(dump), "output dump failed");
}
void run(const std::string &label, const std::vector<Row> &rows,
         int cap = 2051, int repetitions = 0, int64_t active = -1) {
  check(!rows.empty(), "empty fixture batch");
  int64_t largest = 0;
  for (const auto &row : rows) largest = std::max(largest, int64_t(row.n / 4));
  const int64_t blocks = largest + 3; // Includes unused score-stride canaries.
  if (active == 0) active = largest + 1; // Explicit minimal valid launch bound.
  const int64_t nq = int64_t(rows.size());
  std::vector<float> scores(size_t(nq * blocks) + 2 * guard,
                            std::numeric_limits<float>::infinity());
  std::vector<int32_t> steps(size_t(nq) * kStepCount);
  std::vector<std::vector<int32_t>> expected(rows.size());
  for (int64_t q = 0; q < nq; ++q) {
    const auto &row = rows[size_t(q)];
    auto *sc = scores.data() + guard + q * blocks;
    for (int64_t b = 0; b <= row.n / 4; ++b)
      sc[b] = score(row.pattern, b, int(q));
    // A zero-weight spare must never affect selection, even at +infinity.
    if (!(row.n % 4)) sc[row.n / 4] = std::numeric_limits<float>::infinity();
    auto *st = steps.data() + q * kStepCount;
    st[kStepPos] = row.n ? row.n - 1 : -1;
    st[kStepNKv] = row.n;
    st[kStepNBid] = row.n / 4;
    st[kStepWidth] = row.width;
    if (!row.invalid) expected[size_t(q)] = oracle(sc, row.n, row.width);
    switch (row.invalid) {
    case 0: break;
    case 1: st[kStepNKv] = -1; break;
    case 2: ++st[kStepNBid]; break;
    case 3: st[kStepNKv] = int32_t((active == -1 ? blocks : active) * 4);
            st[kStepNBid] = st[kStepNKv] / 4; break;
    case 4: st[kStepWidth] = -1; break;
    case 5: st[kStepWidth] = cap + 1; break;
    case 6: st[kStepWidth] = row.n + 1; break;
    default: throw std::invalid_argument("unknown invalid step case");
    }
  }
  const size_t output_size = size_t(nq) * size_t(cap) + 2 * guard;
  Buffer<float> ds(scores.size());
  Buffer<int32_t> dt(steps.size()), out(output_size);
  Drain drain;
  ds.put(scores); dt.put(steps);
  out.put(std::vector<int32_t>(output_size, sentinel));
  auto invoke = [&] {
    qsa_block_topk_prefill(ds.data() + guard, dt.data(), nq, blocks, cap,
                   qsa_real_shapes(), out.data() + guard,
                   &runtime->compute(), active);
  };
  invoke();
  runtime->wait(); // Completed warmup; no pending work enters timing.
  const auto first = out.get();
  auto validate = [&](const std::vector<int32_t> &actual,
                      const std::vector<std::vector<int32_t>> &references) {
    check(actual.size() == output_size, label + ": output size");
    for (int g = 0; g < guard; ++g)
      check(actual[size_t(g)] == sentinel &&
            actual[actual.size() - 1 - size_t(g)] == sentinel,
            label + ": outer canary");
    for (int64_t q = 0; q < nq; ++q) {
      const auto &ref = references[size_t(q)];
      const auto *got = actual.data() + guard + q * cap;
      for (int64_t j = 0; j < cap; ++j) {
        if (j < int64_t(ref.size())) {
          check(got[j] == ref[size_t(j)], label + ": oracle ID q=" +
                std::to_string(q) + " j=" + std::to_string(j));
          check(got[j] >= 0 && got[j] < steps[size_t(q) * kStepCount + kStepNKv],
                label + ": ID range");
          if (j) check(got[j] > got[j - 1], label + ": ascending unique IDs");
        } else check(got[j] == sentinel, label + ": row tail/invalid canary");
      }
    }
  };
  validate(first, expected);
  if (repetitions) {
    // CPU oracle and copies are outside this submission-plus-drain interval.
    runtime->wait();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) invoke();
    runtime->wait();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    const auto repeated = out.get();
    validate(repeated, expected);
    check(repeated == first, label + ": repeated output changed");
    std::cout << "selector_component label=" << label << " nq=" << nq
              << " n=78453 cap=" << cap << " repeats=" << repetitions
              << " total_submit_drain_ms=" << ms << '\n';
  }
  // Public decode/verification/short-prefill API and the reference entrypoint
  // must keep the original result under each pre-exec candidate environment.
  // Reuse output storage and keep these full-oracle/guard checks outside timing.
  out.put(std::vector<int32_t>(output_size, sentinel));
  qsa_block_topk(ds.data() + guard, dt.data(), nq, blocks, cap,
                 qsa_real_shapes(), out.data() + guard,
                 &runtime->compute(), active);
  runtime->wait();
  const auto original = out.get();
  validate(original, expected);
  check(original == first, label + ": public original differs from prefill");
  // _ref has no active-block argument. An invalid-active-bound row can be
  // valid at its full capacity; compute its actual contract's oracle separately.
  auto reference_expected = expected;
  for (int64_t q = 0; q < nq; ++q) {
    if (rows[size_t(q)].invalid != 3) continue;
    const auto *st = steps.data() + q * kStepCount;
    reference_expected[size_t(q)] = oracle(scores.data() + guard + q * blocks,
                                          st[kStepNKv], st[kStepWidth]);
  }
  out.put(std::vector<int32_t>(output_size, sentinel));
  qsa_block_topk_ref(ds.data() + guard, dt.data(), nq, blocks, cap,
                     qsa_real_shapes(), out.data() + guard, &runtime->compute());
  runtime->wait();
  validate(out.get(), reference_expected);
  std::cout << "PASS public_original_and_ref label=" << label << " nq=" << nq << '\n';
  const auto scores_after = ds.get();
  check(std::memcmp(scores_after.data(), scores.data(),
                    scores.size() * sizeof(float)) == 0, label + ": scores changed");
  check(dt.get() == steps, label + ": steps changed");
  write_dump(first);
  ++cases;
  std::cout << "PASS label=" << label << " nq=" << nq << " cap=" << cap << '\n';
}
void validation(bool dedicated_prefill) {
  const auto select = dedicated_prefill ? qsa_block_topk_prefill : qsa_block_topk;
  auto shapes = qsa_real_shapes();
  Buffer<float> scores(8);
  Buffer<int32_t> steps(kStepCount), ids(8);
  Drain drain;
  scores.put(std::vector<float>(8, 1.f));
  steps.put({0, 4, 1, 1});
  ids.put(std::vector<int32_t>(8, sentinel));
  auto call = [&](int64_t nq, int64_t blocks, int64_t cap,
                  const QsaShapes &s, int64_t active = -1) {
    select(scores.data(), steps.data(), nq, blocks, cap, s,
                   ids.data(), &runtime->compute(), active);
  };
  auto reject = [&](auto f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument &) { caught = true; }
    check(caught, "invalid geometry/span accepted");
  };
  reject([&] { call(0, 8, 8, shapes); });
  reject([&] { call(65536, 8, 8, shapes); });
  reject([&] { call(1, 0, 8, shapes); });
  reject([&] { call(1, int64_t(INT_MAX) + 1, 8, shapes); });
  reject([&] { call(1, 8, 0, shapes); });
  reject([&] { call(1, 8, int64_t(INT_MAX) + 1, shapes); });
  for (int64_t active : {int64_t(-2), int64_t(0), int64_t(9)})
    reject([&] { call(1, 8, 8, shapes, active); });
  for (int which = 0; which < 5; ++which) {
    auto bad = shapes;
    if (which == 0) bad.idx_dim = 127;
    if (which == 1) bad.idx_n_head = 3;
    if (which == 2) bad.idx_block = 3;
    if (which == 3) bad.idx_top_k = -1;
    if (which == 4) bad.idx_top_k = int64_t(INT_MAX) - 2;
    reject([&] { call(1, 8, 8, bad); });
  }
  reject([&] { select(nullptr, steps.data(), 1, 8, 8, shapes,
                              ids.data(), &runtime->compute(), -1); });
  reject([&] { select(scores.data(), nullptr, 1, 8, 8, shapes,
                              ids.data(), &runtime->compute(), -1); });
  reject([&] { select(scores.data(), steps.data(), 1, 8, 8, shapes,
                              nullptr, &runtime->compute(), -1); });
  reject([&] { select(scores.data(), steps.data(), 1, 8, 8, shapes,
                              steps.data(), &runtime->compute(), -1); });
  reject([&] { select(scores.data(), steps.data(), 1, 8, 8, shapes,
                       reinterpret_cast<int32_t *>(scores.data()),
                       &runtime->compute(), -1); });
  reject([&] { select(scores.data(), steps.data(), 1, 8, 8, shapes,
                       reinterpret_cast<int32_t *>(
                           reinterpret_cast<char *>(ids.data()) + 1),
                       &runtime->compute(), -1); });
  runtime->wait();
  check(ids.get() == std::vector<int32_t>(8, sentinel),
        "rejected host calls changed IDs");
  std::cout << "PASS host_validation api="
            << (dedicated_prefill ? "dedicated_prefill" : "public_original") << '\n';
}
int integer(const char *text) {
  char *end = nullptr;
  const long n = std::strtol(text, &end, 10);
  check(end != text && *end == '\0' && n >= 1 && n <= 20,
        "repetitions must be 1..20");
  return int(n);
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool bench = argc == 6 && std::string(argv[1]) == "--bench";
    check(argc == 1 || (argc == 3 && std::string(argv[1]) == "--dump") || bench,
      "usage: fixture [--dump PATH] | --bench varied|equal REPEATS --dump PATH");
    int reps = 0, pattern = 0;
    if (bench) {
      const std::string name = argv[2];
      check(name == "varied" || name == "equal", "benchmark pattern varied|equal");
      pattern = name == "equal" ? 1 : 0;
      reps = integer(argv[3]);
      check(std::string(argv[4]) == "--dump", "benchmark requires full output dump");
    }
    if (argc != 1) {
      dump.open(argv[bench ? 5 : 2], std::ios::binary | std::ios::trunc);
      check(bool(dump), "cannot open output dump");
    }
    // Read only; caller sets all SYCL/driver/experiment variables before exec.
    const char *arm = std::getenv("STRATA_SYCL_SELECT_RADIX4");
    std::cout << "STRATA_SYCL_SELECT_RADIX4=" << (arm ? arm : "<unset>") << '\n';
    const char *cyclic = std::getenv("STRATA_SYCL_SELECT_CYCLIC");
    std::cout << "STRATA_SYCL_SELECT_CYCLIC=" << (cyclic ? cyclic : "<unset>") << '\n';
    runtime = sycl_backend::runtime_for();
    if (bench) {
      run(pattern ? "bench_equal" : "bench_varied",
          std::vector<Row>(256, Row{78453, 2051, pattern}), 2051, reps, 0);
    } else {
      for (int pattern_id = 0; pattern_id < 8; ++pattern_id) {
        std::vector<Row> rows;
        for (int n = 0; n <= 12; ++n)
          for (int width : {0, n / 2, n}) rows.push_back({n, width, pattern_id});
        for (int n = 1016; n <= 1027; ++n)
          rows.push_back({n, 127, pattern_id}); // B = 255,256,257; tails 0..3.
        for (int n = 2050; n <= 2053; ++n) {
          rows.push_back({n, std::min(n, 2051), pattern_id});
          for (int width : {1, 3, 4, 5, 2049}) rows.push_back({n, width, pattern_id});
        }
        run("small_pattern_" + std::to_string(pattern_id), rows, 2051, 0, 0);
        rows = {{0, 0, pattern_id}, {5, 5, pattern_id},
                {78452, 2051, pattern_id}, {78453, 2051, pattern_id}};
        for (int n = 262141; n <= 262144; ++n) rows.push_back({n, 2051, pattern_id});
        // No timing from full physical-context coverage.
        run("long_mixed_pattern_" + std::to_string(pattern_id), rows);
      }
      std::vector<Row> invalid = {{8, 3, 0}, {0, 0, 1}, {7, 7, 2}, {9, 0, 3}};
      for (int kind = 1; kind <= 6; ++kind) invalid.push_back({8, 3, kind, kind});
      run("mixed_invalid_steps", invalid, 16, 0, 3);
      validation(false);
      validation(true);
    }
    runtime->wait();
    if (dump.is_open()) {
      dump.flush(); check(bool(dump), "dump flush failed");
      dump.close(); check(!dump.fail(), "dump close failed");
    }
    std::cout << "PASS cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
