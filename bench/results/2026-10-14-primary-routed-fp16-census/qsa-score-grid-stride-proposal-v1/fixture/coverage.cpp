// CPU-only integer geometry fixture. Proposed helper and extracted real host
// validator are exercised; score FP arithmetic/subgroups require Main's GPU gate.
// UNCOMPILED / UNRUN. No SYCL runtime, model or product implementation import.
#include "strata/kernels/qsa_score_grid_stride.hpp"
#include "extracted_validate.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using strata::kernels::detail::ScoreGridStride;
using strata::kernels::detail::score_grid_stride_enabled;
struct Step { int64_t n, nb; };
static uint64_t cases = 0;

static void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}

static void check(int64_t blocks, int64_t active,
                  const std::vector<Step> &steps) {
  const int64_t nq = int64_t(steps.size());
  const auto shapes = strata::kernels::qsa_real_shapes();
  const int64_t bound = fixture_source::validate(nq, blocks, active, shapes);
  const ScoreGridStride grid{bound};
  require(grid.active_bound == bound, "original bound changed");
  require(grid.group_count == (bound <= 4096 ? bound : 4096), "launch count");
  std::vector<int> expected(size_t(nq * blocks), 0), seen(expected.size(), 0);
  std::vector<int64_t> owner(expected.size(), -1);
  std::vector<int> tail(expected.size(), -1);

  // Independent baseline: enumerate every original group, never grid-stride.
  // Derive validity by multiplication/remainder rather than the helper's n/4.
  for (int64_t qi = 0; qi < nq; ++qi) {
    const auto st = steps[size_t(qi)];
    const bool valid = st.n >= 0 && st.nb >= 0 && st.nb < bound &&
                       st.nb * 4 <= st.n && st.n - st.nb * 4 < 4;
    for (int64_t b = 0; b < bound; ++b)
      if (valid && b <= st.nb) expected[size_t(qi * blocks + b)] = 1;
  }

  // These are the same mapping and traversal used in the proposed device kernel.
  for (int64_t group = 0; group < nq * grid.group_count; ++group) {
    const int64_t qi = grid.query_for_group(group);
    require(qi >= 0 && qi < nq, "query ownership");
    const auto st = steps[size_t(qi)];
    grid.for_each_block(st.n, st.nb, grid.first_block(group), [&](int64_t b) {
      require(b >= 0 && b < bound && b < blocks, "runtime/capacity bound");
      const size_t address = size_t(qi * blocks + b);
      ++seen[address];
      owner[address] = group;
      tail[address] = b == st.nb ? 1 : 0;
    });
  }
  require(seen == expected, "coverage/multiple writers/untouched capacity");
  for (int64_t qi = 0; qi < nq; ++qi) {
    for (int64_t b = 0; b < blocks; ++b) {
      const size_t address = size_t(qi * blocks + b);
      if (!expected[address]) {
        require(owner[address] == -1 && tail[address] == -1, "unexpected write");
        continue;
      }
      // Independent congruence owner; all query/score strides remain distinct.
      require(owner[address] == qi * grid.group_count + b % grid.group_count,
              "block owner");
      require(tail[address] == (b == steps[size_t(qi)].nb ? 1 : 0),
              "dead-key candidate ownership");
    }
  }
  ++cases;
}

static void invalid_host(int64_t nq, int64_t blocks, int64_t active,
                         strata::kernels::QsaShapes shapes) {
  bool rejected = false;
  try { (void)fixture_source::validate(nq, blocks, active, shapes); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "unchanged host validator accepted malformed geometry");
}

int main() {
  require(!score_grid_stride_enabled(nullptr), "missing must be off");
  for (const char *value : {"", "0", "01", "10", "true", "yes", "1 ", " 1", "1\n"})
    require(!score_grid_stride_enabled(value), "non-exact value must be off");
  require(score_grid_stride_enabled("1"), "exact 1 must enable");

  constexpr int64_t capacity = 65538;
  for (int64_t end : {32768, 65536, 262144}) {
    for (int nq = 1; nq <= 4; ++nq) {
      for (int remainder = 0; remainder < 4; ++remainder) {
        std::vector<Step> rows;
        for (int qi = 0; qi < nq; ++qi) {
          const int64_t n = end - nq + qi + 1 + remainder;
          rows.push_back({n, n / 4});
        }
        check(capacity, -1, rows);
        // Explicit smallest legal active bound for the batch; scores still use
        // full capacity as their row stride. Also covers nb > reduced grid count.
        check(capacity, rows.back().nb + 1, rows);
      }
    }
  }
  for (int64_t bound : {1, 2, 4095, 4096, 4097, 8193, 16385, 65538}) {
    for (int remainder = 0; remainder < 4; ++remainder) {
      const int64_t nb = bound - 1;
      check(capacity, bound, {{nb * 4 + remainder, nb}, {0, 0},
                              {(bound / 2) * 4, bound / 2}, {-1, 0}});
      // nb == original active bound must be rejected even though capacity fits.
      check(capacity, bound, {{bound * 4, bound}, {4, 0}, {0, 1}, {0, -1}});
    }
  }
  // Small valid capacities and an empty KV count: n=0 still owns the dead score.
  for (int64_t cap : {1, 2, 4, 4095, 4096, 4097})
    check(cap, -1, {{0, 0}, {3, 0}, {-1, -1}, {4, 1}});
  check(capacity, -1, {{INT32_MIN, 0}, {INT32_MAX, INT32_MAX},
                      {16, 3}, {16, 5}});
  check(capacity, -1, {{INT32_MAX, INT32_MAX / 4}, {0, INT32_MIN}});

  const auto shapes = strata::kernels::qsa_real_shapes();
  for (int64_t nq : {0, -1, 65536}) invalid_host(nq, capacity, -1, shapes);
  for (int64_t blocks : std::vector<int64_t>{0, -1, int64_t(INT_MAX) + 1})
    invalid_host(1, blocks, -1, shapes);
  for (int64_t active : {0, -2, 65539}) invalid_host(1, capacity, active, shapes);
  auto bad = shapes; bad.idx_dim = 127; invalid_host(1, capacity, -1, bad);
  bad = shapes; bad.idx_n_head = 3; invalid_host(1, capacity, -1, bad);
  bad = shapes; bad.idx_block = 8; invalid_host(1, capacity, -1, bad);
  bad = shapes; bad.idx_top_k = -1; invalid_host(1, capacity, -1, bad);
  bad = shapes; bad.idx_top_k = INT_MAX - 2; invalid_host(1, capacity, -1, bad);
  require(fixture_source::validate(65535, INT_MAX, -1, shapes) == INT_MAX,
          "unchanged largest host geometry");
  std::cout << "PASS CPU integer coverage/ownership cases=" << cases
            << "; no GPU/FP/capture correctness claim\n";
}
