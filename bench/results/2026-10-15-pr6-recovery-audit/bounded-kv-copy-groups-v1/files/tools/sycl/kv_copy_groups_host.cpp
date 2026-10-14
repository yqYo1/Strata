#include "kv_copy_groups_oracle.hpp"
#include <iostream>
using namespace kv_copy_fixture;
namespace {
void coverage(int slots, int placed, int limit) {
  const int groups = limit == 0 ? slots : std::min(slots, limit);
  std::vector<int> visited(placed);
  for (int g = 0; g < groups; ++g)
    for (int k = g; k < placed; k += groups) ++visited[k];
  require(std::all_of(visited.begin(), visited.end(), [](int n) { return n == 1; }),
          "each placement visited exactly once");
}
void bytes_exact(const Shape &s, const Case &c, int limit) {
  const auto after = resolve(s, c); invariants(s, after);
  auto actual = initial_payload(s, c.before);
  const auto expected = expected_payload(s, c.before, after);
  const auto lens = s.lengths();
  const int groups = limit == 0 ? s.slots : std::min(s.slots, limit);
  for (int g = 0; g < groups; ++g)
    for (int lane = 0; lane < 128; ++lane)
      for (int k = g; k < after.ctl[2]; k += groups)
        for (size_t a = 0; a < lens.size(); ++a)
          for (size_t i = size_t(lane); i < lens[a]; i += 128)
            actual[a][size_t(after.ms[k])*lens[a]+i] = pattern(int(a), after.mb[k], i);
  require(actual == expected, c.name + ": complete copy spans exact");
}
}
int main() {
  try {
    uint64_t coverage_cases = 0, byte_cases = 0, state_cases = 0;
    for (int slots : {1, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 8192})
      for (int limit : {0, 128, 256, 512, 1024}) {
        for (int placed = 0; placed <= std::min(slots, 513); ++placed) {
          coverage(slots, placed, limit); ++coverage_cases;
        }
        for (int placed : {slots, std::min(slots, limit + 1), std::min(slots, 2*limit+1)}) {
          coverage(slots, placed, limit); ++coverage_cases;
        }
      }
    for (int fmt : {0, 1, 2}) {
      Shape s{1043, 1031, 1, 1, 64, fmt};
      for (int n = 0; n <= 513; ++n) {
        auto c = cold(s, n); auto m = resolve(s, c);
        require(m.ctl[2] == n && u64(m.ctl, 4) == uint64_t(n) &&
                u64(m.ctl, 6) == uint64_t(n) && u64(m.ctl, 8) == 1 && !m.ctl[3],
                "cold placement/counter oracle");
        for (int limit : {0, 128, 256, 512, 1024}) { bytes_exact(s, c, limit); ++byte_cases; }
      }
      for (const auto &c : semantics(s)) {
        auto m = resolve(s, c); invariants(s, m); ++state_cases;
        for (int limit : {0, 128, 256, 512, 1024}) { bytes_exact(s, c, limit); ++byte_cases; }
      }
      const auto cases = semantics(s);
      require(resolve(s, cases[0]).ctl[2] == 0, "all hits copies zero pages");
      auto duplicate = resolve(s, cases[1]);
      require(duplicate.ctl[2] == 3 && u64(duplicate.ctl, 6) == 7, "lookups are page runs, not set cardinality");
      auto overflow = resolve(s, cases[5]);
      require(overflow.ctl[2] == s.slots && overflow.ctl[3] == 1, "bounded pending-list overflow");
      auto protected_full = resolve(s, cases[6]);
      require(protected_full.ctl[2] == 0 && protected_full.ctl[3] == 1, "all protected no victim");
      Shape full{65536, 8192, 4, 2, 256, fmt};
      for (int missing : {0, 1, 128, 513, 1025, 2052}) {
        const auto c = typical_t4(full, missing);
        const auto m = resolve(full, c);
        invariants(full, m);
        require(m.ctl[2] == missing && u64(m.ctl, 6) == 2052 && !m.ctl[3], "T4 oracle");
        ++state_cases;
      }
      Case last("max_logical_cell", full); last.add({262143, 0, 262140, 262143});
      auto m = resolve(full, last); invariants(full, m);
      require(m.ctl[2] == 2 && !m.ctl[3], "256K logical last page oracle only");
      ++state_cases;
      require(full.block_bytes() == size_t(fmt == 0 ? 8192 : fmt == 1 ? 4224 : 2304),
              "real-shape byte accounting");
    }
    // A page can contribute one cell: ceil(width/page_size) is not a generic bound.
    Shape arbitrary{1043, 1031, 4, 1, 64, 1};
    auto c = cold(arbitrary, 513);
    auto m = resolve(arbitrary, c);
    require(m.ctl[2] == 513 && c.rows[0].size() == 513, "arbitrary-ID page bound regression");
    for (int limit : {0, 128, 256, 512, 1024}) { bytes_exact(arbitrary, c, limit); ++byte_cases; }
    // Two nonadjacent partial blocks plus one complete block: width 9, pages 3.
    Case partial("two_partial", arbitrary);
    partial.add({2, 3, 20, 21, 22, 23, 4168, 4169, 4170});
    require(resolve(arbitrary, partial).ctl[2] == 3, "two partial blocks preserved");
    std::cout << "PASS host fixture coverage_cases=" << coverage_cases << " byte_cases=" << byte_cases
              << " state_cases=" << state_cases << "; no SYCL runtime created\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
