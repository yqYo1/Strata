// Model-free scheduling/ownership fixture. Includes the exact candidate helper,
// never a reimplemented planner. This is symbolic row coverage, not GPU FP math.
#include "strata/kernels/qsa_indexer_batch_plan.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Key = std::array<int64_t, 4>; // ordered pooling operands, not their sum
struct State {
  std::array<int64_t, 3> tail{{-10, -11, -12}};
  Key dead{{-21, -22, -23, -24}};
  std::vector<Key> pooled;
  int64_t block_pos = -37;
  explicit State(int64_t capacity)
      : pooled(size_t(capacity / 4 + 2), Key{{-41, -42, -43, -44}}) {}
  bool operator==(const State &rhs) const {
    return tail == rhs.tail && dead == rhs.dead && pooled == rhs.pooled &&
           block_pos == rhs.block_pos;
  }
};
void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
// Independent literal serial transitions, with no proposed planning code.
// Raw row symbols distinguish every position (F16 conversion is deferred to
// GPU parity); complete keys retain ordered inputs rather than hash a sum.
void serial_cell(State &s, int64_t pos, int64_t base) {
  const int slot = int(pos % 4);
  if (slot < 3)
    s.tail[size_t(slot)] = pos;
  if (pos != 0 && slot != 3)
    return;
  const Key key = pos == 0 ? Key{{0, 0, 0, 0}}
                          : Key{{s.tail[0], s.tail[1], s.tail[2], pos}};
  s.pooled[size_t(pos / 4)] = key;
  if (pos == 0)
    s.dead = key;
  else {
    s.pooled[size_t(pos / 4 + 1)] = s.dead;
    s.block_pos = base + 4 * (pos / 4);
  }
}
void serial_batch(State &s, int64_t p0, int64_t n, int64_t base) {
  for (int64_t i = 0; i < n; ++i)
    serial_cell(s, p0 + i, base);
}

struct Coverage {
  uint64_t cases = 0, bulk_blocks = 0, old_tail_reads = 0, continuations = 0;
  std::array<bool, 4> start_mod{}, end_mod{};
  std::array<bool, 4> short_n{};
  bool full_boundary = false, empty = false, no_complete = false;
} coverage;

void planned_batch(State &s, int64_t capacity, int64_t p0, int64_t n, int64_t base) {
  using strata::kernels::detail::qsa_indexer_batch_plan;
  const auto plan = qsa_indexer_batch_plan(p0, n);
  require(plan.end <= capacity, "range exceeds capacity");
  require(plan.completed >= 0, "negative complete count");
  const State before = s;
  if (plan.initialize_dead)
    serial_cell(s, 0, base); // shared initial transition in the proposed kernel
  std::vector<bool> written(s.pooled.size(), false);
  for (int64_t g = 0; g < plan.completed; ++g) {
    const int64_t block = plan.first_block + g;
    require(block >= 0 && size_t(block) < s.pooled.size(), "pooled bounds");
    require(!written[size_t(block)], "overlapping completed writes");
    written[size_t(block)] = true;
    Key key;
    for (int row = 0; row < 4; ++row) {
      const int64_t i = plan.input_row(block, row);
      if (i < 0) {
        require(g == 0 && row < p0 % 4 && row < 3, "illegal old tail read");
        key[size_t(row)] = s.tail[size_t(row)];
        require(s.tail[size_t(row)] == before.tail[size_t(row)], "old tail clobber");
        ++coverage.old_tail_reads;
      } else {
        require(i < n, "raw source beyond batch");
        require(p0 + i == 4 * block + row, "raw mapping");
        key[size_t(row)] = p0 + i;
      }
    }
    s.pooled[size_t(block)] = key;
    ++coverage.bulk_blocks;
  }
  // Model unordered bulk groups by also verifying they have disjoint output
  // rows, share only read-only old-tail inputs, and perform no spare writes.
  for (int slot = 0; slot < 3; ++slot) {
    const int64_t i = plan.tail_input_row(slot);
    if (i >= 0) {
      require(i < n && (p0 + i) % 4 == slot, "tail source bounds/slot");
      require(p0 + i + 4 >= plan.end, "tail source is not last write");
      s.tail[size_t(slot)] = p0 + i;
    } else {
      require(s.tail[size_t(slot)] == before.tail[size_t(slot)], "retained slot changed");
    }
  }
  if (plan.completed) {
    require(plan.spare_block() >= 0 && size_t(plan.spare_block()) < s.pooled.size(), "spare bounds");
    require(!written[size_t(plan.spare_block())], "spare clobbers complete key");
    s.pooled[size_t(plan.spare_block())] = s.dead;
    s.block_pos = base + 4 * plan.last_completed_block();
  }
  // Physical +2 allocation's last row must remain byte-for-byte untouched.
  require(s.pooled.back() == before.pooled.back(), "extra +2 row changed");
}

void check(int64_t capacity, int64_t p0, int64_t n, int64_t base, bool seed_prefix) {
  require(p0 >= 0 && n >= 0 && p0 + n <= capacity, "fixture input range");
  State reference(capacity);
  if (seed_prefix)
    serial_batch(reference, 0, p0, base);
  // Also accept independently initialized continuation buffers: complete keys
  // depend on their preexisting tail, not on an assumed prefix implementation.
  State candidate = reference;
  serial_batch(reference, p0, n, base);
  planned_batch(candidate, capacity, p0, n, base);
  const std::string label = "capacity=" + std::to_string(capacity) + " p0=" +
      std::to_string(p0) + " n=" + std::to_string(n);
  require(reference == candidate, "whole-state mismatch " + label);
  coverage.start_mod[size_t(p0 % 4)] = true;
  coverage.end_mod[size_t((p0 + n) % 4)] = true;
  if (n <= 3)
    coverage.short_n[size_t(n)] = true;
  coverage.empty |= n == 0;
  coverage.no_complete |= n > 0 && (p0 + n) / 4 == p0 / 4;
  coverage.full_boundary |= p0 + n == 262144;
  ++coverage.cases;
  // Subsequent device-position appends are unchanged. Compare every state after
  // each single continuation cell, then exercise a second planned batch/head.
  int64_t next = p0 + n;
  for (int i = 0; i < 5 && next < capacity; ++i, ++next) {
    serial_cell(reference, next, base);
    serial_cell(candidate, next, base);
    require(reference == candidate, "single-head continuation mismatch " + label);
    ++coverage.continuations;
  }
  const int64_t more = capacity - next < 7 ? capacity - next : 7;
  if (more) {
    serial_batch(reference, next, more, base);
    planned_batch(candidate, capacity, next, more, base);
    require(reference == candidate, "batch continuation mismatch " + label);
    ++coverage.continuations;
  }
}
int main() {
  try {
    for (int64_t capacity : {1, 2, 3, 4, 5, 7, 16, 63, 64, 65})
      for (int64_t p0 = 0; p0 <= capacity; ++p0)
        for (int64_t n = 0; n <= capacity - p0; ++n)
          for (bool prefix : {false, true})
            check(capacity, p0, n, 100, prefix);
    // Chunk geometry, odd/short lengths, unaligned partial starts, context
    // endpoints, final valid cell 262143 and full-context zero-work boundary.
    for (int64_t capacity : {32768, 65536, 262144}) {
      for (int64_t p0 : {int64_t(0), int64_t(1), int64_t(2), int64_t(3), int64_t(4),
                         capacity - 8, capacity - 7, capacity - 6, capacity - 5,
                         capacity - 4, capacity - 3, capacity - 2, capacity - 1, capacity}) {
        for (int64_t n : {int64_t(0), int64_t(1), int64_t(2), int64_t(3), int64_t(4),
                          int64_t(5), int64_t(7), int64_t(4095), int64_t(4096), int64_t(4097), capacity - p0})
          if (n <= capacity - p0)
            check(capacity, p0, n, 0, true);
      }
    }
    // Exercise int64 planning at the API's INT32_MAX capacity without allocating
    // a gigantic symbolic state; no completion position ever equals capacity.
    constexpr int64_t max = INT32_MAX;
    for (int64_t p0 = max - 8; p0 <= max; ++p0)
      for (int64_t n = 0; n <= max - p0; ++n) {
        const auto p = strata::kernels::detail::qsa_indexer_batch_plan(p0, n);
        require(p.end <= max, "INT32_MAX end overflow");
        if (p.completed)
          require(4 * p.last_completed_block() + 3 < max, "invalid complete position");
        for (int s = 0; s < 3; ++s) {
          const auto row = p.tail_input_row(s);
          require(row == -1 || (row >= 0 && row < n), "INT32_MAX tail overflow");
        }
      }
    for (size_t r = 0; r < 4; ++r) {
      require(coverage.start_mod[r] && coverage.end_mod[r], "missing start/end remainder");
      require(coverage.short_n[r], "missing n0..3");
    }
    require(coverage.empty && coverage.no_complete && coverage.full_boundary &&
            coverage.old_tail_reads && coverage.continuations, "missing special coverage");
    std::cout << "PASS symbolic state/planning cases=" << coverage.cases
              << " completed_blocks=" << coverage.bulk_blocks
              << " old_tail_reads=" << coverage.old_tail_reads
              << " continuations=" << coverage.continuations << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
