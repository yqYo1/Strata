#pragma once
// Independent host fixture/oracle. No SYCL headers or runtime initialization.
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
namespace kv_copy_fixture {
inline void require(bool x, const std::string &s) {
  if (!x) throw std::runtime_error(s);
}
struct Shape {
  int blocks, slots, page, heads, dim, fmt;
  std::vector<size_t> lengths() const {
    require(blocks > 0 && slots > 0 && page > 0 && heads > 0 && dim > 0,
            "positive fixture dimensions");
    const size_t rows = size_t(page) * heads;
    if (fmt == 0) return {rows * dim * 2, rows * dim * 2};
    if (fmt == 1) {
      require(dim % 64 == 0, "int8 fixture width");
      return {rows * dim, rows * dim, rows * (dim / 64) * 2, rows * (dim / 64) * 2};
    }
    require(fmt == 2 && dim % 32 == 0, "q4 fixture width");
    return {rows * (dim / 32) * 18, rows * (dim / 32) * 18};
  }
  size_t block_bytes() const {
    const auto l = lengths(); return std::accumulate(l.begin(), l.end(), size_t(0));
  }
};
inline uint64_t u64(const std::vector<int32_t> &c, int at) {
  return uint64_t(uint32_t(c.at(at))) | uint64_t(uint32_t(c.at(at + 1))) << 32;
}
inline void put_u64(std::vector<int32_t> &c, int at, uint64_t x) {
  uint32_t lo = uint32_t(x), hi = uint32_t(x >> 32);
  std::memcpy(&c.at(at), &lo, 4); std::memcpy(&c.at(at + 1), &hi, 4);
}
struct State {
  std::vector<int32_t> table, block, stamp, ref, ctl, mb, ms;
  explicit State(const Shape &s)
      : table(s.blocks, -1), block(s.slots, -1), stamp(s.slots, -1),
        ref(s.slots, 0), ctl(16, 0), mb(s.slots, -77), ms(s.slots, -79) {}
  bool operator==(const State &o) const {
    return table == o.table && block == o.block && stamp == o.stamp &&
           ref == o.ref && ctl == o.ctl && mb == o.mb && ms == o.ms;
  }
};
struct Case {
  std::string name;
  State before;
  std::vector<std::vector<int32_t>> rows;
  std::vector<int32_t> widths;
  int cap = 1;
  Case(std::string n, const Shape &s) : name(std::move(n)), before(s) {}
  void add(std::vector<int32_t> cells) {
    cap = std::max(cap, int(cells.size())); widths.push_back(int(cells.size()));
    rows.push_back(std::move(cells));
  }
};
inline uint8_t pattern(int a, int b, size_t i) {
  const uint64_t x = uint64_t(b) * 0x9e3779b97f4a7c15ULL +
                     uint64_t(i) * 0xbf58476d1ce4e5b9ULL + uint64_t(a + 1) * 137;
  return uint8_t((x ^ (x >> 17) ^ (x >> 39)) & 255);
}
inline void seed(State &m, int b, int slot) {
  require(m.table.at(b) == -1 && m.block.at(slot) == -1, "seed collision");
  m.table[b] = slot; m.block[slot] = b; m.stamp[slot] = 0; m.ref[slot] = 1;
}
// An ordered set of misses is determined before CLOCK placement. Adjacency only
// affects lookups; every selected resident is protected before any victim choice.
inline State resolve(const Shape &s, const Case &c) {
  State out = c.before;
  if (c.rows.empty()) return out;
  int epoch = out.ctl[0];
  if (epoch < 0 || epoch == INT_MAX) {
    std::fill(out.stamp.begin(), out.stamp.end(), -1); epoch = 0;
  }
  ++epoch;
  std::vector<int> pending;
  uint64_t lookups = 0;
  for (size_t q = 0; q < c.rows.size(); ++q) {
    int width = c.widths.at(q);
    if (width < 0 || width > c.cap) { out.ctl[3] = 1; continue; }
    int previous = -1;
    for (int i = 0; i < width; ++i) {
      const int cell = c.rows[q].at(i);
      if (cell < 0 || int64_t(cell) / s.page >= s.blocks) { out.ctl[3] = 1; continue; }
      const int b = cell / s.page;
      if (b == previous) continue;
      previous = b; ++lookups;
      const int sl = out.table[b];
      if (sl >= 0 && sl < s.slots) { out.stamp[sl] = epoch; out.ref[sl] = 1; }
      else if (sl == -1) {
        if (pending.size() == size_t(s.slots)) { out.ctl[3] = 1; continue; }
        out.table[b] = -2; out.mb[pending.size()] = b; pending.push_back(b);
      } else if (sl != -2) out.ctl[3] = 1;
    }
  }
  int hand = out.ctl[1];
  if (hand < 0 || hand >= s.slots) hand = 0;
  std::vector<std::pair<int,int>> placed;
  for (int b : pending) {
    int victim = -1;
    for (int64_t scanned = 0; scanned < int64_t(s.slots) * 2; ++scanned) {
      const int sl = hand;
      hand = hand + 1 == s.slots ? 0 : hand + 1;
      if (out.stamp[sl] == epoch) continue;
      if (out.block[sl] < 0 || !out.ref[sl]) { victim = sl; break; }
      out.ref[sl] = 0;
    }
    if (victim < 0) { out.table[b] = -1; out.ctl[3] = 1; continue; }
    const int old = out.block[victim];
    if (old >= 0 && old < s.blocks) out.table[old] = -1;
    out.table[b] = victim; out.block[victim] = b;
    out.stamp[victim] = epoch; out.ref[victim] = 1;
    placed.emplace_back(b, victim);
  }
  for (size_t k = 0; k < placed.size(); ++k) {
    out.mb[k] = placed[k].first; out.ms[k] = placed[k].second;
  }
  out.ctl[0] = epoch; out.ctl[1] = hand; out.ctl[2] = int(placed.size());
  put_u64(out.ctl, 4, u64(out.ctl, 4) + placed.size());
  put_u64(out.ctl, 6, u64(out.ctl, 6) + lookups);
  put_u64(out.ctl, 8, u64(out.ctl, 8) + 1);
  return out;
}
inline void invariants(const Shape &s, const State &m) {
  std::vector<int> seen(s.slots, 0);
  for (int b = 0; b < s.blocks; ++b) {
    const int sl = m.table[b];
    require(sl == -1 || (sl >= 0 && sl < s.slots && m.block[sl] == b), "map inverse");
    if (sl >= 0) ++seen[sl];
  }
  for (int sl = 0; sl < s.slots; ++sl)
    require((m.block[sl] == -1 && seen[sl] == 0) ||
            (m.block[sl] >= 0 && m.block[sl] < s.blocks && seen[sl] == 1), "slot inverse");
  require(m.ctl[2] >= 0 && m.ctl[2] <= s.slots, "placed bound");
  std::vector<int> used(s.slots, 0);
  for (int k = 0; k < m.ctl[2]; ++k) {
    const int b = m.mb[k], sl = m.ms[k];
    require(b >= 0 && b < s.blocks && sl >= 0 && sl < s.slots &&
            m.table[b] == sl && !used[sl]++, "distinct placement slots");
  }
}
inline Case cold(const Shape &s, int count, const std::string &name = "cold") {
  require(count >= 0 && count <= s.blocks, "cold case bound");
  Case c(name + "_" + std::to_string(count), s);
  std::vector<int32_t> ids;
  // 37 is coprime to the provided 1043 and 65536 block dimensions.
  require(std::gcd(37, s.blocks) == 1, "fixture permutation");
  for (int i = 0; i < count; ++i)
    ids.push_back(((int64_t(i) * 37) % s.blocks) * s.page + i % s.page);
  c.add(std::move(ids)); return c;
}
inline std::vector<Case> semantics(const Shape &s) {
  std::vector<Case> out;
  auto hits = cold(s, std::min(s.slots, 513), "all_hits");
  for (size_t i = 0; i < hits.rows[0].size(); ++i) seed(hits.before, hits.rows[0][i] / s.page, int(i));
  out.push_back(hits);
  Case duplicate("nonadjacent_duplicate", s);
  duplicate.add({s.page + 0, s.page + s.page - 1, 3*s.page, s.page, 5*s.page});
  duplicate.add({5*s.page, s.page, 3*s.page}); out.push_back(duplicate);
  Case eviction("eviction_protect_hits_clock_second_chance", s);
  for (int b = 0; b < s.slots; ++b) seed(eviction.before, b, (b + 7) % s.slots);
  eviction.add({0, (s.slots - 1)*s.page, s.slots*s.page}); out.push_back(eviction);
  Case tail("two_partial_blocks_last_allocated_cell", s);
  tail.add({2 % s.page, s.page - 1, 5*s.page, 5*s.page + s.page - 1,
            (s.blocks - 1)*s.page, s.blocks*s.page - 1}); out.push_back(tail);
  auto full = cold(s, s.slots, "all_slots"); out.push_back(full);
  auto over = cold(s, s.slots + 1, "overflow_pending_list"); out.push_back(over);
  Case protected_full("overflow_all_residents_protected", s);
  std::vector<int32_t> ids;
  for (int b = 0; b < s.slots; ++b) { seed(protected_full.before, b, b); ids.push_back(b*s.page); }
  ids.push_back(s.slots*s.page); protected_full.add(ids); out.push_back(protected_full);
  for (int epoch : {-1, INT_MAX}) {
    auto wrap = hits; wrap.name = "epoch_" + std::to_string(epoch); wrap.before.ctl[0] = epoch;
    out.push_back(wrap);
  }
  auto badhand = eviction; badhand.name = "hand_reset"; badhand.before.ctl[1] = INT_MAX; out.push_back(badhand);
  auto carry = cold(s, 3, "u64_counter_carries");
  put_u64(carry.before.ctl, 4, UINT64_MAX - 1); put_u64(carry.before.ctl, 6, UINT32_MAX - 1);
  put_u64(carry.before.ctl, 8, UINT64_MAX); out.push_back(carry);
  Case invalid("invalid_ids_and_widths_sticky_overflow", s);
  invalid.add({-1, s.blocks*s.page, 0}); invalid.add({0}); invalid.widths.back() = -1;
  invalid.add({0}); invalid.widths.back() = invalid.cap + 1; out.push_back(invalid);
  auto sticky = hits; sticky.name = "existing_overflow_stays_set"; sticky.before.ctl[3] = 1; out.push_back(sticky);
  Case empty("nq_zero_noop", s); out.push_back(empty);
  return out;
}
inline Case typical_t4(const Shape &s, int missing) {
  require(s.page == 4 && s.blocks >= 4096 && s.slots >= 2052 && missing <= 2052,
          "T4 synthetic shape");
  Case c("T4_selector_shaped_misses_" + std::to_string(missing), s);
  std::vector<int> referenced;
  for (int q = 0; q < 4; ++q) {
    std::vector<int32_t> ids;
    for (int i = 0; i < 512; ++i) {
      const int b = q*1024 + i;
      referenced.push_back(b);
      for (int j = 0; j < 4; ++j) ids.push_back(b*4 + j);
    }
    const int tail = q*1024 + 777; referenced.push_back(tail);
    for (int j = 0; j < 3; ++j) ids.push_back(tail*4 + j);
    c.add(std::move(ids));
  }
  for (int i = 0; i < int(referenced.size()) - missing; ++i) seed(c.before, referenced[i], i);
  return c;
}
inline std::vector<std::vector<uint8_t>> initial_payload(const Shape &s, const State &m) {
  std::vector<std::vector<uint8_t>> p;
  const auto lens = s.lengths();
  for (size_t a = 0; a < lens.size(); ++a) {
    const size_t len = lens[a]; p.emplace_back(size_t(s.slots)*len, 0xa5);
    for (int sl = 0; sl < s.slots; ++sl)
      if (m.block[sl] >= 0)
        for (size_t i = 0; i < len; ++i) p.back()[size_t(sl)*len+i] = pattern(int(a), m.block[sl], i);
  }
  return p;
}
inline auto expected_payload(const Shape &s, const State &before, const State &after) {
  auto p = initial_payload(s, before); const auto lens = s.lengths();
  for (int k = 0; k < after.ctl[2]; ++k)
    for (size_t a = 0; a < lens.size(); ++a)
      for (size_t i = 0; i < lens[a]; ++i)
        p[a][size_t(after.ms[k])*lens[a]+i] = pattern(int(a), after.mb[k], i);
  return p;
}
} // namespace kv_copy_fixture
