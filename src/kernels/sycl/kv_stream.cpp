#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <vector>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
using Spans = std::vector<Span>;
void spans(const Spans &writes, const Spans &reads = {}) {
  for (size_t i = 0; i < writes.size(); ++i) {
    validate_spans({writes[i]}, {});
    for (size_t j = 0; j < i; ++j)
      validate_spans({writes[i]}, {writes[j]});
    for (auto r : reads)
      validate_spans({writes[i]}, {r});
  }
  for (auto r : reads)
    validate_spans({}, {r});
}
size_t bytes(int64_t count, size_t stride) {
  if (count < 0 || (stride && uint64_t(count) > SIZE_MAX / stride))
    throw std::invalid_argument("SYCL KV stream size overflow");
  return size_t(count) * stride;
}
struct Runs {
  std::array<const uint8_t *, 4> src{};
  std::array<uint8_t *, 4> dst{};
  std::array<size_t, 4> len{};
  int n = 0;
};
Runs runs(const QsaAttnPools &dst, const KvHostPools &src, int fmt,
          const QsaShapes &s) {
  if (s.n_head_kv <= 0 || s.page_size <= 0 || s.head_dim <= 0 ||
      s.head_dim > INT_MAX ||
      (fmt != kKvF16 && fmt != kKvInt8 && fmt != kKvQ4) ||
      (fmt == kKvInt8 && s.head_dim % 64) || (fmt == kKvQ4 && s.head_dim % 32))
    throw std::invalid_argument("invalid SYCL KV streaming format/shape");
  const size_t rows = checked_count(s.n_head_kv, s.page_size);
  Runs r;
  auto add = [&](const void *p, const void *q, size_t rowbytes) {
    r.src[r.n] = static_cast<const uint8_t *>(p);
    r.dst[r.n] = const_cast<uint8_t *>(static_cast<const uint8_t *>(q));
    r.len[r.n++] = bytes(rows, rowbytes);
  };
  if (fmt == kKvF16) {
    add(src.k_pool, dst.k_pool, size_t(s.head_dim) * 2);
    add(src.v_pool, dst.v_pool, size_t(s.head_dim) * 2);
  } else if (fmt == kKvInt8) {
    add(src.k_q, dst.k_q, s.head_dim);
    add(src.v_q, dst.v_q, s.head_dim);
    add(src.k_scale, dst.k_scale, size_t(s.head_dim / 64) * 2);
    add(src.v_scale, dst.v_scale, size_t(s.head_dim / 64) * 2);
  } else {
    add(src.k_q4, dst.k_q4, kv_q4_bytes_per_head(int(s.head_dim)));
    add(src.v_q4, dst.v_q4, kv_q4_bytes_per_head(int(s.head_dim)));
  }
  return r;
}
void run_spans(Runs r, int64_t dstcount, int64_t srccount, Spans &w,
               Spans &rd) {
  for (int i = 0; i < r.n; ++i) {
    w.push_back({r.dst[i], bytes(dstcount, r.len[i])});
    rd.push_back({r.src[i], bytes(srccount, r.len[i])});
  }
}
Spans map_spans(const KvStreamMap &m) {
  if (m.n_blocks <= 0 || m.n_blocks > INT_MAX || m.n_slots <= 0 ||
      m.n_slots > INT_MAX)
    throw std::invalid_argument("invalid SYCL KV residency capacity");
  return {{m.page_table, bytes(m.n_blocks, 4)},
          {m.slot_block, bytes(m.n_slots, 4)},
          {m.slot_stamp, bytes(m.n_slots, 4)},
          {m.slot_ref, bytes(m.n_slots, 4)},
          {m.ctl, kKvCtlInts * 4},
          {m.miss_block, bytes(m.n_slots, 4)},
          {m.miss_slot, bytes(m.n_slots, 4)}};
}
uint64_t counter(const int32_t *c) {
  return uint64_t(uint32_t(c[0])) | (uint64_t(uint32_t(c[1])) << 32);
}
void increment(int32_t *c, uint64_t amount) {
  const uint64_t value = counter(c) + amount;
  c[0] = sycl::bit_cast<int32_t>(uint32_t(value));
  c[1] = sycl::bit_cast<int32_t>(uint32_t(value >> 32));
}
} // namespace
uint64_t kv_block_bytes(const QsaShapes &s, int fmt) {
  const auto r = runs({}, {}, fmt, s);
  uint64_t total = 0;
  for (int i = 0; i < r.n; ++i) {
    if (r.len[i] > UINT64_MAX - total)
      throw std::invalid_argument("KV block overflow");
    total += r.len[i];
  }
  return total;
}
void kv_stream_reset(const KvStreamMap &m, void *stream) {
  spans(map_spans(m));
  const size_t count = std::max<int64_t>({m.n_blocks, m.n_slots, kKvCtlInts});
  auto event = queue_for(stream).parallel_for(sycl::range<1>(count),
                                              [=](sycl::id<1> id) {
                                                const size_t i = id[0];
                                                if (i < size_t(m.n_blocks))
                                                  m.page_table[i] = -1;
                                                if (i < size_t(m.n_slots)) {
                                                  m.slot_block[i] = -1;
                                                  m.slot_stamp[i] = -1;
                                                  m.slot_ref[i] = 0;
                                                }
                                                if (i < kKvCtlInts)
                                                  m.ctl[i] = 0;
                                              });
  finish(stream, event);
}
void kv_stream_resolve(const KvStreamMap &m, const QsaAttnPools &slots,
                       const KvHostPools &host, int fmt, const int32_t *ids,
                       const int32_t *steps, int64_t nq, int64_t cap,
                       const QsaShapes &s, void *stream) {
  if (nq <= 0)
    return;
  if (nq > INT_MAX || cap <= 0 || cap > INT_MAX)
    throw std::invalid_argument("invalid KV selection dimensions");
  auto w = map_spans(m);
  Spans rd{{ids, bytes(checked_count(nq, cap), 4)},
           {steps, bytes(checked_count(nq, kStepCount), 4)}};
  const auto r = runs(slots, host, fmt, s);
  run_spans(r, m.n_slots, m.n_blocks, w, rd);
  spans(w, rd);
  auto &q = queue_for(stream);
  // Deterministic baseline policy. Mark every hit before choosing victims;
  // bounded miss storage also makes an over-capacity selection safe.
  q.single_task([=] {
    int epoch = m.ctl[0];
    if (epoch < 0 || epoch == INT_MAX) {
      for (int64_t i = 0; i < m.n_slots; ++i)
        m.slot_stamp[i] = -1;
      epoch = 0;
    }
    ++epoch;
    int need = 0;
    uint64_t lookups = 0;
    for (int64_t qi = 0; qi < nq; ++qi) {
      const int width = steps[qi * kStepCount + kStepWidth];
      if (width < 0 || width > cap) {
        m.ctl[3] = 1;
        continue;
      }
      int64_t previous = -1;
      for (int i = 0; i < width; ++i) {
        const int cell = ids[qi * cap + i];
        if (cell < 0 || int64_t(cell) / s.page_size >= m.n_blocks) {
          m.ctl[3] = 1;
          continue;
        }
        const int64_t block = cell / s.page_size;
        if (block == previous)
          continue;
        previous = block;
        ++lookups;
        const int slot = m.page_table[block];
        if (slot >= 0 && slot < m.n_slots) {
          m.slot_stamp[slot] = epoch;
          m.slot_ref[slot] = 1;
        } else if (slot == -1) {
          if (need == m.n_slots) {
            m.ctl[3] = 1;
            continue;
          }
          m.page_table[block] = -2;
          m.miss_block[need++] = int(block);
        } else if (slot != -2)
          m.ctl[3] = 1;
      }
    }
    int hand = m.ctl[1];
    if (hand < 0 || hand >= m.n_slots)
      hand = 0;
    int placed = 0;
    for (int k = 0; k < need; ++k) {
      int victim = -1;
      for (int64_t scanned = 0; scanned < 2 * m.n_slots; ++scanned) {
        const int sl = hand;
        hand = int((int64_t(hand) + 1) % m.n_slots);
        if (m.slot_stamp[sl] == epoch)
          continue;
        if (m.slot_block[sl] < 0 || m.slot_ref[sl] == 0) {
          victim = sl;
          break;
        }
        m.slot_ref[sl] = 0;
      }
      const int b = m.miss_block[k];
      if (victim < 0) {
        m.page_table[b] = -1;
        m.ctl[3] = 1;
        continue;
      }
      const int old = m.slot_block[victim];
      if (old >= 0 && old < m.n_blocks)
        m.page_table[old] = -1;
      m.page_table[b] = victim;
      m.slot_block[victim] = b;
      m.slot_stamp[victim] = epoch;
      m.slot_ref[victim] = 1;
      m.miss_block[placed] = b;
      m.miss_slot[placed++] = victim;
    }
    m.ctl[0] = epoch;
    m.ctl[1] = hand;
    m.ctl[2] = placed;
    increment(m.ctl + 4, placed);
    increment(m.ctl + 6, lookups);
    increment(m.ctl + 8, 1);
  });
  auto event = q.parallel_for(
      sycl::nd_range<1>(size_t(m.n_slots) * 128, 128),
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_group_linear_id(), t = it.get_local_linear_id();
        if (k >= size_t(m.ctl[2]))
          return;
        const size_t b = m.miss_block[k], slot = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a)
          for (size_t i = t; i < r.len[a]; i += 128)
            r.dst[a][slot * r.len[a] + i] = r.src[a][b * r.len[a] + i];
      });
  finish(stream, event);
}
void kv_ring_table(int32_t *table, int64_t blocks, int64_t slots,
                   void *stream) {
  if (blocks <= 0 || slots <= 0 || slots > INT_MAX)
    throw std::invalid_argument("invalid KV ring capacity");
  validate_spans({{table, bytes(blocks, 4)}}, {});
  auto event = queue_for(stream).parallel_for(
      sycl::range<1>(size_t(blocks)),
      [=](sycl::id<1> i) { table[i] = int32_t(i[0] % slots); });
  finish(stream, event);
}
void kv_ring_restore(const QsaAttnPools &slots, const KvHostPools &host,
                     int fmt, int64_t b0, int64_t b1, int64_t nslots,
                     const QsaShapes &s, void *stream) {
  if (b0 < 0 || b1 < b0 || nslots <= 0)
    throw std::invalid_argument("invalid KV ring range");
  if (b0 == b1)
    return;
  const auto r = runs(slots, host, fmt, s);
  Spans w, rd;
  run_spans(r, nslots, b1, w, rd);
  spans(w, rd);
  auto &q = queue_for(stream);
  sycl::event event;
  for (int64_t b = b0; b < b1;) {
    const int64_t sl = b % nslots, count = std::min(b1 - b, nslots - sl);
    for (int a = 0; a < r.n; ++a)
      event = q.memcpy(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a],
                       bytes(count, r.len[a]));
    b += count;
  }
  finish(stream, event);
}
void kv_stage_from_host(const QsaAttnPools &dst, const KvHostPools &src,
                        int fmt, int64_t blocks, const QsaShapes &s,
                        void *stream) {
  if (blocks <= 0)
    return;
  const auto r = runs(dst, src, fmt, s);
  Spans w, rd;
  run_spans(r, blocks, blocks, w, rd);
  spans(w, rd);
  auto &q = queue_for(stream);
  sycl::event event;
  for (int a = 0; a < r.n; ++a)
    event = q.memcpy(r.dst[a], r.src[a], bytes(blocks, r.len[a]));
  finish(stream, event);
}
KvStreamCounters kv_stream_counters(const KvStreamMap &m) {
  if (!m.ctl)
    return {};
  validate_spans({}, {{m.ctl, kKvCtlInts * 4}});
  auto rt = runtime_for();
  rt->wait();
  int32_t values[kKvCtlInts];
  rt->wait(rt->compute().memcpy(values, m.ctl, sizeof(values)));
  return {counter(values + 4), counter(values + 6), counter(values + 8),
          values[3] != 0};
}
} // namespace strata::kernels
