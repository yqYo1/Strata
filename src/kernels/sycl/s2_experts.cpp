// Canonical S2 experts use the pack's interleaved gate/up rows. Metadata and
// counts stay on the device, including during graph replay.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <cstdlib>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int H = 2560, FF = 640;
constexpr size_t DC = 1280 * 640, GS = DC + 2560 * 160, DS = GS + 1280 * 40 * 2,
                 BLOB = DS + 2560 * 10 * 2;
std::atomic<int> selected{-1};
thread_local int last_path = -1;
size_t aligned(size_t n) { return (n + 15) & ~size_t(15); }
float half_at(const uint8_t *p) {
  return f32_from_f16(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}
int code4(uint8_t c, const int8_t *x) {
  int dot = 0;
  for (int i = 0; i < 4; ++i)
    dot += int((c >> (2 * i)) & 3) * int(x[i]);
  return dot;
}
float sum32(sycl::sub_group sg, float x) {
  for (int d = 16; d; d >>= 1)
    x += sycl::permute_group_by_xor(sg, x, d);
  return x;
}
bool old() {
  int s = selected.load();
  if (s >= 0)
    return s != 0;
  static bool env = [] {
    auto p = std::getenv("STRATA_OLD_GROUPED");
    return p && p[0] == '1';
  }();
  return env;
}
struct Meta {
  const uint8_t *base = nullptr;
  const int32_t *slots = nullptr;
  size_t blob = 0;
  const unsigned long long *ptr = nullptr;
  const int32_t *start = nullptr;
  const int32_t *count = nullptr;
  const int32_t *dst = nullptr;
  const int32_t *tok = nullptr;
  int groups = 0, entries = 0, k = 0;
};
struct Work {
  float *gu;
  uint8_t *q;
  float *scale;
  size_t bytes;
};
void require(bool value) {
  if (!value)
    throw std::invalid_argument("invalid SYCL canonical expert arguments");
}
Work workspace(void *scratch, int64_t n) {
  size_t b = moe_hit_grouped_scratch_bytes(n, H, FF);
  require(b && scratch);
  auto *p = static_cast<uint8_t *>(scratch);
  size_t gb = aligned(size_t(n) * 2 * FF * 4),
         qb = aligned(size_t(n) * (FF / 32) * 34);
  return {reinterpret_cast<float *>(p), p + gb,
          reinterpret_cast<float *>(p + gb + qb), b};
}
void validate(const Meta &m, const uint8_t *x, const float *xs, void *scratch,
              float *out) {
  require(m.groups > 0 && m.entries > 0 && m.dst &&
          (m.ptr || (m.base && m.slots && m.blob >= BLOB)));
  auto w = workspace(scratch, m.entries);
  require(uintptr_t(scratch) % 4 == 0 && uintptr_t(out) % 4 == 0);
  validate_spans({{scratch, w.bytes}, {out, H * 4}},
                 {{x, (H / 32) * 34}, {m.dst, size_t(m.entries) * 4}}, 2);
  if (xs)
    validate_spans({{scratch, w.bytes}, {out, H * 4}}, {{xs, (H / 32) * 4}});
  if (m.count)
    validate_spans({{scratch, w.bytes}, {out, H * 4}}, {{m.count, 4}});
  if (m.ptr) {
    require(m.start && m.tok && uintptr_t(m.ptr) % 8 == 0);
    validate_spans({{scratch, w.bytes}, {out, H * 4}},
                   {{m.ptr, size_t(m.groups) * 8},
                    {m.start, size_t(m.groups + 1) * 4},
                    {m.tok, size_t(m.entries) * 4}});
  } else
    validate_spans({{scratch, w.bytes}, {out, H * 4}},
                   {{m.base, m.blob}, {m.slots, size_t(m.groups) * 4}});
}
template <bool Down, int Tile>
void projection(Meta m, const uint8_t *act, const float *scales, float *out,
                void *stream) {
  constexpr int NI = Down ? FF : H, NO = Down ? H : 2 * FF;
  queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(m.groups) * NO * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        int group = it.get_group_linear_id() / NO,
            row = it.get_group_linear_id() % NO,
            lane = it.get_local_linear_id();
        if (m.count &&
            (*m.count < 0 || *m.count > m.groups || group >= *m.count))
          return;
        int begin = m.ptr ? m.start[group] : group,
            end = m.ptr ? m.start[group + 1] : group + 1;
        if (begin < 0 || end < begin || end > m.entries)
          return;
        const uint8_t *blob;
        if (m.ptr) {
          blob = reinterpret_cast<const uint8_t *>(m.ptr[group]);
          if (!blob)
            return;
        } else {
          int slot = m.slots[group];
          if (slot < 0)
            return;
          blob = m.base + size_t(slot) * m.blob;
        }
        const uint8_t *codes = blob + (Down ? DC : 0) + size_t(row) * (NI / 4);
        const uint8_t *ws =
            blob + (Down ? DS : GS) + size_t(row) * (NI / 64) * 2;
        for (int tile = begin; tile < end; tile += Tile) {
          float acc[Tile] = {};
          int token[Tile];
          bool live[Tile];
          for (int e = 0; e < Tile; ++e) {
            int i = tile + e;
            live[e] = i < end && m.dst[i] >= 0;
            token[e] = 0;
            if (live[e])
              token[e] =
                  Down ? i : (m.ptr ? m.tok[i] : (m.k ? m.dst[i] / m.k : 0));
            live[e] = live[e] && token[e] >= 0;
          }
          for (int chunk = lane; chunk < NI / 32; chunk += 32) {
            uint8_t packed[8];
            for (int j = 0; j < 8; ++j)
              packed[j] = codes[chunk * 8 + j];
            float dw = half_at(ws + (chunk / 2) * 2);
            for (int e = 0; e < Tile; ++e)
              if (live[e]) {
                const uint8_t *xb =
                    act + (size_t(token[e]) * (NI / 32) + chunk) * 34;
                const auto *xq = reinterpret_cast<const int8_t *>(xb + 2);
                int dot = 0, hx = 0;
                for (int j = 0; j < 8; ++j) {
                  dot += code4(packed[j], xq + 4 * j);
                  for (int k = 0; k < 4; ++k)
                    hx += xq[4 * j + k];
                }
                float dx = scales ? scales[size_t(token[e]) * (NI / 32) + chunk]
                                  : half_at(xb);
                acc[e] = sycl::fma(dw * dx, float(dot - hx), acc[e]);
              }
          }
          for (int e = 0; e < Tile; ++e) {
            float result = sum32(it.get_sub_group(), acc[e]);
            if (lane == 0 && live[e]) {
              size_t at = Down ? size_t(m.dst[tile + e]) * H + row
                               : size_t(row & 1) * m.entries * FF +
                                     size_t(tile + e) * FF + row / 2;
              out[at] = result;
            }
          }
        }
      });
}
void run(Meta m, const uint8_t *x, const float *xs, void *scratch, float *out,
         void *stream) {
  validate(m, x, xs, scratch, out);
  auto work = workspace(scratch, m.entries);
  auto &q = queue_for(stream);
  void *s = &q;
  bool one = old() || uintptr_t(x) % 4 ||
             (m.base && (uintptr_t(m.base) % 8 || m.blob % 8));
  last_path = one ? 0 : 1;
  q.memset(work.gu, 0, size_t(m.entries) * 2 * FF * 4);
  if (one)
    projection<false, 1>(m, x, xs, work.gu, s);
  else
    projection<false, 8>(m, x, xs, work.gu, s);
  const size_t pairs = size_t(m.entries) * FF;
  for_each(pairs, s, [=](size_t i) {
    float g = work.gu[i];
    work.gu[i] = (g / (1.f + sycl::exp(-g))) * work.gu[pairs + i];
  });
  if (xs)
    quantize_q8_0_scaled(work.gu, work.q, work.scale, pairs, s);
  else
    quantize_q8_0(work.gu, work.q, pairs, s);
  if (one)
    projection<true, 1>(m, work.q, xs ? work.scale : nullptr, out, s);
  else
    projection<true, 8>(m, work.q, xs ? work.scale : nullptr, out, s);
  if (!stream)
    runtime_for()->wait(q.ext_oneapi_submit_barrier());
}
float cpu_dot(const uint8_t *codes, const uint8_t *ws, const uint8_t *x,
              const float *xs, int blocks, int lane, sycl::sub_group sg) {
  float acc = 0, corr = 0;
  for (int b = 0; b < blocks; ++b) {
    float d = half_at(ws + 2 * b);
    const auto *lo =
        reinterpret_cast<const int8_t *>(x + size_t(2 * b) * 34 + 2);
    const auto *hi = lo + 34;
    acc = sycl::fma(d * xs[2 * b],
                    float(code4(codes[b * 16 + lane], lo + lane * 4)), acc);
    acc = sycl::fma(d * xs[2 * b + 1],
                    float(code4(codes[b * 16 + 8 + lane], hi + lane * 4)), acc);
    if (lane == 0) {
      int a = 0, z = 0;
      for (int j = 0; j < 32; ++j) {
        a += lo[j];
        z += hi[j];
      }
      float h0 = xs[2 * b] * float(a), h1 = xs[2 * b + 1] * float(z);
      corr += d * (h0 + h1);
    }
  }
  for (int delta : {4, 1, 2})
    acc += sycl::shift_group_left(sg, acc, delta);
  return acc - corr;
}
template <bool Down>
void cpu_projection(Meta m, const uint8_t *x, const float *xs, float *out,
                    void *stream) {
  constexpr int NI = Down ? FF : H, NO = Down ? H : 2 * FF;
  queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(m.entries) * NO * 8, 128),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        size_t index = it.get_global_linear_id() / 8;
        int lane = it.get_local_linear_id() % 8, entry = index / NO,
            row = index % NO, slot = m.slots[entry];
        if (slot < 0 || m.dst[entry] < 0)
          return;
        const auto *blob = m.base + size_t(slot) * m.blob;
        size_t chunk = Down ? size_t(entry) * (FF / 32) : 0;
        float result = cpu_dot(
            blob + (Down ? DC : 0) + size_t(row) * (NI / 4),
            blob + (Down ? DS : GS) + size_t(row) * (NI / 64) * 2,
            x + chunk * 34, xs + chunk, NI / 64, lane, it.get_sub_group());
        if (!lane)
          out[Down ? size_t(m.dst[entry]) * H + row
                   : size_t(row & 1) * m.entries * FF + size_t(entry) * FF +
                         row / 2] = result;
      });
}
Meta hit(const uint8_t *b, const int32_t *slot, const int32_t *dst, int64_t cap,
         int64_t blob, const int32_t *count = nullptr, int k = 0) {
  require(cap > 0 && cap <= INT32_MAX / (2 * FF) && blob >= int64_t(BLOB));
  Meta m;
  m.base = b;
  m.slots = slot;
  m.dst = dst;
  m.blob = blob;
  m.groups = m.entries = int(cap);
  m.count = count;
  m.k = k;
  return m;
}
} // namespace
uint64_t moe_hit_grouped_scratch_bytes(int64_t n, int64_t width, int64_t ff) {
  if (n <= 0)
    return 0;
  if (width != H || ff != FF || n > INT32_MAX / (2 * FF))
    return 0;
  return aligned(size_t(n) * 2 * FF * 4) + aligned(size_t(n) * (FF / 32) * 34) +
         2 * aligned(size_t(n) * (FF / 32) * 4) + aligned((H / 32) * 4);
}
void moe_grouped_select_old(int value) {
  selected.store(value < 0 ? -1 : !!value);
}
int moe_grouped_last_path() {
  int v = last_path;
  last_path = -1;
  return v;
}
void moe_hit_grouped_s2(const uint8_t *b, const int32_t *slot,
                        const int32_t *dst, int64_t n, int64_t bytes,
                        const uint8_t *x, void *w, float *y, void *s,
                        const float *xs) {
  if (n <= 0)
    return;
  run(hit(b, slot, dst, n, bytes), x, xs, w, y, s);
}
void moe_hit_grouped_s2_dev(const uint8_t *b, const int32_t *slot,
                            const int32_t *dst, const int32_t *count, int64_t n,
                            int64_t bytes, const uint8_t *x, void *w, float *y,
                            void *s, const float *xs) {
  if (n <= 0)
    return;
  require(count);
  run(hit(b, slot, dst, n, bytes, count), x, xs, w, y, s);
}
void moe_hit_grouped_s2_multi(const uint8_t *b, const int32_t *slot,
                              const int32_t *dst, const int32_t *count,
                              int64_t n, int64_t bytes, const uint8_t *x,
                              const float *xs, int k, void *w, float *y,
                              void *s) {
  if (n <= 0)
    return;
  require(count && k > 0);
  run(hit(b, slot, dst, n, bytes, count, k), x, xs, w, y, s);
}
void moe_grouped_s2(const unsigned long long *ptr, const int32_t *start,
                    const int32_t *count, const int32_t *dst,
                    const int32_t *tok, int64_t groups, int64_t entries,
                    const uint8_t *x, const float *xs, void *w, float *y,
                    void *s) {
  if (groups <= 0 || entries <= 0)
    return;
  require(groups <= entries && entries <= INT32_MAX / (2 * FF) && count);
  Meta m;
  m.ptr = ptr;
  m.start = start;
  m.count = count;
  m.dst = dst;
  m.tok = tok;
  m.groups = int(groups);
  m.entries = int(entries);
  run(m, x, xs, w, y, s);
}
void moe_hit_grouped_s2_cpu_order(const uint8_t *b, const int32_t *slot,
                                  const int32_t *dst, int64_t n, int64_t bytes,
                                  const uint8_t *x, void *scratch, float *y,
                                  void *stream, const float *xs, float *trace) {
  if (n <= 0)
    return;
  require(xs);
  auto m = hit(b, slot, dst, n, bytes);
  validate(m, x, xs, scratch, y);
  auto w = workspace(scratch, n);
  size_t pairs = size_t(n) * FF;
  if (trace)
    validate_spans(
        {{trace, pairs * 8}, {scratch, w.bytes}, {y, H * 4}},
        {{b, size_t(bytes)}, {x, (H / 32) * 34}, {xs, (H / 32) * 4}});
  auto &q = queue_for(stream);
  void *s = &q;
  q.memset(w.gu, 0, pairs * 8);
  cpu_projection<false>(m, x, xs, w.gu, s);
  if (trace)
    q.memcpy(trace, w.gu, pairs * 8);
  for_each(pairs, s, [=](size_t i) {
    float g = w.gu[i];
    w.gu[i] = (g / (1.f + sycl::exp(-g))) * w.gu[pairs + i];
  });
  quantize_q8_0_scaled(w.gu, w.q, w.scale, pairs, s);
  cpu_projection<true>(m, w.q, w.scale, y, s);
  last_path = 0;
  if (!stream)
    runtime_for()->wait(q.ext_oneapi_submit_barrier());
}
} // namespace strata::kernels
