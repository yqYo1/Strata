// GPU routing and direct packed expert products. The activation ABI is shared
// with the CUDA backend; packed integer dots use portable SYCL SIMD arithmetic.
#include "strata/artifact/dequant.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/sycl/gguf_decode.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <sycl/ext/intel/math.hpp>

namespace strata::prefill::fused {
namespace {
using namespace sycl_backend;
constexpr int AB = 80, WG = 256;
constexpr size_t GC = 1280 * 640, DC = 2560 * 160, GS = GC + DC,
                 DS = GS + 1280 * 40 * 2, BLOB = DS + 2560 * 10 * 2;
struct Tile {
  int expert, row;
};
struct Tables {
  int32_t *cnt, *off, *fill, *ts;
  Tile *tiles;
};
size_t tiles_at(int e) { return (checked_count(e, 16) + 8 + 15) / 16 * 16; }
Tables tables(void *scratch, int e) {
  auto *p = static_cast<int32_t *>(scratch);
  return {
      p, p + e, p + 2 * e + 1, p + 3 * e + 1,
      reinterpret_cast<Tile *>(static_cast<uint8_t *>(scratch) + tiles_at(e))};
}
int perm32(int k) { return 8 * (k & 3) + 4 * (k >> 4) + ((k & 15) >> 2); }
bool flag(const char *name, bool def) {
  const char *v = std::getenv(name);
  return v ? v[0] != '0' : def;
}
bool gu_covered(int t) {
  return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 23;
}
bool d_covered(int t) { return t == 20 || t == 42; }
void dims(int64_t n, int e) {
  if (n < 0 || n > INT_MAX || e <= 0 || e > 1024)
    throw std::invalid_argument("invalid fused MoE grouping dimensions");
}
template <bool Native>
void store_act(sycl::sub_group sg, uint8_t *out, int lane, int half,
               float value) {
  float m =
      sycl::reduce_over_group(sg, sycl::fabs(value), sycl::maximum<float>());
  int q =
      m > 0 ? int(sycl::rint(value * sycl::ext::intel::math::fdiv_rn(127.f, m)))
            : 0;
  out[half * 32 + (Native ? lane : perm32(lane))] = uint8_t(int8_t(q));
  int sum = sycl::reduce_over_group(sg, q, sycl::plus<int>());
  if (lane == 0) {
    auto *sc = reinterpret_cast<float *>(out + 64);
    sc[Native ? half : half * 2] = sycl::ext::intel::math::fdiv_rn(m, 127.f);
    if constexpr (!Native)
      sc[half * 2 + 1] = -(12582912.f + float(sum));
    else
      sc[2 + half] = 0.f;
  }
}
template <bool Native>
void quantize(const float *x, int64_t rows, int64_t cols, void *out,
              void *stream) {
  if (rows == 0)
    return;
  if (rows < 0 || cols <= 0 || cols % 64)
    throw std::invalid_argument("invalid fused activation dimensions");
  const size_t blocks = checked_count(rows, cols / 64);
  validate_spans({{out, act_bytes(rows, cols)}},
                 {{x, checked_count(checked_count(rows, cols), 4)}});
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>((blocks + 7) / 8 * WG, WG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t b = it.get_global_linear_id() / 32;
        const int lane = it.get_local_linear_id() % 32;
        if (b >= blocks)
          return;
        auto sg = it.get_sub_group();
        auto *a = static_cast<uint8_t *>(out) + b * AB;
        store_act<Native>(sg, a, lane, 0, x[b * 64 + lane]);
        store_act<Native>(sg, a, lane, 1, x[b * 64 + 32 + lane]);
      });
  finish(stream, event);
}
int sign_code(int code, unsigned signs, int i) {
  return (signs >> i) & 1 ? -code : code;
}
// One 32-value GGUF unit, as signed int8 codes and two FP32 scales.
template <int T>
void unpack(const uint8_t *row, int unit, int8_t (&codes)[32], float (&sc)[2]) {
  constexpr int BW = T == 20 ? 32 : T == 42 ? 64 : 256;
  constexpr int BB = T == 16   ? 66
                     : T == 17 ? 74
                     : T == 18 ? 98
                     : T == 21 ? 110
                     : T == 22 ? 82
                     : T == 23 ? 136
                               : 18;
  const uint8_t *b = row + size_t(unit / (BW / 32)) * BB;
  const int g = unit % (BW / 32);
  const float d = fp16_to_fp32(iq_read_u16(b));
  if constexpr (T == 16) {
    const uint32_t extra = iq_read_u32(b + 6 + 8 * g);
    sc[0] = sc[1] = d * float(2 * (extra >> 28) + 1) * .125f;
    for (int i = 0; i < 32; ++i) {
      const int group = i / 8;
      const auto grid = iq2_xxs_grid[b[2 + 8 * g + group]];
      unsigned s = iq_even_parity_signs(extra >> (7 * group));
      codes[i] =
          int8_t(sign_code(int((grid >> (8 * (i % 8))) & 255), s, i % 8));
    }
  } else if constexpr (T == 17 || T == 22) {
    const int scales = b[(T == 17 ? 66 : 74) + g];
    sc[0] = d * float(2 * (scales & 15) + 1) * .125f;
    sc[1] = d * float(2 * (scales >> 4) + 1) * .125f;
    for (int i = 0; i < 32; ++i) {
      const int group = i / 8;
      uint64_t grid;
      unsigned s;
      if constexpr (T == 17) {
        const unsigned q = iq_read_u16(b + 2 + 8 * g + 2 * group);
        grid = iq2_xs_grid[q & 511];
        s = iq_even_parity_signs(q >> 9);
      } else {
        const unsigned index =
            b[2 + 4 * g + group] | (((b[66 + g] >> (2 * group)) & 3) << 8);
        grid = iq2_s_grid[index];
        s = b[34 + 4 * g + group];
      }
      codes[i] =
          int8_t(sign_code(int((grid >> (8 * (i % 8))) & 255), s, i % 8));
    }
  } else if constexpr (T == 18 || T == 21) {
    unsigned extra = T == 18 ? iq_read_u32(b + 66 + 4 * g) : 0;
    unsigned ss =
        T == 18 ? extra >> 28 : (b[106 + g / 2] >> (4 * (g % 2))) & 15;
    sc[0] = sc[1] = d * float(2 * ss + 1) * (T == 18 ? .25f : 1.f);
    for (int i = 0; i < 32; ++i) {
      const int word = i / 4;
      unsigned index = b[2 + 8 * g + word];
      if constexpr (T == 21)
        index |= ((b[66 + g] >> word) & 1) << 8;
      const uint32_t grid = T == 18 ? iq3_xxs_grid[index] : iq3_s_grid[index];
      unsigned s = T == 18 ? iq_even_parity_signs(extra >> (7 * (i / 8)))
                           : b[74 + 4 * g + i / 8];
      codes[i] =
          int8_t(sign_code(int((grid >> (8 * (i % 4))) & 255), s, i % 8));
    }
  } else if constexpr (T == 20 || T == 23) {
    const int ls = T == 23 ? int(((b[4 + g / 2] >> (4 * (g % 2))) & 15) |
                                 (((iq_read_u16(b + 2) >> (2 * g)) & 3) << 4)) -
                                 32
                           : 1;
    sc[0] = sc[1] = d * float(ls);
    const auto *q = b + (T == 23 ? 8 + 16 * g : 2);
    for (int i = 0; i < 32; ++i)
      codes[i] = kvalues_iq4nl[(q[i % 16] >> (i < 16 ? 0 : 4)) & 15];
  } else {
    sc[0] = sc[1] = d;
    for (int i = 0; i < 32; ++i) {
      int k = g * 32 + i;
      codes[i] = int8_t(int((b[2 + k / 4] >> (2 * (k % 4))) & 3) - 1);
    }
  }
}
template <int T, bool Native>
float dot(const uint8_t *blob, const NativeGeom &geo, bool gu, int feature,
          bool up, const uint8_t *act) {
  const int k = gu ? 2560 : 640;
  const uint8_t *row = nullptr;
  if constexpr (Native)
    row = blob + (gu ? (up ? geo.up_off : 0) + size_t(feature) * geo.gu_row
                     : geo.down_off + size_t(feature) * geo.d_row);
  float result = 0;
  for (int u = 0; u < k / 32; ++u) {
    int8_t w[32];
    float ws[2];
    if constexpr (Native)
      unpack<T>(row, u, w, ws);
    else {
      const int wr = gu ? 2 * feature + int(up) : feature;
      const int block = u / 2;
      const size_t co = gu ? 0 : GC, so = gu ? GS : DS;
      const int stride = gu ? 640 : 160, blocks = gu ? 40 : 10;
      ws[0] = ws[1] = fp16_to_fp32(
          iq_read_u16(blob + so + (size_t(wr) * blocks + block) * 2));
      for (int i = 0; i < 32; ++i) {
        const int col = (u % 2) * 32 + i;
        w[i] =
            int8_t(int((blob[co + size_t(wr) * stride + block * 16 + col / 4] >>
                        (2 * (col % 4))) &
                       3) -
                   1);
      }
    }
    const auto *a = act + (u / 2) * AB;
    const auto *sc = reinterpret_cast<const float *>(a + 64);
    const float dx = sc[Native ? u % 2 : 2 * (u % 2)];
    int sums[2] = {};
    for (int i = 0; i < 32; ++i)
      sums[i / 16] +=
          int(w[i]) * int(int8_t(a[(u % 2) * 32 + (Native ? i : perm32(i))]));
    result += dx * (ws[0] * float(sums[0]) + ws[1] * float(sums[1]));
  }
  return result;
}
template <int T, bool Native, bool GU>
void project(const Batch &batch, const NativeGeom &geo, int e, int64_t n,
             const void *scratch, const void *activation, const int32_t *src,
             void *hidden, float *dm, void *stream) {
  const Tables tb = tables(const_cast<void *>(scratch), e);
  constexpr int features = GU ? 640 : 2560, blocks = features / 32;
  const size_t tiles = (size_t(n) + 63) / 64 + (batch.e1 - batch.e0);
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(tiles * blocks * WG, WG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t item = it.get_group_linear_id();
        const int tile = tb.ts[batch.e0] + int(item / blocks),
                  fb = item % blocks;
        if (tile >= tb.ts[batch.e1])
          return;
        const Tile t = tb.tiles[tile];
        const uint8_t *blob = batch.blob[t.expert - batch.e0];
        const int lane = it.get_local_linear_id() % 32,
                  warp = it.get_local_linear_id() / 32;
        auto sg = it.get_sub_group();
        for (int r = t.row + warp;
             r < sycl::min(t.row + 64, tb.off[t.expert + 1]); r += 8) {
          const int ar = GU ? src[r] : r;
          const auto *act = static_cast<const uint8_t *>(activation) +
                            size_t(ar) * (GU ? 40 : 10) * AB;
          const int f = fb * 32 + lane;
          float v = dot<T, Native>(blob, geo, GU, f, false, act);
          if constexpr (GU) {
            float up = dot<T, Native>(blob, geo, true, f, true, act);
            v = (v / (1.f + sycl::exp(-v))) * up;
            auto *out =
                static_cast<uint8_t *>(hidden) + (size_t(r) * 10 + f / 64) * AB;
            store_act<Native>(sg, out, lane, (f / 32) % 2, v);
          } else
            dm[size_t(r) * 2560 + f] = v;
        }
      });
  finish(stream, event);
}
void validate_batch(const Batch &b, int e, int64_t n, const void *scratch,
                    const void *xa, const int32_t *src, void *ha, float *dm,
                    size_t blob_bytes) {
  dims(n, e);
  if (b.e0 < 0 || b.e1 > e || b.e1 < b.e0 || b.e1 - b.e0 > kMaxBatch)
    throw std::invalid_argument("invalid fused expert batch");
  validate_spans(
      {{ha, act_bytes(n, 640)}, {dm, checked_count(n, 2560 * 4)}},
      {{scratch, group_bytes(n, e)}, {xa, 80}, {src, checked_count(n, 4)}});
  for (int i = 0; i < b.e1 - b.e0; ++i)
    if (b.blob[i])
      validate_spans(
          {{ha, act_bytes(n, 640)}, {dm, checked_count(n, 2560 * 4)}},
          {{b.blob[i], blob_bytes}}, 2);
}
} // namespace
bool built() { return true; }
bool available() {
  const auto dev = queue_for(nullptr).get_device();
  auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  return dev.has(sycl::aspect::fp16) &&
         dev.get_info<sycl::info::device::max_work_group_size>() >= WG &&
         std::find(sizes.begin(), sizes.end(), 32) != sizes.end();
}
bool enabled() {
  static const bool env = flag("STRATA_PF_FUSED", true);
  return env && available();
}
bool requested() {
  static const bool env = [] {
    const char *v = std::getenv("STRATA_PF_FUSED");
    return v && v[0] == '1';
  }();
  return env && available();
}
bool native_supported(int gu, int down) {
  static const bool on = flag("STRATA_PF_FUSED_NATIVE", true);
  return requested() && on && gu_covered(gu) && d_covered(down);
}
size_t act_bytes(int64_t rows, int64_t cols) {
  if (cols <= 0 || cols % 64)
    throw std::invalid_argument(
        "fused activation columns must be multiples of 64");
  return checked_count(checked_count(rows, cols / 64), AB);
}
size_t group_bytes(int64_t n, int e) {
  dims(n, e);
  return tiles_at(e) + ((size_t(n) + 63) / 64 + size_t(e)) * sizeof(Tile);
}
void quantize_act(const float *x, int64_t rows, int64_t cols, void *xa,
                  void *stream) {
  quantize<false>(x, rows, cols, xa, stream);
}
void quantize_act_native(const float *x, int64_t rows, int64_t cols, void *xa,
                         void *stream) {
  quantize<true>(x, rows, cols, xa, stream);
}
void group(const int32_t *ids, int64_t n, int k, int e, void *scratch,
           int32_t *slot, int32_t *src, void *stream) {
  dims(n, e);
  if (!n)
    return;
  if (k <= 0 || n % k)
    throw std::invalid_argument("invalid fused routing top-k");
  validate_spans({{scratch, group_bytes(n, e)},
                  {slot, checked_count(n, 4)},
                  {src, checked_count(n, 4)}},
                 {{ids, checked_count(n, 4)}});
  auto &q = queue_for(stream);
  auto tb = tables(scratch, e);
  q.memset(tb.cnt, 0, size_t(e) * 4);
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> ii) {
    int expert = ids[ii];
    if (expert >= 0 && expert < e)
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(
          tb.cnt[expert])
          .fetch_add(1);
  });
  q.single_task([=] {
    int r = 0, t = 0;
    for (int expert = 0; expert < e; ++expert) {
      tb.off[expert] = tb.fill[expert] = r;
      tb.ts[expert] = t;
      int c = tb.cnt[expert];
      for (int j = 0; j < c; j += 64)
        tb.tiles[t++] = {expert, r + j};
      r += c;
    }
    tb.off[e] = r;
    tb.ts[e] = t;
  });
  auto event = q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> ii) {
    const int expert = ids[ii];
    if (expert < 0 || expert >= e) {
      slot[ii] = 0;
      return;
    }
    const int r = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>(
                      tb.fill[expert])
                      .fetch_add(1);
    slot[ii] = r;
    src[r] = int32_t(size_t(ii) / k);
  });
  finish(stream, event);
}
void experts(const Batch &b, int e, int64_t n, const void *scratch,
             const void *xa, const int32_t *src, void *ha, float *dm,
             void *stream) {
  if (n == 0 || b.e1 == b.e0)
    return;
  validate_batch(b, e, n, scratch, xa, src, ha, dm, BLOB);
  project<42, false, true>(b, {}, e, n, scratch, xa, src, ha, dm, stream);
  project<42, false, false>(b, {}, e, n, scratch, ha, src, nullptr, dm, stream);
}
void experts_native(const Batch &b, const NativeGeom &g, int e, int64_t n,
                    const void *scratch, const void *xa, const int32_t *src,
                    void *ha, float *dm, void *stream) {
  if (n == 0 || b.e1 == b.e0)
    return;
  auto gu = block_format(g.gu_type), down = block_format(g.d_type);
  if (!gu_covered(g.gu_type) || !d_covered(g.d_type) ||
      g.gu_row != size_t(2560 / gu.width * gu.bytes) ||
      g.d_row != size_t(640 / down.width * down.bytes) ||
      g.up_off < 640 * g.gu_row || g.down_off < g.up_off + 640 * g.gu_row)
    throw std::invalid_argument("unsupported fused native expert geometry");
  validate_batch(b, e, n, scratch, xa, src, ha, dm,
                 g.down_off + 2560 * g.d_row);
  switch (g.gu_type) {
  case 16:
    project<16, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  case 17:
    project<17, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  case 18:
    project<18, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  case 21:
    project<21, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  case 22:
    project<22, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  case 23:
    project<23, true, true>(b, g, e, n, scratch, xa, src, ha, dm, stream);
    break;
  }
  if (g.d_type == 20)
    project<20, true, false>(b, g, e, n, scratch, ha, src, nullptr, dm, stream);
  else
    project<42, true, false>(b, g, e, n, scratch, ha, src, nullptr, dm, stream);
}
} // namespace strata::prefill::fused
