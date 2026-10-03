#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/native_mmq.hpp"
#include <climits>
#include <sycl/ext/intel/math.hpp>

namespace strata::prefill::mmq {
namespace {
using namespace sycl_backend;
struct alignas(4) Q81 {
  uint16_t d, sum;
  int8_t codes[32];
};
static_assert(sizeof(Q81) == 36);
size_t padded_cols(int64_t cols) {
  if (cols <= 0 || cols > INT_MAX - 511 || cols % 32)
    throw std::invalid_argument("invalid SYCL quantized prompt width");
  return (size_t(cols) + 511) / 512 * 512;
}
} // namespace
bool built() { return true; }
bool supported(int type) { return kernels::native_mmvq_supported(type); }
bool fits(int type, int64_t rows) { return supported(type) && rows > 0 && rows <= INT_MAX; }
size_t matrix_bytes(int type, int64_t rows, int64_t cols) {
  if (rows <= 0 || rows > INT_MAX || cols <= 0 || cols > INT_MAX)
    throw std::invalid_argument("invalid SYCL prompt matrix dimensions");
  return kernels::native_mmvq_weight_bytes(type, int(cols), int(rows));
}
size_t q8_bytes(int64_t rows, int64_t cols) {
  if (rows < 0 || rows > INT_MAX)
    throw std::invalid_argument("invalid SYCL quantized prompt row count");
  return checked_count(checked_count(rows, padded_cols(cols) / 32), sizeof(Q81));
}
void quantize(const float *x, const int32_t *ids, void *xq, int type,
              int64_t cols, int64_t ld, int64_t rows, void *stream) {
  const size_t padded = padded_cols(cols), bytes = q8_bytes(rows, cols);
  if (!supported(type) || ld < cols)
    throw std::invalid_argument("invalid SYCL quantized prompt type/stride");
  if (!rows)
    return;
  const size_t input_rows = ids ? 1 : size_t(rows);
  validate_spans({{xq, bytes}},
                 {{x, checked_count(checked_count(input_rows - 1, ld) + cols, 4)}});
  if (ids)
    validate_spans({{xq, bytes}}, {{ids, size_t(rows) * 4}});
  auto *out = static_cast<Q81 *>(xq);
  const size_t total = checked_count(rows, padded);
  auto e = queue_for(stream).parallel_for(
      sycl::nd_range<1>(total, 256),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t i = it.get_global_linear_id(), row = i / padded, col = i % padded;
        const int64_t src = ids ? ids[row] : int64_t(row);
        const float v = col < size_t(cols) && src >= 0 ? x[size_t(src) * ld + col] : 0.f;
        const auto sg = it.get_sub_group();
        float maximum = sycl::fabs(v), sum = v;
        for (int offset = 16; offset; offset >>= 1) {
          maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(sg, maximum, offset));
          sum += sycl::permute_group_by_xor(sg, sum, offset);
        }
        const float scale = sycl::ext::intel::math::fdiv_rn(maximum, 127.f);
        out[i / 32].codes[i % 32] = maximum == 0 ? 0 : int8_t(sycl::round(
            sycl::ext::intel::math::fdiv_rn(v, scale)));
        if (i % 32 == 0) {
          out[i / 32].d = kernels::f16_from_f32(scale);
          out[i / 32].sum = kernels::f16_from_f32(sum);
        }
      });
  finish(stream, e);
}
Context::Context() = default;
Context::~Context() = default;
void Context::run(const Product &p, void *stream) {
  kernels::NativeMmq native;
  native.weights = p.w; native.activation = p.xq; native.type = p.type;
  native.experts = p.n; native.rows = p.w_rows; native.cols = p.w_cols;
  native.total_rows = p.total_rows; native.max_rows = p.max_rows;
  native.ld_output = p.ld_dst; native.expert_bytes = p.expert_bytes;
  native.bounds = p.bounds; native.destinations = p.ids; native.output = p.dst;
  native.scratch_weights = p.scratch_weights;
  kernels::native_mmq(native, stream);
}
void gather_native(const void *gate, const void *up, size_t half_bytes,
                   const void *down, size_t down_bytes, void *gu_dst,
                   void *down_dst, void *stream) {
  if (!half_bytes || !down_bytes || half_bytes > SIZE_MAX / 2)
    throw std::invalid_argument("invalid SYCL prompt gather sizes");
  validate_spans({{gu_dst, half_bytes * 2}, {down_dst, down_bytes}},
                 {{gate, half_bytes}, {up, half_bytes}, {down, down_bytes}}, 2);
  const size_t count = half_bytes * 2 + down_bytes;
  if (count < down_bytes)
    throw std::invalid_argument("SYCL prompt gather overflow");
  // Copy aligned native blocks a full vector at a time. Expert matrices are
  // megabytes long; one work-item per byte wastes most of the memory bandwidth.
  if (((half_bytes | down_bytes | uintptr_t(gate) | uintptr_t(up) |
        uintptr_t(down) | uintptr_t(gu_dst) | uintptr_t(down_dst)) & 15) == 0) {
    for_each(count / 16, stream, [=](size_t i) {
      const size_t byte = i * 16;
      if (byte < half_bytes * 2)
        static_cast<sycl::uint4 *>(gu_dst)[i] = static_cast<const sycl::uint4 *>(
            byte < half_bytes ? gate : up)[(byte % half_bytes) / 16];
      else
        static_cast<sycl::uint4 *>(down_dst)[(byte - half_bytes * 2) / 16] =
            static_cast<const sycl::uint4 *>(down)[(byte - half_bytes * 2) / 16];
    });
    return;
  }
  for_each(count, stream, [=](size_t i) {
    if (i < half_bytes * 2)
      static_cast<uint8_t *>(gu_dst)[i] = static_cast<const uint8_t *>(
          i < half_bytes ? gate : up)[i % half_bytes];
    else
      static_cast<uint8_t *>(down_dst)[i - half_bytes * 2] =
          static_cast<const uint8_t *>(down)[i - half_bytes * 2];
  });
}
bool gather_native_group(const GatherGroup &g, size_t up_off, size_t half_bytes,
                         size_t down_off, size_t down_bytes, void *gu_dst,
                         size_t gu_stride, void *down_dst, size_t down_stride,
                         void *stream) {
  if (g.first < 0 || g.n < g.first || g.n > kGatherGroupMax || !half_bytes ||
      !down_bytes || half_bytes > SIZE_MAX / 2 || gu_stride < 2 * half_bytes ||
      down_stride < down_bytes || ((up_off | half_bytes | down_off | down_bytes |
      gu_stride | down_stride | uintptr_t(gu_dst) | uintptr_t(down_dst)) & 15))
    return false;
  for (int e = g.first; e < g.n; ++e)
    if (!g.blob[e] || (uintptr_t(g.blob[e]) & 15))
      return false;
  if (g.first == g.n)
    return true;
  const size_t per = checked_count(half_bytes, 2) + down_bytes;
  if (per < down_bytes)
    throw std::invalid_argument("SYCL group gather overflow");
  const size_t gu_extent = checked_count(g.n - 1, gu_stride) + half_bytes * 2;
  const size_t down_extent = checked_count(g.n - 1, down_stride) + down_bytes;
  for (int e = g.first; e < g.n; ++e)
    validate_spans({{gu_dst, gu_extent}, {down_dst, down_extent}},
                   {{g.blob[e], half_bytes}, {g.blob[e] + up_off, half_bytes},
                    {g.blob[e] + down_off, down_bytes}}, 16);
  for_each(checked_count(g.n - g.first, per / 16), stream, [=](size_t i) {
    const size_t expert = g.first + i / (per / 16), byte = (i % (per / 16)) * 16;
    if (byte < half_bytes * 2) {
      const size_t src = byte < half_bytes ? byte : up_off + byte - half_bytes;
      reinterpret_cast<sycl::uint4 *>(static_cast<uint8_t *>(gu_dst) + expert * gu_stride)[byte / 16] =
          *reinterpret_cast<const sycl::uint4 *>(g.blob[expert] + src);
    } else {
      *reinterpret_cast<sycl::uint4 *>(static_cast<uint8_t *>(down_dst) + expert * down_stride + byte - half_bytes * 2) =
          *reinterpret_cast<const sycl::uint4 *>(g.blob[expert] + down_off + byte - half_bytes * 2);
    }
  });
  return true;
}
void gather_strata_q2(const uint8_t *blob, void *gu, void *down, void *stream) {
  constexpr size_t GC = 1280 * 640, DC = 2560 * 160;
  constexpr size_t GS = GC + DC, DS = GS + 1280 * 40 * 2;
  constexpr size_t GB = 1280 * 40, DB = 2560 * 10;
  validate_spans({{gu, GB * 18}, {down, DB * 18}}, {{blob, DS + DB * 2}}, 2);
  for_each(GB + DB, stream, [=](size_t i) {
    const bool gate = i < GB;
    const size_t block = gate ? i : i - GB;
    auto *dst = static_cast<uint8_t *>(gate ? gu : down) + block * 18;
    const size_t scale = (gate ? GS : DS) + block * 2;
    dst[0] = blob[scale]; dst[1] = blob[scale + 1];
    for (int j = 0; j < 16; ++j)
      dst[2 + j] = blob[(gate ? 0 : GC) + block * 16 + j];
  });
}
void swiglu(const float *gu, float *h, int64_t rows, int64_t ff,
            bool interleaved, void *stream) {
  if (rows < 0 || ff <= 0)
    throw std::invalid_argument("invalid SYCL prompt SwiGLU geometry");
  const size_t n = checked_count(rows, ff);
  if (!n)
    return;
  validate_spans({{h, checked_count(n, 4)}}, {{gu, checked_count(n, 8)}});
  for_each(n, stream, [=](size_t i) {
    const size_t r = i / ff, f = i % ff;
    const size_t g = r * ff * 2 + (interleaved ? 2 * f : f);
    const float gate = gu[g], up = gu[g + (interleaved ? 1 : ff)];
    h[i] = (gate / (1.f + sycl::exp(-gate))) * up;
  });
}
void iota(int32_t *dst, int64_t n, void *stream) {
  if (n < 0 || n > INT_MAX)
    throw std::invalid_argument("invalid SYCL prompt identity row count");
  if (!n) return;
  validate_spans({{dst, size_t(n) * 4}}, {});
  for_each(n, stream, [=](size_t i) { dst[i] = int32_t(i); });
}
} // namespace strata::prefill::mmq
