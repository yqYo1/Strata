// Paged FP16/Q8/Q4 KV storage. CPU access to host USM requires queue
// completion.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
void geometry(const QsaShapes &s, int fmt) {
  if (s.n_head_kv <= 0 || s.n_head_kv > INT_MAX || s.head_dim <= 0 ||
      s.head_dim > INT_MAX || s.page_size <= 0 || s.page_size > INT_MAX ||
      s.head_dim % 4 || (fmt == kKvInt8 && s.head_dim % 64) ||
      (fmt == kKvQ4 && s.head_dim != 256))
    throw std::invalid_argument("invalid SYCL KV geometry");
  checked_count(checked_count(s.n_head_kv, s.head_dim), s.page_size);
}
struct Pools {
  void *k;
  void *v;
  uint16_t *ks = nullptr;
  uint16_t *vs = nullptr;
};
Pools host_pools(KvHostPools h, int fmt) {
  if (fmt == kKvInt8)
    return {h.k_q, h.v_q, h.k_scale, h.v_scale};
  if (fmt == kKvQ4)
    return {h.k_q4, h.v_q4};
  return {h.k_pool, h.v_pool};
}
void validate_pool(Pools p, int fmt, bool optional) {
  if (optional && !p.k && !p.v && !p.ks && !p.vs)
    return;
  validate_spans({{p.k, 4}, {p.v, 4}}, {});
  if (fmt == kKvInt8)
    validate_spans({{p.k, 4}, {p.v, 4}, {p.ks, 4}, {p.vs, 4}}, {});
}
// Pool and page-table capacities are not carried by the public API. Callers
// guarantee valid resident pages, cell indices and sufficiently large pools.
// Every dynamic position/count is re-read from device memory on each launch.
template <int Format>
void append(Pools dst, const int32_t *table, const int32_t *step, int64_t pos0,
            int64_t tokens, const float *k, const float *v, const QsaShapes &s,
            void *stream, KvHostPools host = {}, KvHostPools stage = {}) {
  geometry(s, Format);
  if (tokens < 0 || pos0 < 0 || tokens > INT_MAX || pos0 > INT_MAX - tokens)
    throw std::invalid_argument("invalid SYCL KV append range");
  if (!tokens)
    return;
  validate_pool(dst, Format, false);
  const Pools hp = host_pools(host, Format), sp = host_pools(stage, Format);
  validate_pool(hp, Format, true);
  validate_pool(sp, Format, true);
  const size_t elems =
      checked_count(tokens, checked_count(s.n_head_kv, s.head_dim));
  if (elems > SIZE_MAX / 4)
    throw std::invalid_argument("SYCL KV input overflow");
  validate_spans({}, {{table, 4}, {k, elems * 4}, {v, elems * 4}});
  if (step)
    validate_spans({}, {{step, kStepCount * 4}});
  constexpr int group = Format == kKvInt8 ? 64 : 32;
  const size_t groups = (s.head_dim + group - 1) / group;
  const size_t count =
      checked_count(tokens, checked_count(s.n_head_kv, groups * 2));
  auto event = queue_for(stream).submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> maxima(2, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(count * group, group),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const size_t item = it.get_group_linear_id();
          const int lane = it.get_local_linear_id();
          const size_t g = item % groups, h = (item / groups) % s.n_head_kv;
          const bool isv = (item / groups / s.n_head_kv) % 2;
          const size_t token = item / (groups * s.n_head_kv * 2);
          const int64_t pos = step ? step[kStepPos] : pos0 + token;
          if (pos < 0)
            return;
          const int64_t page = table[pos / s.page_size];
          const size_t row =
              ((pos / s.page_size) * s.n_head_kv + h) * s.page_size +
              pos % s.page_size;
          const size_t prow =
              page < 0
                  ? 0
                  : (page * s.n_head_kv + h) * s.page_size + pos % s.page_size;
          // FP16 also uses 32-element groups. All supported widths are
          // multiples of 4; tail lanes must not access the next head.
          const size_t d = g * group + lane;
          const float x =
              d < size_t(s.head_dim)
                  ? (isv ? v : k)[(token * s.n_head_kv + h) * s.head_dim + d]
                  : 0.f;
          auto sg = it.get_sub_group();
          if constexpr (Format == kKvF16) {
            if (d >= size_t(s.head_dim))
              return;
            const uint16_t bits = f16_from_f32(x);
            if (page >= 0)
              static_cast<uint16_t *>(
                  isv ? dst.v : dst.k)[prow * s.head_dim + d] = bits;
            if (hp.k)
              static_cast<uint16_t *>(isv ? hp.v : hp.k)[row * s.head_dim + d] =
                  bits;
            if (sp.k)
              static_cast<uint16_t *>(isv ? sp.v : sp.k)[row * s.head_dim + d] =
                  bits;
          } else if constexpr (Format == kKvInt8) {
            float a = sycl::fabs(x);
            for (int off = 16; off; off /= 2)
              a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, off));
            if (lane % 32 == 0)
              maxima[lane / 32] = a;
            it.barrier(sycl::access::fence_space::local_space);
            const uint16_t bits = f16_from_f32(sycl::ext::intel::math::fdiv_rn(
                sycl::fmax(maxima[0], maxima[1]), 127.f));
            const float scale = f32_from_f16(bits);
            int code =
                scale > 0.f
                    ? int(sycl::rint(sycl::ext::intel::math::fdiv_rn(x, scale)))
                    : 0;
            code = sycl::clamp(code, -127, 127);
            auto write = [&](Pools p, size_t r) {
              static_cast<int8_t *>(isv ? p.v : p.k)[r * s.head_dim + d] =
                  int8_t(code);
              if (!lane)
                (isv ? p.vs : p.ks)[r * groups + g] = bits;
            };
            if (page >= 0)
              write(dst, prow);
            if (hp.k)
              write(hp, row);
            if (sp.k)
              write(sp, row);
          } else {
            float a = sycl::fabs(x), m = x;
            for (int off = 16; off; off /= 2) {
              const float oa = sycl::permute_group_by_xor(sg, a, off),
                          om = sycl::permute_group_by_xor(sg, m, off);
              if (oa > a || (oa == a && om > m)) {
                a = oa;
                m = om;
              }
            }
            const float scale = m * -.125f;
            const float inv = scale != 0.f
                                  ? sycl::ext::intel::math::fdiv_rn(1.f, scale)
                                  : 0.f;
            const int code = sycl::clamp(int(x * inv + 8.5f), 0, 15);
            const int high =
                sycl::select_from_group(sg, code, (lane + 16) % 32);
            auto write = [&](Pools p, size_t r) {
              auto *block = reinterpret_cast<block_q4_0 *>(isv ? p.v : p.k) +
                            r * groups + g;
              if (!lane)
                block->d = f16_from_f32(scale);
              if (lane < 16)
                block->qs[lane] = uint8_t(code | (high << 4));
            };
            if (page >= 0)
              write(dst, prow);
            if (hp.k)
              write(hp, row);
            if (sp.k)
              write(sp, row);
          }
        });
  });
  finish(stream, event);
}
template <int Format>
void gather(Pools src, const int32_t *table, const int32_t *ids,
            const int32_t *step, int64_t capacity, const QsaShapes &s,
            uint16_t *k, uint16_t *v, void *stream) {
  geometry(s, Format);
  if (capacity < 0 || capacity > INT_MAX)
    throw std::invalid_argument("invalid SYCL KV gather capacity");
  if (!capacity)
    return;
  validate_pool(src, Format, false);
  const size_t n =
      checked_count(capacity, checked_count(s.n_head_kv, s.head_dim));
  if (n > SIZE_MAX / 2)
    throw std::invalid_argument("SYCL KV gather overflow");
  validate_spans(
      {{k, n * 2}, {v, n * 2}},
      {{table, 4}, {ids, size_t(capacity) * 4}, {src.k, 4}, {src.v, 4}});
  if (step)
    validate_spans({{k, n * 2}, {v, n * 2}}, {{step, kStepCount * 4}});
  auto e =
      queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
        const size_t i = tid[0], d = i % s.head_dim,
                     h = (i / s.head_dim) % s.n_head_kv,
                     id = i / (s.head_dim * s.n_head_kv);
        const int64_t width = step ? step[kStepWidth] : capacity;
        if (int64_t(id) >= width)
          return;
        const int64_t cell = ids[id];
        const int64_t page = cell < 0 ? -1 : table[cell / s.page_size];
        if (page < 0) {
          k[i] = 0x7e00;
          v[i] = 0x7e00;
          return;
        }
        const size_t row =
            (page * s.n_head_kv + h) * s.page_size + cell % s.page_size;
        auto read = [&](const void *p, const uint16_t *scales) {
          if constexpr (Format == kKvF16)
            return static_cast<const uint16_t *>(p)[row * s.head_dim + d];
          else if constexpr (Format == kKvInt8)
            return f16_from_f32(
                float(static_cast<const int8_t *>(p)[row * s.head_dim + d]) *
                f32_from_f16(scales[row * (s.head_dim / 64) + d / 64]));
          else {
            const auto &block = reinterpret_cast<const block_q4_0 *>(
                p)[row * (s.head_dim / 32) + d / 32];
            const uint8_t b = block.qs[d % 16];
            const int q = int(d % 32 < 16 ? b & 15 : b >> 4) - 8;
            return f16_from_f32(float(q) * f32_from_f16(block.d));
          }
        };
        k[i] = read(src.k, src.ks);
        v[i] = read(src.v, src.vs);
      });
  finish(stream, e);
}
} // namespace
void qsa_step_fill(int32_t *step, int64_t pos, const QsaShapes &s) {
  if (!step || pos < 0 || pos >= INT_MAX || s.idx_block <= 0 ||
      s.idx_block > INT_MAX || s.idx_top_k < 0 ||
      s.idx_top_k > INT_MAX - s.idx_block + 1)
    throw std::invalid_argument("invalid QSA step");
  step[kStepPos] = int32_t(pos);
  step[kStepNKv] = int32_t(pos + 1);
  step[kStepNBid] = int32_t((pos + 1) / s.idx_block);
  step[kStepWidth] = int32_t(qsa_selection_width(pos + 1, s));
}
void kv_append(uint16_t *k, uint16_t *v, const int32_t *t, int64_t pos,
               const float *x, const float *y, const QsaShapes &s,
               void *stream) {
  append<kKvF16>({k, v}, t, nullptr, pos, 1, x, y, s, stream);
}
void kv_append_step(uint16_t *k, uint16_t *v, const int32_t *t,
                    const int32_t *step, const float *x, const float *y,
                    const QsaShapes &s, void *stream, const KvHostPools *host) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  append<kKvF16>({k, v}, t, step, 0, 1, x, y, s, stream,
                 host ? *host : KvHostPools{});
}
void kv_append_q8_step(int8_t *k, int8_t *v, uint16_t *ks, uint16_t *vs,
                       const int32_t *t, const int32_t *step, const float *x,
                       const float *y, const QsaShapes &s, void *stream,
                       const KvHostPools *host) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  append<kKvInt8>({k, v, ks, vs}, t, step, 0, 1, x, y, s, stream,
                  host ? *host : KvHostPools{});
}
void kv_append_q4_step(uint8_t *k, uint8_t *v, const int32_t *t,
                       const int32_t *step, const float *x, const float *y,
                       const QsaShapes &s, void *stream,
                       const KvHostPools *host) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  append<kKvQ4>({k, v}, t, step, 0, 1, x, y, s, stream,
                host ? *host : KvHostPools{});
}
void kv_append_q4(uint8_t *k, uint8_t *v, const int32_t *t, int64_t pos,
                  int64_t tokens, const float *x, const float *y,
                  const QsaShapes &s, void *stream, const KvHostPools *host,
                  const KvHostPools *stage) {
  append<kKvQ4>({k, v}, t, nullptr, pos, tokens, x, y, s, stream,
                host ? *host : KvHostPools{}, stage ? *stage : KvHostPools{});
}
void kv_gather(const uint16_t *k, const uint16_t *v, const int32_t *t,
               const int32_t *ids, int64_t count, const QsaShapes &s,
               uint16_t *x, uint16_t *y, void *stream) {
  gather<kKvF16>({const_cast<uint16_t *>(k), const_cast<uint16_t *>(v)}, t, ids,
                 nullptr, count, s, x, y, stream);
}
void kv_gather_step(const uint16_t *k, const uint16_t *v, const int32_t *t,
                    const int32_t *ids, const int32_t *step, int64_t cap,
                    const QsaShapes &s, uint16_t *x, uint16_t *y,
                    void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  gather<kKvF16>({const_cast<uint16_t *>(k), const_cast<uint16_t *>(v)}, t, ids,
                 step, cap, s, x, y, stream);
}
void kv_gather_q8_step(const int8_t *k, const int8_t *v, const uint16_t *ks,
                       const uint16_t *vs, const int32_t *t, const int32_t *ids,
                       const int32_t *step, int64_t cap, const QsaShapes &s,
                       uint16_t *x, uint16_t *y, void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  gather<kKvInt8>({const_cast<int8_t *>(k), const_cast<int8_t *>(v),
                   const_cast<uint16_t *>(ks), const_cast<uint16_t *>(vs)},
                  t, ids, step, cap, s, x, y, stream);
}
void kv_gather_q4_step(const uint8_t *k, const uint8_t *v, const int32_t *t,
                       const int32_t *ids, const int32_t *step, int64_t cap,
                       const QsaShapes &s, uint16_t *x, uint16_t *y,
                       void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  gather<kKvQ4>({const_cast<uint8_t *>(k), const_cast<uint8_t *>(v)}, t, ids,
                step, cap, s, x, y, stream);
}
void fwht256_cuda(const float *src, float *dst, int64_t rows, void *stream) {
  if (rows < 0)
    throw std::invalid_argument("negative FWHT rows");
  if (!rows)
    return;
  const size_t count = sycl_backend::checked_count(rows, 256);
  if (count > SIZE_MAX / 4)
    throw std::invalid_argument("FWHT overflow");
  sycl_backend::validate_spans({}, {{src, count * 4}, {dst, count * 4}});
  if (src != dst)
    sycl_backend::validate_spans({{dst, count * 4}}, {{src, count * 4}});
  auto e = sycl_backend::queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(rows) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t row = it.get_group_linear_id();
        const int lane = it.get_local_linear_id();
        auto sg = it.get_sub_group();
        float reg[8];
        for (int j = 0; j < 8; ++j)
          reg[j] = src[row * 256 + j * 32 + lane] * .0625f;
        for (int h = 1; h < 32; h *= 2)
          for (int j = 0; j < 8; ++j) {
            float a = reg[j], b = sycl::permute_group_by_xor(sg, a, h);
            reg[j] = (lane & h) ? b - a : a + b;
          }
        for (int h = 1; h < 8; h *= 2)
          for (int j = 0; j < 8; j += 2 * h)
            for (int k = 0; k < h; ++k) {
              float a = reg[j + k], b = reg[j + k + h];
              reg[j + k] = a + b;
              reg[j + k + h] = a - b;
            }
        for (int j = 0; j < 8; ++j)
          dst[row * 256 + j * 32 + lane] = reg[j];
      });
  sycl_backend::finish(stream, e);
}
} // namespace strata::kernels

namespace strata::prefill {
void kv_append(const float *K, const float *V, int64_t T, int64_t pos0,
               const int32_t *table, int64_t page_size, uint16_t *k,
               uint16_t *v, int8_t *kq, int8_t *vq, uint16_t *ks, uint16_t *vs,
               void *stream, const strata::kernels::KvHostPools *host,
               const strata::kernels::KvHostPools *stage) {
  using namespace strata::kernels;
  QsaShapes shape{};
  shape.n_head_kv = 2;
  shape.head_dim = 256;
  shape.page_size = page_size;
  if (k)
    append<kKvF16>({k, v}, table, nullptr, pos0, T, K, V, shape, stream,
                   host ? *host : KvHostPools{},
                   stage ? *stage : KvHostPools{});
  else
    append<kKvInt8>({kq, vq, ks, vs}, table, nullptr, pos0, T, K, V, shape,
                    stream, host ? *host : KvHostPools{},
                    stage ? *stage : KvHostPools{});
}
} // namespace strata::prefill
