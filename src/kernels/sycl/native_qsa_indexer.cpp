// Native QSA pooling follows the F16 cache / F32 arithmetic contract.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/sycl/launch.hpp"
#include <array>
#include <atomic>
#include <cmath>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
struct Angles {
  std::array<float, 32> frequencies;
  RopeKernelArgs args;
  RopeTab table;
  const int32_t *mrope;
};
Angles validate(const float *raw, int64_t n, const int32_t *pos, int base,
                const float *gamma, float eps, const QsaIndexerBuffers &b,
                const QsaShapes &s, int64_t capacity, const RopeScaling &sc,
                void *stream) {
  if (!stream || s.idx_dim != 128 || s.idx_block != 4 || s.n_rot != 64 ||
      capacity <= 0 || capacity > INT32_MAX || base < 0 || base % 4 ||
      int64_t(base) + capacity > INT32_MAX || !std::isfinite(eps) || eps <= 0 ||
      rope_scaling_invalid(sc))
    throw std::invalid_argument("invalid native SYCL QSA indexer parameters");
  // All spans, including read-only inputs, must be disjoint in this API.
  sycl_backend::validate_spans(
      {{raw, size_t(n) * 512},
       {gamma, 512},
       {b.tail, 1536},
       {b.dead, 512},
       {b.pooled, size_t(capacity / 4 + 1) * 512},
       {b.block_pos, 4}},
      pos ? std::initializer_list<sycl_backend::Span>{{pos, 4}}
          : std::initializer_list<sycl_backend::Span>{});
  auto &q = sycl_backend::queue_for(stream);
  Angles a;
  a.args = sc.kernel_args(64);
  a.table = rope_table_for(sc);
  a.mrope = mrope_table();
  if ((a.table.cos || a.mrope) &&
      q.get_context() != sycl_backend::runtime_for()->context())
    throw std::invalid_argument(
        "native QSA registered tables require runtime context");
  if (a.table.cos)
    sycl_backend::validate_spans(
        {{b.tail, 1536},
         {b.dead, 512},
         {b.pooled, size_t(capacity / 4 + 1) * 512},
         {b.block_pos, 4}},
        {{a.table.cos, size_t(a.table.max_pos) * 128},
         {a.table.sin, size_t(a.table.max_pos) * 128}});
  const float theta = std::pow(float(sc.freq_base), -2.f / 64);
  for (int i = 0; i < 32; ++i)
    a.frequencies[i] = std::pow(theta, float(i));
  return a;
}
void append(sycl::queue &q, const float *raw, const int32_t *position,
            int fixed_position, int base, const float *gamma, float eps,
            QsaIndexerBuffers b, int capacity, Angles a) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> values(128, h), partials(8, h);
    h.parallel_for(
        sycl::nd_range<1>(256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int pos = position ? *position : fixed_position;
          if (pos < 0 || pos >= capacity)
            return;
          const int d = item.get_local_linear_id(), slot = pos % 4;
          float incoming = 0, mean = 0;
          if (d < 128) {
            incoming = float(sycl::half(raw[d]));
            if (slot < 3)
              b.tail[slot * 128 + d] = incoming;
          }
          if (pos != 0 && slot != 3)
            return;
          if (d < 128) {
            float sum = pos == 0 ? incoming : b.tail[d];
            for (int j = 1; j < 4; ++j)
              sum += pos == 0 || j == 3 ? incoming : b.tail[j * 128 + d];
            mean = sycl::fma(.25f, sum, 0.f);
          }
          auto sg = item.get_sub_group();
          float ss = mean * mean;
          for (int offset = 16; offset; offset /= 2)
            ss += sycl::permute_group_by_xor(sg, ss, offset);
          if (d % 32 == 0)
            partials[d / 32] = ss;
          item.barrier(sycl::access::fence_space::local_space);
          ss = d % 32 < 8 ? partials[d % 32] : 0.f;
          for (int offset = 16; offset; offset /= 2)
            ss += sycl::permute_group_by_xor(sg, ss, offset);
          if (d < 128)
            values[d] = (sycl::rsqrt(ss / 128.f + eps) * mean) * gamma[d];
          item.barrier(sycl::access::fence_space::local_space);
          if (d >= 128)
            return;
          const int block = pos / 4, rp = pos == 0 ? 0 : base + 4 * block;
          float y = values[d];
          if (d < 64) {
            const int pair = d % 32;
            const int p = pos == 0  ? 0
                          : a.mrope ? a.mrope[size_t(rp) * 3 + pair % 3]
                                    : rp;
            float c, s;
            if (a.table.cos && p >= 0 && p < a.table.max_pos) {
              c = a.table.cos[size_t(p) * 32 + pair];
              s = a.table.sin[size_t(p) * 32 + pair];
            } else {
              const float extrap = float(p) * a.frequencies[pair];
              float angle = a.args.freq_scale * extrap,
                    magnitude = a.args.attn_factor;
              if (a.args.ext_factor != 0) {
                const float ramp =
                    rope_yarn_ramp(a.args.corr_low, a.args.corr_high, pair) *
                    a.args.ext_factor;
                angle = angle * (1.f - ramp) + extrap * ramp;
                magnitude *= 1.f + .1f * sycl::log(1.f / a.args.freq_scale);
              }
              c = sycl::cos(angle) * magnitude;
              s = sycl::sin(angle) * magnitude;
            }
            const float x = values[pair], z = values[pair + 32];
            y = d < 32 ? x * c - z * s : x * s + z * c;
          }
          b.pooled[size_t(block) * 128 + d] = y;
          if (pos == 0)
            b.dead[d] = y;
          else
            b.pooled[size_t(block + 1) * 128 + d] = b.dead[d];
          if (d == 0 && pos != 0)
            *b.block_pos = rp;
        });
  });
}
} // namespace
void native_qsa_indexer_set_enabled(bool value) { enabled.store(value); }
bool native_qsa_indexer_enabled() { return enabled.load(); }
void native_qsa_indexer_append(const float *raw, const int32_t *position,
                               int32_t base, const float *gamma, float eps,
                               const QsaIndexerBuffers &b, const QsaShapes &s,
                               int64_t capacity, const RopeScaling &sc,
                               void *stream) {
  if (!position)
    throw std::invalid_argument("null native QSA position");
  const auto a =
      validate(raw, 1, position, base, gamma, eps, b, s, capacity, sc, stream);
  append(sycl_backend::queue_for(stream), raw, position, 0, base, gamma, eps, b,
         int(capacity), a);
}
void native_qsa_indexer_append_batch(const float *raw, int64_t n, int64_t p0,
                                     int32_t base, const float *gamma,
                                     float eps, const QsaIndexerBuffers &b,
                                     const QsaShapes &s, int64_t capacity,
                                     const RopeScaling &sc, void *stream) {
  if (n <= 0)
    return;
  if (p0 < 0 || p0 > capacity || n > capacity - p0)
    throw std::invalid_argument("invalid native QSA batch range");
  const auto a =
      validate(raw, n, nullptr, base, gamma, eps, b, s, capacity, sc, stream);
  auto &q = sycl_backend::queue_for(stream);
  // Baseline: preserve the exact single-cell state transition on an in-order
  // queue. A future batched kernel can reduce launch overhead against this
  // path.
  for (int64_t i = 0; i < n; ++i)
    append(q, raw + i * 128, nullptr, int(p0 + i), base, gamma, eps, b,
           int(capacity), a);
}
} // namespace strata::kernels
