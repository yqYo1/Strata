#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>

namespace strata::prefill {
namespace {
using namespace sycl_backend;
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, FF = 640;
void need(bool v) {
  if (!v)
    throw std::invalid_argument("invalid SYCL prefill post-operation geometry");
}
size_t count(int64_t rows, int64_t width) {
  auto n = checked_count(rows, width);
  need(n <= SIZE_MAX / 4);
  return n;
}
float sigmoid(float x) { return 1.f / (1.f + sycl::exp(-x)); }
uint16_t bf(float f) {
  auto u = sycl::bit_cast<uint32_t>(f);
  return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
float unbf(uint16_t u) { return sycl::bit_cast<float>(uint32_t(u) << 16); }
uint16_t hf(float f) { return strata::kernels::f16_from_f32(f); }
uint16_t hsat(float f) {
  return hf(sycl::isnan(f) ? f : sycl::clamp(f, -65504.f, 65504.f));
}
float sum32(sycl::sub_group sg, float f) {
  for (int d = 16; d; d >>= 1)
    f += sycl::permute_group_by_xor(sg, f, d);
  return f;
}
void lo_check(uint16_t *hi, uint16_t *lo, size_t n, const float *input,
              size_t input_n) {
  if (lo)
    validate_spans({{hi, n * 2}, {lo, n * 2}}, {{input, input_n * 4}});
}
void image(float x, size_t i, uint16_t *hi, uint16_t *lo,
           uint16_t *half = nullptr) {
  if (hi) {
    uint16_t b = bf(x);
    hi[i] = b;
    if (lo)
      lo[i] = bf(x - unbf(b));
  }
  if (half)
    half[i] = hf(x);
}
template <bool scales>
void norm(const float *R, const float *w, float eps, float *result,
          uint16_t *hi, int64_t T, void *stream, uint16_t *lo) {
  const auto n = count(T, D);
  need(T > 0 && eps >= 0 && std::isfinite(eps));
  validate_spans({{result, size_t(T) * (scales ? HC : D) * 4}, {hi, n * 2}},
                 {{R, n * 4}, {w, D * 4}});
  lo_check(hi, lo, n, R, n);
  auto e = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(8, h);
    h.parallel_for(sycl::nd_range<1>(size_t(T) * HC * 256, 256),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                     const size_t row = it.get_group_linear_id();
                     const int tid = it.get_local_linear_id(), lane = tid % 32;
                     float ss = 0;
                     for (int d = tid; d < N; d += 256) {
                       float v = R[row * N + d];
                       ss = sycl::fma(v, v, ss);
                     }
                     ss = sum32(it.get_sub_group(), ss);
                     if (!lane)
                       part[tid / 32] = ss;
                     it.barrier(sycl::access::fence_space::local_space);
                     ss =
                         sum32(it.get_sub_group(), lane < 8 ? part[lane] : 0.f);
                     float rs = sycl::rsqrt(ss / float(N) + eps);
                     if constexpr (scales) {
                       if (!tid)
                         result[row] = rs;
                     }
                     for (int d = tid; d < N; d += 256) {
                       size_t i = row * N + d;
                       float v = (R[i] * rs) * w[(row % HC) * N + d];
                       if constexpr (!scales)
                         result[i] = v;
                       image(v, i, hi, lo);
                     }
                   });
  });
  finish(stream, e);
}
template <bool recompute>
void mix(const float *x, const float *rs, const float *w, const float *g,
         float *out, uint16_t *hi, int64_t T, void *stream, uint16_t *half,
         uint16_t *lo) {
  size_t n = count(T, N);
  need(!lo || hi);
  validate_spans({{out, n * 4}}, {{x, n * HC * 4}, {g, n * HC * 4}});
  if (hi)
    validate_spans({{out, n * 4}, {hi, n * 2}},
                   {{x, n * HC * 4}, {g, n * HC * 4}});
  if (lo)
    validate_spans({{hi, n * 2}, {lo, n * 2}, {out, n * 4}},
                   {{x, n * HC * 4}, {g, n * HC * 4}});
  if (half)
    validate_spans({{out, n * 4}, {half, n * 2}},
                   {{x, n * HC * 4}, {g, n * HC * 4}});
  if constexpr (recompute)
    validate_spans({{out, n * 4}}, {{rs, size_t(T) * HC * 4}, {w, D * 4}});
  for_each(n, stream, [=](size_t i) {
    size_t t = i / N, d = i % N;
    float v = 0;
    for (int c = 0; c < HC; ++c) {
      size_t j = t * D + c * N + d;
      float a = x[j];
      if constexpr (recompute)
        a = (a * rs[t * HC + c]) * w[c * N + d];
      v = sycl::fma(a, sigmoid(g[j]), v);
    }
    v /= float(HC);
    out[i] = v;
    image(v, i, hi, lo, half);
  });
}
} // namespace
void to_f16(const float *x, uint16_t *y, int64_t n, void *stream) {
  auto z = count(n, 1);
  validate_spans({{y, z * 2}}, {{x, z * 4}});
  for_each(n, stream, [=](size_t i) { y[i] = hf(x[i]); });
}
void to_bf16(const float *x, uint16_t *y, int64_t n, void *stream,
             uint16_t *lo) {
  auto z = count(n, 1);
  validate_spans({{y, z * 2}}, {{x, z * 4}});
  lo_check(y, lo, z, x, z);
  for_each(n, stream, [=](size_t i) { image(x[i], i, y, lo); });
}
void round_f16(const float *x, float *y, int64_t n, void *stream) {
  auto z = count(n, 1);
  if (x == y)
    validate_spans({{y, z * 4}}, {});
  else
    validate_spans({{y, z * 4}}, {{x, z * 4}});
  for_each(n, stream,
           [=](size_t i) { y[i] = strata::kernels::f32_from_f16(hf(x[i])); });
}
void gr_norm(const float *R, const float *w, float eps, float *x, uint16_t *hi,
             int64_t T, void *s, uint16_t *lo) {
  norm<false>(R, w, eps, x, hi, T, s, lo);
}
void gr_norm_rs(const float *R, const float *w, float eps, float *rs,
                uint16_t *hi, int64_t T, void *s, uint16_t *lo) {
  norm<true>(R, w, eps, rs, hi, T, s, lo);
}
void gr_mix(const float *x, const float *g, float *y, uint16_t *hi, int64_t T,
            void *s, uint16_t *half, uint16_t *lo) {
  mix<false>(x, nullptr, nullptr, g, y, hi, T, s, half, lo);
}
void gr_mix_r(const float *R, const float *rs, const float *w, const float *g,
              float *y, uint16_t *hi, int64_t T, void *s, uint16_t *half,
              uint16_t *lo) {
  mix<true>(R, rs, w, g, y, hi, T, s, half, lo);
}
void gr_write(float *R, const float *bo, const float *inj, int64_t ld,
              int64_t T, void *s) {
  need(ld >= HC);
  size_t n = count(T, D);
  validate_spans({{R, n * 4}},
                 {{bo, size_t(T) * N * 4}, {inj, count(T, ld) * 4}});
  for_each(n, s, [=](size_t i) {
    size_t t = i / D, c = i % D / N, d = i % N;
    R[i] = sycl::fma(bo[t * N + d], 2.f * sigmoid(inj[t * ld + c] / float(HC)),
                     R[i]);
  });
}
void gr_write_norm_rs(float *R, const float *bo, const float *inj, int64_t ld,
                      const float *w, float eps, float *rs, uint16_t *hi,
                      int64_t T, void *s, uint16_t *lo) {
  gr_write(R, bo, inj, ld, T, s);
  gr_norm_rs(R, w, eps, rs, hi, T, s, lo);
}
void gr_silu(const float *x, uint16_t *y, int64_t T, void *s, uint16_t *lo) {
  size_t n = count(T, LR);
  validate_spans({{y, n * 2}}, {{x, n * 4}});
  lo_check(y, lo, n, x, n);
  for_each(n, s, [=](size_t i) {
    float a = x[i] / float(HC);
    image(a / (1.f + sycl::exp(-a)), i, y, lo);
  });
}
void gr_broadcast(const float *x, float *y, int64_t T, void *s) {
  size_t n = count(T, D);
  validate_spans({{y, n * 4}}, {{x, size_t(T) * N * 4}});
  for_each(n, s, [=](size_t i) { y[i] = x[(i / D) * N + i % N]; });
}
void gdn_gates(const float *ab, const float *dt, const float *a, float *gate,
               float *beta, int64_t T, void *s) {
  size_t n = count(T, 48);
  validate_spans({{gate, n * 4}, {beta, n * 4}},
                 {{ab, n * 8}, {dt, 48 * 4}, {a, 48 * 4}});
  for_each(n, s, [=](size_t i) {
    size_t t = i / 48, h = i % 48;
    float v = ab[t * 96 + h] + dt[h];
    gate[i] = (v > 20.f ? v : sycl::log1p(sycl::exp(v))) * a[h];
    beta[i] = sigmoid(ab[t * 96 + 48 + h]);
  });
}
void swiglu_pair(const float *g, const float *u, uint16_t *y, int64_t T,
                 void *s) {
  size_t n = count(T, FF);
  validate_spans({{y, n * 2}}, {{g, n * 4}, {u, n * 4}});
  for_each(n, s, [=](size_t i) {
    y[i] = hsat((g[i] / (1.f + sycl::exp(-g[i]))) * u[i]);
  });
}
void swiglu_interleaved(const float *gu, uint16_t *y, int64_t T, void *s) {
  size_t n = count(T, FF);
  validate_spans({{y, n * 2}}, {{gu, n * 8}});
  for_each(n, s, [=](size_t i) {
    float g = gu[2 * i];
    y[i] = hsat((g / (1.f + sycl::exp(-g))) * gu[2 * i + 1]);
  });
}
void copy_i32(int32_t *y, const int32_t *x, int64_t n, void *s) {
  size_t z = count(n, 1);
  if (x == y)
    return;
  validate_spans({{y, z * 4}}, {{x, z * 4}});
  for_each(n, s, [=](size_t i) { y[i] = x[i]; });
}
void gather_rows16(const uint16_t *x, const int32_t *rows, uint16_t *y,
                   int64_t n, int64_t width, void *s) {
  need(width > 0);
  size_t z = count(n, width);
  validate_spans({{y, z * 2}}, {{x, size_t(width) * 2}, {rows, size_t(n) * 4}});
  for_each(z, s, [=](size_t i) {
    int r = rows[i / width];
    y[i] = r < 0 ? 0 : x[size_t(r) * width + i % width];
  });
}
void moe_combine(const float *d, const int32_t *slot, const float *w,
                 const float *shared, const float *sg, float *bo, int64_t T,
                 void *s) {
  size_t n = count(T, N);
  validate_spans({{bo, n * 4}}, {{d, N * 4},
                                 {slot, size_t(T) * 10 * 4},
                                 {w, size_t(T) * 10 * 4},
                                 {shared, n * 4},
                                 {sg, size_t(T) * 4}});
  for_each(n, s, [=](size_t i) {
    size_t t = i / N, c = i % N;
    float v = 0;
    for (int k = 0; k < 10; ++k) {
      int r = slot[t * 10 + k];
      if (r >= 0)
        v = sycl::fma(w[t * 10 + k], d[size_t(r) * N + c], v);
    }
    bo[i] = v + shared[i] * sigmoid(sg[t]);
  });
}
void rms_rows(float *x, const float *w, int64_t rows, int64_t cols, int64_t ld,
              float eps, void *s) {
  need(cols > 0 && ld >= cols && eps >= 0 && std::isfinite(eps));
  size_t n = count(rows, ld);
  validate_spans({{x, n * 4}}, {{w, size_t(cols) * 4}});
  auto e = queue_for(s).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(8, h);
    h.parallel_for(sycl::nd_range<1>(size_t(rows) * 256, 256),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                     size_t row = it.get_group_linear_id();
                     int tid = it.get_local_linear_id(), lane = tid % 32;
                     float sum = 0;
                     for (int64_t c = tid; c < cols; c += 256) {
                       float v = x[row * ld + c];
                       sum = sycl::fma(v, v, sum);
                     }
                     sum = sum32(it.get_sub_group(), sum);
                     if (!lane)
                       part[tid / 32] = sum;
                     it.barrier(sycl::access::fence_space::local_space);
                     sum =
                         sum32(it.get_sub_group(), lane < 8 ? part[lane] : 0.f);
                     float rs = sycl::rsqrt(sum / float(cols) + eps);
                     for (int64_t c = tid; c < cols; c += 256)
                       x[row * ld + c] = (rs * x[row * ld + c]) * w[c];
                   });
  });
  finish(s, e);
}
void split_q(const float *x, float *y, int64_t T, void *s) {
  size_t n = count(T, 24 * 256);
  validate_spans({{y, n * 4}}, {{x, n * 8}});
  for_each(n, s, [=](size_t i) {
    y[i] = x[(i / (24 * 256)) * (24 * 512) + (i / 256 % 24) * 512 + i % 256];
  });
}
void gate_attn(const float *a, const float *q, uint16_t *y, int64_t T,
               void *s) {
  size_t n = count(T, 24 * 256);
  validate_spans({{y, n * 2}}, {{a, n * 4}, {q, n * 8}});
  for_each(n, s, [=](size_t i) {
    y[i] = hf(a[i] * sigmoid(q[(i / (24 * 256)) * (24 * 512) +
                               (i / 256 % 24) * 512 + 256 + i % 256]));
  });
}
void route(const float *l, int32_t *ids, float *w, int64_t T, int64_t E,
           void *s) {
  need(T > 0 && T <= INT32_MAX && E >= 10 && E <= INT32_MAX);
  if (E == 512) {
    auto &q = queue_for(s);
    strata::kernels::native_router_top10_multi(l, ids, w, int(T), &q);
    if (!s)
      q.wait_and_throw();
  } else {
    strata::kernels::router_top10(l, int(T), int(E), 10, ids, w, s);
  }
}
} // namespace strata::prefill
