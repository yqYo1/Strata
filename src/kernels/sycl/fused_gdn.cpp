// Portable translation of Strata's CUDA fused_gdn and verify GDN arithmetic.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
void dims(bool ok) {
  if (!ok)
    throw std::invalid_argument("invalid SYCL fused GDN geometry");
}
void eps_check(float eps) { dims(std::isfinite(eps) && eps >= 0); }
float xor_sum(sycl::sub_group sg, float x) {
  for (int off = 16; off; off >>= 1)
    x += sycl::permute_group_by_xor(sg, x, off);
  return x;
}
void conv(const float *hist, float *update, const float *x, const float *w,
          float *out, int C, int qkh, float eps, int T, int begin,
          void *stream) {
  dims(C > 0 && C <= INT32_MAX / 4 && C % 128 == 0 && qkh >= 0 &&
       qkh <= C / 128 && T > 0 && begin >= 0 && begin <= INT32_MAX - T &&
       int64_t(begin + T) * (C / 128) <= INT32_MAX &&
       (!update || (T == 1 && begin == 0)));
  eps_check(eps);
  const size_t b = size_t(C) * 4, total = b * (begin + T);
  if (update)
    validate_spans({{out, total}, {update, b * 3}}, {{x, total}, {w, b * 4}});
  else
    validate_spans({{out, total}}, {{hist, b * 3}, {x, total}, {w, b * 4}});
  auto e = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(4, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(T) * C, 128),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(),
                    block = it.get_group_linear_id();
          const int t = begin + block / (C / 128), head = block % (C / 128),
                    c = head * 128 + tid;
          float v[4];
          for (int j = 0; j < 3; ++j) {
            int src = t + j;
            v[j] = src < 3 ? hist[c * 3 + src] : x[size_t(src - 3) * C + c];
          }
          v[3] = x[size_t(t) * C + c];
          float sum = v[0] * w[c * 4];
          for (int j = 1; j < 4; ++j)
            sum = sycl::fma(v[j], w[c * 4 + j], sum);
          if (update)
            for (int j = 0; j < 3; ++j)
              update[c * 3 + j] = v[j + 1];
          float y = sum / (1.f + sycl::exp(-sum));
          if (head < qkh) {
            float sq = xor_sum(it.get_sub_group(), y * y);
            if (tid % 32 == 0)
              part[tid / 32] = sq;
            it.barrier(sycl::access::fence_space::local_space);
            y *= sycl::rsqrt(((part[0] + part[1]) + part[2]) + part[3] + eps);
          }
          out[size_t(t) * C + c] = y;
        });
  });
  finish(stream, e);
}
void recurrence(float *state, const float *q, const float *k, const float *v,
                int stride, const float *gate, const float *beta,
                const float *z, const float *gamma, float eps, float *y, int hk,
                int hv, int T, const int32_t *keep, bool single, int begin,
                void *stream, uint16_t *y16 = nullptr) {
  dims(hk > 0 && hv > 0 && hv % hk == 0 && hv <= INT32_MAX / 16384 && T > 0 &&
       (single || T <= kVerifyMaxT) && begin >= 0 && begin <= T);
  eps_check(eps);
  const size_t vb = size_t(hv) * 128 * 4, qb = size_t(hk) * 128 * 4;
  const size_t tail = size_t(stride) * (T - 1) * 4;
  validate_spans({{state, vb * 128}, {y, vb * T}}, {{q, qb + tail},
                                                    {k, qb + tail},
                                                    {v, vb + tail},
                                                    {gate, size_t(hv) * T * 4},
                                                    {beta, size_t(hv) * T * 4},
                                                    {z, vb * T},
                                                    {gamma, 128 * 4}});
  if (y16)
    validate_spans({{state, vb * 128}, {y, vb * T}, {y16, vb * T / 2}},
                   {{q, qb + tail},
                    {k, qb + tail},
                    {v, vb + tail},
                    {gate, size_t(hv) * T * 4},
                    {beta, size_t(hv) * T * 4},
                    {z, vb * T},
                    {gamma, 128 * 4}});
  if (keep)
    validate_spans({{state, vb * 128}, {y, vb * T}}, {{keep, 4}});
  auto e = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sk(128, h), sq(128, h), red(512, h),
        part(16, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(hv) * 512, 512),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(),
                    head = it.get_group_linear_id(), rg = tid / 128,
                    col = tid % 128;
          const size_t base = (size_t(rg * 32) * hv + head) * 128 + col,
                       rs = size_t(hv) * 128;
          float s[32];
          for (int r = 0; r < 32; ++r)
            s[r] = state[base + r * rs];
          const int n = keep ? sycl::clamp(*keep, 0, T) : T;
          for (int t = 0; t < n; ++t) {
            it.barrier(sycl::access::fence_space::local_space);
            if (tid < 128) {
              sk[tid] = k[size_t(t) * stride + (head % hk) * 128 + tid];
              sq[tid] = q[size_t(t) * stride + (head % hk) * 128 + tid];
            }
            it.barrier(sycl::access::fence_space::local_space);
            const float decay = sycl::exp(gate[size_t(t) * hv + head]);
            float kv = 0;
            for (int r = 0; r < 32; ++r)
              kv = sycl::fma(s[r], sk[rg * 32 + r], kv);
            red[tid] = kv;
            it.barrier(sycl::access::fence_space::local_space);
            const float proj =
                ((red[col] + red[128 + col]) + red[256 + col]) + red[384 + col];
            const float delta =
                sycl::fma(-decay, proj,
                          v[size_t(t) * stride + head * 128 + col]) *
                beta[size_t(t) * hv + head];
            float o = 0;
            for (int r = 0; r < 32; ++r) {
              s[r] = sycl::fma(decay, s[r], sk[rg * 32 + r] * delta);
              o = sycl::fma(s[r], sq[rg * 32 + r], o);
            }
            it.barrier(sycl::access::fence_space::local_space);
            red[tid] = o;
            it.barrier(sycl::access::fence_space::local_space);
            float oc = 0, sumsq = 0;
            if (rg == 0) {
              oc = (((red[col] + red[128 + col]) + red[256 + col]) +
                    red[384 + col]) *
                   sycl::rsqrt(128.f);
              sumsq = oc * oc;
            }
            if (t < begin)
              continue;
            sumsq = xor_sum(it.get_sub_group(), sumsq);
            if (tid % 32 == 0)
              part[tid / 32] = sumsq;
            it.barrier(sycl::access::fence_space::local_space);
            if (rg == 0) {
              const float ss = ((part[0] + part[1]) + part[2]) + part[3];
              const size_t idx = (size_t(t) * hv + head) * 128 + col;
              const float value =
                  ((oc * sycl::rsqrt(ss / 128.f + eps)) * gamma[col]) *
                  (1.f / (1.f + sycl::exp(-z[idx])));
              y[idx] = value;
              if (y16)
                y16[idx] = f16_from_f32(value);
            }
          }
          if ((single || keep) && n > 0)
            for (int r = 0; r < 32; ++r)
              state[base + r * rs] = s[r];
        });
  });
  finish(stream, e);
}
} // namespace
void fused_gdn_conv_l2(float *hist, const float *x, const float *w, float *out,
                       int C, int qkh, float eps, void *stream) {
  conv(hist, hist, x, w, out, C, qkh, eps, 1, 0, stream);
}
void gdn_conv_l2_multi(const float *hist, const float *x, const float *w,
                       float *out, int C, int qkh, float eps, int T,
                       void *stream, int begin) {
  dims(T > 0 && T <= kVerifyMaxT && begin >= 0 && begin <= kVerifyMaxT - T);
  conv(hist, nullptr, x, w, out, C, qkh, eps, T, begin, stream);
}
void gdn_conv_commit(float *hist, const float *x, int C, const int32_t *keep,
                     void *stream) {
  dims(C > 0);
  validate_spans({{hist, size_t(C) * 12}}, {{x, size_t(C) * 4}, {keep, 4}});
  for_each(C, stream, [=](size_t c) {
    const int n = *keep;
    if (n <= 0 || n > kVerifyMaxT)
      return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
      int src = n + j;
      v[j] = src < 3 ? hist[c * 3 + src] : x[size_t(src - 3) * C + c];
    }
    for (int j = 0; j < 3; ++j)
      hist[c * 3 + j] = v[j];
  });
}
void gdn_ab_multi(const float *x, const uint16_t *wa, const uint16_t *wb,
                  const float *dt, const float *a, float *gate, float *beta,
                  int N, int hv, int T, void *stream) {
  dims(N > 0 && N % 8 == 0 && hv > 0 && hv <= INT32_MAX / 2 && T > 0 &&
       T <= kVerifyMaxT);
  validate_spans({{gate, size_t(hv) * T * 4}, {beta, size_t(hv) * T * 4}},
                 {{x, size_t(N) * T * 4},
                  {wa, size_t(N) * hv * 2},
                  {wb, size_t(N) * hv * 2},
                  {dt, size_t(hv) * 4},
                  {a, size_t(hv) * 4}});
  auto e = queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(2 * hv) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int row = it.get_group_linear_id(),
                  lane = it.get_local_linear_id(), r = row % hv;
        const uint16_t *w = (row >= hv ? wb : wa) + size_t(r) * N;
        float acc[kVerifyMaxT] = {};
        for (int j = lane; j < N / 8; j += 32)
          for (int i = 0; i < 8; ++i) {
            float weight = f32_from_bf16(w[j * 8 + i]);
            for (int t = 0; t < T; ++t)
              acc[t] = sycl::fma(weight, x[size_t(t) * N + j * 8 + i], acc[t]);
          }
        for (int t = 0; t < T; ++t) {
          const float sum = xor_sum(it.get_sub_group(), acc[t]);
          if (lane == 0) {
            if (row >= hv)
              beta[size_t(t) * hv + r] = 1.f / (1.f + sycl::exp(-sum));
            else {
              const float v = sum + dt[r];
              gate[size_t(t) * hv + r] =
                  (v > 20.f ? v : sycl::log1p(sycl::exp(v))) * a[r];
            }
          }
        }
      });
  finish(stream, e);
}
void fused_gdn_ab(const float *x, const uint16_t *wa, const uint16_t *wb,
                  const float *dt, const float *a, float *gate, float *beta,
                  int N, int hv, void *stream) {
  gdn_ab_multi(x, wa, wb, dt, a, gate, beta, N, hv, 1, stream);
}
void fused_gdn_step_norm(float *state, const float *q, const float *k,
                         const float *v, const float *gate, const float *beta,
                         const float *z, const float *gamma, float eps,
                         float *y, int hk, int hv, void *stream) {
  recurrence(state, q, k, v, 0, gate, beta, z, gamma, eps, y, hk, hv, 1,
             nullptr, true, 0, stream);
}
void gdn_step_norm_multi(float *state, const float *h, int C, const float *gate,
                         const float *beta, const float *z, const float *gamma,
                         float eps, float *y, int hk, int hv, int T,
                         const int32_t *keep, void *stream, int begin) {
  dims(hk > 0 && hv > 0 && int64_t(C) == int64_t(2 * int64_t(hk) + hv) * 128);
  recurrence(state, h, h + size_t(hk) * 128, h + size_t(hk) * 256, C, gate,
             beta, z, gamma, eps, y, hk, hv, T, keep, false, begin, stream);
}
} // namespace strata::kernels

namespace strata::prefill {
void gdn_conv(float *history, const float *qkv, const float *w, float *out,
              int64_t T, float eps, void *stream) {
  constexpr int C = 10240;
  strata::kernels::dims(T > 0 && T <= INT32_MAX / (C / 128));
  strata::sycl_backend::validate_spans(
      {{history, size_t(C) * 12}, {out, size_t(T) * C * 4}},
      {{qkv, size_t(T) * C * 4}, {w, size_t(C) * 16}});
  // Every cell reads the old history before a separate ordered update.
  strata::kernels::conv(history, nullptr, qkv, w, out, C, 32, eps, int(T), 0,
                        stream);
  strata::sycl_backend::for_each(C, stream, [=](size_t c) {
    float v[3];
    for (int j = 0; j < 3; ++j) {
      int64_t t = T - 3 + j;
      v[j] = t < 0 ? history[c * 3 + size_t(t + 3)] : qkv[size_t(t) * C + c];
    }
    for (int j = 0; j < 3; ++j)
      history[c * 3 + j] = v[j];
  });
}
void gdn_recurrence(float *state, const float *h, const float *gate,
                    const float *beta, const float *z, const float *gamma,
                    float eps, float *y, uint16_t *y16, int64_t T,
                    void *stream) {
  strata::kernels::dims(T > 0 && T <= INT32_MAX && y16);
  strata::kernels::recurrence(state, h, h + 16 * 128, h + 32 * 128, 10240, gate,
                              beta, z, gamma, eps, y, 16, 48, int(T), nullptr,
                              true, 0, stream, y16);
}
} // namespace strata::prefill
