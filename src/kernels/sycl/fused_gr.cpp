// Three-stage hyper-connection read with normalized activations reconstructed
// inside the projections, using the native GR/MMVF reduction layout.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <array>
#include <cmath>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int N = 2560, HC = 4, LR = 320, D = N * HC;
float sum32(sycl::sub_group sg, float x) {
  for (int i = 16; i; i >>= 1)
    x += sycl::permute_group_by_xor(sg, x, i);
  return x;
}
float sigmoid(float x) { return 1.f / (1.f + sycl::exp(-x)); }
void valid(const FusedGrArgs &a) {
  if (!std::isfinite(a.eps) || a.eps < 0 || (a.apply && a.R_out != a.R))
    throw std::invalid_argument(
        "invalid SYCL fused GR epsilon/residual destination");
  std::array<Span, 5> out{{{a.lo, LR * 4}, {a.rs, HC * 4}, {a.mixed, N * 4}}};
  std::array<Span, 6> in{
      {{a.w_norm, D * 4}, {a.w_down, D * LR * 2}, {a.w_up, D * LR * 2}}};
  int nw = 3, nr = 3;
  if (a.apply) {
    out[nw++] = {a.R_out, D * 4};
    in[nr++] = {a.bo_prev, N * 4};
    in[nr++] = {a.inj_prev, HC * 4};
  } else
    in[nr++] = {a.R, D * 4};
  if (a.w_inject) {
    out[nw++] = {a.inject_out, HC * 4};
    in[nr++] = {a.w_inject, HC * D * 2};
  }
  for (int i = 0; i < nw; ++i) {
    validate_spans({out[i]}, {});
    for (int j = 0; j < i; ++j)
      validate_spans({out[i]}, {out[j]});
    for (int j = 0; j < nr; ++j)
      validate_spans({out[i]}, {in[j]});
  }
}
void launch(FusedGrArgs a, void *stream) {
  auto &q = queue_for(stream);
  float norm_width = float(N);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(32, h);
    h.parallel_for(
        sycl::nd_range<1>(HC * 1024, 1024),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          int tid = it.get_local_linear_id(), c = it.get_group_linear_id(),
              lane = tid % 32;
          float sum = 0;
          const float weight =
              a.apply
                  ? sycl::fma(2.f, sigmoid(sycl::fma(.25f, a.inj_prev[c], 0.f)),
                              0.f)
                  : 0;
          for (int d = tid; d < N; d += 1024) {
            const int i = c * N + d;
            float v = a.R[i];
            if (a.apply) {
              v = sycl::fma(a.bo_prev[d], weight, v);
              a.R_out[i] = v;
            }
            sum = sycl::fma(v, v, sum);
          }
          sum = sum32(it.get_sub_group(), sum);
          if (!lane)
            part[tid / 32] = sum;
          it.barrier(sycl::access::fence_space::local_space);
          sum = sum32(it.get_sub_group(), part[lane]);
          if (!tid)
            a.rs[c] = sycl::rsqrt(sum / norm_width + a.eps);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(32, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(LR + (a.w_inject ? HC : 0)) * 256, 256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(),
                    row = it.get_group_linear_id();
          const auto sg = it.get_sub_group();
          const uint16_t *w = row < LR ? a.w_down + size_t(row) * D
                                       : a.w_inject + size_t(row - LR) * D;
          if (tid < 32)
            part[tid] = 0;
          it.barrier(sycl::access::fence_space::local_space);
          float acc = 0;
          for (int pair = tid; pair < D / 2; pair += 256)
            for (int j = 0; j < 2; ++j) {
              const int i = 2 * pair + j;
              const float x = (a.rs[i / N] * a.R[i]) * a.w_norm[i];
              acc = sycl::fma(f32_from_bf16(w[i]), x, acc);
            }
          acc = sum32(sg, acc);
          if (tid % 32 == 0)
            part[tid / 32] = acc;
          it.barrier(sycl::access::fence_space::local_space);
          if (tid < 32)
            acc = sum32(sg, part[tid]);
          if (!tid) {
            if (row < LR) {
              float x = sycl::fma(.25f, acc, 0.f);
              a.lo[row] = x / (1.f + sycl::exp(-x));
            } else
              a.inject_out[row - LR] = acc;
          }
        });
  });
  auto done = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> gates(HC, h);
    h.parallel_for(
        sycl::nd_range<1>(N * 128, 128),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(), lane = tid % 32,
                    c = tid / 32, d = it.get_group_linear_id();
          const auto sg = it.get_sub_group();
          const uint16_t *w = a.w_up + size_t(c * N + d) * LR;
          // One subgroup per residual stream. Preserve the original five
          // 32-lane partial reductions and their zero-padded reduction tree.
          float partial[5];
#pragma unroll
          for (int k = 0; k < 5; ++k) {
            float acc = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int i = (k * 32 + lane) * 2 + j;
              acc = sycl::fma(f32_from_bf16(w[i]), a.lo[i], acc);
            }
            partial[k] = sum32(sg, acc);
          }
          if (!lane) {
            const float acc = ((partial[0] + partial[4]) + partial[2]) +
                              (partial[1] + partial[3]);
            gates[c] = sigmoid(acc);
          }
          it.barrier(sycl::access::fence_space::local_space);
          if (!tid) {
            float mixed = 0;
            for (int stream = 0; stream < HC; ++stream) {
              const int i = stream * N + d;
              const float x = (a.rs[stream] * a.R[i]) * a.w_norm[i];
              if (a.w_inject)
                mixed = sycl::fma(x, gates[stream], mixed);
              else {
                const float product = x * gates[stream];
                mixed = stream ? mixed + product : product;
              }
            }
            a.mixed[d] = sycl::fma(.25f, mixed, 0.f);
          }
        });
  });
  finish(stream, done);
}
} // namespace
bool fused_gr_supported(int64_t n, int64_t hc, int64_t lr) {
  return n == N && hc == HC && lr == LR;
}
void fused_gr_read(const FusedGrArgs &a, void *stream) {
  valid(a);
  launch(a, stream);
}
void fused_gr_read_multi(const FusedGrArgs *a, int T, float *scratch,
                         void *stream, unsigned long long *stamps, int) {
  if (!a || T < 1 || T > kFusedGrMaxT || stamps)
    throw std::invalid_argument(
        "SYCL fused GR requires 1..8 tokens and event-based profiling");
  // The portable path reconstructs each normalized value, so no XN scratch is
  // used.
  (void)scratch;
  for (int t = 0; t < T; ++t) {
    valid(a[t]);
    if (a[t].w_norm != a[0].w_norm || a[t].w_down != a[0].w_down ||
        a[t].w_up != a[0].w_up || a[t].w_inject != a[0].w_inject ||
        a[t].eps != a[0].eps)
      throw std::invalid_argument(
          "SYCL fused GR batch must share weights and epsilon");
  }
  for (int t = 0; t < T; ++t)
    launch(a[t], stream);
}
void fused_gr_check() { (void)sycl_backend::runtime_for(); }
int fused_gr_variant() {
  return 0;
} // One portable implementation; no CUDA variants.
} // namespace strata::kernels
