// Three-stage hyper-connection read using the native GR/MMVF reduction layout.
// An optional workspace computes normalized activations once in the norm stage.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <array>
#include <cmath>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int N = 2560, HC = 4, LR = 320, D = N * HC;
float sigmoid(float x) { return 1.f / (1.f + sycl::exp(-x)); }
namespace e = sycl::ext::intel::esimd;
namespace ex = sycl::ext::intel::experimental::esimd;
using F16 = e::simd<float, 16>;
float sum32(sycl::sub_group sg, float x);
float scalar_fma(float a, float b, float c) SYCL_ESIMD_FUNCTION {
  return ex::fma(e::simd<float, 1>(a), e::simd<float, 1>(b),
                 e::simd<float, 1>(c))[0];
}
float reduce16(F16 v) SYCL_ESIMD_FUNCTION {
  e::simd<float, 8> s8 = v.select<8, 1>(0) + v.select<8, 1>(8);
  e::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
  e::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
  return s2[0] + s2[1];
}
float down_warp_dot(const float *x, const uint16_t *w, int warp) SYCL_ESIMD_FUNCTION {
  F16 acc[2] = {F16(0.f), F16(0.f)};
  for (int first = warp * 32; first < D / 2; first += 256) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int pair = first + j * 16;
      const auto bits = e::block_load<uint32_t, 16>(
          reinterpret_cast<const uint32_t *>(w) + pair);
      auto values = e::block_load<float, 32>(x + 2 * pair);
      e::simd<uint32_t, 16> b0 = bits << 16, b1 = bits & 0xffff0000u;
      acc[j] = ex::fma(F16(b0.bit_cast_view<float>()), F16(values.select<16, 2>(0)), acc[j]);
      acc[j] = ex::fma(F16(b1.bit_cast_view<float>()), F16(values.select<16, 2>(1)), acc[j]);
    }
  }
  return reduce16(acc[0] + acc[1]);
}
void down_esimd(FusedGrArgs a, void *stream) {
  const int rows = LR + (a.w_inject ? HC : 0);
  queue_for(stream).parallel_for(
      sycl::nd_range<1>(rows * 8, 16),
      [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        // Two rows per workgroup, eight programs per row. Every group is
        // complete because both supported row counts (320 and 324) are even.
        e::slm_init<16 * sizeof(float)>();
        const int tid = it.get_local_linear_id();
        const int row = it.get_global_linear_id() / 8, warp = tid % 8;
        const uint16_t *w = row < LR ? a.w_down + size_t(row) * D
                                    : a.w_inject + size_t(row - LR) * D;
        const float partial = down_warp_dot(a.xn, w, warp);
        e::slm_scalar_store<float>(tid * sizeof(float), partial);
        e::barrier();
        if (!warp) {
          F16 sums = 0.f;
          sums.select<8, 1>(0) = e::slm_block_load<float, 8>(tid * sizeof(float));
          const float dot = reduce16(sums + F16(0.f));
          if (row < LR) {
            const float v = scalar_fma(.25f, dot, 0.f);
            a.lo[row] = v / (1.f + sycl::exp(-v));
          } else a.inject_out[row - LR] = dot;
        }
      });
}
float up_dot(const float *x, const uint16_t *w) SYCL_ESIMD_FUNCTION {
  F16 acc[10];
#pragma unroll
  for (int j = 0; j < 10; ++j) {
    const auto bits = e::block_load<uint32_t, 16>(
        reinterpret_cast<const uint32_t *>(w) + j * 16);
    auto values = e::block_load<float, 32>(x + j * 32);
    e::simd<uint32_t, 16> b0 = bits << 16, b1 = bits & 0xffff0000u;
    acc[j] = ex::fma(F16(b0.bit_cast_view<float>()), F16(values.select<16, 2>(0)), F16(0.f));
    acc[j] = ex::fma(F16(b1.bit_cast_view<float>()), F16(values.select<16, 2>(1)), acc[j]);
  }
  F16 partial = 0.f;
#pragma unroll
  for (int k = 0; k < 5; ++k)
    partial[k] = reduce16(acc[2 * k] + acc[2 * k + 1]);
  // Match the zero-padded subgroup XOR tree, including its rounded additions.
  return reduce16(partial + F16(0.f));
}
void up_esimd(FusedGrArgs a, void *stream) {
  auto done = queue_for(stream).parallel_for(
      sycl::nd_range<1>(N, 16),
      [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const int d = it.get_global_linear_id();
        float mixed = 0.f;
        for (int c = 0; c < HC; ++c) {
          const int row = c * N + d;
          const float dot = up_dot(a.lo, a.w_up + size_t(row) * LR);
          if (a.gates) a.gates[row] = dot;
          const float gate = 1.f / (1.f + sycl::exp(-dot));
          if (a.w_inject)
            mixed = scalar_fma(a.xn[row], gate, mixed);
          else {
            const float product = scalar_fma(a.xn[row], gate, 0.f);
            mixed = c ? mixed + product : product;
          }
        }
        a.mixed[d] = scalar_fma(.25f, mixed, 0.f);
      });
  finish(stream, done);
}
float sum32(sycl::sub_group sg, float x) {
  for (int i = 16; i; i >>= 1)
    x += sycl::permute_group_by_xor(sg, x, i);
  return x;
}
void valid(const FusedGrArgs &a) {
  if (!std::isfinite(a.eps) || a.eps < 0 || (a.apply && a.R_out != a.R) ||
      (a.gates && !a.xn))
    throw std::invalid_argument(
        "invalid SYCL fused GR epsilon, residual destination or workspace");
  std::array<Span, 7> out{{{a.lo, LR * 4}, {a.rs, HC * 4}, {a.mixed, N * 4}}};
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
  if (a.xn)
    out[nw++] = {a.xn, D * 4};
  if (a.gates)
    out[nw++] = {a.gates, D * 4};
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
          if (a.xn) {
            // Only the first subgroup reads this reduction. It finishes its
            // shuffles before lane zero reuses part[0] for the scale.
            if (tid < 32) {
              sum = sum32(it.get_sub_group(), part[lane]);
              if (!tid) {
                const float scale = sycl::rsqrt(sum / norm_width + a.eps);
                a.rs[c] = scale;
                part[0] = scale;
              }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int d = tid; d < N; d += 1024) {
              const int i = c * N + d;
              // Preserve both FP32 product boundaries from the reconstructed
              // path, including when the previous residual write was folded in.
              a.xn[i] = sycl::fma(sycl::fma(part[0], a.R[i], 0.f), a.w_norm[i], 0.f);
            }
          } else {
            sum = sum32(it.get_sub_group(), part[lane]);
            if (!tid)
              a.rs[c] = sycl::rsqrt(sum / norm_width + a.eps);
          }
        });
  });
  if (a.xn) {
    down_esimd(a, stream);
    up_esimd(a, stream);
    return;
  }
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
  std::array<FusedGrArgs, kFusedGrMaxT> args;
  for (int t = 0; t < T; ++t) {
    args[t] = a[t];
    if (scratch)
      args[t].xn = scratch + size_t(t) * D;
    valid(args[t]);
    if (a[t].w_norm != a[0].w_norm || a[t].w_down != a[0].w_down ||
        a[t].w_up != a[0].w_up || a[t].w_inject != a[0].w_inject ||
        a[t].eps != a[0].eps)
      throw std::invalid_argument(
          "SYCL fused GR batch must share weights and epsilon");
  }
  if (scratch) {
    // The full batch workspace must also avoid other tokens' buffers.
    for (int t = 0; t < T; ++t) {
      const auto &b = a[t];
      validate_spans({{scratch, size_t(T) * D * 4}},
                     {{b.R, D * 4}, {b.lo, LR * 4}, {b.rs, HC * 4},
                      {b.mixed, N * 4}, {b.w_norm, D * 4},
                      {b.w_down, D * LR * 2}, {b.w_up, D * LR * 2}});
      if (b.apply)
        validate_spans({{scratch, size_t(T) * D * 4}},
                       {{b.bo_prev, N * 4}, {b.inj_prev, HC * 4}});
      if (b.w_inject)
        validate_spans({{scratch, size_t(T) * D * 4}},
                       {{b.w_inject, HC * D * 2}, {b.inject_out, HC * 4}});
      if (b.gates)
        validate_spans({{scratch, size_t(T) * D * 4}}, {{b.gates, D * 4}});
    }
  }
  for (int t = 0; t < T; ++t)
    launch(args[t], stream);
}
void fused_gr_check() { (void)sycl_backend::runtime_for(); }
int fused_gr_variant() {
  return 0;
} // One portable implementation; no CUDA variants.
} // namespace strata::kernels
