// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/sycl/sform.hpp"
#include <algorithm>
#include <atomic>
#include <vector>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
std::atomic<bool> native_bf16{false}, native_combine{false};
using Spans = std::vector<Span>;
void disjoint(const Spans &writes, const Spans &reads) {
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
void dimensions(int64_t n, int64_t f) {
  if (n <= 0 || f <= 0 || n > INT_MAX || f > INT_MAX || n % 32 || f % 32)
    throw std::invalid_argument("invalid SYCL shared expert dimensions");
}
size_t aligned(size_t value) { return (value + 15) & ~size_t(15); }
void scalar_gate(sycl::queue &q, const float *x, const uint16_t *xb,
                 const uint16_t *weight, float *gate, int n, int tokens,
                 bool native) {
  if (native) {
    bf16_gemv_fp32_mmvf_multi(x, n, weight, gate, 1, n, 1, tokens, &q);
    q.parallel_for(sycl::range<1>(tokens), [=](sycl::id<1> i) {
      gate[i] = 1.f / (1.f + sycl::exp(-gate[i]));
    });
  } else {
    q.parallel_for(sycl::nd_range<1>(size_t(tokens) * 256, 256),
                   [=](sycl::nd_item<1> it) {
                     const size_t t = it.get_group_linear_id();
                     double sum = 0;
                     for (int i = it.get_local_linear_id(); i < n; i += 256)
                       sum += double(f32_from_bf16(xb[t * n + i])) *
                              double(f32_from_bf16(weight[i]));
                     sum = sycl::reduce_over_group(it.get_group(), sum,
                                                   sycl::plus<double>());
                     if (it.get_local_linear_id() == 0)
                       gate[t] = float(1. / (1. + sycl::exp(-sum)));
                   });
  }
}
void swiglu(sycl::queue &q, float *gate, const float *up, size_t count,
            bool native) {
  q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
    const float x = gate[id];
    const float act = native ? x / (1.f + sycl::exp(-x))
                             : float(double(x) / (1. + sycl::exp(-double(x))));
    gate[id] = act * up[id];
  });
}
sycl::event scale(sycl::queue &q, float *out, const float *gate, int n,
                  int tokens) {
  return q.parallel_for(sycl::range<1>(size_t(n) * tokens),
                        [=](sycl::id<1> id) { out[id] *= gate[id[0] / n]; });
}
void combine(const float *parts, const float *weights, const float *shared,
             float *out, int64_t n, int64_t k, int tokens, void *stream,
             bool native) {
  if (n <= 0 || n > INT_MAX || k <= 0 || k > (native ? 15 : 64) ||
      tokens <= 0 || tokens > 65535 || (native && !stream))
    throw std::invalid_argument("invalid SYCL MoE reduction dimensions/stream");
  const size_t outbytes = checked_count(n, tokens) * 4;
  validate_spans({{out, outbytes}},
                 {{parts, checked_count(int64_t(tokens) * k, n) * 4},
                  {weights, size_t(tokens) * k * 4}});
  if (shared)
    validate_spans({{out, outbytes}}, {{shared, outbytes}});
  auto event = queue_for(stream).parallel_for(
      sycl::range<1>(size_t(tokens) * n), [=](sycl::id<1> id) {
        const size_t t = id[0] / n, d = id[0] % n;
        if (native) {
          float sum = parts[t * k * n + d] * weights[t * k];
          for (int e = 1; e < k; ++e)
            sum =
                sycl::fma(parts[(t * k + e) * n + d], weights[t * k + e], sum);
          if (shared)
            sum += shared[id];
          out[id] = sum;
        } else {
          double sum = 0;
          for (int e = 0; e < k; ++e)
            sum +=
                double(parts[(t * k + e) * n + d]) * double(weights[t * k + e]);
          if (shared)
            sum += double(shared[id]);
          out[id] = float(sum);
        }
      });
  finish(stream, event);
}
} // namespace
void shared_expert_set_native_bf16(bool enabled) { native_bf16.store(enabled); }
void native_moe_combine_set_enabled(bool enabled) {
  native_combine.store(enabled);
}
bool native_moe_combine_enabled() { return native_combine.load(); }
void native_moe_combine(const float *p, const float *w, const float *s,
                        float *o, int64_t n, int64_t k, void *st) {
  combine(p, w, s, o, n, k, 1, st, true);
}
void native_moe_combine_multi(const float *p, const float *w, const float *s,
                              float *o, int64_t n, int64_t k, int t, void *st) {
  combine(p, w, s, o, n, k, t, st, true);
}
void moe_combine(const float *p, const float *w, const float *s, float *o,
                 int64_t n, int64_t k, void *st) {
  combine(p, w, s, o, n, k, 1, st, false);
}
uint64_t shared_expert_scratch_bytes(int64_t f) {
  dimensions(32, f);
  return aligned(size_t(f) * 4) * 2 + aligned(size_t(f / 32) * 34) +
         aligned(size_t(f / 256) * 292) + 32;
}
void shared_expert_multi(int tokens, const float *x, const uint16_t *xb,
                         const NativeSharedWeights &nw, const uint16_t *wi,
                         float *gate, float *up, float *g, float *out,
                         int64_t n, int64_t f, void *stream) {
  dimensions(n, f);
  if (!stream || tokens < 1 || tokens > 8)
    throw std::invalid_argument(
        "shared expert multi needs 1..8 tokens and explicit queue");
  const bool ng = native_bf16.load();
  Spans rd{{x, size_t(tokens) * n * 4},
           {wi, size_t(n) * 2},
           {nw.gate_data, native_mmvq_weight_bytes(nw.gate_type, n, f)},
           {nw.up_data, native_mmvq_weight_bytes(nw.up_type, n, f)},
           {nw.down_data, native_mmvq_weight_bytes(nw.down_type, f, n)}};
  if (!ng)
    rd.push_back({xb, size_t(tokens) * n * 2});
  disjoint({{gate, size_t(tokens) * f * 4},
            {up, size_t(tokens) * f * 4},
            {g, size_t(tokens) * 4},
            {out, size_t(tokens) * n * 4},
            {nw.q8_1, size_t(tokens) * native_q8_1_bytes(std::max(n, f))}},
           rd);
  if (reinterpret_cast<uintptr_t>(x) % 8)
    throw std::invalid_argument("shared input requires 8-byte alignment");
  auto &q = queue_for(stream);
  native_quantize_q8_1(x, nw.q8_1, n, tokens, &q);
  native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, n, f, tokens, &q);
  native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, n, f, tokens, &q);
  swiglu(q, gate, up, size_t(tokens) * f, true);
  native_quantize_q8_1(gate, nw.q8_1, f, tokens, &q);
  native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, f, n, tokens, &q);
  scalar_gate(q, x, xb, wi, g, n, tokens, ng);
  scale(q, out, g, n, tokens);
}
void shared_expert(const uint8_t *x0, const uint8_t *xk, const uint16_t *xb,
                   const SForm &gf, const uint8_t *gc, const float *gs,
                   const float *go, const SForm &uf, const uint8_t *uc,
                   const float *us, const float *uo, const SForm &df,
                   const uint8_t *dc, const float *ds, const float *doff,
                   const uint16_t *wi, float *scratch, float *out, int64_t n,
                   int64_t f, int threads, void *stream, const float *x,
                   const NativeSharedWeights *nw) {
  dimensions(n, f);
  const bool ng = native_bf16.load();
  const bool gn = nw && nw->gate_data && native_mmvq_supported(nw->gate_type),
             un = nw && nw->up_data && native_mmvq_supported(nw->up_type),
             dn = nw && nw->down_data && native_mmvq_supported(nw->down_type),
             any = gn || un || dn;
  if ((any && !stream) || threads <= 0 || (threads & (threads - 1)))
    throw std::invalid_argument("invalid shared expert stream/work group");
  Spans rd{{wi, size_t(n) * 2}},
      wr{{scratch, size_t(shared_expert_scratch_bytes(f))},
         {out, size_t(n) * 4}};
  if (ng || gn || un) {
    rd.push_back({x, size_t(n) * 4});
    if (reinterpret_cast<uintptr_t>(x) % 8)
      throw std::invalid_argument("shared input requires 8-byte alignment");
  }
  if (!ng)
    rd.push_back({xb, size_t(n) * 2});
  if (any)
    wr.push_back({nw->q8_1, native_q8_1_bytes(std::max(n, f))});
  auto projection = [&](bool native, int type, const void *data,
                        const SForm &form, const uint8_t *codes,
                        const float *scales, const float *offset, int64_t ni,
                        int64_t no, bool input) {
    if (native)
      rd.push_back({data, native_mmvq_weight_bytes(type, ni, no)});
    else {
      validate_sform(form, ni, no);
      if (form.act_kind == 1 && ni % 256)
        throw std::invalid_argument(
            "Q8_K activation dimension must divide 256");
      rd.push_back({codes, checked_count(no, ni / (8 / form.code_bits))});
      rd.push_back({scales, checked_count(no, ni / form.group_elems) * 4});
      if (form.has_offset)
        rd.push_back({offset, checked_count(no, ni / form.group_elems) * 4});
      if (input)
        rd.push_back(form.act_kind == 1 ? Span{xk, size_t(ni / 256) * 292}
                                        : Span{x0, size_t(ni / 32) * 34});
    }
  };
  projection(gn, nw ? nw->gate_type : -1, nw ? nw->gate_data : nullptr, gf, gc,
             gs, go, n, f, true);
  projection(un, nw ? nw->up_type : -1, nw ? nw->up_data : nullptr, uf, uc, us,
             uo, n, f, true);
  projection(dn, nw ? nw->down_type : -1, nw ? nw->down_data : nullptr, df, dc,
             ds, doff, f, n, false);
  disjoint(wr, rd);
  auto &q = queue_for(stream);
  void *st = &q;
  if (size_t(threads) >
      q.get_device().get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument("unsupported shared work group");
  auto base = reinterpret_cast<uint8_t *>(scratch);
  const size_t a = aligned(size_t(f) * 4),
               q0bytes = aligned(size_t(f / 32) * 34),
               qkbytes = aligned(size_t(f / 256) * 292);
  auto gate = scratch;
  auto up = reinterpret_cast<float *>(base + a);
  auto h0 = base + 2 * a, hk = h0 + q0bytes;
  auto g = reinterpret_cast<float *>(hk + qkbytes);
  auto canonical = [&](const SForm &form, const uint8_t *codes,
                       const float *scales, const float *offset,
                       const uint8_t *a0, const uint8_t *ak, float *y,
                       int64_t ni, int64_t no) {
    if (form.code_bits == 2 && form.code_bias == -1 && form.group_elems == 64 &&
        !form.has_offset && form.act_kind == 0)
      s2_gemv_q8(a0, codes, scales, y, ni, no, threads, st);
    else if (form.act_kind == 1)
      s_gemv_q8k_split(ak, codes, scales, offset, y, ni, no, form, st);
    else
      s_gemv_q8_0_split(a0, codes, scales, offset, y, ni, no, form, st);
  };
  if (gn || un)
    native_quantize_q8_1(x, nw->q8_1, n, 1, st);
  if (gn)
    native_mmvq(nw->gate_type, nw->gate_data, nw->q8_1, gate, n, f, 1, st);
  else
    canonical(gf, gc, gs, go, x0, xk, gate, n, f);
  if (un)
    native_mmvq(nw->up_type, nw->up_data, nw->q8_1, up, n, f, 1, st);
  else
    canonical(uf, uc, us, uo, x0, xk, up, n, f);
  swiglu(q, gate, up, f, any);
  if (dn) {
    native_quantize_q8_1(gate, nw->q8_1, f, 1, st);
    native_mmvq(nw->down_type, nw->down_data, nw->q8_1, out, f, n, 1, st);
  } else {
    if (df.act_kind == 1)
      quantize_q8_K(gate, hk, f, st);
    else
      quantize_q8_0(gate, h0, f, st);
    canonical(df, dc, ds, doff, h0, hk, out, f, n);
  }
  scalar_gate(q, x, xb, wi, g, n, 1, ng);
  finish(stream, scale(q, out, g, n, 1));
}
} // namespace strata::kernels
