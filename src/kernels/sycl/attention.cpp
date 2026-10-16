// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
//
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <climits>
#include <cmath>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
std::atomic<bool> native_enabled{false};
size_t shape(int64_t heads, int64_t dim) {
  const size_t n = checked_count(heads, dim);
  if (heads <= 0 || dim <= 0 || n > INT_MAX)
    throw std::invalid_argument("invalid SYCL attention shape");
  return n;
}
float sum32(sycl::sub_group sg, float x) {
  for (int o = 16; o; o /= 2)
    x += sycl::permute_group_by_xor(sg, x, o);
  return x;
}
float max32(sycl::sub_group sg, float x) {
  for (int o = 16; o; o /= 2)
    x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, o));
  return x;
}
void attend(const float *query, const uint16_t *keys, const uint16_t *values,
            const int32_t *step, int64_t cap, const QsaShapes &s, float *out,
            float *weights, void *stream) {
  const size_t n = shape(s.n_head, s.head_dim);
  if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv || cap < 0 || cap > INT_MAX)
    throw std::invalid_argument("invalid SYCL attention capacity/GQA");
  const size_t kv = checked_count(cap, checked_count(s.n_head_kv, s.head_dim));
  if (kv > SIZE_MAX / 2)
    throw std::invalid_argument("SYCL attention size overflow");
  validate_spans({{out, n * 4}}, {{query, n * 4}});
  if (cap)
    validate_spans({{out, n * 4}}, {{keys, kv * 2}, {values, kv * 2}});
  if (step)
    validate_spans({{out, n * 4}}, {{step, kStepCount * 4}});
  if (weights && cap) {
    const size_t wn = checked_count(s.n_head, cap);
    validate_spans({{weights, wn * 4}, {out, n * 4}},
                   {{query, n * 4}, {keys, kv * 2}, {values, kv * 2}});
    if (step)
      validate_spans({{weights, wn * 4}}, {{step, kStepCount * 4}});
  }
  auto &queue = queue_for(stream);
  if ((size_t(cap) + 32) * sizeof(float) >
      queue.get_device().get_info<sycl::info::device::local_mem_size>())
    throw std::invalid_argument(
        "SYCL gathered attention exceeds local memory; use split attention");
  const float scale = 1.f / std::sqrt(float(s.head_dim));
  auto e = queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> w(size_t(cap) + 32, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(size_t(s.n_head) * 256, 256),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int h = it.get_group_linear_id(), t = it.get_local_linear_id(),
                    lane = t % 32, warp = t / 32;
          const int64_t count = step ? step[kStepWidth] : cap;
          const int kvh = h / (s.n_head / s.n_head_kv);
          auto sg = it.get_sub_group();
          if (count <= 0 || count > cap) {
            for (int64_t d = t; d < s.head_dim; d += 256)
              out[h * s.head_dim + d] = count == 0 ? 0.f : sycl::nan(0u);
            return;
          }
          for (int64_t j = t; j < count; j += 256) {
            float dot = 0;
            for (int64_t d = 0; d < s.head_dim; ++d)
              dot +=
                  f32_from_f16(keys[(j * s.n_head_kv + kvh) * s.head_dim + d]) *
                  query[h * s.head_dim + d];
            w[j + 32] = dot * scale;
          }
          it.barrier(sycl::access::fence_space::local_space);
          float mx = -INFINITY;
          for (int64_t j = t; j < count; j += 256)
            mx = sycl::fmax(mx, w[j + 32]);
          mx = max32(sg, mx);
          if (!lane)
            w[warp] = mx;
          it.barrier(sycl::access::fence_space::local_space);
          if (!warp) {
            mx = max32(sg, lane < 8 ? w[lane] : -INFINITY);
            if (!lane)
              w[0] = mx;
          }
          it.barrier(sycl::access::fence_space::local_space);
          mx = w[0];
          it.barrier(sycl::access::fence_space::local_space);
          float sum = 0;
          for (int64_t j = t; j < count; j += 256) {
            float v = sycl::exp(w[j + 32] - mx);
            w[j + 32] = v;
            sum += v;
          }
          sum = sum32(sg, sum);
          if (!lane)
            w[warp] = sum;
          it.barrier(sycl::access::fence_space::local_space);
          if (!warp) {
            sum = sum32(sg, lane < 8 ? w[lane] : 0.f);
            if (!lane)
              w[0] = 1.f / sum;
          }
          it.barrier(sycl::access::fence_space::local_space);
          const float inv = w[0];
          for (int64_t d = t; d < s.head_dim; d += 256) {
            float acc = 0;
            for (int64_t j = 0; j < count; ++j)
              acc += (w[j + 32] * inv) *
                     f32_from_f16(
                         values[(j * s.n_head_kv + kvh) * s.head_dim + d]);
            out[h * s.head_dim + d] = acc;
          }
          if (weights)
            for (int64_t j = t; j < count; j += 256)
              weights[h * count + j] = w[j + 32] * inv;
        });
  });
  finish(stream, e);
}
template <bool Native, typename Output>
void gate(const float *attn, const float *full, int64_t heads, int64_t dim,
          Output *out, void *stream) {
  const size_t n = shape(heads, dim);
  if constexpr (Native)
    if (!stream)
      throw std::invalid_argument("native QSA requires explicit queue");
  validate_spans({{out, n * sizeof(Output)}}, {{full, n * 8}});
  validate_spans({{attn, n * 4}}, {{full, n * 8}});
  if (static_cast<const void *>(attn) != static_cast<void *>(out))
    validate_spans({{out, n * sizeof(Output)}}, {{attn, n * 4}});
  else if constexpr (sizeof(Output) != 4)
    throw std::invalid_argument("QSA FP16 output cannot alias FP32 input");
  auto e =
      queue_for(stream).parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t i = id[0], h = i / dim, d = i % dim;
        const float raw = full[h * 2 * dim + dim + d];
        float value;
        if constexpr (Native)
          value = attn[i] * (1.f / (1.f + sycl::exp(-raw)));
        else
          value =
              float(double(attn[i]) * (1. / (1. + sycl::exp(-double(raw)))));
        if constexpr (sizeof(Output) == 2)
          out[i] = f16_from_f32(value);
        else
          out[i] = value;
      });
  finish(stream, e);
}
} // namespace
void qsa_attend(const float *q, const uint16_t *k, const uint16_t *v,
                int64_t count, const QsaShapes &s, float *a, float *w,
                void *stream) {
  attend(q, k, v, nullptr, count, s, a, w, stream);
}
void qsa_attend_step(const float *q, const uint16_t *k, const uint16_t *v,
                     const int32_t *step, int64_t cap, const QsaShapes &s,
                     float *a, float *w, void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  attend(q, k, v, step, cap, s, a, w, stream);
}
void qsa_gate_apply(const float *a, const float *f, const QsaShapes &s,
                    uint16_t *out, void *stream) {
  gate<false>(a, f, s.n_head, s.head_dim, out, stream);
}
void qsa_gate_apply_f32(const float *a, const float *f, const QsaShapes &s,
                        float *out, void *stream) {
  gate<false>(a, f, s.n_head, s.head_dim, out, stream);
}
void native_qsa_set_enabled(bool value) { native_enabled.store(value); }
bool native_qsa_enabled() { return native_enabled.load(); }
void native_qsa_gate_apply(const float *a, const float *f, float *out,
                           int heads, int dim, void *stream) {
  gate<true>(a, f, heads, dim, out, stream);
}
void native_qsa_rms_norm_weighted(const float *input, const float *gamma,
                                  float *output, int cols, int rows, float eps,
                                  void *stream) {
  const size_t n = shape(rows, cols);
  if (!stream || !std::isfinite(eps) || eps < 0)
    throw std::invalid_argument(
        "native QSA requires explicit queue and finite nonnegative epsilon");
  sycl_backend::validate_spans({{output, n * 4}}, {{gamma, size_t(cols) * 4}});
  sycl_backend::validate_spans({{input, n * 4}}, {{gamma, size_t(cols) * 4}});
  if (input != output)
    sycl_backend::validate_spans({{output, n * 4}}, {{input, n * 4}});
  const size_t wg = cols < 1024 ? 256 : 1024;
  auto &q = sycl_backend::queue_for(stream);
  if (wg > q.get_device().get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument("native QSA work group unsupported");
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sums(32, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(size_t(rows) * wg, wg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const size_t row = it.get_group_linear_id(),
                       t = it.get_local_linear_id(), lane = t % 32;
          float partial = 0;
          for (size_t c = t; c < size_t(cols); c += wg) {
            float x = input[row * cols + c];
            partial = sycl::fma(x, x, partial);
          }
          auto sg = it.get_sub_group();
          partial = sum32(sg, partial);
          if (!lane)
            sums[t / 32] = partial;
          it.barrier(sycl::access::fence_space::local_space);
          partial = sum32(sg, lane < wg / 32 ? sums[lane] : 0.f);
          const float scale = sycl::rsqrt(partial / cols + eps);
          for (size_t c = t; c < size_t(cols); c += wg)
            output[row * cols + c] = (scale * input[row * cols + c]) * gamma[c];
        });
  });
}
} // namespace strata::kernels
