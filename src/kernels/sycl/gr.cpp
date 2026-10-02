// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32
// RMSNorm.
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

#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <cmath>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
std::atomic<bool> fp32_activations{false}, native_mmvf{false};
size_t shape(int64_t n, int64_t hc) {
  const auto count = checked_count(n, hc);
  if (n <= 0 || hc <= 0 || count > INT32_MAX)
    throw std::invalid_argument("SYCL GR requires positive bounded dimensions");
  return count;
}
void epsilon_check(float eps) {
  if (!std::isfinite(eps) || eps < 0)
    throw std::invalid_argument("SYCL GR requires finite nonnegative epsilon");
}
float subgroup_sum(sycl::sub_group group, float value, bool xor_tree) {
  const int lane = int(group.get_local_linear_id());
  for (int offset = 16; offset; offset >>= 1) {
    if (xor_tree)
      value += sycl::permute_group_by_xor(group, value, offset);
    else {
      const auto other = sycl::shift_group_left(group, value, offset);
      if (lane + offset < 32)
        value += other;
    }
  }
  return sycl::group_broadcast(group, value, 0);
}
float sigmoid(float x) { return 1.f / (1.f + sycl::exp(-x)); }
void norm(const float *input, const float *gamma, float *output, int cols,
          int rows, float eps, bool native, void *stream) {
  const int block = native && cols >= 1024 ? 1024 : 256;
  const auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partials(32, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(rows) * block, block),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = int(item.get_local_linear_id()), lane = tid % 32;
          const size_t start = item.get_group_linear_id() * cols;
          const auto group = item.get_sub_group();
          float sum = 0;
          for (int col = tid; col < cols; col += block)
            sum = sycl::fma(input[start + col], input[start + col], sum);
          sum = subgroup_sum(group, sum, native);
          if (!lane)
            partials[tid / 32] = sum;
          item.barrier(sycl::access::fence_space::local_space);
          sum = lane < block / 32 ? partials[lane] : 0;
          sum = subgroup_sum(group, sum, native);
          const float rs = sycl::rsqrt(sum / float(cols) + eps);
          for (int col = tid; col < cols; col += block)
            output[start + col] =
                (rs * input[start + col]) * gamma[start + col];
        });
  });
  finish(stream, event);
}
float activation(float v) { return v; }
float activation(uint16_t v) { return f32_from_bf16(v); }
template <typename T>
void project(const T *x, const uint16_t *w, float *y, int width, int rows,
             int threads, void *stream) {
  // Legacy GR uses 256 lanes for down and one subgroup for up/injection.
  const auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partials(8, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(rows) * threads, threads),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = int(item.get_local_linear_id()), lane = tid % 32;
          const auto row = item.get_group_linear_id();
          const auto group = item.get_sub_group();
          float sum = 0;
          for (int i = tid; i < width; i += threads)
            sum = sycl::fma(activation(x[i]), f32_from_bf16(w[row * width + i]),
                            sum);
          sum = subgroup_sum(group, sum, false);
          if (threads > 32) {
            if (!lane)
              partials[tid / 32] = sum;
            item.barrier(sycl::access::fence_space::local_space);
            sum = lane < threads / 32 ? partials[lane] : 0;
            sum = subgroup_sum(group, sum, false);
          }
          if (!tid)
            y[row] = sum;
        });
  });
  finish(stream, event);
}
} // namespace
void native_gr_rms_norm_weighted(const float *input, const float *gamma,
                                 float *output, int cols, int rows, float eps,
                                 void *stream) {
  const auto bytes = shape(cols, rows) * 4;
  epsilon_check(eps);
  validate_spans({{output, bytes}}, {{input, bytes}, {gamma, bytes}});
  norm(input, gamma, output, cols, rows, eps, true, stream);
}
void native_gr_down_silu(float *lo, int count, int hc, void *stream) {
  shape(count, hc);
  validate_spans({{lo, size_t(count) * 4}}, {});
  const float scale = 1.f / float(hc);
  for_each(count, stream, [=](size_t i) {
    const float x = sycl::fma(scale, lo[i], 0.f);
    lo[i] = x / (1.f + sycl::exp(-x));
  });
}
void native_gr_pre_gated(const float *xn, float *gate, float *mixed, int n,
                         int hc, bool fused, void *stream) {
  const auto bytes = shape(n, hc) * 4;
  validate_spans({{gate, bytes}, {mixed, size_t(n) * 4}}, {{xn, bytes}});
  const float scale = 1.f / float(hc);
  for_each(n, stream, [=](size_t d) {
    float sum = 0;
    for (int c = 0; c < hc; ++c) {
      const size_t i = size_t(c) * n + d;
      const float x = xn[i], w = sigmoid(gate[i]);
      const float product = x * w;
      gate[i] = product;
      if (fused)
        sum = sycl::fma(x, w, sum);
      else
        sum = c == 0 ? product : sum + product;
    }
    mixed[d] = fused ? scale * sum : sycl::fma(scale, sum, 0.f);
  });
}
void native_gr_post(const float *residual, const float *block,
                    const float *inject, float *output, int n, int hc,
                    void *stream) {
  const auto count = shape(n, hc);
  if (output == residual)
    validate_spans({{output, count * 4}},
                   {{block, size_t(n) * 4}, {inject, size_t(hc) * 4}});
  else
    validate_spans({{output, count * 4}}, {{residual, count * 4},
                                           {block, size_t(n) * 4},
                                           {inject, size_t(hc) * 4}});
  const float scale = 1.f / float(hc);
  for_each(count, stream, [=](size_t i) {
    const float weight =
        sycl::fma(2.f, sigmoid(sycl::fma(scale, inject[i / n], 0.f)), 0.f);
    output[i] = sycl::fma(block[i % n], weight, residual[i]);
  });
}
void gr_set_fp32_activations(bool enabled) {
  fp32_activations.store(enabled, std::memory_order_relaxed);
}
void gr_set_native_mmvf(bool enabled) {
  native_mmvf.store(enabled, std::memory_order_relaxed);
}
size_t gr_workspace_init(const GrShapes &s, void *base, GrWorkspace &out) {
  const auto dim = shape(s.n_embd, s.hc);
  shape(s.hc_lr, 1);
  if (base && reinterpret_cast<uintptr_t>(base) % 16)
    throw std::invalid_argument("SYCL GR workspace requires 16-byte alignment");
  const size_t sizes[] = {dim * 4, dim * 2, size_t(s.hc_lr) * 2, dim * 4,
                          size_t(s.hc_lr) * 4};
  uintptr_t ptrs[5]{};
  size_t total = 0;
  const uintptr_t address = reinterpret_cast<uintptr_t>(base);
  for (int i = 0; i < 5; ++i) {
    ptrs[i] = base ? address + total : 0;
    total += (sizes[i] + 15) & ~size_t(15);
  }
  if (total > UINTPTR_MAX - address)
    throw std::invalid_argument("SYCL GR workspace address overflow");
  out = {reinterpret_cast<float *>(ptrs[0]),
         reinterpret_cast<uint16_t *>(ptrs[1]),
         reinterpret_cast<uint16_t *>(ptrs[2]),
         reinterpret_cast<float *>(ptrs[3]),
         reinterpret_cast<float *>(ptrs[4]),
         total};
  return total;
}
void gr_read(const float *R, const float *wn, const uint16_t *wd,
             const uint16_t *wu, const uint16_t *wi, float eps,
             const GrShapes &s, const GrWorkspace &ws, float *mixed,
             float *inject, void *stream) {
  const auto dim = shape(s.n_embd, s.hc);
  shape(s.hc_lr, 1);
  epsilon_check(eps);
  GrWorkspace expected;
  const auto need = gr_workspace_init(s, ws.xn, expected);
  if (!ws.xn || ws.bytes < need || expected.xq != ws.xq ||
      expected.lq != ws.lq || expected.gated != ws.gated ||
      expected.lo != ws.lo)
    throw std::invalid_argument(
        "SYCL GR workspace is uninitialized or too small");
  const auto matrix = checked_count(checked_count(dim, s.hc_lr), 2);
  if (wi)
    validate_spans({{ws.xn, need},
                    {mixed, size_t(s.n_embd) * 4},
                    {inject, size_t(s.hc) * 4}},
                   {{R, dim * 4},
                    {wn, dim * 4},
                    {wd, matrix},
                    {wu, matrix},
                    {wi, checked_count(checked_count(dim, s.hc), 2)}});
  else
    validate_spans({{ws.xn, need}, {mixed, size_t(s.n_embd) * 4}},
                   {{R, dim * 4}, {wn, dim * 4}, {wd, matrix}, {wu, matrix}});
  const int n = int(s.n_embd), hc = int(s.hc), rank = int(s.hc_lr),
            width = int(dim);
  const bool use_native = native_mmvf.load(std::memory_order_relaxed);
  const bool use_float =
      use_native || fp32_activations.load(std::memory_order_relaxed);
  if (use_native && ((width & 1) || (rank & 1)))
    throw std::invalid_argument(
        "SYCL native GR requires even projection widths");
  // Resolve null once; composition stays asynchronous until the final wait.
  auto &queue = queue_for(stream);
  void *st = &queue;
  norm(R, wn, ws.xn, n, hc, eps, use_native, st);
  if (use_native) {
    bf16_gemv_fp32_mmvf(ws.xn, wd, ws.lo, width, rank, st);
    native_gr_down_silu(ws.lo, rank, hc, st);
    bf16_gemv_fp32_mmvf(ws.lo, wu, ws.gated, rank, width, st);
    native_gr_pre_gated(ws.xn, ws.gated, mixed, n, hc, wi != nullptr, st);
    if (wi)
      bf16_gemv_fp32_mmvf(ws.xn, wi, inject, width, hc, st);
  } else {
    if (use_float)
      project(ws.xn, wd, ws.lo, width, rank, 256, st);
    else {
      for_each(dim, st, [=](size_t i) { ws.xq[i] = bf16_from_f32(ws.xn[i]); });
      project(ws.xq, wd, ws.lo, width, rank, 256, st);
    }
    for_each(rank, st, [=](size_t i) {
      const float v = ws.lo[i] / float(hc);
      ws.lo[i] = v / (1.f + sycl::exp(-v));
      if (!use_float)
        ws.lq[i] = bf16_from_f32(ws.lo[i]);
    });
    if (use_float)
      project(ws.lo, wu, ws.gated, rank, width, 32, st);
    else
      project(ws.lq, wu, ws.gated, rank, width, 32, st);
    for_each(n, st, [=](size_t d) {
      float sum = 0;
      for (int c = 0; c < hc; ++c) {
        const auto i = size_t(c) * n + d;
        const float v = ws.xn[i] * sigmoid(ws.gated[i]);
        ws.gated[i] = v;
        sum += v;
      }
      mixed[d] = sum / float(hc);
    });
    if (wi) {
      if (use_float)
        project(ws.xn, wi, inject, width, hc, 32, st);
      else
        project(ws.xq, wi, inject, width, hc, 32, st);
    }
  }
  if (!stream)
    runtime_for()->wait();
}
void gr_write(const float *R, const float *block, const float *inject,
              const GrShapes &s, float *out, void *stream) {
  const auto count = shape(s.n_embd, s.hc);
  if (native_mmvf.load(std::memory_order_relaxed)) {
    native_gr_post(R, block, inject, out, int(s.n_embd), int(s.hc), stream);
    return;
  }
  if (out == R)
    validate_spans({{out, count * 4}},
                   {{block, size_t(s.n_embd) * 4}, {inject, size_t(s.hc) * 4}});
  else
    validate_spans({{out, count * 4}}, {{R, count * 4},
                                        {block, size_t(s.n_embd) * 4},
                                        {inject, size_t(s.hc) * 4}});
  const int n = int(s.n_embd), hc = int(s.hc);
  for_each(count, stream, [=](size_t i) {
    const float weight = 2.f * sigmoid(inject[i / n] / float(hc));
    out[i] = sycl::fma(block[i % n], weight, R[i]);
  });
}
} // namespace strata::kernels
