// Native recurrence arithmetic is adapted from Strata's CUDA port of llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d gated_delta_net.cu.
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
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>

namespace strata::kernels {
using namespace sycl_backend;
namespace {
std::atomic<bool> native_enabled{false};

void shape_check(const GdnShapes &shape) {
  if (shape.S <= 0 || shape.S > 128 || shape.h_k <= 0 || shape.h_v <= 0 ||
      shape.h_v % shape.h_k)
    throw std::invalid_argument(
        "GDN requires S in [1,128] and divisible positive head counts");
  checked_count(checked_count(shape.S, shape.h_v), shape.S);
}

void native_count(int64_t count, void *stream) {
  if (!stream || count < 1 || count > 65535)
    throw std::invalid_argument(
        "native GDN requires a queue and count in [1,65535]");
}

void native_norm(int64_t rows, int64_t cols, float epsilon, void *stream) {
  native_count(rows, stream);
  if (cols != 128 || !std::isfinite(epsilon) || epsilon < 0)
    throw std::invalid_argument("native GDN normalization requires width 128 "
                                "and nonnegative finite epsilon");
}

template <typename Acc, bool mean, bool scale_after>
void normalize(const float *input, const float *gamma, const float *z,
               float *output, int64_t rows, int64_t cols, float epsilon,
               void *stream) {
  checked_count(rows, cols);
  if (!rows)
    return;
  if (cols <= 0 || !std::isfinite(epsilon) || epsilon < 0)
    throw std::invalid_argument("invalid GDN normalization geometry");
  constexpr size_t width = 128;
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(checked_count(rows, width), width),
      [=](sycl::nd_item<1> item) {
        const size_t row = item.get_group_linear_id(),
                     lane = item.get_local_linear_id();
        Acc sum = 0;
        for (size_t c = lane; c < size_t(cols); c += width) {
          const Acc value = input[row * cols + c];
          sum += value * value;
        }
        sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<Acc>());
        if constexpr (mean)
          sum /= Acc(cols);
        const float inverse = float(Acc(1) / sycl::sqrt(sum + Acc(epsilon)));
        for (size_t c = lane; c < size_t(cols); c += width) {
          float value = input[row * cols + c] * inverse;
          if constexpr (scale_after)
            value = sycl::fma(value, 1.f / sycl::sqrt(float(cols)), 0.f);
          if (gamma)
            value *= gamma[c];
          if (z)
            value *= 1.f / (1.f + sycl::exp(-z[row * cols + c]));
          output[row * cols + c] = value;
        }
      });
  finish(stream, event);
}
} // namespace

void gdn_step(float *state, const float *q, const float *k, const float *v,
              const float *gate, const float *beta, float *output,
              const GdnShapes &shape, void *stream) {
  shape_check(shape);
  const auto s = shape.S, heads = shape.h_v, key_heads = shape.h_k;
  for_each(checked_count(heads, s), stream, [=](size_t column) {
    const size_t head = column / s, source = head % key_heads;
    const float decay = sycl::exp(gate[head]);
    const size_t stride = heads * s;
    float projected = 0;
    for (int64_t i = 0; i < s; ++i) {
      const float value = state[i * stride + column] * decay;
      state[i * stride + column] = value;
      projected += value * k[source * s + i];
    }
    const float delta = (v[column] - projected) * beta[head];
    float result = 0;
    for (int64_t i = 0; i < s; ++i) {
      const float value =
          state[i * stride + column] + k[source * s + i] * delta;
      state[i * stride + column] = value;
      result += value * q[source * s + i];
    }
    output[column] = result;
  });
}

void gdn_conv_step(float *history, const float *input, const float *weights,
                   float *output, int64_t channels, int64_t taps,
                   void *stream) {
  if (taps < 1)
    throw std::invalid_argument("GDN convolution requires at least one tap");
  checked_count(channels, taps);
  for_each(channels, stream, [=](size_t c) {
    float sum = 0;
    for (int64_t t = 0; t < taps - 1; ++t)
      sum += history[c * (taps - 1) + t] * weights[c * taps + t];
    sum += input[c] * weights[c * taps + taps - 1];
    output[c] = sum;
    for (int64_t t = 0; t < taps - 2; ++t)
      history[c * (taps - 1) + t] = history[c * (taps - 1) + t + 1];
    if (taps > 1)
      history[c * (taps - 1) + taps - 2] = input[c];
  });
}
void gdn_l2_norm(float *input, int64_t rows, int64_t cols, float epsilon,
                 void *stream) {
  normalize<double, false, false>(input, nullptr, nullptr, input, rows, cols,
                                  epsilon, stream);
}
void gdn_beta_gate(float *beta, int64_t heads, void *stream) {
  for_each(heads, stream,
           [=](size_t i) { beta[i] = 1.f / (1.f + sycl::exp(-beta[i])); });
}
void gdn_out_norm(const float *output, const float *z, const float *gamma,
                  float *destination, int64_t heads, int64_t cols,
                  float epsilon, void *stream) {
  normalize<double, true, false>(output, gamma, z, destination, heads, cols,
                                 epsilon, stream);
}

void native_gdn_set_enabled(bool value) {
  native_enabled.store(value, std::memory_order_relaxed);
}
bool native_gdn_enabled() {
  return native_enabled.load(std::memory_order_relaxed);
}

void native_gdn_step(float *state, const float *q, const float *k,
                     const float *v, const float *gate, const float *beta,
                     float *output, const GdnShapes &shape, void *stream) {
  shape_check(shape);
  native_count(shape.h_v, stream);
  if (shape.S != 128)
    throw std::invalid_argument("native GDN requires S=128");
  const auto heads = shape.h_v, key_heads = shape.h_k;
  const size_t state_bytes = size_t(heads) * 128 * 128 * 4,
               vector_bytes = size_t(heads) * 128 * 4;
  const size_t qk_bytes = size_t(key_heads) * 128 * 4,
               head_bytes = size_t(heads) * 4;
  validate_spans({{state, state_bytes}, {output, vector_bytes}},
                 {{q, qk_bytes},
                  {k, qk_bytes},
                  {v, vector_bytes},
                  {gate, head_bytes},
                  {beta, head_bytes}});
  auto &queue = queue_for(stream);
  const auto sizes =
      queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(sizes.begin(), sizes.end(), size_t{32}) == sizes.end())
    throw std::runtime_error("native GDN requires subgroup size 32");
  queue.parallel_for(
      sycl::nd_range<1>(size_t(heads) * 128 * 32, 128),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const auto group = item.get_sub_group();
        const size_t lane = group.get_local_linear_id();
        const size_t column = item.get_global_linear_id() / 32;
        const size_t head = column / 128, source = head % key_heads;
        float shards[4], keys[4], queries[4];
        float projected = 0;
        for (int r = 0; r < 4; ++r) {
          const size_t row = r * 32 + lane;
          shards[r] = state[row * heads * 128 + column];
          keys[r] = k[source * 128 + row];
          queries[r] = q[source * 128 + row];
          projected = sycl::fma(shards[r], keys[r], projected);
        }
        for (int mask = 16; mask; mask >>= 1)
          projected += sycl::permute_group_by_xor(group, projected, mask);
        const float decay = sycl::exp(gate[head]);
        const float delta =
            sycl::fma(-decay, projected, v[column]) * beta[head];
        float result = 0;
        for (int r = 0; r < 4; ++r) {
          shards[r] = sycl::fma(keys[r], delta, decay * shards[r]);
          result = sycl::fma(shards[r], queries[r], result);
          state[(r * 32 + lane) * heads * 128 + column] = shards[r];
        }
        for (int mask = 16; mask; mask >>= 1)
          result += sycl::permute_group_by_xor(group, result, mask);
        if (!lane)
          output[column] = result * (1.f / sycl::sqrt(128.f));
      });
}

void native_gdn_conv_silu(float *history, const float *input,
                          const float *weights, float *raw, float *activated,
                          int64_t channels, int64_t taps, void *stream) {
  native_count(channels, stream);
  if (taps != 4)
    throw std::invalid_argument("native GDN convolution requires four taps");
  const size_t bytes = size_t(channels) * 4;
  validate_spans({{history, bytes * 3}, {raw, bytes}, {activated, bytes}},
                 {{input, bytes}, {weights, bytes * 4}});
  for_each(channels, stream, [=](size_t c) {
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2],
                       input[c]};
    float sum = 0;
    for (int t = 0; t < 4; ++t)
      sum = sycl::fma(values[t], weights[c * 4 + t], sum);
    sum += 0.f;
    raw[c] = sum;
    activated[c] = sum / (1.f + sycl::exp(-sum));
    for (int t = 0; t < 3; ++t)
      history[c * 3 + t] = values[t + 1];
  });
}
void native_gdn_l2_norm(float *input, int64_t rows, int64_t cols, float epsilon,
                        void *stream) {
  native_norm(rows, cols, epsilon, stream);
  validate_spans({{input, size_t(rows) * 128 * 4}}, {});
  normalize<float, true, true>(input, nullptr, nullptr, input, rows, cols,
                               epsilon / 128.f, stream);
}
void native_gdn_beta_gate(float *beta, int64_t heads, void *stream) {
  native_count(heads, stream);
  validate_spans({{beta, size_t(heads) * 4}}, {});
  gdn_beta_gate(beta, heads, stream);
}
void native_gdn_gate(const float *alpha, const float *dt, const float *a,
                     float *gate, int64_t heads, void *stream) {
  native_count(heads, stream);
  const size_t bytes = size_t(heads) * 4;
  validate_spans({{gate, bytes}}, {{alpha, bytes}, {dt, bytes}, {a, bytes}});
  gdn_gate(alpha, dt, a, gate, 1, heads, stream);
}
void native_gdn_out_norm(const float *output, const float *z,
                         const float *gamma, float *destination, int64_t heads,
                         int64_t cols, float epsilon, void *stream) {
  native_norm(heads, cols, epsilon, stream);
  const size_t bytes = size_t(heads) * 128 * 4;
  validate_spans({{destination, bytes}},
                 {{output, bytes}, {z, bytes}, {gamma, 128 * 4}});
  normalize<float, true, false>(output, gamma, z, destination, heads, cols,
                                epsilon, stream);
}
} // namespace strata::kernels
