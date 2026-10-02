#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/sycl/launch.hpp"

// Pair accumulation and XOR reductions adapted from Strata native_bf16.cu,
// based on llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d MMVF.
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

namespace strata::kernels {
namespace {
using namespace sycl_backend;
void validate(const void *x, const uint16_t *w, float *y, int64_t n_in,
              int64_t n_out, int element_bytes, int64_t ldx, int64_t ldy,
              int tokens) {
  if (n_in <= 0 || n_out <= 0 || n_in > INT32_MAX || n_out > INT32_MAX ||
      tokens < 1 || tokens > 8 || ldx < n_in || ldy < n_out)
    throw std::invalid_argument("invalid SYCL BF16 GEMV geometry");
  // Include stride padding, so overlapping writable columns are rejected.
  const auto xbytes = checked_count(checked_count(tokens, ldx), element_bytes);
  const auto ybytes = checked_count(checked_count(tokens, ldy), 4);
  validate_spans(
      {{y, ybytes}},
      {{x, xbytes}, {w, checked_count(checked_count(n_in, n_out), 2)}});
}
float xor_sum(sycl::sub_group group, float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += sycl::permute_group_by_xor(group, value, offset);
  return value;
}
int block_size(int64_t n) {
  int best = 32;
  int64_t iterations = (n + 63) / 64;
  for (int candidate = 64; candidate <= 256; candidate += 32) {
    const auto next = (n + 2 * candidate - 1) / (2 * candidate);
    if (next < iterations) {
      iterations = next;
      best = candidate;
    }
  }
  return best;
}
template <int NT>
void native(const float *x, int64_t ldx, const uint16_t *w, float *y,
            int64_t ldy, int64_t n_in, int64_t n_out, int tokens,
            void *stream) {
  const int block = block_size(n_in);
  const auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partials(NT * 32, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(n_out) * block, block),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int t = int(item.get_local_linear_id());
          const auto row = item.get_group_linear_id();
          auto group = item.get_sub_group();
          float acc[NT] = {};
          if (block > 32) {
            if (t < 32)
              for (int k = 0; k < NT; ++k)
                partials[k * 32 + t] = 0;
            item.barrier(sycl::access::fence_space::local_space);
          }
          for (int64_t pair = t; pair < n_in / 2; pair += block) {
            const float w0 = f32_from_bf16(w[row * n_in + 2 * pair]);
            const float w1 = f32_from_bf16(w[row * n_in + 2 * pair + 1]);
            for (int k = 0; k < NT; ++k)
              if (k < tokens) {
                acc[k] = sycl::fma(w0, x[k * ldx + 2 * pair], acc[k]);
                acc[k] = sycl::fma(w1, x[k * ldx + 2 * pair + 1], acc[k]);
              }
          }
          for (int k = 0; k < NT; ++k)
            acc[k] = xor_sum(group, acc[k]);
          if (block > 32) {
            if (!(t & 31))
              for (int k = 0; k < NT; ++k)
                partials[k * 32 + t / 32] = acc[k];
            item.barrier(sycl::access::fence_space::local_space);
            if (t < 32)
              for (int k = 0; k < NT; ++k)
                acc[k] = xor_sum(group, partials[k * 32 + t]);
          }
          if (t == 0)
            for (int k = 0; k < NT; ++k)
              if (k < tokens)
                y[k * ldy + row] = acc[k];
        });
  });
  finish(stream, event);
}
void split(const uint16_t *x, const uint16_t *w, float *y, int64_t n_in,
           int64_t n_out, int threads, void *stream) {
  const auto event = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partials(threads, h);
    h.parallel_for(sycl::nd_range<1>(size_t(n_out) * threads, threads),
                   [=](sycl::nd_item<1> item) {
                     const int t = int(item.get_local_linear_id());
                     const auto row = item.get_group_linear_id();
                     float sum = 0;
                     for (int64_t i = t; i < n_in; i += threads)
                       sum = sycl::fma(f32_from_bf16(x[i]),
                                       f32_from_bf16(w[row * n_in + i]), sum);
                     partials[t] = sum;
                     item.barrier(sycl::access::fence_space::local_space);
                     for (int offset = threads / 2; offset; offset >>= 1) {
                       if (t < offset)
                         partials[t] += partials[t + offset];
                       item.barrier(sycl::access::fence_space::local_space);
                     }
                     if (!t)
                       y[row] = partials[0];
                   });
  });
  finish(stream, event);
}
} // namespace

void bf16_gemv_fp32_mmvf_multi(const float *x, int64_t ldx, const uint16_t *w,
                               float *y, int64_t ldy, int64_t n_in,
                               int64_t n_out, int tokens, void *stream) {
  validate(x, w, y, n_in, n_out, 4, ldx, ldy, tokens);
  if ((n_in & 1) || (ldx & 1) || (reinterpret_cast<uintptr_t>(x) & 7))
    throw std::invalid_argument("SYCL MMVF requires even input width/stride "
                                "and 8-byte input alignment");
  if (tokens == 1)
    native<1>(x, ldx, w, y, ldy, n_in, n_out, tokens, stream);
  else if (tokens <= 4)
    native<4>(x, ldx, w, y, ldy, n_in, n_out, tokens, stream);
  else
    native<8>(x, ldx, w, y, ldy, n_in, n_out, tokens, stream);
}
void bf16_gemv_fp32_mmvf(const float *x, const uint16_t *w, float *y,
                         int64_t n_in, int64_t n_out, void *stream) {
  bf16_gemv_fp32_mmvf_multi(x, n_in, w, y, n_out, n_in, n_out, 1, stream);
}
void bf16_gemv(const uint16_t *x, const uint16_t *w, float *y, int64_t n_in,
               int64_t n_out, void *stream) {
  validate(x, w, y, n_in, n_out, 2, n_in, n_out, 1);
  if (n_out >= 64) {
    split(x, w, y, n_in, n_out, 32, stream);
    return;
  }
  for_each(n_out, stream, [=](size_t row) {
    float sum = 0;
    for (int64_t i = 0; i < n_in; ++i)
      sum =
          sycl::fma(f32_from_bf16(x[i]), f32_from_bf16(w[row * n_in + i]), sum);
    y[row] = sum;
  });
}
void bf16_gemv_split(const uint16_t *x, const uint16_t *w, float *y,
                     int64_t n_in, int64_t n_out, int threads, void *stream) {
  validate(x, w, y, n_in, n_out, 2, n_in, n_out, 1);
  if (threads <= 0 || (threads & (threads - 1)) ||
      size_t(threads) >
          queue_for(stream)
              .get_device()
              .get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument(
        "SYCL BF16 split requires a supported power-of-two group size");
  split(x, w, y, n_in, n_out, threads, stream);
}
} // namespace strata::kernels
