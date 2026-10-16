// Specialized from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cuda/{fattn-vec.cuh,fattn-common.cuh,common.cuh}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <array>
#include <cfloat>
#include <cmath>
namespace strata::kernels {
namespace {
float rounded_product(float a, float b) { return a * b; }
template <int Width> float warp_sum(sycl::sub_group sg, float x) {
  for (int o = Width / 2; o; o >>= 1)
    x += sycl::permute_group_by_xor(sg, x, o);
  return x;
}
float warp_max(sycl::sub_group sg, float x) {
  for (int o = 16; o; o >>= 1)
    x = sycl::fmax(x, sycl::permute_group_by_xor(sg, x, o));
  return x;
}
} // namespace
void native_flash_attn_short_step(const float *q, const uint16_t *k,
                                  const uint16_t *v, const int32_t *step,
                                  int64_t capacity, int max_context,
                                  const QsaShapes &shapes, float *out,
                                  int32_t *status, const uint16_t *mask,
                                  void *stream) {
  using namespace sycl_backend;
  if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 ||
      shapes.head_dim != 256 || shapes.idx_block != 4 ||
      shapes.idx_top_k < 256 || capacity < 256 ||
      uint64_t(capacity) > SIZE_MAX / 1024 || max_context < 1 ||
      max_context > 256)
    throw std::invalid_argument("SYCL native short attention requires "
                                "Q24x256/KV2x256 and context 1..256");
  struct S {
    const void *p;
    size_t bytes, align;
  };
  std::array<S, 7> spans = {{{q, 24 * 256 * 4, 4},
                             {k, size_t(capacity) * 1024, 2},
                             {v, size_t(capacity) * 1024, 2},
                             {step, kStepCount * 4, 4},
                             {out, 24 * 256 * 4, 4},
                             {status, 4, 4}}};
  if (mask)
    spans[6] = {mask, 256 * 2, 2};
  for (size_t i = 0; i < size_t(mask ? 7 : 6); ++i) {
    auto a = spans[i];
    validate_spans({{a.p, a.bytes}}, {}, a.align);
    for (size_t j = 0; j < i; ++j)
      validate_spans({{a.p, a.bytes}}, {{spans[j].p, spans[j].bytes}}, 1);
  }
  queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile(4 * 4 * 256, h), max_shared(32, h),
        sum_shared(32, h);
    h.parallel_for(
        sycl::nd_range<1>(24 * 128, 128),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          constexpr int padded_length = 256;
          constexpr float scale = .0625f;
          const int tid = item.get_local_linear_id(), lane = tid % 32,
                    warp = tid / 32;
          const auto sg = item.get_sub_group();
          const int head = item.get_group_linear_id(), kv = head / 12;
          const int width = step[kStepWidth], nkv = step[kStepNKv];
          const bool valid = width >= 1 && width <= max_context &&
                             nkv == width && step[kStepPos] == width - 1 &&
                             step[kStepNBid] == width / 4;
          if (head == 0 && tid == 0)
            *status = valid ? kNativeFlashAttnSuccess
                            : kNativeFlashAttnUnsupportedStep;
          if (!valid) {
            out[head * 256 + tid] = sycl::nan(0u);
            out[head * 256 + tid + 128] = sycl::nan(0u);
            return;
          }
          sycl::float2 qreg[16];
          sycl::float2 vkq[16] = {};
          float maximum = -FLT_MAX / 2.0f, sum = 0.0f;
#pragma unroll
          for (int i0 = 0; i0 < 128; i0 += 32) {
            const int i = i0 + (lane % 8) * 4;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int d = 2 * (i + j);
              qreg[i0 / 8 + j] = sycl::float2(q[head * 256 + d] * scale,
                                              q[head * 256 + d + 1] * scale);
            }
          }
          for (int base = 0; base < padded_length; base += 128) {
            float score = 0.0f, next_max = maximum;
#pragma unroll
            for (int row = 0; row < 8; ++row) {
              const int cell = base + warp * 32 + (lane & ~7) + row;
              float dot = 0.0f;
#pragma unroll
              for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const int d = 2 * (i + j);
                  const float a =
                      cell < width ? f32_from_f16(k[(cell * 2 + kv) * 256 + d])
                                   : 0.0f;
                  const float b =
                      cell < width
                          ? f32_from_f16(k[(cell * 2 + kv) * 256 + d + 1])
                          : 0.0f;
                  dot = sycl::fma(a, qreg[i0 / 8 + j].x(), dot);
                  dot = sycl::fma(b, qreg[i0 / 8 + j].y(), dot);
                }
              }
              dot = warp_sum<8>(sg, dot);
              dot += cell < width ? (mask ? f32_from_f16(mask[cell]) : 0.0f)
                                  : -INFINITY;
              next_max = sycl::fmax(next_max, dot + (3.0f * 0.6931f));
              if (lane % 8 == row)
                score = dot;
            }
#pragma unroll
            for (int offset = 8; offset < 32; offset <<= 1)
              next_max = sycl::fmax(
                  next_max, sycl::permute_group_by_xor(sg, next_max, offset));
            const float rescale = sycl::exp(maximum - next_max);
            maximum = next_max;
            score = sycl::exp(score - maximum);
            sum = sycl::fma(sum, rescale, score);
            tile[tid] = score;
            sycl::group_barrier(sg, sycl::memory_scope::sub_group);
#pragma unroll
            for (int k0 = 0; k0 < 32; k0 += 4) {
              const int local = warp * 32 + k0 + lane / 8, cell = base + local;
              const float weight = tile[local];
#pragma unroll
              for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const int d = 2 * (i + j);
                  const float a =
                      cell < width ? f32_from_f16(v[(cell * 2 + kv) * 256 + d])
                                   : 0.0f;
                  const float b =
                      cell < width
                          ? f32_from_f16(v[(cell * 2 + kv) * 256 + d + 1])
                          : 0.0f;
                  // The pinned sm120a binary contracts the old-accumulator
                  // rescale with the FIRST addition:
                  // fma(rescale,old,round(V*w)). Later columns use
                  // fma(V,w,acc). Explicit intrinsics retain that order despite
                  // this adapter's masked-load branches.
                  if (k0 == 0) {
                    vkq[i0 / 8 + j].x() =
                        sycl::fma(rescale, vkq[i0 / 8 + j].x(),
                                  rounded_product(a, weight));
                    vkq[i0 / 8 + j].y() =
                        sycl::fma(rescale, vkq[i0 / 8 + j].y(),
                                  rounded_product(b, weight));
                  } else {
                    vkq[i0 / 8 + j].x() =
                        sycl::fma(a, weight, vkq[i0 / 8 + j].x());
                    vkq[i0 / 8 + j].y() =
                        sycl::fma(b, weight, vkq[i0 / 8 + j].y());
                  }
                }
              }
            }
          }
          if (warp == 0) {
            max_shared[lane] = -FLT_MAX / 2.0f;
            sum_shared[lane] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (lane == 0)
            max_shared[warp] = maximum;
          item.barrier(sycl::access::fence_space::local_space);
          const float global_max = warp_max(sg, max_shared[lane]);
          const float rescale = sycl::exp(maximum - global_max);
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            vkq[i].x() = rounded_product(vkq[i].x(), rescale);
            vkq[i].y() = rounded_product(vkq[i].y(), rescale);
          }
#pragma unroll
          for (int i0 = 0; i0 < 128; i0 += 32) {
            const int start =
                warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              tile[start + 2 * j] = vkq[i0 / 8 + j].x();
              tile[start + 2 * j + 1] = vkq[i0 / 8 + j].y();
            }
          }
          sum *= rescale;
          sum = warp_sum<32>(sg, sum);
          if (lane == 0)
            sum_shared[warp] = sum;
          item.barrier(sycl::access::fence_space::local_space);
          sum = warp_sum<32>(sg, sum_shared[lane]);
#pragma unroll
          for (int i0 = 0; i0 < 256; i0 += 128) {
            float result = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
              for (int group = 0; group < 4; ++group)
                result += tile[w * 4 * 256 + group * 256 + i0 + tid];
            }
            out[head * 256 + i0 + tid] = result / sum;
          }
        });
  });
}
} // namespace strata::kernels
