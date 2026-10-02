// SYCL port of src/kernels/cuda/native_router.cu, adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d topk-moe.cu/common.cuh.
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
#include "strata/kernels/native_router.hpp"
#include "strata/sycl/launch.hpp"

#include <algorithm>
#include <atomic>
#include <limits>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
float subgroup_sum(sycl::sub_group group, float value) {
  for (int mask = 16; mask; mask >>= 1)
    value += sycl::permute_group_by_xor(group, value, mask);
  return value;
}
bool valid(const void *pointer, size_t bytes) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  return pointer && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const auto ap = reinterpret_cast<uintptr_t>(a),
             bp = reinterpret_cast<uintptr_t>(b);
  return ap < bp + bn && bp < ap + an;
}
} // namespace

void native_router_set_enabled(bool value) {
  enabled.store(value, std::memory_order_relaxed);
}
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float *logits, int32_t *ids, float *weights,
                         void *stream) {
  native_router_top10_multi(logits, ids, weights, 1, stream);
}
void native_router_top10_multi(const float *logits, int32_t *ids,
                               float *weights, int tokens, void *stream) {
  if (!stream || tokens < 1)
    throw std::invalid_argument(
        "native SYCL router requires a queue and tokens");
  const auto input_bytes = size_t(tokens) * 512 * 4,
             output_bytes = size_t(tokens) * 10 * 4;
  if (!valid(logits, input_bytes) || !valid(ids, output_bytes) ||
      !valid(weights, output_bytes) ||
      overlaps(logits, input_bytes, ids, output_bytes) ||
      overlaps(logits, input_bytes, weights, output_bytes) ||
      overlaps(ids, output_bytes, weights, output_bytes))
    throw std::invalid_argument(
        "native SYCL router requires aligned disjoint spans");
  auto &queue = sycl_backend::queue_for(stream);
  const auto sizes =
      queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(sizes.begin(), sizes.end(), size_t{32}) == sizes.end())
    throw std::runtime_error("native SYCL router requires subgroup size 32");
  const float negative_inf = -std::numeric_limits<float>::infinity();
  queue.parallel_for(
      sycl::nd_range<1>(size_t(tokens) * 32, 32),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const auto group = item.get_sub_group();
        const auto lane = item.get_local_linear_id();
        const auto token = item.get_group_linear_id();
        float values[16];
        float maximum = negative_inf;
        for (int i = 0; i < 16; ++i) {
          values[i] = logits[token * 512 + lane + i * 32];
          maximum = sycl::fmax(maximum, values[i]);
        }
        for (int mask = 16; mask; mask >>= 1)
          maximum = sycl::fmax(
              maximum, sycl::permute_group_by_xor(group, maximum, mask));
        float sum = 0;
        for (int i = 0; i < 16; ++i) {
          values[i] = sycl::exp(values[i] - maximum);
          sum += values[i];
        }
        const float inverse = 1.f / subgroup_sum(group, sum);
        for (int i = 0; i < 16; ++i)
          values[i] *= inverse;
        float selected = 0, selected_sum = 0;
        for (int rank = 0; rank < 10; ++rank) {
          float best = values[0];
          int expert = int(lane);
          for (int i = 1; i < 16; ++i) {
            if (values[i] > best) {
              best = values[i];
              expert = int(lane) + i * 32;
            }
          }
          for (int mask = 16; mask; mask >>= 1) {
            const float other = sycl::permute_group_by_xor(group, best, mask);
            const int other_id =
                sycl::permute_group_by_xor(group, expert, mask);
            if (other > best || (other == best && other_id < expert)) {
              best = other;
              expert = other_id;
            }
          }
          if (size_t(expert & 31) == lane) {
            values[expert / 32] = negative_inf;
            ids[token * 10 + rank] = expert;
            selected_sum += best;
          }
          if (size_t(rank) == lane)
            selected = best;
        }
        selected_sum = sycl::fmax(subgroup_sum(group, selected_sum), 0x1p-14f);
        if (lane < 10)
          weights[token * 10 + lane] = selected * (1.f / selected_sum);
      });
}
} // namespace strata::kernels
