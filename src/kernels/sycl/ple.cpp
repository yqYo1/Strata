// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
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

#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <cmath>
#include <sycl/ext/intel/math.hpp>
#include <vector>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int N = NG_N_EMBD, H = NG_HC, D = NG_HC_DIM, HIST = NG_HIST;
std::atomic<bool> native_bf16{false}, native_postops{false};
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
Spans weights(const PleWeights &w) {
  return {{w.norm_key, D * 4},
          {w.norm_query, D * 4},
          {w.norm_conv, D * 4},
          {w.conv1d_f16, D * 8}};
}
void norm(sycl::queue &q, const float *x, const float *gamma, float *out,
          int tokens, bool native) {
  const int wg = native ? 1024 : 256;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> fp(32, h);
    sycl::local_accessor<double, 1> dp(32, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(tokens) * H * wg, wg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const size_t row = it.get_group_linear_id(), start = row * N;
          const int tid = it.get_local_linear_id(), lane = tid % 32;
          auto sg = it.get_sub_group();
          float scale;
          if (native) {
            float sum = 0;
            for (int j = tid; j < N; j += wg)
              sum = sycl::fma(x[start + j], x[start + j], sum);
            for (int off = 16; off; off /= 2)
              sum += sycl::permute_group_by_xor(sg, sum, off);
            if (!lane)
              fp[tid / 32] = sum;
            it.barrier(sycl::access::fence_space::local_space);
            sum = lane < wg / 32 ? fp[lane] : 0;
            for (int off = 16; off; off /= 2)
              sum += sycl::permute_group_by_xor(sg, sum, off);
            scale = sycl::rsqrt(sum / float(N) + NG_RMS_EPS);
          } else {
            double sum = 0;
            for (int j = tid; j < N; j += wg) {
              const float square = x[start + j] * x[start + j];
              sum += double(square);
            }
            for (int off = 16; off; off /= 2)
              sum += sycl::permute_group_by_xor(sg, sum, off);
            if (!lane)
              dp[tid / 32] = sum;
            it.barrier(sycl::access::fence_space::local_space);
            sum = lane < wg / 32 ? dp[lane] : 0;
            for (int off = 16; off; off /= 2)
              sum += sycl::permute_group_by_xor(sg, sum, off);
            const float mean =
                float(sycl::ext::intel::math::ddiv_rn(sum, double(N)));
            scale = 1.f / sycl::sqrt(mean + NG_RMS_EPS);
          }
          // All inputs have been read before any output is written, including
          // in-place batch keys.
          it.barrier(sycl::access::fence_space::local_space);
          for (int j = tid; j < N; j += wg)
            out[start + j] =
                (native ? scale * x[start + j] : x[start + j] * scale) *
                gamma[(row % H) * N + j];
        });
  });
}
void gate(sycl::queue &q, const float *key, const float *query, float *out,
          int tokens, bool native) {
  const int wg = native ? 512 : 256;
  const float factor = 1.f / std::sqrt(float(N));
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> fp(16, h);
    sycl::local_accessor<double, 1> dp(8, h);
    h.parallel_for(sycl::nd_range<1>(size_t(tokens) * H * wg, wg),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                     const size_t row = it.get_group_linear_id(),
                                  start = row * N;
                     const int tid = it.get_local_linear_id(), lane = tid % 32;
                     auto sg = it.get_sub_group();
                     float total;
                     if (native) {
                       float sum = 0;
                       for (int j = 0; j < 8; ++j) {
                         const int d = tid + j * 512;
                         const float product =
                             d < N ? key[start + d] * query[start + d] : 0.f;
                         sum += product;
                       }
                       for (int off = 16; off; off /= 2)
                         sum += sycl::permute_group_by_xor(sg, sum, off);
                       if (!lane)
                         fp[tid / 32] = sum;
                       it.barrier(sycl::access::fence_space::local_space);
                       sum = lane < 16 ? fp[lane] : 0;
                       for (int off = 16; off; off /= 2)
                         sum += sycl::permute_group_by_xor(sg, sum, off);
                       total = sycl::fma(factor, sum, 0.f);
                     } else {
                       double sum = 0;
                       for (int d = tid; d < N; d += wg)
                         sum += double(key[start + d] * query[start + d]);
                       for (int off = 16; off; off /= 2)
                         sum += sycl::permute_group_by_xor(sg, sum, off);
                       if (!lane)
                         dp[tid / 32] = sum;
                       it.barrier(sycl::access::fence_space::local_space);
                       sum = lane < 8 ? dp[lane] : 0;
                       for (int off = 16; off; off /= 2)
                         sum += sycl::permute_group_by_xor(sg, sum, off);
                       total = float(sum) * factor;
                     }
                     if (!tid) {
                       const float sign = float((total > 0) - (total < 0));
                       const float mag =
                           sycl::sqrt(sycl::fmax(sycl::fabs(total), 1e-6f));
                       out[row] = 1.f / (1.f + sycl::exp(-sign * mag));
                     }
                   });
  });
}
void broadcast(sycl::queue &q, const float *value, const float *g, float *out,
               int tokens) {
  q.parallel_for(sycl::range<1>(size_t(tokens) * D), [=](sycl::id<1> id) {
    const size_t i = id[0], t = i / D, d = i % D;
    out[i] = value[t * N + d % N] * g[t * H + d / N];
  });
}
sycl::event convolve(sycl::queue &q, const float *history,
                     const float *normalized, const uint16_t *w,
                     const float *hidden, const float *gated, float *conv,
                     float *result, int tokens, bool native) {
  return q.parallel_for(
      sycl::range<1>(size_t(tokens) * D), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int t = int(i / D), c = int(i % D);
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
          const int p = t - HIST + 3 * k;
          const float x = p < 0 ? history[size_t(c) * HIST + HIST + p]
                                : normalized[size_t(p) * D + c];
          const float term = x * f32_from_f16(w[c * 4 + k]);
          sum = k == 0 ? term : sum + term;
        }
        const float activation = sum / (1.f + sycl::exp(-sum));
        if (conv)
          conv[i] = activation;
        result[i] = native ? hidden[i] + (gated[i] + activation)
                           : (hidden[i] + gated[i]) + activation;
      });
}
sycl::event history_advance(sycl::queue &q, float *history,
                            const float *normalized, int tokens) {
  return q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
    const size_t c = id[0];
    float rows[HIST];
    for (int r = 0; r < HIST; ++r) {
      const int p = tokens - HIST + r;
      rows[r] = p < 0 ? history[c * HIST + tokens + r]
                      : normalized[size_t(p) * D + c];
    }
    for (int r = 0; r < HIST; ++r)
      history[c * HIST + r] = rows[r];
  });
}
} // namespace
void native_ple_postops(const float *key, const float *hidden,
                        const float *value, const float *history,
                        const PleWeights &w, const NativePlePostopsBuffers &b,
                        void *stream) {
  if (!stream)
    throw std::invalid_argument("native PLE requires an explicit queue");
  auto rd = weights(w);
  rd.insert(rd.end(), {{key, D * 4}, {value, N * 4}, {history, HIST * D * 4}});
  if (b.result != hidden)
    rd.push_back({hidden, D * 4});
  Spans wr{{b.key, D * 4},   {b.query, D * 4}, {b.gate, H * 4},
           {b.gated, D * 4}, {b.conv, D * 4},  {b.result, D * 4}};
  if (b.query != b.normalized)
    wr.push_back({b.normalized, D * 4});
  disjoint(wr, rd);
  auto &q = queue_for(stream);
  norm(q, key, w.norm_key, b.key, 1, true);
  norm(q, hidden, w.norm_query, b.query, 1, true);
  gate(q, b.key, b.query, b.gate, 1, true);
  broadcast(q, value, b.gate, b.gated, 1);
  norm(q, b.gated, w.norm_conv, b.normalized, 1, true);
  convolve(q, history, b.normalized, w.conv1d_f16, hidden, b.gated, b.conv,
           b.result, 1, true);
}
void native_ple_postops_batch(float *key, float *hidden, const float *value,
                              float *history, const PleWeights &w, float *query,
                              float *gated, float *g, int tokens,
                              void *stream) {
  if (!stream || tokens <= 0 || tokens > 65535)
    throw std::invalid_argument("invalid native PLE batch");
  const size_t bytes = size_t(tokens) * D * 4;
  auto rd = weights(w);
  rd.push_back({value, size_t(tokens) * N * 4});
  disjoint({{key, bytes},
            {hidden, bytes},
            {history, HIST * D * 4},
            {query, bytes},
            {gated, bytes},
            {g, size_t(tokens) * H * 4}},
           rd);
  auto &q = queue_for(stream);
  norm(q, key, w.norm_key, key, tokens, true);
  norm(q, hidden, w.norm_query, query, tokens, true);
  gate(q, key, query, g, tokens, true);
  broadcast(q, value, g, gated, tokens);
  norm(q, gated, w.norm_conv, query, tokens, true);
  convolve(q, history, query, w.conv1d_f16, hidden, gated, nullptr, hidden,
           tokens, true);
  history_advance(q, history, query, tokens);
}
void ple_set_native_bf16(bool value) { native_bf16.store(value); }
void ple_set_native_postops(bool value) { native_postops.store(value); }
bool ple_native_postops_enabled() { return native_postops.load(); }
bool ple_block_available() { return !sycl_backend::Runtime::devices().empty(); }
uint64_t ple_block_scratch_bytes() {
  const size_t f = size_t(5 * D + N + H) * 4, act = size_t(N / 32) * 34;
  return ((f + 15) & ~size_t(15)) + ((act + 15) & ~size_t(15)) + N * 2 + 256;
}
void ple_history_advance(float *hist, const float *normalized, void *stream) {
  validate_spans({{hist, HIST * D * 4}}, {{normalized, D * 4}});
  finish(stream, history_advance(queue_for(stream), hist, normalized, 1));
}
void ple_block(const float *emb, const float *hidden, const float *history,
               const PleWeights &w, PleOut &out, void *scratch, void *stream) {
  auto rd = weights(w);
  rd.insert(rd.end(), {{emb, N * 4},
                       {history, HIST * D * 4},
                       {w.value_bf16, size_t(N) * N * 2}});
  if (out.result != hidden)
    rd.push_back({hidden, D * 4});
  const bool nk = w.key_native_data && !w.key_bf16;
  if (nk && !stream)
    throw std::invalid_argument("native PLE key requires an explicit queue");
  if (w.key_bf16)
    rd.push_back({w.key_bf16, size_t(N) * D * 2});
  else if (nk)
    rd.push_back(
        {w.key_native_data, native_mmvq_weight_bytes(w.key_native_type, N, D)});
  else
    rd.insert(rd.end(), {{w.key_codes, size_t(D) * N / 4},
                         {w.key_scales, size_t(D) * (N / 64) * 4}});
  Spans wr{{scratch, size_t(ple_block_scratch_bytes())}, {out.result, D * 4}};
  for (auto span : Spans{{out.key, D * 4},
                         {out.value, N * 4},
                         {out.gate, H * 4},
                         {out.gated, D * 4},
                         {out.normalized, D * 4},
                         {out.conv, D * 4}})
    if (span.pointer)
      wr.push_back(span);
  if (nk)
    wr.push_back({w.key_native_q8_1, native_q8_1_bytes(N)});
  disjoint(wr, rd);
  if (reinterpret_cast<uintptr_t>(emb) % 8)
    throw std::invalid_argument("PLE embedding requires 8-byte alignment");
  auto &q = queue_for(stream);
  void *st = &q;
  auto base = static_cast<uint8_t *>(scratch);
  float *key = reinterpret_cast<float *>(base), *query = key + D,
        *normalized = query + D, *gated = normalized + D, *conv = gated + D,
        *value = conv + D, *g = value + N;
  auto act = base + ((size_t(5 * D + N + H) * 4 + 15) & ~size_t(15));
  auto emb16 = reinterpret_cast<uint16_t *>(
      act + ((size_t(N / 32) * 34 + 15) & ~size_t(15)));
  if (w.key_bf16)
    bf16_gemv_fp32_mmvf(emb, w.key_bf16, key, N, D, st);
  else if (nk) {
    native_quantize_q8_1(emb, w.key_native_q8_1, N, 1, st);
    native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1, key, N,
                D, 1, st);
  } else {
    quantize_q8_0(emb, act, N, st);
    s2_gemv_q8(act, w.key_codes, w.key_scales, key, N, D, 8, st);
  }
  if (native_bf16.load())
    bf16_gemv_fp32_mmvf(emb, w.value_bf16, value, N, N, st);
  else {
    q.parallel_for(sycl::range<1>(N),
                   [=](sycl::id<1> i) { emb16[i] = bf16_from_f32(emb[i]); });
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
      const size_t row = id[0];
      double sum = 0;
      for (int j = 0; j < N; ++j)
        sum += double(f32_from_bf16(emb16[j])) *
               double(f32_from_bf16(w.value_bf16[row * N + j]));
      value[row] = float(sum);
    });
  }
  const bool native = native_postops.load();
  norm(q, key, w.norm_key, key, 1, native);
  norm(q, hidden, w.norm_query, query, 1, native);
  gate(q, key, query, g, 1, native);
  broadcast(q, value, g, gated, 1);
  norm(q, gated, w.norm_conv, normalized, 1, native);
  auto event = convolve(q, history, normalized, w.conv1d_f16, hidden, gated,
                        conv, out.result, 1, native);
  const float *srcs[] = {key, value, g, gated, normalized, conv};
  float *dsts[] = {out.key,   out.value,      out.gate,
                   out.gated, out.normalized, out.conv};
  const size_t sizes[] = {D * 4, N * 4, H * 4, D * 4, D * 4, D * 4};
  for (int i = 0; i < 6; ++i)
    if (dsts[i])
      event = q.memcpy(dsts[i], srcs[i], sizes[i]);
  finish(stream, event);
}
} // namespace strata::kernels
