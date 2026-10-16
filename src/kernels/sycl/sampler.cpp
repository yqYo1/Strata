// Strata's sampler chain and counter mapping, implemented directly in SYCL.
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int threads = 256;
float uniform(uint64_t seed, uint64_t counter) {
  uint32_t c0 = uint32_t(counter), c1 = uint32_t(counter >> 32),
           c2 = uint32_t(seed), c3 = uint32_t(seed >> 32);
  for (uint32_t r = 0; r < 10; ++r) {
    const uint32_t n0 = sycl::mul_hi(0xBB67AE85u, c2) ^ c1 ^ r,
                   n1 = 0xBB67AE85u * c2;
    const uint32_t n2 = sycl::mul_hi(0x9E3779B9u, c0) ^ c3,
                   n3 = 0x9E3779B9u * c0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
  }
  return float(c0 >> 8) * (1.f / 16777216.f);
}
float penalize(float v, int count, const SamplerParams &p) {
  if (count > 0) {
    v = v <= 0 ? v * p.penalty_repeat : v / p.penalty_repeat;
    v -= sycl::fma(float(count), p.penalty_freq, p.penalty_present);
  }
  return v;
}
int count_history(const int *h, int n, int id) {
  int count = 0;
  for (int i = 0; i < n; ++i)
    count += h[i] == id;
  return count;
}
void first(float &v, int &id, float other, int oi) {
  if (other > v || (other == v && oi < id)) {
    v = other;
    id = oi;
  }
}
void subgroup_first(sycl::sub_group sg, float &v, int &id) {
  for (int off = 16; off; off >>= 1) {
    const float ov = sycl::permute_group_by_xor(sg, v, off);
    const int oi = sycl::permute_group_by_xor(sg, id, off);
    first(v, id, ov, oi);
  }
}
// Selection is parallel across each vocabulary row. The <=64-entry tail uses
// ordered FP64 sums, matching the engine's distribution contract.
template <bool coupled>
void launch(const float *logits, int T, int V, const int *history,
            int history_len, SamplerParams params,
            const SamplerParams *dynamic_params, const int32_t *step,
            const int32_t *mapping, int *out, float *probability, int32_t *ring,
            int ring_at, void *stream) {
  const bool bitmap =
      !coupled && history && params.penalty_last_n > 0 && V <= 262144;
  const size_t words = bitmap ? (size_t(V) + 31) / 32 : 1;
  auto e = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> bits(words, h);
    sycl::local_accessor<float, 1> part(8, h), values(64, h);
    sycl::local_accessor<int, 1> ids(8, h), selected(64, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(T) * threads, threads),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(),
                    row = it.get_group_linear_id(), lane = tid % 32;
          SamplerParams p = params;
          if constexpr (coupled) {
            p = *dynamic_params;
            p.counter = uint64_t(int64_t(step[0]) + 1);
          }
          const int len =
              history ? sycl::clamp(p.penalty_last_n, 0, history_len) : 0;
          const int *tail =
              history ? history + size_t(row) * history_len + history_len - len
                      : nullptr;
          if (bitmap) {
            for (size_t i = tid; i < words; i += threads)
              bits[i] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            for (int i = tid; i < len; i += threads) {
              const int id = tail[i];
              if (id >= 0 && id < V) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>
                    a(bits[id / 32]);
                a.fetch_or(1u << (id % 32));
              }
            }
            it.barrier(sycl::access::fence_space::local_space);
          }
          const bool greedy = p.greedy || p.temperature <= 0;
          const int K =
              greedy ? 1
                     : sycl::min(V, p.top_k > 0 && p.top_k < 64 ? p.top_k : 64);
          float prev = INFINITY;
          int prev_id = -1;
          for (int choice = 0; choice < K; ++choice) {
            float best = -INFINITY;
            int best_id = V;
            for (int v = tid; v < V; v += threads) {
              float value = logits[size_t(row) * V + v];
              if constexpr (!coupled) {
                if (len && (!bitmap || (bits[v / 32] & (1u << (v % 32)))))
                  value = penalize(value, count_history(tail, len, v), p);
              }
              if ((value < prev || (value == prev && v > prev_id)) &&
                  value > best) {
                best = value;
                best_id = v;
              }
            }
            subgroup_first(it.get_sub_group(), best, best_id);
            if (!lane) {
              part[tid / 32] = best;
              ids[tid / 32] = best_id;
            }
            it.barrier(sycl::access::fence_space::local_space);
            best = lane < 8 ? part[lane] : -INFINITY;
            best_id = lane < 8 ? ids[lane] : V;
            subgroup_first(it.get_sub_group(), best, best_id);
            if (tid == 0) {
              values[choice] = best;
              selected[choice] = best_id < V ? best_id : 0;
            }
            prev = best;
            prev_id = best_id < V ? best_id : 0;
            it.barrier(sycl::access::fence_space::local_space);
          }
          if (tid == 0) {
            int pick = selected[0];
            float prob = 1;
            if (!greedy) {
              double ex[64];
              int keep = K;
              if (p.top_p < 1) {
                double sum = 0;
                for (int i = 0; i < K; ++i) {
                  ex[i] = sycl::exp(double(values[i]) - double(values[0]));
                  sum += ex[i];
                }
                double cum = 0;
                for (int i = 0; i < K; ++i) {
                  cum += ex[i] / sum;
                  if (cum >= double(p.top_p)) {
                    keep = i + 1;
                    break;
                  }
                }
                if (keep < p.min_keep)
                  keep = sycl::min(p.min_keep, K);
              }
              if (p.min_p > 0) {
                const float threshold = values[0] + sycl::log(p.min_p);
                for (int i = 0; i < keep; ++i)
                  if (values[i] < threshold) {
                    keep = i;
                    break;
                  }
              }
              const float inv = 1.f / p.temperature, max = values[0] * inv;
              double sum = 0;
              for (int i = 0; i < keep; ++i) {
                ex[i] = sycl::exp(double(values[i] * inv) - double(max));
                sum += ex[i];
              }
              for (int i = 0; i < keep; ++i)
                ex[i] /= sum;
              int index = keep > 0 ? keep - 1 : 0;
              double cum = 0;
              const float u = uniform(p.seed, p.counter + uint64_t(row));
              for (int i = 0; i < keep; ++i) {
                cum += ex[i];
                if (double(u) < cum) {
                  index = i;
                  break;
                }
              }
              pick = selected[index];
              prob = keep > 0 ? float(ex[index]) : 1.f;
            }
            if constexpr (coupled) {
              pick = mapping ? mapping[pick] : pick;
              ring[ring_at] = pick;
              *probability = prob;
            }
            out[row] = pick;
          }
        });
  });
  finish(stream, e);
}
} // namespace
void sample_tokens(const float *logits, int T, int V, const int *history,
                   int history_len, const SamplerParams &p, int *out,
                   void *stream) {
  if (T <= 0 || V <= 0)
    return;
  if (history_len < 0 || (p.penalty_last_n > 0 && (!history || !history_len)))
    throw std::invalid_argument("SYCL sampling penalties require a history");
  validate_spans({{out, size_t(T) * 4}}, {{logits, checked_count(T, V) * 4}});
  if (history && history_len)
    validate_spans({{out, size_t(T) * 4}},
                   {{history, checked_count(T, history_len) * 4}});
  launch<false>(logits, T, V, history, history_len, p, nullptr, nullptr,
                nullptr, out, nullptr, nullptr, 0, stream);
}
size_t coupled_draft_scratch_bytes(int V) {
  // Keep the public split-sampler storage contract. This allocation-free
  // baseline selects within one work-group and does not consume the reserved
  // workspace.
  return V > 0 && V <= 262144 ? size_t((V + 4095) / 4096) * 64 * 8 : 0;
}
void coupled_draft_stage(const SamplerParams *src, const int32_t *hist,
                         SamplerParams *dst, int32_t *ring, int cap,
                         void *stream) {
  if (cap < 0)
    throw std::invalid_argument("negative SYCL draft history capacity");
  validate_spans({{dst, sizeof(SamplerParams)}, {ring, size_t(cap) * 4}},
                 {{src, sizeof(SamplerParams)}, {hist, size_t(cap) * 4}});
  // Caller publishes host USM before launch; no concurrent mapped-host polling.
  auto e = queue_for(stream).single_task([=] {
    *dst = *src;
    const int n = sycl::clamp(src->penalty_last_n, 0, cap);
    for (int i = cap - n; i < cap; ++i)
      ring[i] = hist[i];
  });
  finish(stream, e);
}
void coupled_draft_sample(float *logits, int V, const int32_t *mapping,
                          const int32_t *inverse, int id_vocab,
                          const SamplerParams *params, int32_t *ring, int cap,
                          int j, const int32_t *step, void *scratch,
                          int32_t *out, float *prob, void *stream) {
  const auto bytes = coupled_draft_scratch_bytes(V);
  if (!bytes || !scratch || cap < 0 || j < 0 || j > INT32_MAX - cap - 1 ||
      id_vocab <= 0 || (!mapping && V > id_vocab))
    throw std::invalid_argument("invalid SYCL coupled sampling geometry");
  validate_spans({{logits, size_t(V) * 4},
                  {ring, size_t(cap + j + 1) * 4},
                  {out, 4},
                  {prob, 4},
                  {scratch, bytes}},
                 {{params, sizeof(SamplerParams)}, {step, 4}});
  if (mapping)
    validate_spans({{logits, size_t(V) * 4},
                    {ring, size_t(cap + j + 1) * 4},
                    {out, 4},
                    {prob, 4}},
                   {{mapping, size_t(V) * 4}});
  if (inverse)
    validate_spans({{logits, size_t(V) * 4},
                    {ring, size_t(cap + j + 1) * 4},
                    {out, 4},
                    {prob, 4}},
                   {{inverse, size_t(id_vocab) * 4}});
  for_each(V, stream, [=](size_t i) {
    const int id = mapping ? mapping[i] : int(i);
    if (id < 0 || id >= id_vocab || (inverse && inverse[id] != int(i)))
      return;
    const auto p = *params;
    const int n = sycl::clamp(p.penalty_last_n, 0, cap);
    logits[i] =
        penalize(logits[i], count_history(ring + cap + j - n, n, id), p);
  });
  launch<true>(logits, 1, V, nullptr, 0, SamplerParams{}, params, step, mapping,
               out, prob, ring, cap + j, stream);
}
} // namespace strata::kernels
