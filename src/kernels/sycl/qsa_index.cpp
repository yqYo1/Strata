#include "strata/kernels/mrope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/sycl/launch.hpp"
#include <bit>
#include <climits>
#include <cmath>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
void geometry(const QsaShapes &s) {
  if (s.idx_dim <= 0 || s.idx_dim > 1024 || s.idx_n_head <= 0 ||
      s.idx_n_head > 32 || s.idx_block <= 0 || s.idx_block > INT_MAX ||
      s.idx_top_k < 0 || s.idx_top_k > INT_MAX - s.idx_block + 1)
    throw std::invalid_argument("invalid SYCL indexer geometry");
}
void score(const float *pooled, const float *query, const float *bias,
           const QsaShapes &s, const int32_t *step, int64_t nbid, int64_t nkv,
           int64_t blocks, float *out, void *stream) {
  geometry(s);
  if (blocks <= 0 || blocks > INT_MAX)
    throw std::invalid_argument("invalid SYCL indexer capacity");
  const size_t capacity = checked_count(blocks, s.idx_block);
  if (capacity > SIZE_MAX / 4)
    throw std::invalid_argument("SYCL indexer overflow");
  validate_spans({{out, capacity * 4}},
                 {{pooled, checked_count(blocks, s.idx_dim) * 4},
                  {query, checked_count(s.idx_n_head, s.idx_dim) * 4}});
  if (step)
    validate_spans({{out, capacity * 4}}, {{step, kStepCount * 4}});
  if (bias)
    validate_spans({{out, capacity * 4}}, {{bias, size_t(blocks) * 4}});
  const size_t wg = s.idx_n_head * 32;
  auto &q = queue_for(stream);
  if (wg > q.get_device().get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument("indexer work group unsupported");
  auto e = q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> dots(size_t(s.idx_n_head), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(size_t(blocks) * wg, wg),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int64_t b = it.get_group_linear_id(),
                        n = step ? step[kStepNKv] : nkv,
                        nb = step ? step[kStepNBid] : nbid;
          if (b > nb || n < 0 || uint64_t(n) > capacity || nb < 0 ||
              nb >= blocks)
            return;
          const size_t t = it.get_local_linear_id(), head = t / 32,
                       lane = t % 32;
          double dot = 0;
          for (int64_t d = lane; d < s.idx_dim; d += 32)
            dot += double(pooled[b * s.idx_dim + d]) *
                   double(query[head * s.idx_dim + d]);
          auto sg = it.get_sub_group();
          for (int off = 16; off; off /= 2)
            dot += sycl::permute_group_by_xor(sg, dot, off);
          if (!lane)
            dots[head] = dot;
          it.barrier(sycl::access::fence_space::local_space);
          if (t)
            return;
          double total = 0;
          for (int64_t h = 0; h < s.idx_n_head; ++h)
            total += dots[h] > 0 ? dots[h] : 0.;
          if (bias)
            total += bias[b];
          float value = float(total);
          if (b == nb && n % s.idx_block)
            value += 1e9f;
          const int64_t end = b == nb ? n : sycl::min(n, (b + 1) * s.idx_block);
          for (int64_t j = b * s.idx_block; j < end; ++j)
            out[j] = value;
        });
  });
  finish(stream, e);
}
uint32_t key(float value) {
  // Canonicalize signed zero by bits: default host/device optimization may
  // remove an arithmetic +0, which would otherwise break the ID tie rule.
  uint32_t b = sycl::bit_cast<uint32_t>(value);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    return 0; // NaNs rank below -inf.
  if ((b & 0x7fffffffu) == 0)
    b = 0;
  return b & 0x80000000u ? ~b : b | 0x80000000u;
}

void select(const float *scores, const QsaShapes &s, int64_t cap,
            const int32_t *step, int64_t nkv, int32_t *ids, void *stream) {
  geometry(s);
  if (cap < 0 || cap > kTopkMaxCells)
    throw std::invalid_argument("invalid SYCL top-k capacity");
  if (!cap)
    return;
  const int64_t maxcells = step ? kTopkMaxCells : nkv;
  validate_spans({{ids, size_t(cap) * 4}}, {{scores, size_t(maxcells) * 4}});
  if (step)
    validate_spans({{ids, size_t(cap) * 4}}, {{step, kStepCount * 4}});
  auto e = queue_for(stream).parallel_for(
      sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
        const int64_t n = step ? step[kStepNKv] : nkv,
                      width =
                          step ? step[kStepWidth] : qsa_selection_width(nkv, s);
        if (n < 0 || n > maxcells || width < 0 || width > cap || width > n)
          return;
        const int t = it.get_local_linear_id();
        if (width == n) {
          for (int64_t j = t; j < n; j += 256)
            ids[j] = int32_t(j);
          return;
        }
        if (width == 0)
          return;
        const int64_t chunk = (n + 255) / 256, lo = t * chunk,
                      hi = sycl::min(n, lo + chunk);
        uint32_t threshold = 0;
        for (int bit = 31; bit >= 0; --bit) {
          uint32_t candidate = threshold | (1u << bit);
          int c = 0;
          for (int64_t j = lo; j < hi; ++j)
            c += key(scores[j]) >= candidate;
          int total =
              sycl::reduce_over_group(it.get_group(), c, sycl::plus<int>());
          if (total >= width)
            threshold = candidate;
        }
        int gt = 0, eq = 0;
        for (int64_t j = lo; j < hi; ++j) {
          auto k = key(scores[j]);
          gt += k > threshold;
          eq += k == threshold;
        }
        const int budget =
            int(width) -
            sycl::reduce_over_group(it.get_group(), gt, sycl::plus<int>());
        const int eqstart = sycl::exclusive_scan_over_group(it.get_group(), eq,
                                                            sycl::plus<int>());
        const int chosen = gt + sycl::min(eq, sycl::max(0, budget - eqstart));
        int offset = sycl::exclusive_scan_over_group(it.get_group(), chosen,
                                                     sycl::plus<int>()),
            eqrank = eqstart;
        for (int64_t j = lo; j < hi; ++j) {
          auto k = key(scores[j]);
          if (k > threshold)
            ids[offset++] = int32_t(j);
          else if (k == threshold && eqrank++ < budget)
            ids[offset++] = int32_t(j);
        }
      });
  finish(stream, e);
}
} // namespace
void qsa_index_step(const float *p, const float *q, const float *bias,
                    const QsaShapes &s, const int32_t *step, int64_t maxblocks,
                    float *out, void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  score(p, q, bias, s, step, 0, 0, maxblocks, out, stream);
}
void qsa_index(const float *p, int64_t nbid, const float *q, const float *bias,
               const QsaShapes &s, int64_t nkv, float *out, void *stream) {
  geometry(s);
  if (nkv < 0 || nkv > INT_MAX || nbid != nkv / s.idx_block)
    throw std::invalid_argument("inconsistent QSA index counts");
  score(p, q, bias, s, nullptr, nbid, nkv, nbid + 1, out, stream);
}
void topk_512_step(const float *scores, const QsaShapes &s, int64_t cap,
                   const int32_t *step, int32_t *ids, void *stream) {
  sycl_backend::validate_spans({}, {{step, kStepCount * 4}});
  if (cap < qsa_selection_width(kTopkMaxCells, s))
    throw std::invalid_argument("top-k step capacity too small");
  select(scores, s, cap, step, 0, ids, stream);
}
void topk_512(const float *scores, int64_t nkv, const QsaShapes &s, int64_t cap,
              int32_t *ids, void *stream) {
  geometry(s);
  if (nkv < 0 || nkv > kTopkMaxCells || cap < qsa_selection_width(nkv, s))
    throw std::invalid_argument("invalid top-k count/capacity");
  select(scores, s, cap, nullptr, nkv, ids, stream);
}
void indexer_key_append(const float *raw, const int32_t *posdev,
                        int32_t posbase, const float *gamma, float eps,
                        const QsaIndexerBuffers &b, const QsaShapes &s,
                        const float *costab, const float *sintab,
                        void *stream) {
  geometry(s);
  if (posbase < 0 || !std::isfinite(eps) || eps <= 0 || s.n_rot <= 0 ||
      s.n_rot % 2 || s.n_rot > s.idx_dim)
    throw std::invalid_argument("invalid SYCL indexer pooling parameters");
  const size_t bytes = size_t(s.idx_dim) * 4;
  sycl_backend::validate_spans({{b.tail, size_t(s.idx_block - 1) * bytes},
                                {b.dead, bytes},
                                {b.pooled, bytes},
                                {b.block_pos, 4}},
                               {{raw, bytes},
                                {posdev, 4},
                                {gamma, bytes},
                                {costab, size_t(s.n_rot / 2) * 4},
                                {sintab, size_t(s.n_rot / 2) * 4}});
  auto &q = sycl_backend::queue_for(stream);
  const auto mtab = mrope_table();
  if (mtab && q.get_context() != sycl_backend::runtime_for()->context())
    throw std::invalid_argument("SYCL MRoPE table belongs to runtime 0");
  size_t wg = 32;
  while (wg < size_t(s.idx_dim))
    wg *= 2;
  if (wg > q.get_device().get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument("indexer pooling work group unsupported");
  auto e = q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> means(size_t(s.idx_dim), cgh);
    sycl::local_accessor<float, 1> normalized(size_t(s.idx_dim), cgh);
    cgh.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
      const int64_t d = it.get_local_linear_id(), pos = *posdev,
                    slot = pos % s.idx_block;
      if (pos < 0)
        return;
      if (d < s.idx_dim) {
        if (slot < s.idx_block - 1)
          b.tail[slot * s.idx_dim + d] = raw[d];
        if (pos == 0) {
          double ss = 0;
          for (int64_t i = 0; i < s.idx_dim; ++i) {
            double v = raw[i];
            ss += v * v;
          }
          const double mean =
              sycl::ext::intel::math::ddiv_rn(ss, double(s.idx_dim));
          const double inv = sycl::ext::intel::math::ddiv_rn(
              1., sycl::ext::intel::math::dsqrt_rn(mean + double(eps)));
          float y = float(double(raw[d]) * inv * double(gamma[d]));
          if (d < s.n_rot)
            y *= costab[d % (s.n_rot / 2)];
          b.dead[d] = y;
          b.pooled[d] = y;
        }
      }
      if (slot != s.idx_block - 1)
        return;
      if (d < s.idx_dim) {
        double m = 0;
        for (int64_t j = 0; j < s.idx_block - 1; ++j)
          m += b.tail[j * s.idx_dim + d];
        m += raw[d];
        means[d] = sycl::ext::intel::math::ddiv_rn(m, double(s.idx_block));
      }
      it.barrier(sycl::access::fence_space::local_space);
      const int64_t block = pos / s.idx_block;
      if (d < s.idx_dim) {
        double ss = 0;
        for (int64_t i = 0; i < s.idx_dim; ++i)
          ss += means[i] * means[i];
        double mean = sycl::ext::intel::math::ddiv_rn(ss, double(s.idx_dim));
        double inv = sycl::ext::intel::math::ddiv_rn(
            1., sycl::ext::intel::math::dsqrt_rn(mean + double(eps)));
        normalized[d] = float(means[d] * inv * double(gamma[d]));
        b.pooled[(block + 1) * s.idx_dim + d] = b.dead[d];
      }
      if (!d)
        *b.block_pos = int32_t(posbase + block * s.idx_block);
      it.barrier(sycl::access::fence_space::local_space);
      const int64_t half = s.n_rot / 2;
      if (d < half) {
        const int64_t cell = posbase + block * s.idx_block;
        const int p = mtab ? mtab[cell * 3 + d % 3] : int(cell);
        const float c = p < 0 ? sycl::nan(0u) : costab[size_t(p) * half + d],
                    si = p < 0 ? sycl::nan(0u) : sintab[size_t(p) * half + d];
        rope_neox_pair(normalized[d], normalized[d + half], c, si,
                       b.pooled[block * s.idx_dim + d],
                       b.pooled[block * s.idx_dim + half + d]);
      } else if (d >= s.n_rot && d < s.idx_dim)
        b.pooled[block * s.idx_dim + d] = normalized[d];
    });
  });
  sycl_backend::finish(stream, e);
}
} // namespace strata::kernels
