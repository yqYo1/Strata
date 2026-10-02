// Device-side sequence records and data movement for shared engine callers.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
void require(bool v) {
  if (!v)
    throw std::invalid_argument("invalid SYCL sequence geometry");
}
size_t count(int64_t a, int64_t b = 1) {
  auto n = checked_count(a, b);
  require(n <= size_t(INT64_MAX) / 4);
  return n;
}
} // namespace
void copy_i32_from_mapped_unless(int32_t *dst, const int32_t *src, long long n,
                                 const uint32_t *skip, uint32_t value,
                                 void *stream) {
  const auto bytes = count(n) * 4;
  validate_spans({{dst, bytes}}, {{src, bytes}, {skip, 4}});
  for_each(n, stream, [=](size_t i) {
    if (*skip != value)
      dst[i] = src[i];
  });
}
void copy_or_zero_from_mapped(float *dst, const float *src, long long n,
                              const uint32_t *skip, uint32_t value,
                              void *stream) {
  const auto bytes = count(n) * 4;
  validate_spans({{dst, bytes}}, {{src, bytes}, {skip, 4}});
  for_each(n, stream,
           [=](size_t i) { dst[i] = *skip == value ? 0.f : src[i]; });
}
void wait_flag_ge(const uint32_t *, uint32_t, void *) {
  throw std::runtime_error("SYCL requires event-completed CPU/GPU handoff; "
                           "mapped flag polling is unsupported");
}
void wait_flag_ge_or(const uint32_t *, uint32_t, const uint32_t *, void *) {
  throw std::runtime_error("SYCL requires event-completed CPU/GPU handoff; "
                           "mapped flag polling is unsupported");
}
void gpu_stamp(unsigned long long *, int, void *) {
  throw std::runtime_error("SYCL device global-timer stamps are unsupported; "
                           "use profiling events outside capture");
}
void broadcast_streams(const float *x, float *out, int64_t n, int hc, int T,
                       void *stream) {
  require(n > 0 && hc > 0 && T > 0);
  const auto per = count(n, hc), total = count(per, T);
  validate_spans({{out, total * 4}}, {{x, count(n, T) * 4}});
  for_each(total, stream, [=](size_t i) { out[i] = x[i / per * n + i % n]; });
}
void add_streams_broadcast(const float *input, const float *x, float *out,
                           int64_t n, int hc, int T, void *stream) {
  require(n > 0 && hc > 0 && T > 0);
  const auto per = count(n, hc), total = count(per, T);
  validate_spans({{out, total * 4}},
                 {{x, count(n, T) * 4}, {input, total * 4}});
  for_each(total, stream,
           [=](size_t i) { out[i] = input[i] + x[i / per * n + i % n]; });
}
void copy_indexed(float *dst, const float *src, int64_t stride,
                  const int32_t *index, int64_t n, void *stream) {
  require(stride >= n && n >= 0);
  const auto bytes = count(n) * 4;
  validate_spans({{dst, bytes}}, {{src, bytes}, {index, 4}});
  // Positive indices and the corresponding source extent are caller-owned.
  for_each(n, stream, [=](size_t i) {
    if (*index >= 0)
      dst[i] = src[size_t(*index) * stride + i];
  });
}
void gather_rows(const uint8_t *src, int64_t row_bytes, const int32_t *ids,
                 int64_t n, uint8_t *dst, void *stream) {
  require(row_bytes > 0 && reinterpret_cast<uintptr_t>(ids) % 4 == 0);
  const auto bytes = checked_count(row_bytes, n);
  validate_spans({{dst, bytes}},
                 {{src, size_t(row_bytes)}, {ids, count(n) * 4}}, 1);
  for_each(bytes, stream, [=](size_t i) {
    int row = ids[i / row_bytes];
    dst[i] = row < 0 ? 0 : src[size_t(row) * row_bytes + i % row_bytes];
  });
}
void fetch_blobs(const unsigned long long *src, const int32_t *n, uint8_t *dst,
                 int64_t bytes, int cap, void *stream) {
  require(bytes > 0 && bytes % 16 == 0 && cap >= 0);
  const auto total = checked_count(bytes, cap);
  validate_spans({{dst, total}}, {{src, size_t(cap) * 8}, {n, 4}}, 1);
  require(reinterpret_cast<uintptr_t>(src) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(n) % 4 == 0);
  for_each(total, stream, [=](size_t i) {
    const int live = sycl::clamp(*n, 0, cap);
    const size_t row = i / bytes;
    if (row < size_t(live)) {
      auto ptr = reinterpret_cast<const uint8_t *>(src[row]);
      dst[i] = ptr ? ptr[i % bytes] : 0;
    }
  });
}
void rebase_ptrs(unsigned long long *ptr, const int32_t *n, uint8_t *base,
                 int64_t bytes, void *stream) {
  require(bytes > 0 && reinterpret_cast<uintptr_t>(ptr) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(n) % 4 == 0);
  validate_spans({{ptr, 8}}, {{n, 4}, {base, size_t(bytes)}}, 1);
  auto e = queue_for(stream).single_task([=] {
    for (int i = 0; i < *n; ++i)
      ptr[i] = reinterpret_cast<uintptr_t>(base + size_t(i) * bytes);
  });
  finish(stream, e);
}
void map_ids(int32_t *ids, const int32_t *table, int n, void *stream) {
  validate_spans({{ids, count(n) * 4}}, {{table, 4}});
  for_each(n, stream, [=](size_t i) {
    if (ids[i] >= 0)
      ids[i] = table[ids[i]];
  });
}
void ident_hits(const int32_t *ids, int n, int32_t *slot, int32_t *dst,
                int32_t *total, void *stream) {
  require(n > 0);
  const auto bytes = count(n) * 4;
  validate_spans({{slot, bytes}, {dst, bytes}, {total, 4}}, {{ids, bytes}});
  for_each(n, stream, [=](size_t i) {
    slot[i] = ids[i];
    dst[i] = int(i);
    if (!i)
      *total = n;
  });
}
void dense_steps(const int32_t *cells, int n, int32_t *steps, void *stream) {
  validate_spans({{steps, count(n, kStepCount) * 4}}, {{cells, count(n) * 4}});
  for_each(n, stream, [=](size_t i) {
    const int c = cells[i];
    int32_t *s = steps + i * kStepCount;
    const int width = c >= 0 && c < INT32_MAX ? c + 1 : 0;
    s[kStepPos] = c;
    s[kStepNKv] = width;
    s[kStepNBid] = width / 4;
    s[kStepWidth] = width;
  });
}
void window_ids(int32_t *steps, int n, int window, int32_t *ids, int64_t stride,
                void *stream) {
  require(n >= 0 && window > 0 && stride >= window);
  validate_spans(
      {{steps, count(n, kStepCount) * 4}, {ids, count(n, stride) * 4}}, {});
  for_each(n, stream, [=](size_t i) {
    int32_t *s = steps + i * kStepCount;
    const int end = sycl::max(s[kStepNKv], 0), width = sycl::min(end, window);
    for (int j = 0; j < width; ++j)
      ids[i * stride + j] = end - width + j;
    s[kStepWidth] = width;
  });
}
void mtp_select(const float *src, int64_t stride, const int32_t *ids,
                const int32_t *row, float *dst, int32_t *tok, int32_t *out,
                int j, void *stream, const float *probs, float *outp) {
  require(stride > 0 && j >= 0 && j < INT32_MAX);
  const auto bytes = count(stride) * 4;
  validate_spans({{dst, bytes}, {tok, 4}}, {{src, bytes}, {ids, 4}, {row, 4}});
  if (out)
    validate_spans({{dst, bytes}, {tok, 4}, {out, size_t(j + 1) * 4}},
                   {{src, bytes}, {ids, 4}, {row, 4}});
  if (probs && outp)
    validate_spans({{outp, size_t(j + 1) * 4}, {dst, bytes}, {tok, 4}},
                   {{probs, 4}, {src, bytes}, {ids, 4}, {row, 4}});
  for_each(stride, stream, [=](size_t i) {
    const int r = *row;
    if (r < 0)
      return;
    dst[i] = src[size_t(r) * stride + i];
    if (!i) {
      *tok = ids[r];
      if (out)
        out[j] = ids[r];
      if (probs && outp)
        outp[j] = probs[r];
    }
  });
}
void row_top_prob(const float *logits, int rows, int vocab, const int32_t *ids,
                  float *probs, void *stream) {
  require(rows >= 0 && vocab > 0);
  validate_spans({{probs, count(rows) * 4}},
                 {{logits, count(rows, vocab) * 4}, {ids, count(rows) * 4}});
  auto e = queue_for(stream).submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(32, h);
    h.parallel_for(sycl::nd_range<1>(size_t(rows) * 1024, 1024),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                     const int tid = it.get_local_linear_id(),
                               row = it.get_group_linear_id();
                     const int id = ids[row];
                     if (id < 0 || id >= vocab) {
                       if (!tid)
                         probs[row] = 0;
                       return;
                     }
                     const float *l = logits + size_t(row) * vocab;
                     const float m = l[id];
                     float sum = 0;
                     for (int i = tid; i < vocab; i += 1024)
                       sum += sycl::exp(l[i] - m);
                     for (int o = 16; o; o >>= 1)
                       sum += sycl::permute_group_by_xor(it.get_sub_group(),
                                                         sum, o);
                     if (tid % 32 == 0)
                       part[tid / 32] = sum;
                     it.barrier(sycl::access::fence_space::local_space);
                     if (!tid) {
                       float total = 0;
                       for (int w = 0; w < 32; ++w)
                         total += part[w];
                       probs[row] = 1.f / total;
                     }
                   });
  });
  finish(stream, e);
}
void embedding_gather_dev(const uint8_t *codes, const float *scales,
                          const float *offsets, const int32_t *tokens, int T,
                          int64_t n, int bits, int bias, int group,
                          uint64_t row_codes, uint64_t row_groups, float *out,
                          void *stream) {
  require(T >= 0 && n > 0 && (bits == 2 || bits == 4 || bits == 8) &&
          group > 0);
  require(row_codes >= (uint64_t(n) + 8 / bits - 1) / (8 / bits) &&
          row_groups >= (uint64_t(n) + group - 1) / group);
  require(row_codes <= SIZE_MAX && row_groups <= SIZE_MAX / 4);
  require(reinterpret_cast<uintptr_t>(out) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(scales) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(tokens) % 4 == 0);
  const auto total = count(T, n);
  validate_spans({{out, total * 4}},
                 {{codes, size_t(row_codes)},
                  {scales, size_t(row_groups) * 4},
                  {tokens, count(T) * 4}},
                 1);
  if (offsets)
    validate_spans({{out, total * 4}}, {{offsets, size_t(row_groups) * 4}});
  for_each(total, stream, [=](size_t i) {
    const int token = tokens[i / n];
    if (token < 0) {
      out[i] = 0;
      return;
    }
    const size_t c = i % n;
    const unsigned code = (codes[size_t(token) * row_codes + c / (8 / bits)] >>
                           ((c % (8 / bits)) * bits)) &
                          ((1u << bits) - 1);
    const size_t g = size_t(token) * row_groups + c / group;
    const float product = float(int(code) + bias) * scales[g];
    out[i] = product + (offsets ? offsets[g] : 0.f);
  });
}
void resident_plan(const int32_t *ids, int n, int k, const int32_t *res,
                   int experts, const uint8_t *base,
                   const unsigned long long *offsets, long long blob,
                   int32_t *plan, long long cap, uint32_t *skip, uint32_t ring,
                   void *stream) {
  require(n >= 0 && k > 0 && experts > 0 && cap >= n && cap <= INT32_MAX / 16 &&
          blob > 0 && reinterpret_cast<uintptr_t>(plan) % 8 == 0);
  require(reinterpret_cast<uintptr_t>(ids) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(res) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(skip) % 4 == 0 &&
          (!offsets || reinterpret_cast<uintptr_t>(offsets) % 8 == 0));
  const size_t ptr_off = (4 + (cap + 1) + 2 * cap + 1) & ~size_t(1),
               ints = ptr_off + 5 * cap + 1;
  validate_spans(
      {{plan, ints * 4}, {skip, 4}},
      {{ids, count(n) * 4}, {res, count(experts) * 4}, {base, size_t(blob)}},
      1);
  if (offsets)
    validate_spans({{plan, ints * 4}, {skip, 4}}, {{offsets, 8}}, 1);
  auto e = queue_for(stream).single_task([=] {
    for (int i = 0; i < n; ++i)
      if (ids[i] < 0 || ids[i] >= experts || res[ids[i]] < 0) {
        *skip = 0;
        return;
      }
    int32_t *starts = plan + 4;
    int32_t *dst = starts + cap + 1;
    int32_t *tok = dst + cap;
    auto ptr = reinterpret_cast<unsigned long long *>(plan + ptr_off);
    int32_t *start2 = plan + ptr_off + 4 * cap;
    int groups = 0, entries = 0;
    for (int first = 0; first < n; ++first) {
      bool seen = false;
      for (int j = 0; j < first; ++j)
        seen |= ids[j] == ids[first];
      if (seen)
        continue;
      const int slot = res[ids[first]];
      ptr[groups] = reinterpret_cast<uintptr_t>(
          base + (offsets ? size_t(offsets[slot]) : size_t(slot) * blob));
      starts[groups] = entries;
      for (int j = first; j < n; ++j)
        if (ids[j] == ids[first]) {
          dst[entries] = j;
          tok[entries] = j / k;
          ++entries;
        }
      ++groups;
    }
    starts[groups] = entries;
    start2[0] = entries;
    plan[0] = groups;
    plan[1] = entries;
    plan[2] = 0;
    *skip = ring;
  });
  finish(stream, e);
}
} // namespace strata::kernels
