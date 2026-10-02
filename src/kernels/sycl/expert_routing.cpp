// Stable device-side expert routing. Positive blob and destination extents
// remain caller-owned: the public API does not carry those capacities.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl/launch.hpp"

namespace strata::kernels {
using namespace sycl_backend;
namespace {
void require(bool value) {
  if (!value)
    throw std::invalid_argument("invalid SYCL expert routing geometry");
}
} // namespace
void moe_hit_select_multi(const int32_t *ids, const int32_t *resident, int n,
                          int experts, int32_t *slot, int32_t *dst,
                          int32_t *count, void *stream) {
  require(n >= 1 && n <= 128 && experts > 0);
  const size_t bytes = size_t(n) * 4;
  validate_spans({{slot, bytes}, {dst, bytes}, {count, 4}},
                 {{ids, bytes}, {resident, size_t(experts) * 4}});
  auto event = queue_for(stream).single_task([=] {
    int hits = 0;
    for (int i = 0; i < n; ++i) {
      const int e = ids[i];
      if (e >= 0 && e < experts && resident[e] >= 0) {
        slot[hits] = resident[e];
        dst[hits++] = i;
      }
    }
    *count = hits;
  });
  finish(stream, event);
}
void moe_hit_select(const int32_t *ids, const int32_t *resident, int n,
                    int experts, int32_t *slot, int32_t *dst, int32_t *count,
                    void *stream) {
  require(n >= 1 && n <= 32);
  moe_hit_select_multi(ids, resident, n, experts, slot, dst, count, stream);
}
void moe_group_resident(const int32_t *ids, int n, int k, const uint8_t *base,
                        int64_t blob, unsigned long long *ptr, int32_t *start,
                        int32_t *counts, int32_t *dst, int32_t *tok,
                        void *stream) {
  require(n >= 1 && n <= 128 && k > 0 && blob > 0);
  require(reinterpret_cast<uintptr_t>(ptr) % 8 == 0);
  const size_t bytes = size_t(n) * 4;
  validate_spans({{ptr, size_t(n) * 8},
                  {start, bytes + 4},
                  {counts, 8},
                  {dst, bytes},
                  {tok, bytes}},
                 {{ids, bytes}});
  validate_spans({}, {{base, size_t(blob)}}, 1);
  auto event = queue_for(stream).single_task([=] {
    int groups = 0, entries = 0;
    for (int i = 0; i < n; ++i) {
      const int expert = ids[i];
      if (expert < 0)
        continue;
      bool first = true;
      for (int j = 0; j < i; ++j)
        if (ids[j] == expert)
          first = false;
      if (!first)
        continue;
      ptr[groups] = reinterpret_cast<unsigned long long>(base) +
                    static_cast<unsigned long long>(expert) * blob;
      start[groups++] = entries;
      for (int j = i; j < n; ++j)
        if (ids[j] == expert) {
          dst[entries] = j;
          tok[entries++] = j / k;
        }
    }
    start[groups] = entries;
    counts[0] = groups;
    counts[1] = entries;
  });
  finish(stream, event);
}
void moe_hit_add(float *parts, const float *hits, const int32_t *dst,
                 const int32_t *count, int64_t cap, int64_t width,
                 void *stream) {
  require(cap >= 0 && width > 0);
  if (!cap)
    return;
  const auto total = checked_count(cap, width);
  require(total <= size_t(INT64_MAX) / 4);
  validate_spans(
      {{parts, size_t(width) * 4}},
      {{hits, size_t(width) * 4}, {dst, size_t(cap) * 4}, {count, 4}});
  // The routed destinations are distinct (as produced by hit_select).
  for_each(total, stream, [=](size_t i) {
    const size_t h = i / width;
    if (*count <= 0 || *count > cap || h >= size_t(*count) || dst[h] < 0)
      return;
    const size_t at = size_t(dst[h]) * width + i % width;
    parts[at] += hits[at];
  });
}
} // namespace strata::kernels
