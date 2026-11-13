#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>

namespace strata::kernels {
using namespace sycl_backend;

void embedding_gather(const uint8_t *codes, const float *scales,
                      const float *offsets, int64_t n, int bits, int bias,
                      int group, float *out, void *stream) {
  if ((bits != 2 && bits != 4 && bits != 8) || group <= 0 || n < 0 || n % group)
    throw std::invalid_argument("invalid embedding quantization geometry");
  for_each(n, stream, [=](size_t i) {
    const unsigned code = (codes[i / (8 / bits)] >> ((i % (8 / bits)) * bits)) &
                          ((1u << bits) - 1);
    const float product = float(int(code) + bias) * scales[i / group];
    out[i] = product + (offsets ? offsets[i / group] : 0.0f);
  });
}

void gdn_gate(const float *alpha, const float *dt, const float *ssm_a,
              float *gate, int64_t tokens, int64_t heads, void *stream) {
  const auto count = checked_count(tokens, heads);
  for_each(count, stream, [=](size_t i) {
    const auto head = i % heads;
    const float x = alpha[i] + dt[head];
    gate[i] = (x > 20.f ? x : sycl::log1p(sycl::exp(x))) * ssm_a[head];
  });
}

void scale_inplace(float *x, int64_t n, float scale, void *stream) {
  for_each(n, stream, [=](size_t i) { x[i] *= scale; });
}
void add_inplace(float *dst, const float *src, int64_t n, void *stream) {
  for_each(n, stream, [=](size_t i) { dst[i] += src[i]; });
}
void copy_hit_miss(float *dst, const float *misses, const float *hits,
                   const uint32_t *add_hits, int64_t n, void *stream) {
  const size_t count = checked_count(n, 1);
  if (!count) return;
  queue_for(stream).memcpy(dst, misses, count * sizeof(float));
  for_each(n, stream, [=](size_t i) {
    if (*add_hits) dst[i] += hits[i];
  });
}
void f32_to_f16_bulk(const float *x, uint16_t *y, int64_t n, void *stream) {
  for_each(n, stream, [=](size_t i) { y[i] = f16_from_f32(x[i]); });
}
void f32_to_bf16_bulk(const float *x, uint16_t *y, int64_t n, void *stream) {
  for_each(n, stream, [=](size_t i) { y[i] = bf16_from_f32(x[i]); });
}
void silu_inplace(float *x, int64_t n, void *stream) {
  for_each(n, stream, [=](size_t i) {
    const double value = x[i];
    x[i] = float(value / (1.0 + sycl::exp(-value)));
  });
}

void rms_norm_weighted(float *x, const float *w, int64_t rows, int64_t cols,
                       float eps, void *stream) {
  checked_count(rows, cols);
  if (!rows)
    return;
  if (cols <= 0 || eps < 0)
    throw std::invalid_argument("invalid RMS normalization geometry");
  constexpr size_t width = 128;
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(checked_count(rows, width), width),
      [=](sycl::nd_item<1> item) {
        const size_t row = item.get_group_linear_id();
        const size_t lane = item.get_local_linear_id();
        auto *values = x + row * cols;
        float sum = 0;
        for (size_t c = lane; c < size_t(cols); c += width)
          sum += values[c] * values[c];
        sum =
            sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
        const float inverse = sycl::rsqrt(sum / float(cols) + eps);
        for (size_t c = lane; c < size_t(cols); c += width)
          values[c] = (w ? values[c] * w[c] : values[c]) * inverse;
      });
  finish(stream, event);
}

void copy_from_mapped(float *dst, const float *src, int64_t n, void *stream) {
  // Host USM is legal here only in alternating CPU/GPU phases. The scheduler
  // must complete this command before the host modifies the source again.
  for_each(n, stream, [=](size_t i) { dst[i] = src[i]; });
}
void scatter_rows_f32(const float *src, float *dst, const int32_t *rows,
                       int64_t n, int64_t width, void *stream) {
  // The caller owns the destination extents and supplies distinct valid rows.
  for_each(checked_count(n, width), stream, [=](size_t i) {
    dst[size_t(rows[i / width]) * width + i % width] = src[i];
  });
}
void copy_i32_from_mapped(int32_t *dst, const int32_t *src, int64_t n,
                          void *stream) {
  for_each(n, stream, [=](size_t i) { dst[i] = src[i]; });
}
void copy_rows_from_mapped(float *dst, const float *src, int64_t rows,
                           int64_t width, const int32_t *hit_rows,
                           const int32_t *count, void *stream) {
  for_each(checked_count(rows, width), stream, [=](size_t i) {
    bool hit = false;
    for (int j = 0; j < *count; ++j)
      hit |= hit_rows[j] == int64_t(i / width);
    dst[i] = hit ? 0.f : src[i];
  });
}

bool copy_rows_from_mapped_subgroup(float *dst, const float *src, int64_t rows,
                                   int64_t width, const int32_t *hit_rows,
                                   const int32_t *count, void *stream) {
  const size_t elements = checked_count(rows, width);
  if (!elements)
    return false;
  const size_t data_bytes = checked_count(elements, sizeof(float));
  validate_spans({{dst, data_bytes}},
                 {{src, data_bytes}, {hit_rows, checked_count(rows, sizeof(int32_t))},
                  {count, sizeof(int32_t)}});
  auto &queue = queue_for(stream);
  validate_device_span(dst, data_bytes, queue);
  validate_host_span(src, data_bytes, queue);
  validate_device_span(hit_rows, checked_count(rows, sizeof(int32_t)), queue);
  validate_device_span(count, sizeof(int32_t), queue);
  if (rows > INT_MAX || width % 128 ||
      ((uintptr_t(dst) | uintptr_t(src)) & 15)) {
    copy_rows_from_mapped(dst, src, rows, width, hit_rows, count, stream);
    return false;
  }
  const size_t vectors = elements / 4;
  const size_t row_vectors = size_t(width) / 4;
  const size_t rounded = ((vectors + 255) / 256) * 256;
  auto event = queue.parallel_for(
      sycl::nd_range<1>(rounded, 256),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const auto subgroup = item.get_sub_group();
        // Build the logical index from subgroup IDs rather than assuming the
        // implementation assigns consecutive global IDs to subgroup lanes.
        const size_t v =
            (item.get_group_linear_id() * 8 + subgroup.get_group_id()[0]) * 32 +
            subgroup.get_local_linear_id();
        // width/4 and vectors are multiples of 32. A subgroup stays in one
        // row and the padded tail contains only wholly inactive subgroups.
        // Every active lane therefore reaches the same broadcast, as required
        // by https://github.khronos.org/SYCL_Reference/iface/group-functions.html.
        if (v >= vectors)
          return;
        uint32_t hit = 0;
        if (subgroup.get_local_linear_id() == 0) {
          const int64_t row = int64_t(v / row_vectors);
          // Production plans require this bound. Also keep malformed counts
          // from causing an unbounded device scan beyond the admitted span.
          const int n = sycl::clamp(*count, 0, int(rows));
          for (int j = 0; j < n; ++j)
            hit |= hit_rows[j] == row;
        }
        hit = sycl::group_broadcast(subgroup, hit);
        // Scalar float accesses preserve C++ aliasing/lifetime rules; four
        // adjacent values per lane allow coalescing without vector type-puns.
#pragma unroll
        for (size_t k = 0; k < 4; ++k) {
          const size_t i = 4 * v + k;
          dst[i] = hit ? 0.f : src[i];
        }
      });
  finish(stream, event);
  return true;
}

// These publish only data. CPU readers must wait for the producing event; no
// volatile polling protocol or concurrent host/device atomic contract is
// implied.
void doorbell_ring(uint32_t *sequence, void *stream) {
  finish(stream, queue_for(stream).single_task([=] { ++*sequence; }));
}
void doorbell_publish(const float *x, const int32_t *ids, const float *weights,
                      int64_t n, int64_t k, float *x_out, int32_t *ids_out,
                      float *weights_out, uint32_t *sequence, void *stream) {
  copy_from_mapped(x_out, x, n, stream);
  copy_i32_from_mapped(ids_out, ids, k, stream);
  copy_from_mapped(weights_out, weights, k, stream);
  doorbell_ring(sequence, stream);
}
void doorbell_wait(const uint32_t *, const uint32_t *, void *) {
  throw std::logic_error("SYCL requires event-based CPU expert scheduling; GPU "
                         "doorbell waits are unsupported");
}
} // namespace strata::kernels
