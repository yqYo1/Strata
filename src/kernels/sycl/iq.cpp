#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/gguf_decode.hpp"
#include "strata/sycl/launch.hpp"
#include <atomic>
#include <climits>
#include <cstdlib>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
std::atomic<bool> old{[] {
  const char *v = std::getenv("STRATA_OLD_IQ_MMVQ");
  return v && std::atoi(v) != 0;
}()};
template <class Out> Out convert(float value) {
  if constexpr (sizeof(Out) == 2)
    return f16_from_f32(value);
  else
    return value;
}
template <class Out>
void dequant(int type, const void *src, int64_t n, Out *out, void *stream) {
  const auto f = block_format(type);
  if (!f.width || n <= 0 || n % f.width)
    throw std::invalid_argument("invalid SYCL GGUF dequantization type/shape");
  if (uintptr_t(out) % alignof(Out))
    throw std::invalid_argument("misaligned SYCL GGUF dequantization output");
  // Packed rows such as Q8_0 have a 34-byte stride, so a row slice can
  // begin at a 2-byte-aligned address even when the original tensor is aligned.
  validate_spans({{out, checked_count(n, sizeof(Out))}},
                 {{src, checked_count(n / f.width, f.bytes)}}, 2);
  auto event = queue_for(stream).parallel_for(
      sycl::range<1>(size_t(n / f.width)), [=](sycl::id<1> id) {
        float values[256];
        decode_block(type, static_cast<const uint8_t *>(src) + id[0] * f.bytes,
                     values);
        for (int j = 0; j < f.width; ++j)
          out[id[0] * f.width + j] = convert<Out>(values[j]);
      });
  finish(stream, event);
}
} // namespace
void iq_set_old_kernels(bool value) { old.store(value); }
bool iq_old_kernels() { return old.load(); }
bool iq_supported(int type) noexcept { return native_mmvq_supported(type); }
bool embed_type_supported(int type) noexcept {
  return type == 30 || iq_supported(type);
}
size_t iq_row_bytes(int type, int64_t n) noexcept {
  const auto f = block_format(type);
  if (!f.width || n <= 0 || n % f.width ||
      uint64_t(n / f.width) > SIZE_MAX / f.bytes)
    return 0;
  return size_t(n / f.width) * f.bytes;
}
void quantize_q8_1_rows(const float *x, int64_t rows, int64_t cols, void *out,
                        void *stream) {
  const size_t count = checked_count(rows, cols);
  if (!rows || !cols || cols % 32 || count > INT_MAX)
    throw std::invalid_argument("invalid SYCL Q8_1 row dimensions");
  auto &q = queue_for(stream);
  native_quantize_q8_1(x, out, int(count), 1, &q);
  if (!stream)
    q.wait_and_throw();
}
void iq_mmvq(int type, const void *w, const void *x, float *y, int ni, int no,
             int cols, void *stream) {
  auto &q = queue_for(stream);
  // Validate the whole call before the first column is submitted.
  const size_t wb = native_mmvq_weight_bytes(type, ni, no),
               xb = native_q8_1_bytes(ni, cols);
  validate_spans({{y, checked_count(no, cols) * 4}}, {{w, wb}, {x, xb}});
  if (iq_old_kernels()) {
    const size_t stride = native_q8_1_bytes(ni);
    for (int c = 0; c < cols; ++c)
      native_mmvq(type, w, static_cast<const uint8_t *>(x) + c * stride,
                  y + size_t(c) * no, ni, no, 1, &q);
  } else
    native_mmvq(type, w, x, y, ni, no, cols, &q);
  if (!stream)
    q.wait_and_throw();
}
void iq_dequant_f16(int type, const void *src, int64_t n, uint16_t *out,
                    void *stream) {
  dequant(type, src, n, out, stream);
}
void iq_dequant_f32(int type, const void *src, int64_t n, float *out,
                    void *stream) {
  dequant(type, src, n, out, stream);
}
void iq_embed_rows(int type, const void *table, size_t stride,
                   const int32_t *ids, int64_t rows, int64_t width, float *out,
                   void *stream) {
  const auto f = block_format(type);
  const size_t rowbytes = iq_row_bytes(type, width);
  if (!rowbytes || rows <= 0 || rows > 65535 || width % 256 ||
      stride < rowbytes || stride > SIZE_MAX / INT_MAX)
    throw std::invalid_argument("invalid SYCL embedding type/stride/shape");
  validate_spans({{out, checked_count(rows, width) * 4}},
                 {{table, rowbytes}, {ids, size_t(rows) * 4}});
  auto event = queue_for(stream).parallel_for(
      sycl::range<2>(size_t(rows), size_t(width / f.width)),
      [=](sycl::id<2> id) {
        const int token = ids[id[0]];
        float values[256] = {};
        if (token >= 0)
          decode_block(type,
                       static_cast<const uint8_t *>(table) +
                           size_t(token) * stride + id[1] * f.bytes,
                       values);
        for (int j = 0; j < f.width; ++j)
          out[id[0] * width + id[1] * f.width + j] = values[j];
      });
  finish(stream, event);
}
void iq_dequant_gu_f16(int type, const void *gate, const void *up, int64_t rows,
                       int64_t width, uint16_t *out, void *stream) {
  const auto f = block_format(type);
  const size_t rowbytes = iq_row_bytes(type, width);
  if (!rowbytes || rows <= 0 || width % 256)
    throw std::invalid_argument(
        "invalid SYCL gate/up dequantization dimensions");
  const size_t count = checked_count(rows, width);
  if (count > SIZE_MAX / 4)
    throw std::invalid_argument("SYCL dequantization size overflow");
  validate_spans({{out, count * 4}}, {{gate, checked_count(rows, rowbytes)},
                                      {up, checked_count(rows, rowbytes)}});
  auto event = queue_for(stream).parallel_for(
      sycl::range<2>(size_t(rows) * 2, size_t(width / f.width)),
      [=](sycl::id<2> id) {
        const size_t row = id[0] / 2;
        const auto ptr = static_cast<const uint8_t *>(id[0] % 2 ? up : gate);
        float values[256];
        decode_block(type, ptr + row * rowbytes + id[1] * f.bytes, values);
        for (int j = 0; j < f.width; ++j)
          out[id[0] * width + id[1] * f.width + j] = f16_from_f32(values[j]);
      });
  finish(stream, event);
}
} // namespace strata::kernels
