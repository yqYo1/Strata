#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/gguf_decode.hpp"
#include "strata/sycl/iq_parallel_decode.hpp"
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
// Read once before use; callers keep the exec-time environment immutable.
// Only the exact string "1" opts in. OFF adds no allocation or queue work.
const bool parallel_iq_dequant = [] {
  const char *v = std::getenv("STRATA_SYCL_PARALLEL_IQ_DEQUANT");
  return v && v[0] == '1' && v[1] == '\0';
}();

template <int Type, bool GU>
void parallel_iq_launch(const void *gate, const void *up, size_t count,
                        size_t width, uint16_t *out, void *stream) {
  constexpr size_t B = parallel_iq_width<Type>, S = parallel_iq_bytes<Type>;
  auto event = queue_for(stream).parallel_for(
      sycl::range<1>(count), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const uint8_t *block;
        unsigned j;
        if constexpr (GU) {
          const size_t row = i / width, col = i % width;
          const auto *base = static_cast<const uint8_t *>(row % 2 ? up : gate);
          block = base + (row / 2) * (width / B * S) + col / B * S;
          j = unsigned(col % B);
        } else {
          block = static_cast<const uint8_t *>(gate) + i / B * S;
          j = unsigned(i % B);
        }
        out[i] = f16_from_f32(parallel_iq_element<Type>(block, j));
      });
  finish(stream, event);
}

template <bool GU>
bool try_parallel_iq(int type, const void *gate, const void *up, size_t count,
                     size_t width, uint16_t *out, void *stream) {
  if (!parallel_iq_dequant)
    return false;
  // Admit only the role/format pairs that won the resident component screen.
  // IQ3_XXS was slower; other role/format combinations were not measured.
  if constexpr (GU) {
    switch (type) {
    case 21: parallel_iq_launch<21, true>(gate, up, count, width, out, stream); break;
    case 22: parallel_iq_launch<22, true>(gate, up, count, width, out, stream); break;
    case 23: parallel_iq_launch<23, true>(gate, up, count, width, out, stream); break;
    default: return false;
    }
  } else {
    switch (type) {
    case 20: parallel_iq_launch<20, false>(gate, up, count, width, out, stream); break;
    case 42: parallel_iq_launch<42, false>(gate, up, count, width, out, stream); break;
    default: return false;
    }
  }
  return true;
}
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
  if constexpr (sizeof(Out) == 2) {
    if (try_parallel_iq<false>(type, src, nullptr, size_t(n), 0, out, stream))
      return;
  }
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
    strata::sycl_backend::cpu_wait_profile::blocking(
        strata::sycl_backend::cpu_wait_profile::direct_queue, [&] { q.wait_and_throw(); });
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
    strata::sycl_backend::cpu_wait_profile::blocking(
        strata::sycl_backend::cpu_wait_profile::direct_queue, [&] { q.wait_and_throw(); });
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
  if (try_parallel_iq<true>(type, gate, up, count * 2, size_t(width), out,
                             stream))
    return;
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
