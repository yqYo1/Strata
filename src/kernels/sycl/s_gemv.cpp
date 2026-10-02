#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/sform.hpp"

namespace strata::kernels {
namespace {
using namespace sycl_backend;
template <int Kind> float activation(const void *ptr, int64_t i) {
  if constexpr (Kind == 0)
    return f32_from_f16(static_cast<const uint16_t *>(ptr)[i]);
  else {
    const auto p = static_cast<const uint8_t *>(ptr);
    if constexpr (Kind == 1) {
      const auto b = p + (i / 32) * 34;
      const float scale = f32_from_f16(uint16_t(b[0]) | uint16_t(b[1]) << 8);
      return scale * float(sycl::bit_cast<int8_t>(b[2 + i % 32]));
    } else {
      const auto b = p + (i / 256) * 292;
      const float scale = *reinterpret_cast<const float *>(b);
      return scale * float(sycl::bit_cast<int8_t>(b[4 + i % 256]));
    }
  }
}
template <int Kind>
void gemv(const void *x, const uint8_t *codes, const float *scales,
          const float *offset, float *out, int64_t ni, int64_t no, SForm f,
          int threads, void *stream, bool stage = false) {
  validate_sform(f, ni, no);
  if ((Kind == 1 && ni % 32) || (Kind == 2 && ni % 256) || threads <= 0 ||
      (threads & (threads - 1)))
    throw std::invalid_argument(
        "invalid SYCL canonical GEMV activation/work group");
  const size_t outbytes = size_t(no) * 4;
  const size_t xbytes = Kind == 0   ? size_t(ni) * 2
                        : Kind == 1 ? size_t(ni / 32) * 34
                                    : size_t(ni / 256) * 292;
  validate_spans({{out, outbytes}},
                 {{x, xbytes},
                  {codes, checked_count(no, ni / (8 / f.code_bits))},
                  {scales, checked_count(no, ni / f.group_elems) * 4}});
  if (f.has_offset)
    validate_spans({{out, outbytes}},
                   {{offset, checked_count(no, ni / f.group_elems) * 4}});
  auto &q = queue_for(stream);
  if (size_t(threads) >
          q.get_device().get_info<sycl::info::device::max_work_group_size>() ||
      (stage &&
       (size_t(ni) + threads) * 4 >
           q.get_device().get_info<sycl::info::device::local_mem_size>()))
    throw std::invalid_argument("unsupported canonical GEMV local storage");
  auto event = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partial(threads, h),
        input(stage ? size_t(ni) : 1, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(no) * threads, threads),
        [=](sycl::nd_item<1> it) {
          const size_t row = it.get_group_linear_id(),
                       tid = it.get_local_linear_id();
          if (stage) {
            for (int64_t i = tid; i < ni; i += threads)
              input[i] = activation<Kind>(x, i);
            it.barrier(sycl::access::fence_space::local_space);
          }
          float sum = 0;
          for (int64_t i = tid; i < ni; i += threads) {
            const float v = stage ? input[i] : activation<Kind>(x, i);
            sum = sycl::fma(
                canonical_value(codes, scales, offset, row, i, ni, f), v, sum);
          }
          partial[tid] = sum;
          it.barrier(sycl::access::fence_space::local_space);
          for (int stride = threads / 2; stride; stride /= 2) {
            if (tid < size_t(stride))
              partial[tid] += partial[tid + stride];
            it.barrier(sycl::access::fence_space::local_space);
          }
          if (!tid)
            out[row] = partial[0];
        });
  });
  finish(stream, event);
}
} // namespace
void s_gemv(const uint16_t *x, const uint8_t *c, const float *s, const float *b,
            float *y, int64_t ni, int64_t no, const SForm &f) {
  gemv<0>(x, c, s, b, y, ni, no, f, 1, nullptr);
}
void s_gemv_split(const uint16_t *x, const uint8_t *c, const float *s,
                  const float *b, float *y, int64_t ni, int64_t no,
                  const SForm &f, int threads) {
  gemv<0>(x, c, s, b, y, ni, no, f, threads, nullptr);
}
void s_gemv_split_async(const uint16_t *x, const uint8_t *c, const float *s,
                        const float *b, float *y, int64_t ni, int64_t no,
                        const SForm &f, int threads, void *stream) {
  gemv<0>(x, c, s, b, y, ni, no, f, threads, stream);
}
void s2_gemv_quads(const uint16_t *x, const uint8_t *c, const float *s,
                   float *y, int64_t ni, int64_t no, int threads) {
  gemv<0>(x, c, s, nullptr, y, ni, no, SForm{}, threads, nullptr);
}
void s2_gemv_fast(const uint16_t *x, const uint8_t *c, const float *s, float *y,
                  int64_t ni, int64_t no, int threads, bool stage) {
  gemv<0>(x, c, s, nullptr, y, ni, no, SForm{}, threads, nullptr, stage);
}
void s_gemv_q8k(const uint8_t *x, const uint8_t *c, const float *s,
                const float *b, float *y, int64_t ni, int64_t no,
                const SForm &f, void *stream) {
  gemv<2>(x, c, s, b, y, ni, no, f, 1, stream);
}
void s_gemv_q8k_split(const uint8_t *x, const uint8_t *c, const float *s,
                      const float *b, float *y, int64_t ni, int64_t no,
                      const SForm &f, void *stream) {
  gemv<2>(x, c, s, b, y, ni, no, f, 32, stream);
}
void s_gemv_q8_0_split(const uint8_t *x, const uint8_t *c, const float *s,
                       const float *b, float *y, int64_t ni, int64_t no,
                       const SForm &f, void *stream) {
  gemv<1>(x, c, s, b, y, ni, no, f, 32, stream);
}
} // namespace strata::kernels
