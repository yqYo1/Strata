#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"

namespace strata::kernels {
void s2_gemv_q8(const uint8_t *act, const uint8_t *codes, const float *scales,
                float *out, int64_t ni, int64_t no, int threads, void *stream) {
  using namespace sycl_backend;
  if (ni <= 0 || no <= 0 || ni % 64 || threads <= 0 ||
      (threads & (threads - 1)))
    throw std::invalid_argument("invalid SYCL S2/Q8 GEMV geometry");
  validate_spans({{out, checked_count(no, 4)}},
                 {{act, checked_count(ni / 32, 34)},
                  {codes, checked_count(no, ni / 4)},
                  {scales, checked_count(no, ni / 64) * 4}});
  auto &q = queue_for(stream);
  if (size_t(threads) >
      q.get_device().get_info<sycl::info::device::max_work_group_size>())
    throw std::invalid_argument("unsupported S2/Q8 work group");
  auto event = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> partial(threads, h);
    h.parallel_for(
        sycl::nd_range<1>(size_t(no) * threads, threads),
        [=](sycl::nd_item<1> it) {
          const size_t row = it.get_group_linear_id(),
                       tid = it.get_local_linear_id();
          float sum[4] = {};
          for (int64_t quad = tid; quad < ni / 4; quad += threads) {
            const uint8_t code = codes[row * (ni / 4) + quad];
            const float scale = scales[row * (ni / 64) + quad / 16];
            const auto block = act + (quad / 8) * 34;
            const float dx =
                f32_from_f16(uint16_t(block[0]) | uint16_t(block[1]) << 8);
            for (int j = 0; j < 4; ++j) {
              const float w = float(int((code >> (2 * j)) & 3) - 1) * scale;
              const float x =
                  float(sycl::bit_cast<int8_t>(block[2 + (quad % 8) * 4 + j])) *
                  dx;
              sum[j] = sycl::fma(w, x, sum[j]);
            }
          }
          partial[tid] = (sum[0] + sum[1]) + (sum[2] + sum[3]);
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
} // namespace strata::kernels
