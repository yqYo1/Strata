#include "strata/sycl/native_q2.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/math.hpp>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
constexpr int H = 2560, FF = 640;
float half_at(const uint8_t *p) {
  return f32_from_f16(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}
int dot4(uint8_t code, const int8_t *x) {
  int result = 0;
  for (int j = 0; j < 4; ++j)
    result += int((code >> (2 * j)) & 3) * int(x[j]);
  return result;
}
template <bool Down>
void projection(const unsigned long long *pointers, const int32_t *destinations,
                const int32_t *count, int cap, const uint8_t *x,
                const float *scales, float *out, sycl::queue &queue) {
  constexpr int NI = Down ? FF : H, NO = Down ? H : 2 * FF;
  constexpr size_t row_bytes = NI / 64 * 18;
  queue.parallel_for(
      sycl::nd_range<1>(size_t(cap) * NO * 8, 128),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t index = it.get_global_linear_id() / 8;
        const int lane = it.get_local_linear_id() % 8, entry = index / NO,
                  row = index % NO;
        const int n = *count;
        if (n < 0 || n > cap || entry >= n || destinations[entry] < 0 ||
            destinations[entry] >= cap || !pointers[entry])
          return;
        const auto *blob = reinterpret_cast<const uint8_t *>(pointers[entry]);
        const auto *weight = blob +
                             (Down ? size_t(2 * FF) * (H / 64 * 18) : 0) +
                             size_t(row) * row_bytes;
        const size_t offset = Down ? size_t(entry) * (FF / 32) : 0;
        const auto *act = x + offset * 34;
        const auto *scale = scales + offset;
        float acc = 0, corr = 0;
        for (int b = 0; b < NI / 64; ++b) {
          const auto *w = weight + size_t(b) * 18;
          const float d = half_at(w);
          const auto *lo =
              reinterpret_cast<const int8_t *>(act + size_t(2 * b) * 34 + 2);
          const auto *hi = lo + 34;
          acc = sycl::fma(d * scale[2 * b],
                          float(dot4(w[2 + lane], lo + lane * 4)), acc);
          acc = sycl::fma(d * scale[2 * b + 1],
                          float(dot4(w[10 + lane], hi + lane * 4)), acc);
          if (!lane) {
            int sum0 = 0, sum1 = 0;
            for (int j = 0; j < 32; ++j) {
              sum0 += lo[j];
              sum1 += hi[j];
            }
            const float hx0 = scale[2 * b] * float(sum0),
                        hx1 = scale[2 * b + 1] * float(sum1);
            corr = sycl::fma(d, hx0 + hx1, corr);
          }
        }
        // AVX2: low128+high128, then (lane0+lane2)+(lane1+lane3).
        for (int delta : {4, 2, 1})
          acc += sycl::permute_group_by_xor(it.get_sub_group(), acc, delta);
        if (!lane) {
          const size_t at = Down ? size_t(destinations[entry]) * H + row
                                 : size_t(row / FF) * cap * FF +
                                       size_t(entry) * FF + row % FF;
          out[at] = acc - corr;
        }
      });
}
} // namespace
void native_q2_quantize_cpu_order(const float *input, uint8_t *blocks,
                                  float *scales, int64_t n, void *stream) {
  if (n <= 0 || n > INT32_MAX || n % 32)
    throw std::invalid_argument("invalid native Q2 CPU-order activation width");
  validate_spans({{blocks, size_t(n / 32) * 34}, {scales, size_t(n / 32) * 4}},
                 {{input, size_t(n) * 4}});
  for_each(n / 32, stream, [=](size_t b) {
    const float *row = input + b * 32;
    uint8_t *block = blocks + b * 34;
    float maximum = 0;
    for (int i = 0; i < 32; ++i)
      maximum = sycl::fmax(maximum, sycl::fabs(row[i]));
    const float scale = maximum * (1.f / 127.f);
    const float inverse =
        scale > 0 ? sycl::ext::intel::math::fdiv_rn(1.f, scale) : 0;
    scales[b] = scale;
    const uint16_t half = f16_from_f32(scale);
    block[0] = uint8_t(half);
    block[1] = uint8_t(half >> 8);
    for (int i = 0; i < 32; ++i) {
      const float value = row[i] * inverse;
      const float rounded = value + (value >= 0 ? .5f : -.5f);
      block[2 + i] = uint8_t(int8_t(sycl::clamp(int(rounded), -127, 127)));
    }
  });
}
void native_q2_expert_cpu_order(const unsigned long long *pointers,
                                const int32_t *destinations,
                                const int32_t *count, int cap, const uint8_t *x,
                                const float *scales, void *scratch,
                                float *output, void *stream) {
  if (cap <= 0 || cap > INT32_MAX / (2 * FF))
    throw std::invalid_argument("invalid native Q2 CPU-order capacity");
  const size_t pairs = size_t(cap) * FF,
               bytes = native_expert_scratch_bytes(cap, FF);
  validate_spans({{scratch, bytes}, {output, size_t(cap) * H * 4}},
                 {{pointers, size_t(cap) * 8},
                  {destinations, size_t(cap) * 4},
                  {count, 4},
                  {x, H / 32 * 34},
                  {scales, H / 32 * 4}});
  if (reinterpret_cast<uintptr_t>(pointers) % 8)
    throw std::invalid_argument("unaligned native Q2 pointer table");
  auto &q = queue_for(stream);
  auto *gu = static_cast<float *>(scratch);
  auto *hq = reinterpret_cast<uint8_t *>(scratch) + pairs * 8;
  auto *hs =
      reinterpret_cast<float *>(hq + ((pairs / 32 * 34 + 15) & ~size_t(15)));
  q.memset(gu, 0, pairs * 8);
  projection<false>(pointers, destinations, count, cap, x, scales, gu, q);
  q.parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> id) {
    const size_t i = id[0];
    const float g = gu[i];
    gu[i] = sycl::ext::intel::math::fdiv_rn(
                g, 1.f + sycl::ext::intel::math::ha::exp(-g)) *
            gu[pairs + i];
  });
  native_q2_quantize_cpu_order(gu, hq, hs, pairs, &q);
  projection<true>(pointers, destinations, count, cap, hq, hs, output, q);
  if (!stream)
    runtime_for()->wait(q.ext_oneapi_submit_barrier());
}
} // namespace strata::kernels
