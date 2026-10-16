// Independent varying-input integer DPAS checks for Q2_0 expert packing.
// Build with icpx -std=c++20 -fsycl -O2.
// Run with ONEAPI_DEVICE_SELECTOR=level_zero:gpu.
#include <cstdint>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>
namespace e = sycl::ext::intel::esimd;
namespace x = e::xmx;
template <int M> int test(sycl::queue &q) {
  auto *weights = sycl::malloc_device<uint16_t>(16 * 9, q);
  auto *act = sycl::malloc_device<int32_t>(M * 8, q);
  auto *correction = sycl::malloc_device<int32_t>(M * 16, q);
  auto *output = sycl::malloc_device<int32_t>(M * 16, q);
  if (!weights || !act || !correction || !output)
    throw std::runtime_error("allocation failed");
  int errors = 0;
  uint32_t rng = 0xabcddcba;
  auto next = [&]() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
  };
  for (int trial = 0; trial < 16; ++trial) {
    std::vector<uint16_t> w(16 * 9, 0x55aa);
    int codes[16][64];
    for (int row = 0; row < 16; ++row)
      for (int k = 0; k < 64; ++k) {
        const int code = trial < 4 ? trial : next() % 4;
        codes[row][k] = code;
        const int word = row * 9 + 1 + k / 8, shift = (k % 8) * 2;
        w[word] =
            (w[word] & ~(uint16_t(3) << shift)) | (uint16_t(code) << shift);
      }
    for (int half = 0; half < 2; ++half) {
      std::vector<int32_t> packed(M * 8), c(M * 16), result(M * 16);
      int a[M][32];
      for (int row = 0; row < M; ++row)
        for (int k = 0; k < 32; ++k) {
          int val = trial == 4   ? -128
                    : trial == 5 ? 127
                    : trial == 6 ? 0
                                 : int(next() % 256) - 128;
          a[row][k] = val;
          uint32_t p = uint32_t(packed[row * 8 + k / 4]);
          p |= uint32_t(uint8_t(val)) << ((k % 4) * 8);
          packed[row * 8 + k / 4] = int32_t(p);
          for (int n = 0; n < 16; ++n)
            c[row * 16 + n] -= val;
        }
      q.memcpy(weights, w.data(), w.size() * 2).wait_and_throw();
      q.memcpy(act, packed.data(), packed.size() * 4).wait_and_throw();
      q.memcpy(correction, c.data(), c.size() * 4).wait_and_throw();
      q.single_task([=]() SYCL_ESIMD_KERNEL {
         const e::simd<uint32_t, 16> row(0, 1);
         const auto offsets = row * 18u + 2u + uint32_t(half) * 8u;
         e::simd<uint32_t, 32> bb;
         bb.template select<16, 1>(0) =
             e::gather<uint16_t, 16>(weights, offsets) |
             (e::convert<uint32_t>(
                  e::gather<uint16_t, 16>(weights, offsets + 2u))
              << 16);
         bb.template select<16, 1>(16) =
             e::gather<uint16_t, 16>(weights, offsets + 4u) |
             (e::convert<uint32_t>(
                  e::gather<uint16_t, 16>(weights, offsets + 6u))
              << 16);
         e::simd<int32_t, M * 8> aa;
         aa.copy_from(act);
         e::simd<int32_t, M * 16> cc;
         cc.copy_from(correction);
         auto rr =
             x::dpas<8, M, int32_t, int32_t, uint32_t, int32_t,
                     x::dpas_argument_type::u2, x::dpas_argument_type::s8>(
                 cc, bb, aa);
         rr.copy_to(output);
       }).wait_and_throw();
      q.memcpy(result.data(), output, result.size() * 4).wait_and_throw();
      for (int row = 0; row < M; ++row)
        for (int n = 0; n < 16; ++n) {
          int ref = 0;
          for (int k = 0; k < 32; ++k)
            ref += (codes[n][half * 32 + k] - 1) * a[row][k];
          if (ref != result[row * 16 + n]) {
            if (errors < 4)
              std::cerr << trial << ' ' << half << ' ' << row << ' ' << n << ' '
                        << ref << ' ' << result[row * 16 + n] << '\n';
            ++errors;
          }
        }
    }
  }
  sycl::free(weights, q);
  sycl::free(act, q);
  sycl::free(correction, q);
  sycl::free(output, q);
  std::cout << "mixed s8/u2 M" << M << " N16 K32, 16 trials, two block halves, "
            << M * 16 * 32 << " results: errors=" << errors << '\n';
  return errors;
}
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    return test<1>(q) + test<4>(q) ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
