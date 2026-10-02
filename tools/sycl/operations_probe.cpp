// Bounded correctness samples, not throughput or full conformance tests.
// Build: icpx -std=c++20 -fsycl -O2 tools/sycl/operations_probe.cpp -o
// /tmp/sycl-ops
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <vector>
namespace es = sycl::ext::intel::esimd;
namespace xmx = es::xmx;

template <int Bits> int packed_dpas(sycl::queue &q) {
  constexpr int M = 4, N = 16, K = 8 * ((32 / Bits > 8) ? 8 : 32 / Bits);
  constexpr int Pack = 32 / Bits, AN = M * K / Pack, BN = K * N / Pack;
  constexpr auto P = Bits == 2   ? xmx::dpas_argument_type::u2
                     : Bits == 4 ? xmx::dpas_argument_type::u4
                                 : xmx::dpas_argument_type::u8;
  std::vector<uint32_t> a(AN), b(BN), c(M * N);
  auto av = [](int m, int k) {
    return uint32_t((m * 3 + k * 7 + 1) & ((1 << Bits) - 1));
  };
  auto bv = [](int k, int n) {
    return uint32_t((k * 5 + n * 3 + 2) & ((1 << Bits) - 1));
  };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      a[(m * K + k) / Pack] |= av(m, k) << (((m * K + k) % Pack) * Bits);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      b[(k / Pack) * N + n] |= bv(k, n) << ((k % Pack) * Bits);
  auto da = sycl::malloc_device<uint32_t>(AN, q);
  auto db = sycl::malloc_device<uint32_t>(BN, q);
  auto dc = sycl::malloc_device<uint32_t>(M * N, q);
  if (!da || !db || !dc)
    throw std::runtime_error("allocation failed");
  q.memcpy(da, a.data(), a.size() * 4).wait_and_throw();
  q.memcpy(db, b.data(), b.size() * 4).wait_and_throw();
  q.single_task([=]() SYCL_ESIMD_KERNEL {
     es::simd<uint32_t, AN> aa;
     aa.copy_from(da);
     es::simd<uint32_t, BN> bb;
     bb.copy_from(db);
     es::simd<uint32_t, M * N> cc(3);
     auto result =
         xmx::dpas<8, M, uint32_t, uint32_t, uint32_t, uint32_t, P, P>(cc, bb,
                                                                       aa);
     result.copy_to(dc);
   }).wait_and_throw();
  q.memcpy(c.data(), dc, c.size() * 4).wait_and_throw();
  int errors = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      uint32_t ref = 3;
      for (int k = 0; k < K; ++k)
        ref += av(m, k) * bv(k, n);
      errors += c[m * N + n] != ref;
    }
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);
  std::cout << "esimd_u" << Bits << "_dpas M=" << M << " N=" << N << " K=" << K
            << " errors=" << errors << std::endl;
  return errors;
}
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v};
    std::cout << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    auto out = sycl::malloc_shared<double>(8, q);
    auto in = sycl::malloc_shared<double>(1, q);
    if (!out || !in)
      throw std::runtime_error("allocation failed");
    *in = 2.0;
    q.single_task([=] {
       double x = *in;
       out[0] = sycl::fma(x, 3.0, 1.0);
       out[1] = sycl::sqrt(x * x);
       out[2] = sycl::fma(float(x), 3.0f, 1.0f);
       out[3] = float(sycl::half(x) * sycl::half(3));
       out[4] = sycl::popcount(uint32_t(x) + 13u);
       out[5] = sycl::mul_hi(uint32_t(0xffffffffu), uint32_t(x));
     }).wait_and_throw();
    int errors = out[0] != 7 || out[1] != 2 || out[2] != 7 || out[3] != 6 ||
                 out[4] != 4 || out[5] != 1;
    std::cout << "scalar_fp64_fp32_fp16_integer errors=" << errors << std::endl;
    q.parallel_for(sycl::nd_range<1>(16, 16),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                     auto sg = it.get_sub_group();
                     int sum = sycl::reduce_over_group(
                         sg, int(it.get_local_linear_id()), sycl::plus<int>());
                     if (it.get_local_linear_id() == 0)
                       out[6] = sum;
                   })
        .wait_and_throw();
    errors += out[6] != 120;
    std::cout << "subgroup_sum=" << out[6] << " expected=120" << std::endl;
    sycl::free(out, q);
    sycl::free(in, q);
    errors += packed_dpas<8>(q);
    errors += packed_dpas<4>(q);
    errors += packed_dpas<2>(q);
    return errors ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
