// Standalone research probe; the multiply checks one FP16 tile, not model
// accuracy.
#include <iostream>
#include <stdexcept>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>
namespace mx = sycl::ext::oneapi::experimental::matrix;
const char *name(mx::matrix_type t) {
  switch (t) {
  case mx::matrix_type::bf16:
    return "bf16";
  case mx::matrix_type::fp16:
    return "fp16";
  case mx::matrix_type::tf32:
    return "tf32";
  case mx::matrix_type::fp32:
    return "fp32";
  case mx::matrix_type::sint8:
    return "s8";
  case mx::matrix_type::uint8:
    return "u8";
  case mx::matrix_type::sint32:
    return "s32";
  default:
    return "other";
  }
}
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v};
    for (auto c : q.get_device()
                      .get_info<sycl::ext::oneapi::experimental::info::device::
                                    matrix_combinations>())
      std::cout << name(c.atype) << "," << name(c.btype) << "," << name(c.ctype)
                << " max=" << c.max_msize << "," << c.max_nsize << ","
                << c.max_ksize << " exact=" << c.msize << "," << c.nsize << ","
                << c.ksize << "\n";
    constexpr int M = 8, N = 16, K = 16;
    auto a = sycl::malloc_shared<sycl::half>(M * K, q);
    auto b = sycl::malloc_shared<sycl::half>(K * N, q);
    auto c = sycl::malloc_shared<float>(M * N, q);
    if (!a || !b || !c)
      throw std::runtime_error("USM allocation failed");
    for (int i = 0; i < M * K; ++i)
      a[i] = 1;
    for (int i = 0; i < K * N; ++i)
      b[i] = 2;
    q.parallel_for(
         sycl::nd_range<1>{16, 16},
         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
           auto sg = it.get_sub_group();
           mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, M, K,
                            mx::layout::row_major>
               ja;
           mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, K, N,
                            mx::layout::row_major>
               jb;
           mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, M, N>
               jc;
           mx::joint_matrix_load(sg, ja,
                                 sycl::address_space_cast<
                                     sycl::access::address_space::global_space,
                                     sycl::access::decorated::no>(a),
                                 K);
           mx::joint_matrix_load(sg, jb,
                                 sycl::address_space_cast<
                                     sycl::access::address_space::global_space,
                                     sycl::access::decorated::no>(b),
                                 N);
           mx::joint_matrix_fill(sg, jc, 0.0f);
           mx::joint_matrix_mad(sg, jc, ja, jb, jc);
           mx::joint_matrix_store(sg, jc,
                                  sycl::address_space_cast<
                                      sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(c),
                                  N, mx::layout::row_major);
         })
        .wait_and_throw();
    int errors = 0;
    for (int i = 0; i < M * N; ++i)
      errors += c[i] != 32;
    std::cout << "fp16 joint_matrix errors=" << errors << "\n";
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return errors != 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
