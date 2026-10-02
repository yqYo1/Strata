// Small correctness samples only. See the catalog for link flags.
#include <iostream>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/mkl/blas.hpp>
#include <stdexcept>
#include <sycl/sycl.hpp>
int main() {
  try {
    sycl::queue q{sycl::gpu_selector_v};
    auto a = sycl::malloc_shared<float>(4, q);
    auto b = sycl::malloc_shared<float>(4, q);
    auto c = sycl::malloc_shared<float>(4, q);
    auto s = sycl::malloc_shared<int>(4, q);
    if (!a || !b || !c || !s)
      throw std::runtime_error("allocation failed");
    for (int i = 0; i < 4; ++i) {
      a[i] = float(i + 1);
      b[i] = float(i + 5);
      c[i] = 0;
      s[i] = 4 - i;
    }
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans,
                                       oneapi::mkl::transpose::nontrans, 2, 2,
                                       2, 1.0f, a, 2, b, 2, 0.0f, c, 2)
        .wait_and_throw();
    int errors = c[0] != 19 || c[1] != 22 || c[2] != 43 || c[3] != 50;
    std::cout << "onemkl_fp32_gemm errors=" << errors << std::endl;
    oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q), s, s + 4);
    q.wait_and_throw();
    int sort_errors = 0;
    for (int i = 0; i < 4; ++i)
      sort_errors += s[i] != i + 1;
    std::cout << "onedpl_sort errors=" << sort_errors << std::endl;
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    sycl::free(s, q);
    return errors || sort_errors;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
