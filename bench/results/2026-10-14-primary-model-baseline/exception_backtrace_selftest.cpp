#include <sycl/exception.hpp>
#include <iostream>

int main() {
  try {
    throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                          "intentional host-only diagnostic observer test");
  } catch (const sycl::exception &error) {
    std::cout << "original SYCL exception caught: " << error.what() << '\n';
    return 0;
  }
  return 1;
}
