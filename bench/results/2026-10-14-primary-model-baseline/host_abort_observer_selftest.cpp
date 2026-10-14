// Direct libc entry-point checks, without inducing a buffer overflow or a GPU run.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sycl/exception.hpp>

extern "C" [[noreturn]] void __stack_chk_fail();

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "abort") == 0) std::abort();
  if (argc == 2 && std::strcmp(argv[1], "stack") == 0) __stack_chk_fail();
  try {
    throw sycl::exception(sycl::make_error_code(sycl::errc::nd_range),
                          "intentional host-only diagnostic observer test");
  } catch (const sycl::exception &error) {
    std::cout << "original SYCL exception caught: " << error.what() << '\n';
    return 0;
  }
  return 1;
}
