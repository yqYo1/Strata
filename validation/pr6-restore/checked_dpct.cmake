# Project LICENSE. Transforms the already hash-checked HISTORICAL dependency
# into a private build artifact; never writes the external vendor installation.
# The artifact remains Intel Apache-2.0 WITH LLVM-exception dependency code,
# with its original notice intact. No vendor implementation is added to Git.
file(READ "${PR6_VENDOR_ROOT}/sycl/include/dpct/device.hpp" pr6_device)
string(FIND "${pr6_device}" "inline auto exception_handler =" pr6_handler_start)
string(FIND "${pr6_device}" "\ntypedef sycl::event" pr6_handler_end)
if(pr6_handler_start LESS 0 OR pr6_handler_end LESS pr6_handler_start)
  message(FATAL_ERROR "Pinned DPCT async-handler span not found")
endif()
string(SUBSTRING "${pr6_device}" 0 ${pr6_handler_start} pr6_before)
string(SUBSTRING "${pr6_device}" ${pr6_handler_end} -1 pr6_after)
set(pr6_handler "inline auto exception_handler = [](sycl::exception_list errors) {\n  ::strata::pr6_async_fail_stop(errors);\n};\n")
set(pr6_device "${pr6_before}${pr6_handler}${pr6_after}")
# Register the actual owning shared_ptr; stale pointer addresses cannot authorize
# a later unrelated queue. Every selected default/create_queue(true) uses this.
set(pr6_return "return _queues.back().get();")
# Exclude the C++ semicolon from the match: CMake would treat it as a list
# separator and count an extra empty item under CMP0007 NEW.
string(REGEX MATCHALL "return _queues[.]back[(][)][.]get[(][)]" pr6_returns "${pr6_device}")
list(LENGTH pr6_returns pr6_return_count)
if(NOT pr6_return_count EQUAL 1)
  message(FATAL_ERROR "Pinned DPCT queue-creation return is not unique")
endif()
string(REPLACE "${pr6_return}"
  "if (enable_exception_handler) ::strata::pr6_register_queue(_queues.back());\n    ${pr6_return}"
  pr6_device "${pr6_device}")
set(pr6_checked_header "${CMAKE_CURRENT_BINARY_DIR}/pr6-checked-dpct.hpp")
# Force this header before any vendor relative #include "device.hpp". The same
# original guard then suppresses the unmodified external header in every TU.
file(WRITE "${pr6_checked_header}"
  "// Modified for isolated Strata PR6 validation: fail-stop async handling and checked queue registration.\n#define STRATA_PR6_CHECKED_DPCT 1\n#ifndef DPCT_PROFILING_ENABLED\n#define DPCT_PROFILING_ENABLED\n#endif\n#include \"strata/sycl_async_contract.hpp\"\n${pr6_device}")
file(SHA256 "${pr6_checked_header}" pr6_checked_hash)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/pr6-checked-dpct.sha256" "${pr6_checked_hash}\n")
message(STATUS "PR6 private checked DPCT header SHA256=${pr6_checked_hash}")
