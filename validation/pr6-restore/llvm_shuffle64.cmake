# Independently authored build adapter; repository LICENSE applies.
# Only generated, build-private dependency text retains the original Intel
# Apache-2.0 WITH LLVM-exception notice. No vendor file is modified or committed.
if(NOT PR6_OPEN_SOURCE_LLVM_ONEMKL OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "PR6 shuffle adapter requires the reviewed Linux LLVM/oneMKL mode")
endif()
set(pr6_shuffle_install "/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install")
foreach(lang C CXX)
  file(REAL_PATH "${CMAKE_${lang}_COMPILER}" pr6_shuffle_compiler_real)
  file(SHA256 "${CMAKE_${lang}_COMPILER}" pr6_shuffle_compiler_hash)
  if(NOT pr6_shuffle_compiler_real STREQUAL "${pr6_shuffle_install}/bin/clang-22" OR
     NOT pr6_shuffle_compiler_hash STREQUAL "1290881b4e12a5e002e5b0de0787f85328ee5fe19f253a54be1cdd97dc2f26eb")
    message(FATAL_ERROR "PR6 shuffle adapter requires the pinned LLVM 22.1.0 compiler")
  endif()
endforeach()
# Dependency API definitions only; no source from the enclosing checkout.
foreach(pin
    "sycl/aliases.hpp|b7339db39e3234ca06be4610c31ad620e09e2f1122784a9c63b0dab6cb8a9d8f"
    "sycl/detail/generic_type_traits.hpp|f468d7738636ecd239fa2c89920d30908aff2c9129af5e479b952e84a483faa4"
    "sycl/__spirv/spirv_ops.hpp|4a449dfb36cc9d977a2c9567131b7e2dd666a54b56e4cdf0f1a1dd42ab8c7625")
  string(REPLACE "|" ";" pin_parts "${pin}")
  list(GET pin_parts 0 header)
  list(GET pin_parts 1 expected)
  file(SHA256 "${pr6_shuffle_install}/include/${header}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "PR6 shuffle dependency API hash mismatch: ${header}")
  endif()
endforeach()
set(pr6_shuffle_input "${PR6_VENDOR_ROOT}/sycl/include/dpct/util.hpp")
file(SHA256 "${pr6_shuffle_input}" pr6_shuffle_input_hash)
if(NOT pr6_shuffle_input_hash STREQUAL "c76e74a7d18721523839e30c8d77e89f108423e49044aebb8f50ae1fbd3f424b")
  message(FATAL_ERROR "PR6 shuffle adapter requires the original pinned DPCT util.hpp")
endif()
file(READ "${pr6_shuffle_input}" pr6_shuffle_util)
# Exact byte span [20690,21001), original lines 470-475. No regex or broad
# replacement: a different revision, match count or position fails configure.
set(pr6_shuffle_old [=[#if defined(__SPIR__)
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  unsigned logical_remote_id =
      start_index + remote_local_id % logical_sub_group_size;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);]=])
string(SHA256 pr6_shuffle_old_hash "${pr6_shuffle_old}")
string(LENGTH "${pr6_shuffle_old}" pr6_shuffle_old_length)
string(FIND "${pr6_shuffle_util}" "${pr6_shuffle_old}" pr6_shuffle_start)
if(NOT pr6_shuffle_old_hash STREQUAL "40ad5a113889ddcc32d74a677e3e920dc6d853d54a98866e7cbb7adfa111c42f" OR
   NOT pr6_shuffle_old_length EQUAL 311 OR NOT pr6_shuffle_start EQUAL 20690)
  message(FATAL_ERROR "PR6 pinned shuffle adaptation span mismatch")
endif()
string(SUBSTRING "${pr6_shuffle_util}" 21001 -1 pr6_shuffle_after)
string(FIND "${pr6_shuffle_after}" "${pr6_shuffle_old}" pr6_shuffle_duplicate)
if(NOT pr6_shuffle_duplicate EQUAL -1)
  message(FATAL_ERROR "PR6 pinned shuffle adaptation span is not unique")
endif()
set(pr6_shuffle_new [=[#if defined(__SPIR__)
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  unsigned logical_remote_id =
      start_index + remote_local_id % logical_sub_group_size;
  // Strata PR6 compatibility: adapt only C++ long-long scalar storage to
  // the pinned compiler's native OpenCL 64-bit scalar. Same SPIR-V operation.
  if constexpr (std::is_same_v<T, long long> ||
                std::is_same_v<T, unsigned long long>) {
    using pr6_shuffle_scalar = std::conditional_t<
        std::is_signed_v<T>, sycl::opencl::cl_long, sycl::opencl::cl_ulong>;
    static_assert(sizeof(T) == 8 && sizeof(pr6_shuffle_scalar) == 8,
                  "PR6 shuffle requires 64-bit scalar storage");
    static_assert(std::is_same_v<pr6_shuffle_scalar,
                                std::conditional_t<std::is_signed_v<T>,
                                                   long, unsigned long>>,
                  "PR6 shuffle requires the pinned native OpenCL scalar ABI");
    return sycl::bit_cast<T>(__spirv_GroupNonUniformShuffle(
        __spv::Scope::Subgroup, sycl::bit_cast<pr6_shuffle_scalar>(x),
        logical_remote_id));
  } else {
    return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
  }]=])
string(SUBSTRING "${pr6_shuffle_util}" 0 20690 pr6_shuffle_before)
set(pr6_shuffle_util "${pr6_shuffle_before}${pr6_shuffle_new}${pr6_shuffle_after}")
string(SHA256 pr6_shuffle_util_hash "${pr6_shuffle_util}")
string(SHA256 pr6_shuffle_new_hash "${pr6_shuffle_new}")
if(NOT pr6_shuffle_util_hash STREQUAL "cb93a6d8ecb064bf8d693fad0c9c5e64ce753d6f4b126f5e086969d9b1228609" OR
   NOT pr6_shuffle_new_hash STREQUAL "46b520f81366735619fa66e77211bdd464996eb1cd147cd654adc15571c70ece")
  message(FATAL_ERROR "PR6 generated shuffle adaptation hash mismatch")
endif()
# util.hpp requires memory.hpp's declarations. Load it after the checked device
# body and before any image.hpp/dpct.hpp relative include can see original util.
# Its original guard suppresses that original header in all later includes.
set(pr6_shuffle_suffix "\n// Modified for isolated Strata PR6 LLVM 22 64-bit shuffle compatibility.\n#ifdef __DPCT_UTIL_HPP__\n#error PR6_original_DPCT_util_was_included_before_the_checked_adapter\n#endif\n#include <dpct/memory.hpp>\n${pr6_shuffle_util}")
set(pr6_shuffle_proof "{\"schema\":\"strata-pr6-llvm-shuffle64-v1\",\"input_sha256\":\"${pr6_shuffle_input_hash}\",\"original_span_start\":20690,\"original_span_end\":21001,\"original_span_count\":1,\"original_span_sha256\":\"${pr6_shuffle_old_hash}\",\"replacement_sha256\":\"${pr6_shuffle_new_hash}\",\"generated_util_body_sha256\":\"${pr6_shuffle_util_hash}\"}")
message(STATUS "PR6 private shuffle util body SHA256=${pr6_shuffle_util_hash}")
