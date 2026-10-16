# Authored for Strata; repository LICENSE (MIT) applies.
# Called only with STRATA_SYCL_VLLM_XE2_NATIVE16=ON, after strata_prefill exists.
set(SYCL_TLA_SOURCE_DIR "" CACHE PATH "Existing Intel sycl-tla 87f6850680a580654b9ea2c80dbc01aeb36ad231 source")
if(NOT IS_ABSOLUTE "${SYCL_TLA_SOURCE_DIR}" OR NOT IS_DIRECTORY "${SYCL_TLA_SOURCE_DIR}/include")
  message(FATAL_ERROR "VLLM Xe2 needs explicit absolute existing SYCL_TLA_SOURCE_DIR; no dependency is fetched")
endif()
# Enforce the exact R810 include bytes, including inactive literal branches.
# Hash pinning covers dirty files too; no Git, network or dependency CMake runs.
file(READ "${CMAKE_CURRENT_LIST_DIR}/vllm_xe2_tla_headers.json" _strata_tla_pins)
string(JSON _strata_tla_count LENGTH "${_strata_tla_pins}" headers)
math(EXPR _strata_tla_last "${_strata_tla_count} - 1")
foreach(i RANGE 0 ${_strata_tla_last})
  string(JSON path GET "${_strata_tla_pins}" headers ${i} path)
  string(JSON expected GET "${_strata_tla_pins}" headers ${i} sha256)
  if(NOT EXISTS "${SYCL_TLA_SOURCE_DIR}/${path}")
    message(FATAL_ERROR "Pinned TLA header missing: ${path}")
  endif()
  file(SHA256 "${SYCL_TLA_SOURCE_DIR}/${path}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "TLA header differs from 87f68506/R810: ${path}")
  endif()
endforeach()
add_library(strata_vllm_xe2_native16 OBJECT src/prefill/sycl/vllm_xe2/native16.cpp)
# The existing production target supplies the sole Runtime/Allocation registry.
# Never add R810 numerics_chain.cpp or src/sycl/runtime.cpp to this target.
target_link_libraries(strata_vllm_xe2_native16 PRIVATE strata_sycl_runtime)
target_include_directories(strata_vllm_xe2_native16 SYSTEM PRIVATE "${SYCL_TLA_SOURCE_DIR}/include")
target_compile_features(strata_vllm_xe2_native16 PRIVATE cxx_std_20)
target_compile_definitions(strata_vllm_xe2_native16 PRIVATE
  CUTLASS_ENABLE_HEADERS_ONLY CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
target_compile_options(strata_vllm_xe2_native16 PRIVATE
  -O3 -fsycl -fsycl-targets=spir64_gen -fno-sycl-rdc
  -fno-fast-math -ffp-contract=off
  -fno-sycl-instrument-device-code -ftemplate-backtrace-limit=0
  "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  "SHELL:-Xsycl-target-backend=spir64_gen \"-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread\"")
# Eagerly finish only this translation unit's device image during -c. Keep the
# resulting fat object as a DIRECT final-link input, outside the generic engine
# archives. No AOT/GRF flags propagate to the engine's existing JIT images.
# Intel LLVM CompilerAndRuntimeDesign: "Device Link during compilation".
# Exact installed-driver mixed-link acceptance still requires Main's tests.
target_compile_definitions(strata_prefill PRIVATE STRATA_SYCL_VLLM_XE2_NATIVE16=1)
add_dependencies(strata_prefill strata_vllm_xe2_native16)
target_link_libraries(strata_prefill INTERFACE "$<TARGET_OBJECTS:strata_vllm_xe2_native16>")
message(STATUS "Strata: opt-in vLLM Xe2 native16; isolated BMG G21 AOT object; existing engine JIT; external TLA pinned; UNQUALIFIED")

function(strata_add_vllm_native16_tests)
  add_executable(sycl_vllm_native16_test EXCLUDE_FROM_ALL tests/sycl/vllm_native16.cpp)
  target_include_directories(sycl_vllm_native16_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/sycl/vllm_xe2")
  target_link_libraries(sycl_vllm_native16_test PRIVATE strata_prefill)
  target_compile_options(sycl_vllm_native16_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_native16_paired_gemm_bench EXCLUDE_FROM_ALL bench/sycl/native16_gemm.cpp)
  target_include_directories(sycl_native16_paired_gemm_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/sycl/vllm_xe2"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl/compat")
  target_link_libraries(sycl_native16_paired_gemm_bench PRIVATE strata_prefill)
  target_compile_options(sycl_native16_paired_gemm_bench PRIVATE -fno-fast-math -ffp-contract=off)
endfunction()
