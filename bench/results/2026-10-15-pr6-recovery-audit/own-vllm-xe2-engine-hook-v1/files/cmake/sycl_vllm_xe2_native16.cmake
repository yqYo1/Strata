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
add_library(strata_vllm_xe2_native16 STATIC src/prefill/sycl/vllm_xe2/native16.cpp)
# The existing production target supplies the sole Runtime/Allocation registry.
# Never add R810 numerics_chain.cpp or src/sycl/runtime.cpp to this target.
target_link_libraries(strata_vllm_xe2_native16 PUBLIC strata_sycl_runtime)
target_include_directories(strata_vllm_xe2_native16 SYSTEM PRIVATE "${SYCL_TLA_SOURCE_DIR}/include")
target_compile_features(strata_vllm_xe2_native16 PRIVATE cxx_std_20)
target_compile_definitions(strata_vllm_xe2_native16 PRIVATE
  CUTLASS_ENABLE_HEADERS_ONLY CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
target_compile_options(strata_vllm_xe2_native16 PRIVATE
  -O3 -fsycl -fsycl-targets=spir64_gen -fno-fast-math -ffp-contract=off
  -fno-sycl-instrument-device-code -ftemplate-backtrace-limit=0)
# The Runtime target supplies one canonical BMG G21 backend/GRF option string.
# Translator flags below are the exact R810 proposal flags; acceptance UNRUN.
target_link_options(strata_vllm_xe2_native16 INTERFACE
  -fsycl -fsycl-targets=spir64_gen
  "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
target_compile_definitions(strata_prefill PRIVATE STRATA_SYCL_VLLM_XE2_NATIVE16=1)
target_link_libraries(strata_prefill PRIVATE strata_vllm_xe2_native16)
message(STATUS "Strata: opt-in R810 vLLM Xe2 native16; BMG G21 AOT; external TLA include bytes pinned; UNQUALIFIED")
