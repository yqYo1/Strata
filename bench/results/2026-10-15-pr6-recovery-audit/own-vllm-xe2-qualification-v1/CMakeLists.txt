# Authored for Strata; repository LICENSE (MIT) applies. Main-only / UNRUN.
cmake_minimum_required(VERSION 3.24)
project(strata_r816_qualification LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(R810_PACKAGE "" CACHE PATH "Pinned existing R810 source package")
set(V2_BUILD "" CACHE PATH "Immutable qualified v2 archives")
if(NOT IS_ABSOLUTE "${R810_PACKAGE}" OR NOT IS_ABSOLUTE "${V2_BUILD}")
  message(FATAL_ERROR "Explicit R810_PACKAGE and V2_BUILD required")
endif()
set(STRATA_ENABLE_VLLM_XE2_PROPOSAL ON CACHE BOOL "" FORCE)
set(STRATA_VLLM_XE2_BUILD_FIXTURE ON CACHE BOOL "" FORCE)
add_subdirectory("${R810_PACKAGE}" "${CMAKE_BINARY_DIR}/r810")
# Retain the original fixture, original current runtime.cpp, all numerics, guards
# and drains. Only the entrypoint name changes to allow own-process DSO evidence.
set_property(SOURCE "${R810_PACKAGE}/numerics_chain.cpp"
  TARGET_DIRECTORY strata_vllm_xe2_numerics APPEND PROPERTY COMPILE_DEFINITIONS main=r810_fixture_main)
target_sources(strata_vllm_xe2_numerics PRIVATE fixture_entry.cpp)
target_include_directories(strata_vllm_xe2_numerics PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(paired_gemm paired_gemm.cpp)
target_compile_definitions(paired_gemm PRIVATE STRATA_ENABLE_SYCL=1 STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.38" MKL_ILP64)
target_include_directories(paired_gemm PRIVATE
  "${STRATA_SOURCE_DIR}/include/strata/sycl/compat"
  "/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011/ggml/include")
target_include_directories(paired_gemm SYSTEM PRIVATE "/opt/intel/oneapi/mkl/2026.1/include")
target_compile_options(paired_gemm PRIVATE -O3 -DNDEBUG -fsycl -fsycl-targets=spir64_gen
  -fsycl-id-queries-range=size_t -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror)
target_compile_features(paired_gemm PRIVATE cxx_std_20)
# No runtime.cpp object here. One registry/Runtime ABI from EXISTING v2 archive.
# Exactly the prior v2 archive order; these are file inputs, never build targets.
target_link_libraries(paired_gemm PRIVATE strata_vllm_xe2_native16
  "${V2_BUILD}/libstrata_prefill.a" "${V2_BUILD}/libstrata_engine.a"
  "${V2_BUILD}/libstrata_kernels.a" "${V2_BUILD}/libstrata_core.a"
  "${V2_BUILD}/libstrata_sycl_runtime.a" "${V2_BUILD}/libstrata_kernels_cpu.a"
  "${V2_BUILD}/ggml/src/libggml-cpu.a" "${V2_BUILD}/ggml/src/libggml-base.a"
  m -fsycl -fsycl-allow-device-image-dependencies
  "/opt/intel/oneapi/mkl/2026.1/lib/libmkl_sycl_blas.so"
  "/opt/intel/oneapi/mkl/2026.1/lib/libmkl_intel_ilp64.so"
  "/opt/intel/oneapi/mkl/2026.1/lib/libmkl_sequential.so"
  "/opt/intel/oneapi/mkl/2026.1/lib/libmkl_core.so" m dl pthread)
target_link_options(paired_gemm PRIVATE -fsycl-id-queries-range=size_t
  -Wl,-rpath,/opt/intel/oneapi/mkl/2026.1/lib)
