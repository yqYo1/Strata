# Authored for Strata; repository LICENSE (MIT) applies.
cmake_minimum_required(VERSION 3.24)
project(strata_vllm_xe2_proposal LANGUAGES CXX)
option(STRATA_ENABLE_VLLM_XE2_PROPOSAL "Build unqualified standalone proposal" OFF)
option(STRATA_VLLM_XE2_BUILD_FIXTURE "Build separate numerics/chain fixture" OFF)
if(NOT STRATA_ENABLE_VLLM_XE2_PROPOSAL)
  return()
endif()

set(STRATA_SOURCE_DIR "" CACHE PATH "Explicit Strata source with pinned Runtime ABI")
set(SYCL_TLA_SOURCE_DIR "" CACHE PATH "Existing pristine TLA 87f68506 worktree; never fetched")
set(STRATA_XE2_AOT_DEVICE "bmg-g21-a0" CACHE STRING "BMG device: bmg-g21-a0 or bmg-g31-a0")
if(NOT STRATA_XE2_AOT_DEVICE MATCHES "^bmg-g(21|31)-a0$")
  message(FATAL_ERROR "Only explicit BMG G21/G31 AOT targets are accepted")
endif()
if(NOT IS_ABSOLUTE "${STRATA_SOURCE_DIR}" OR NOT IS_ABSOLUTE "${SYCL_TLA_SOURCE_DIR}")
  message(FATAL_ERROR "Supply absolute existing STRATA_SOURCE_DIR and SYCL_TLA_SOURCE_DIR")
endif()

# Pin raw source, not a mutable branch or a silently overridden FetchContent.
# Conservative TLA include closure includes inactive conditional branches.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/manifest.json" proposal_manifest)
function(check_files key root)
  string(JSON count LENGTH "${proposal_manifest}" source_pins "${key}")
  math(EXPR last "${count} - 1")
  foreach(i RANGE 0 ${last})
    string(JSON path GET "${proposal_manifest}" source_pins "${key}" ${i} path)
    string(JSON expected GET "${proposal_manifest}" source_pins "${key}" ${i} sha256)
    if(NOT EXISTS "${root}/${path}")
      message(FATAL_ERROR "Pinned source missing: ${root}/${path}")
    endif()
    file(SHA256 "${root}/${path}" actual)
    if(NOT actual STREQUAL expected)
      message(FATAL_ERROR "Pinned source changed: ${root}/${path}")
    endif()
  endforeach()
endfunction()
check_files(strata "${STRATA_SOURCE_DIR}")
check_files(tla_headers "${SYCL_TLA_SOURCE_DIR}")
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SYCL_TLA_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE pin_status OUTPUT_VARIABLE tla_revision OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT pin_status EQUAL 0 OR NOT tla_revision STREQUAL "87f6850680a580654b9ea2c80dbc01aeb36ad231")
  message(FATAL_ERROR "TLA must be the exact detached 87f68506 source revision")
endif()

add_library(strata_vllm_xe2_native16 STATIC native16.cpp)
target_include_directories(strata_vllm_xe2_native16 PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${STRATA_SOURCE_DIR}/include")
target_include_directories(strata_vllm_xe2_native16 SYSTEM PRIVATE "${SYCL_TLA_SOURCE_DIR}/include")
target_compile_features(strata_vllm_xe2_native16 PUBLIC cxx_std_20)
target_compile_definitions(strata_vllm_xe2_native16 PRIVATE
  CUTLASS_ENABLE_HEADERS_ONLY CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
target_compile_options(strata_vllm_xe2_native16 PRIVATE
  -O3 -fsycl -fsycl-targets=spir64_gen -fno-fast-math -ffp-contract=off
  -fno-sycl-instrument-device-code -ftemplate-backtrace-limit=0)
# Match relevant vLLM Xe2 split-barrier/block2D/DPAS translator extensions and
# 256-GRF backend option. No CUDA, Torch, oneMKL, oneDNN or Ze-loader API call.
target_link_options(strata_vllm_xe2_native16 INTERFACE
  -fsycl -fsycl-targets=spir64_gen
  "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_XE2_AOT_DEVICE} -internal_options -cl-intel-256-GRF-per-thread\"")

if(STRATA_VLLM_XE2_BUILD_FIXTURE)
  find_package(Threads REQUIRED)
  # A separate fixture runtime, with its own registry. Do not link this object
  # into an engine that already defines Runtime/Allocation/validate_device_span.
  add_executable(strata_vllm_xe2_numerics numerics_chain.cpp
    "${STRATA_SOURCE_DIR}/src/sycl/runtime.cpp")
  target_link_libraries(strata_vllm_xe2_numerics PRIVATE strata_vllm_xe2_native16 Threads::Threads)
  target_compile_options(strata_vllm_xe2_numerics PRIVATE -O2 -fsycl
    -fsycl-targets=spir64_gen -fno-fast-math -ffp-contract=off)
endif()
