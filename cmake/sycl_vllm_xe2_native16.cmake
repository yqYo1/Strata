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
  -O3 -fsycl -fsycl-targets=spir64_gen
  -fno-fast-math -ffp-contract=off
  -fno-sycl-instrument-device-code -ftemplate-backtrace-limit=0
  "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  "SHELL:-Xsycl-target-backend=spir64_gen \"-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread\"")
# TLA declares SPIR-V builtins with SYCL_EXTERNAL, so -fno-sycl-rdc cannot be
# used. Finish this translation unit in a separate, ordinary -fsycl-link step.
# That command produces a device registration object, not the original host
# methods. Extract the host slice separately and link BOTH plain host objects
# directly. No relocatable native device IR reaches the generic engine link.
# Intel LLVM UsersManual "Link options" and sycl-offload-old-model.cpp cover
# -fsycl-link for spir64_gen. Exact mixed-image acceptance requires Main tests.
# Installed Intel SDKs place the bundler below bin/compiler; standalone LLVM
# installs place it beside clang. Use this compiler's own tool lookup.
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-prog-name=clang-offload-bundler
  OUTPUT_VARIABLE _strata_offload_bundler OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT IS_ABSOLUTE "${_strata_offload_bundler}" OR NOT EXISTS "${_strata_offload_bundler}" OR
   IS_DIRECTORY "${_strata_offload_bundler}")
  message(FATAL_ERROR "SYCL compiler did not resolve an installed clang-offload-bundler: ${_strata_offload_bundler}")
endif()
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine
  OUTPUT_VARIABLE _strata_host_triple OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(_strata_native_image "${CMAKE_CURRENT_BINARY_DIR}/strata_vllm_xe2_image.o")
set(_strata_native_host "${CMAKE_CURRENT_BINARY_DIR}/strata_vllm_xe2_host.o")
add_custom_command(OUTPUT "${_strata_native_image}" "${_strata_native_host}"
  COMMAND "${CMAKE_CXX_COMPILER}" -O3 -fsycl -fsycl-targets=spir64_gen
    -fsycl-link -fno-sycl-instrument-device-code
    -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
    -Xsycl-target-backend=spir64_gen "-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread"
    $<TARGET_OBJECTS:strata_vllm_xe2_native16> -o "${_strata_native_image}"
  COMMAND "${_strata_offload_bundler}" -unbundle -type=o
    "-targets=host-${_strata_host_triple}"
    "-input=$<TARGET_OBJECTS:strata_vllm_xe2_native16>"
    "-output=${_strata_native_host}"
  DEPENDS strata_vllm_xe2_native16 $<TARGET_OBJECTS:strata_vllm_xe2_native16>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(strata_vllm_xe2_image DEPENDS "${_strata_native_image}" "${_strata_native_host}")
target_compile_definitions(strata_prefill PRIVATE STRATA_SYCL_VLLM_XE2_NATIVE16=1)
add_dependencies(strata_prefill strata_vllm_xe2_image)
target_link_libraries(strata_prefill INTERFACE "${_strata_native_image}" "${_strata_native_host}")
option(STRATA_SYCL_PREFILL_NATIVE_COPY "Build the experimental Level Zero prefill copy transport" OFF)
if(STRATA_SYCL_PREFILL_NATIVE_COPY)
  target_compile_definitions(strata_prefill PRIVATE STRATA_SYCL_PREFILL_NATIVE_COPY=1)
  target_link_libraries(strata_prefill PRIVATE ze_loader)
endif()
message(STATUS "Strata: opt-in vLLM Xe2 native16; isolated BMG G21 AOT object; existing engine JIT; external TLA pinned; UNQUALIFIED")

function(strata_add_vllm_native16_tests)
  if(STRATA_SYCL_PREFILL_NATIVE_COPY)
    target_compile_definitions(sycl_expert_bank_test PRIVATE STRATA_SYCL_PREFILL_NATIVE_COPY=1)
    target_link_libraries(sycl_expert_bank_test PRIVATE ze_loader)
  endif()
  # Pure CPU proofs for the opt-in long window; manual targets, no GPU loader.
  add_executable(prefill_moe_routes_test EXCLUDE_FROM_ALL tests/moe_accum_routes.cpp)
  add_executable(prefill_moe_workspace_test EXCLUDE_FROM_ALL tests/prefill_moe_workspace.cpp)
  add_executable(prefill_fullfit_upload_plan_test EXCLUDE_FROM_ALL tests/prefill/fullfit_upload_plan_test.cpp)
  foreach(target IN ITEMS prefill_moe_routes_test prefill_moe_workspace_test prefill_fullfit_upload_plan_test)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(${target} PRIVATE cxx_std_17)
  endforeach()
  # Qualify the GU wrapper against the same freshly built engine archives.
  add_executable(sycl_gu_policy_test EXCLUDE_FROM_ALL bench/sycl/gu_compact.cpp)
  target_link_libraries(sycl_gu_policy_test PRIVATE strata_prefill)
  target_compile_options(sycl_gu_policy_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_moe_accum_test EXCLUDE_FROM_ALL tests/sycl/moe_accum.cpp)
  target_link_libraries(sycl_moe_accum_test PRIVATE strata_prefill)
  target_compile_options(sycl_moe_accum_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_prefill_attention_bank_test EXCLUDE_FROM_ALL tests/sycl/prefill_attention_bank.cpp)
  target_link_libraries(sycl_prefill_attention_bank_test PRIVATE strata_prefill)
  target_compile_features(sycl_prefill_attention_bank_test PRIVATE cxx_std_20)
  target_compile_options(sycl_prefill_attention_bank_test PRIVATE -fno-fast-math -ffp-contract=off)
  # Component fixtures link a second, isolated prefill kernel. The measurement
  # controller generates and hashes this source before configuring the build.
  function(strata_qsa_prefill_fixture_kernel target source_path)
    add_library(${target} OBJECT EXCLUDE_FROM_ALL "${source_path}")
    target_link_libraries(${target} PRIVATE strata_core strata_artifact)
    target_compile_features(${target} PRIVATE cxx_std_20)
    get_target_property(qsa_kernel_options strata_kernels COMPILE_OPTIONS)
    get_source_file_property(qsa_prefix_options src/kernels/sycl/qsa_bf16_prefix.cpp COMPILE_OPTIONS)
    target_compile_options(${target} PRIVATE ${qsa_kernel_options})
    set_source_files_properties("${source_path}" PROPERTIES COMPILE_OPTIONS "${qsa_prefix_options}")
  endfunction()
  if(STRATA_QSA_PREFILL_PV_SOURCE)
    strata_qsa_prefill_fixture_kernel(sycl_qsa_prefill_pv_kernel "${STRATA_QSA_PREFILL_PV_SOURCE}")
    add_executable(sycl_qsa_prefill_pv_test EXCLUDE_FROM_ALL tests/sycl/qsa_bf16_prefix.cpp)
    add_executable(sycl_qsa_prefill_pv_bench EXCLUDE_FROM_ALL tests/sycl/qsa_bf16_prefix_layout_bench.cpp)
    target_compile_definitions(sycl_qsa_prefill_pv_test PRIVATE STRATA_QSA_PACKED_GUARD=17)
    foreach(target IN ITEMS sycl_qsa_prefill_pv_test sycl_qsa_prefill_pv_bench)
      target_link_libraries(${target} PRIVATE sycl_qsa_prefill_pv_kernel strata_kernels)
      target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench/sycl")
      target_compile_features(${target} PRIVATE cxx_std_20)
      target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off)
      target_compile_definitions(${target} PRIVATE STRATA_QSA_PREFILL_PV_COMPONENT=1)
    endforeach()
  endif()
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
  add_executable(sycl_moe_down_scatter_bench EXCLUDE_FROM_ALL bench/sycl/moe_down_scatter.cpp)
  target_include_directories(sycl_moe_down_scatter_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/sycl/vllm_xe2")
  target_link_libraries(sycl_moe_down_scatter_bench PRIVATE strata_prefill)
  target_compile_options(sycl_moe_down_scatter_bench PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_moe_gu_swiglu_bench EXCLUDE_FROM_ALL bench/sycl/moe_gu_swiglu.cpp)
  target_include_directories(sycl_moe_gu_swiglu_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/sycl/vllm_xe2")
  target_link_libraries(sycl_moe_gu_swiglu_bench PRIVATE strata_prefill)
  target_compile_options(sycl_moe_gu_swiglu_bench PRIVATE -fno-fast-math -ffp-contract=off)
endfunction()
