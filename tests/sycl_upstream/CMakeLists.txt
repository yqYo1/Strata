# Isolated component validation; this does not enable a full SYCL engine.
cmake_minimum_required(VERSION 3.24)
project(strata_sycl_upstream LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(STRATA_GGML_DIR "" CACHE PATH "GGML checkout pinned by Strata")
set(STRATA_SYCL_DEVICE_ARCH "" CACHE STRING "Optional AOT device, e.g. bmg_g21")
get_filename_component(STRATA_ROOT ../.. ABSOLUTE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_oracle.py
    --ggml ${STRATA_GGML_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}/upstream_layout.hpp
    COMMAND_ERROR_IS_FATAL ANY)
add_executable(mmq_quantize_test mmq_quantize.cpp ${STRATA_ROOT}/src/sycl_upstream/mmq_quantize.cpp)
target_include_directories(mmq_quantize_test PRIVATE ${STRATA_ROOT}/include
    ${STRATA_GGML_DIR}/ggml/include ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(mmq_quantize_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off)
target_link_options(mmq_quantize_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    if(NOT STRATA_SYCL_DEVICE_ARCH MATCHES "^[A-Za-z0-9_.-]+$")
        message(FATAL_ERROR "Invalid AOT device name")
    endif()
    target_compile_options(mmq_quantize_test PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(mmq_quantize_test PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()

enable_testing()
add_test(NAME mmq_quantize_upstream_source COMMAND mmq_quantize_test)
set_tests_properties(mmq_quantize_upstream_source PROPERTIES TIMEOUT 90)

execute_process(COMMAND ${Python3_EXECUTABLE} ${STRATA_ROOT}/tools/sycl/extract_mmq_loaders.py
    --ggml ${STRATA_GGML_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
add_library(mmq_product STATIC ${STRATA_ROOT}/src/sycl_upstream/mmq_product.cpp)
target_include_directories(mmq_product PUBLIC ${STRATA_ROOT}/include ${STRATA_GGML_DIR}/ggml/include
    ${STRATA_GGML_DIR}/ggml/src ${CMAKE_CURRENT_BINARY_DIR})
# Original CUDA loaders intentionally reinterpret packed fields and SLM.
# Default strict aliasing miscompiled IQ2_XS on B570; the independent CPU
# dequantizer test detects this (see docs/sycl-audit/mmq-product.md).
target_compile_options(mmq_product PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_product PRIVATE -fsycl-targets=spir64_gen)
endif()
add_executable(mmq_product_test mmq_product.cpp ${STRATA_ROOT}/src/sycl_upstream/mmq_quantize.cpp)
target_link_libraries(mmq_product_test PRIVATE mmq_product)
target_compile_options(mmq_product_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
target_link_options(mmq_product_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_product_test PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(mmq_product_test PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()
add_test(NAME mmq_product_upstream_cpu COMMAND mmq_product_test)
set_tests_properties(mmq_product_upstream_cpu PROPERTIES TIMEOUT 120)

add_executable(mmq_stages_test mmq_stages.cpp
    ${STRATA_ROOT}/src/sycl_upstream/mmq_stages.cpp
    ${STRATA_ROOT}/src/sycl_upstream/mmq_quantize.cpp)
target_link_libraries(mmq_stages_test PRIVATE mmq_product)
target_compile_options(mmq_stages_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
target_link_options(mmq_stages_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_stages_test PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(mmq_stages_test PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()
add_test(NAME mmq_surrounding_stages COMMAND mmq_stages_test)
set_tests_properties(mmq_surrounding_stages PROPERTIES TIMEOUT 120)

# Use the pinned GGML's actual type table for the public matrix_bytes API,
# including non-MMQ formats. Do not duplicate a reduced size/type table.
set(GGML_CPU OFF CACHE BOOL "Component tests need only GGML base" FORCE)
set(GGML_CUDA OFF CACHE BOOL "No CUDA backend in component tests" FORCE)
set(GGML_SYCL OFF CACHE BOOL "Strata supplies its own faithful MMQ component" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

add_library(strata_sycl_mmq_kernels STATIC
    ${STRATA_ROOT}/src/sycl_upstream/mmq_stages.cpp
    ${STRATA_ROOT}/src/sycl_upstream/mmq_quantize.cpp)
target_link_libraries(strata_sycl_mmq_kernels PUBLIC mmq_product ggml-base)
target_compile_options(strata_sycl_mmq_kernels PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
target_link_options(strata_sycl_mmq_kernels PUBLIC -fsycl)
add_library(strata_sycl_mmq STATIC ${STRATA_ROOT}/src/sycl_upstream/mmq_adapter.cpp)
target_link_libraries(strata_sycl_mmq PUBLIC strata_sycl_mmq_kernels)
target_compile_options(strata_sycl_mmq PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
target_link_options(strata_sycl_mmq PUBLIC -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_mmq PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(strata_sycl_mmq_kernels PRIVATE -fsycl-targets=spir64_gen)
    target_compile_definitions(strata_sycl_mmq PRIVATE
        STRATA_SYCL_MMQ_AOT_ARCH=sycl::ext::oneapi::experimental::architecture::intel_gpu_${STRATA_SYCL_DEVICE_ARCH})
    target_link_options(strata_sycl_mmq PUBLIC -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()

add_executable(mmq_public_stages_test mmq_stages.cpp)
target_compile_definitions(mmq_public_stages_test PRIVATE STRATA_TEST_PUBLIC_MMQ=1)
target_link_libraries(mmq_public_stages_test PRIVATE strata_sycl_mmq)
target_compile_options(mmq_public_stages_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_public_stages_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME mmq_public_stages COMMAND mmq_public_stages_test)
set_tests_properties(mmq_public_stages PROPERTIES TIMEOUT 120)
add_executable(mmq_context_test mmq_context.cpp)
target_link_libraries(mmq_context_test PRIVATE strata_sycl_mmq)
target_compile_options(mmq_context_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_context_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME mmq_context_lifetime COMMAND mmq_context_test)
set_tests_properties(mmq_context_lifetime PROPERTIES TIMEOUT 120)

add_library(strata_sycl_runtime STATIC ${STRATA_ROOT}/src/sycl_upstream/runtime.cpp)
target_include_directories(strata_sycl_runtime PUBLIC ${STRATA_ROOT}/include)
target_compile_options(strata_sycl_runtime PRIVATE -fsycl)
target_link_options(strata_sycl_runtime PUBLIC -fsycl)
add_executable(runtime_test runtime.cpp)
target_link_libraries(runtime_test PRIVATE strata_sycl_runtime)
target_compile_options(runtime_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_runtime PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(runtime_test PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(strata_sycl_runtime PUBLIC -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()
add_test(NAME runtime_stream_event_ordering COMMAND runtime_test)
set_tests_properties(runtime_stream_event_ordering PROPERTIES TIMEOUT 45)

add_executable(graph_test graph.cpp)
target_link_libraries(graph_test PRIVATE strata_sycl_runtime strata_sycl_mmq)
target_compile_options(graph_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(graph_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME runtime_graph_capture_replay COMMAND graph_test)
set_tests_properties(runtime_graph_capture_replay PROPERTIES TIMEOUT 90)

add_library(strata_sycl_handoff STATIC ${STRATA_ROOT}/src/sycl_upstream/handoff.cpp)
target_include_directories(strata_sycl_handoff PUBLIC ${STRATA_ROOT}/include)
target_compile_options(strata_sycl_handoff PRIVATE -fsycl)
target_link_options(strata_sycl_handoff PUBLIC -fsycl)
add_executable(handoff_test handoff.cpp)
target_link_libraries(handoff_test PRIVATE strata_sycl_runtime strata_sycl_handoff)
target_compile_options(handoff_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_handoff PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(handoff_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME runtime_mapped_handoff COMMAND handoff_test)
set_tests_properties(runtime_mapped_handoff PROPERTIES TIMEOUT 90)

# Opt-in diagnostics: the variant without a polling fence may deliberately fail.
foreach(poll_fence IN ITEMS 0 1)
    add_executable(handoff_probe_${poll_fence} EXCLUDE_FROM_ALL handoff_probe.cpp)
    target_compile_definitions(handoff_probe_${poll_fence} PRIVATE STRATA_HANDOFF_POLL_FENCE=${poll_fence})
    target_link_libraries(handoff_probe_${poll_fence} PRIVATE strata_sycl_runtime)
    target_compile_options(handoff_probe_${poll_fence} PRIVATE -fsycl)
    if(STRATA_SYCL_DEVICE_ARCH)
        target_compile_options(handoff_probe_${poll_fence} PRIVATE -fsycl-targets=spir64_gen)
    endif()
endforeach()

find_library(STRATA_LEVEL_ZERO_LOADER ze_loader REQUIRED)
add_library(strata_sycl_memory STATIC ${STRATA_ROOT}/src/sycl_upstream/memory.cpp)
target_include_directories(strata_sycl_memory PUBLIC ${STRATA_ROOT}/include)
target_link_libraries(strata_sycl_memory PUBLIC strata_sycl_runtime PRIVATE ${STRATA_LEVEL_ZERO_LOADER})
target_compile_options(strata_sycl_memory PRIVATE -fsycl)
add_executable(memory_test memory.cpp)
target_link_libraries(memory_test PRIVATE strata_sycl_memory)
target_compile_options(memory_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_memory PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(memory_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME runtime_memory_lifetime COMMAND memory_test)
set_tests_properties(runtime_memory_lifetime PROPERTIES TIMEOUT 90)

# Run explicitly on old/new drivers; the old driver's retained mapping fails.
add_executable(memory_import_probe EXCLUDE_FROM_ALL memory_import_probe.cpp)
target_link_libraries(memory_import_probe PRIVATE strata_sycl_runtime ${STRATA_LEVEL_ZERO_LOADER})
target_compile_options(memory_import_probe PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(memory_import_probe PRIVATE -fsycl-targets=spir64_gen)
endif()

add_library(strata_sycl_cuda_frontend STATIC ${STRATA_ROOT}/src/sycl_upstream/cuda_frontend.cpp)
target_include_directories(strata_sycl_cuda_frontend PUBLIC ${STRATA_ROOT}/include/strata/sycl_upstream/cuda)
target_link_libraries(strata_sycl_cuda_frontend PUBLIC strata_sycl_memory)
target_compile_options(strata_sycl_cuda_frontend PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_cuda_frontend PRIVATE -fsycl-targets=spir64_gen)
endif()
# Compile the original host translation units without editing their processing.
set_source_files_properties(${STRATA_ROOT}/src/core/pinned.cu PROPERTIES LANGUAGE CXX)
add_library(strata_sycl_original_host STATIC ${STRATA_ROOT}/src/core/graph.cpp
    ${STRATA_ROOT}/src/core/pinned.cu ${STRATA_ROOT}/src/platform/memory.cpp)
target_link_libraries(strata_sycl_original_host PUBLIC strata_sycl_cuda_frontend)
add_executable(original_pinned_shared_test ${STRATA_ROOT}/src/core/pinned_shared_test.cpp)
target_link_libraries(original_pinned_shared_test PRIVATE strata_sycl_original_host)
add_test(NAME original_pinned_shared_backing COMMAND original_pinned_shared_test)
set_tests_properties(original_pinned_shared_backing PROPERTIES TIMEOUT 60)
add_executable(cuda_frontend_test cuda_frontend.cpp)
target_link_libraries(cuda_frontend_test PRIVATE strata_sycl_original_host strata_sycl_handoff)
target_compile_options(cuda_frontend_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(cuda_frontend_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_host_cuda_frontend COMMAND cuda_frontend_test)
set_tests_properties(original_host_cuda_frontend PROPERTIES TIMEOUT 90)

# Engine-facing original MMQ API: opaque CUDA frontend handles, including
# stream zero. The raw-queue adapter above remains a component-test target.
# Never link both public adapters into one executable.
add_library(strata_sycl_mmq_frontend STATIC ${STRATA_ROOT}/src/sycl_upstream/mmq_adapter.cpp)
target_compile_definitions(strata_sycl_mmq_frontend PRIVATE STRATA_SYCL_MMQ_CUDA_FRONTEND=1)
target_link_libraries(strata_sycl_mmq_frontend PUBLIC strata_sycl_mmq_kernels strata_sycl_cuda_frontend)
target_compile_options(strata_sycl_mmq_frontend PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_mmq_frontend PRIVATE -fsycl-targets=spir64_gen)
    target_compile_definitions(strata_sycl_mmq_frontend PRIVATE
        STRATA_SYCL_MMQ_AOT_ARCH=sycl::ext::oneapi::experimental::architecture::intel_gpu_${STRATA_SYCL_DEVICE_ARCH})
    target_link_options(strata_sycl_mmq_frontend PUBLIC -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()
add_executable(mmq_frontend_test mmq_frontend.cpp)
target_link_libraries(mmq_frontend_test PRIVATE strata_sycl_mmq_frontend strata_sycl_original_host)
target_compile_options(mmq_frontend_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(mmq_frontend_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_mmq_stream_frontend COMMAND mmq_frontend_test)
set_tests_properties(original_mmq_stream_frontend PROPERTIES TIMEOUT 120)
add_test(NAME original_mmq_fatal_errors COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/check_mmq_errors.py $<TARGET_FILE:mmq_frontend_test>)
set_tests_properties(original_mmq_fatal_errors PROPERTIES TIMEOUT 75)

# Original layer/session host code now compiles through the timing frontend.
# The timing test links only the original stage timer; unused unresolved engine
# functions are discarded. This target is not a fully linked inference engine.
add_library(strata_sycl_original_timing_host STATIC ${STRATA_ROOT}/src/core/layer.cpp
    ${STRATA_ROOT}/src/core/session.cpp)
target_compile_definitions(strata_sycl_original_timing_host PRIVATE STRATA_NATIVE_EXPERTS=1)
target_compile_options(strata_sycl_original_timing_host PRIVATE -ffunction-sections -fdata-sections)
target_link_libraries(strata_sycl_original_timing_host PUBLIC strata_sycl_cuda_frontend)
add_executable(runtime_timing_test runtime_timing.cpp)
target_compile_options(runtime_timing_test PRIVATE -fsycl)
target_link_options(runtime_timing_test PRIVATE -Wl,--gc-sections)
target_link_libraries(runtime_timing_test PRIVATE strata_sycl_original_timing_host strata_sycl_original_host)
add_test(NAME original_host_event_timing COMMAND runtime_timing_test)
set_tests_properties(original_host_event_timing PROPERTIES TIMEOUT 60)
add_executable(profiling_tag_probe EXCLUDE_FROM_ALL profiling_tag_probe.cpp)
target_compile_options(profiling_tag_probe PRIVATE -fsycl)
target_link_options(profiling_tag_probe PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(runtime_timing_test PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(profiling_tag_probe PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(profiling_tag_probe PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()
add_executable(graph_copy2d_probe EXCLUDE_FROM_ALL graph_copy2d_probe.cpp)
target_link_libraries(graph_copy2d_probe PRIVATE strata_sycl_runtime ${STRATA_LEVEL_ZERO_LOADER})
target_compile_options(graph_copy2d_probe PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(graph_copy2d_probe PRIVATE -fsycl-targets=spir64_gen)
endif()

# Link and run the original expert cache through native device/memory queries.
add_library(strata_sycl_original_cache_host STATIC ${STRATA_ROOT}/src/core/expert_cache.cpp)
target_link_libraries(strata_sycl_original_cache_host PUBLIC strata_sycl_cuda_frontend)
add_executable(runtime_devices_test runtime_devices.cpp)
target_link_libraries(runtime_devices_test PRIVATE strata_sycl_original_cache_host)
target_compile_options(runtime_devices_test PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(runtime_devices_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_cache_device_memory COMMAND runtime_devices_test)
set_tests_properties(original_cache_device_memory PROPERTIES TIMEOUT 60)
add_test(NAME runtime_empty_device_visibility COMMAND ${CMAKE_COMMAND} -E env
    ONEAPI_DEVICE_SELECTOR=opencl:gpu $<TARGET_FILE:runtime_devices_test> --no-level-zero)
set_tests_properties(runtime_empty_device_visibility PROPERTIES TIMEOUT 20)
add_test(NAME runtime_legacy_sysman_rejection COMMAND ${CMAKE_COMMAND} -E env
    ZES_ENABLE_SYSMAN=1 $<TARGET_FILE:runtime_devices_test> --legacy-sysman)
set_tests_properties(runtime_legacy_sysman_rejection PROPERTIES TIMEOUT 20)

# Read-only native diagnostic: includes the raw physical/free and usable-memory
# values, plus observed allocation-pool retention after sycl::free.
add_executable(sysman_probe EXCLUDE_FROM_ALL sysman_probe.cpp)
target_link_libraries(sysman_probe PRIVATE ${STRATA_LEVEL_ZERO_LOADER})
target_compile_options(sysman_probe PRIVATE -fsycl)
target_link_options(sysman_probe PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(sysman_probe PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(sysman_probe PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()

# Original prefill and peer/remote host objects compile with their real MMQ,
# fused and native declarations. No disabled-feature source stubs are injected.
add_library(strata_sycl_original_peer_host STATIC ${STRATA_ROOT}/src/prefill/prefill.cpp
    ${STRATA_ROOT}/src/core/peer_experts.cpp ${STRATA_ROOT}/src/core/remote_experts.cpp)
target_compile_definitions(strata_sycl_original_peer_host PRIVATE
    STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
target_compile_options(strata_sycl_original_peer_host PRIVATE -ffunction-sections -fdata-sections)
target_link_libraries(strata_sycl_original_peer_host PUBLIC strata_sycl_mmq_frontend strata_sycl_original_cache_host)
add_executable(runtime_peer_test runtime_peer.cpp)
target_compile_options(runtime_peer_test PRIVATE -fsycl)
target_link_options(runtime_peer_test PRIVATE -Wl,--gc-sections)
target_link_libraries(runtime_peer_test PRIVATE strata_sycl_original_peer_host)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(runtime_peer_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_host_peer_api COMMAND runtime_peer_test)
set_tests_properties(original_host_peer_api PROPERTIES TIMEOUT 60)

add_executable(original_prefill_stager_test prefill_stager.cpp)
target_compile_definitions(original_prefill_stager_test PRIVATE
    STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
target_compile_options(original_prefill_stager_test PRIVATE -fsycl -ffunction-sections -fdata-sections)
target_link_options(original_prefill_stager_test PRIVATE -Wl,--gc-sections)
target_link_libraries(original_prefill_stager_test PRIVATE strata_sycl_mmq_frontend)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(original_prefill_stager_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_prefill_stager_ring COMMAND original_prefill_stager_test)
set_tests_properties(original_prefill_stager_ring PROPERTIES TIMEOUT 60)

# The real ExpertSource key function/typeinfo and layout are retained when the
# stager test supplies a transient source through the original virtual interface.
add_library(strata_sycl_original_source_host STATIC ${STRATA_ROOT}/src/core/expert_source.cpp
    ${STRATA_ROOT}/src/kernels/cpu/expert_layout.cpp)
target_compile_definitions(strata_sycl_original_source_host PRIVATE STRATA_NATIVE_EXPERTS=1)
target_compile_options(strata_sycl_original_source_host PRIVATE -ffunction-sections -fdata-sections)
target_link_libraries(strata_sycl_original_source_host PUBLIC strata_sycl_mmq_frontend)
target_link_libraries(original_prefill_stager_test PRIVATE strata_sycl_original_source_host)

add_test(NAME original_prefill_stager_model_size COMMAND original_prefill_stager_test --model-sized)
set_tests_properties(original_prefill_stager_model_size PROPERTIES TIMEOUT 60)

add_library(strata_sycl_original_graph_host STATIC ${STRATA_ROOT}/src/core/mtp.cpp
    ${STRATA_ROOT}/src/core/verify.cpp)
target_compile_definitions(strata_sycl_original_graph_host PRIVATE
    STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
target_compile_options(strata_sycl_original_graph_host PRIVATE -ffunction-sections -fdata-sections)
target_link_libraries(strata_sycl_original_graph_host PUBLIC strata_sycl_mmq_frontend)

add_executable(runtime_graph_api_test runtime_graph_api.cpp)
target_compile_definitions(runtime_graph_api_test PRIVATE
    STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
target_compile_options(runtime_graph_api_test PRIVATE -fsycl -ffunction-sections -fdata-sections)
target_link_options(runtime_graph_api_test PRIVATE -Wl,--gc-sections)
target_link_libraries(runtime_graph_api_test PRIVATE strata_sycl_original_graph_host)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(runtime_graph_api_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_graph_upload_inspection COMMAND runtime_graph_api_test)
set_tests_properties(original_graph_upload_inspection PROPERTIES TIMEOUT 60)

add_executable(graph_prepare_probe EXCLUDE_FROM_ALL graph_prepare_probe.cpp)
target_compile_options(graph_prepare_probe PRIVATE -fsycl)
target_link_options(graph_prepare_probe PRIVATE -fsycl)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(graph_prepare_probe PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(graph_prepare_probe PRIVATE -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_SYCL_DEVICE_ARCH}\"")
endif()

# Native BF16/FP16 BLAS and original quantized-weight GEMM orchestration.
set(STRATA_MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "oneMKL installation")
find_path(STRATA_MKL_INCLUDE oneapi/mkl.hpp HINTS ${STRATA_MKL_ROOT}/include REQUIRED)
foreach(component IN ITEMS sycl_blas intel_lp64 sequential core)
    find_library(STRATA_MKL_${component} mkl_${component} HINTS ${STRATA_MKL_ROOT}/lib REQUIRED)
endforeach()
execute_process(COMMAND ${Python3_EXECUTABLE} ${STRATA_ROOT}/tools/sycl/extract_dequant.py
    --ggml ${STRATA_GGML_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
add_library(strata_sycl_blas_frontend STATIC ${STRATA_ROOT}/src/sycl_upstream/blas_frontend.cpp)
target_include_directories(strata_sycl_blas_frontend PRIVATE ${STRATA_MKL_INCLUDE})
target_link_libraries(strata_sycl_blas_frontend PUBLIC strata_sycl_cuda_frontend
    ${STRATA_MKL_sycl_blas} ${STRATA_MKL_intel_lp64} ${STRATA_MKL_sequential} ${STRATA_MKL_core})
target_compile_options(strata_sycl_blas_frontend PRIVATE -fsycl -fno-fast-math -ffp-contract=off)
add_library(strata_sycl_dequant STATIC ${STRATA_ROOT}/src/sycl_upstream/dequant.cpp)
target_include_directories(strata_sycl_dequant PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
target_link_libraries(strata_sycl_dequant PUBLIC strata_sycl_cuda_frontend)
target_compile_options(strata_sycl_dequant PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
set_source_files_properties(${STRATA_ROOT}/src/prefill/gemm.cu PROPERTIES LANGUAGE CXX)
add_library(strata_sycl_original_gemm STATIC ${STRATA_ROOT}/src/prefill/gemm.cu)
target_link_libraries(strata_sycl_original_gemm PUBLIC strata_sycl_blas_frontend strata_sycl_dequant)
add_executable(original_gemm_test gemm.cpp)
target_include_directories(original_gemm_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
target_link_libraries(original_gemm_test PRIVATE strata_sycl_original_gemm ggml-base)
target_compile_options(original_gemm_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off -fno-strict-aliasing)
add_test(NAME original_native_gemm COMMAND original_gemm_test)
set_tests_properties(original_native_gemm PROPERTIES TIMEOUT 120)
add_executable(original_dequant_model_test EXCLUDE_FROM_ALL
    ${STRATA_ROOT}/src/kernels/dequant_bf16_test.cpp)
target_link_libraries(original_dequant_model_test PRIVATE strata_sycl_dequant ggml-base)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_blas_frontend PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(strata_sycl_dequant PRIVATE -fsycl-targets=spir64_gen)
    target_compile_options(original_gemm_test PRIVATE -fsycl-targets=spir64_gen)
endif()

# Regenerate whole native sources whenever either mechanical lowerer changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${STRATA_ROOT}/tools/sycl/lower_cuda_products.py
    ${STRATA_ROOT}/tools/sycl/lower_cuda_primitives.py)

# Keep original native decode products, IQ grouped experts and BF16 variants.
# Generated sources lower launch/intrinsic syntax; original files remain pinned.
execute_process(COMMAND ${Python3_EXECUTABLE} ${STRATA_ROOT}/tools/sycl/lower_cuda_products.py
    --ggml ${STRATA_GGML_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
add_library(strata_sycl_original_products STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/original_native_mmvq.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/original_iq_kernels.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/original_native_bf16.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/original_bf16_gemv.cpp)
target_include_directories(strata_sycl_original_products PRIVATE
    ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
target_link_libraries(strata_sycl_original_products PUBLIC strata_sycl_cuda_frontend)
# Q8_1 uses explicit fdiv_rn in the mechanical lowering: the compiler precision
# flag alone still changed a half-tie on this device. Preserve native FMA here.
# This is not proof of bit identity with CUDA's --use_fast_math device baseline.
target_compile_options(strata_sycl_original_products PRIVATE -fsycl -fno-fast-math -ffp-contract=off
    -fno-strict-aliasing)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/original_native_mmvq.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/original_native_bf16.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast")
target_link_libraries(strata_sycl_dequant PUBLIC strata_sycl_original_products)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_original_products PRIVATE -fsycl-targets=spir64_gen)
endif()

add_executable(original_products_test products.cpp)
target_include_directories(original_products_test PRIVATE ${STRATA_GGML_DIR}/ggml/include)
target_link_libraries(original_products_test PRIVATE strata_sycl_original_products ggml-base)
target_compile_options(original_products_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(original_products_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_native_products COMMAND original_products_test)
set_tests_properties(original_native_products PROPERTIES TIMEOUT 300)

# Native state updates and attention primitives retain whole original files.
execute_process(COMMAND ${Python3_EXECUTABLE} ${STRATA_ROOT}/tools/sycl/lower_cuda_primitives.py
    --ggml ${STRATA_GGML_DIR} --output ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY)
set(_strata_native_primitives native_gr_norm native_gr_postops native_moe native_gdn
    native_gdn_preprocess native_qsa native_router native_rope native_ple_postops
    native_flash_attn native_qsa_indexer native_qsa_score)
set(_strata_native_primitive_sources)
foreach(_name IN LISTS _strata_native_primitives)
    list(APPEND _strata_native_primitive_sources ${CMAKE_CURRENT_BINARY_DIR}/original_${_name}.cpp)
endforeach()
add_library(strata_sycl_original_primitives STATIC ${_strata_native_primitive_sources})
target_include_directories(strata_sycl_original_primitives BEFORE PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/native_include ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
target_compile_definitions(strata_sycl_original_primitives PRIVATE STRATA_USE_SYCL=1)
target_compile_options(strata_sycl_original_primitives PRIVATE -fsycl -fno-fast-math
    -ffp-contract=fast -fno-strict-aliasing)
target_link_libraries(strata_sycl_original_primitives PUBLIC strata_sycl_original_products)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(strata_sycl_original_primitives PRIVATE -fsycl-targets=spir64_gen)
endif()
add_executable(original_primitives_test native_primitives.cpp)
target_include_directories(original_primitives_test PRIVATE ${STRATA_GGML_DIR}/ggml/include)
target_link_libraries(original_primitives_test PRIVATE strata_sycl_original_primitives ggml-base)
target_compile_options(original_primitives_test PRIVATE -fsycl -fno-fast-math -ffp-contract=off)
if(STRATA_SYCL_DEVICE_ARCH)
    target_compile_options(original_primitives_test PRIVATE -fsycl-targets=spir64_gen)
endif()
add_test(NAME original_native_primitives COMMAND original_primitives_test)
set_tests_properties(original_native_primitives PROPERTIES TIMEOUT 300)
