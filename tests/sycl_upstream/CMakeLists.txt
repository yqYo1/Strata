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
