cmake_minimum_required(VERSION 3.20)
project(packed_iq_xmx_probe LANGUAGES CXX)
set(MKL_LINK dynamic)
set(MKL_THREADING sequential)
find_package(MKL CONFIG REQUIRED)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(packed_iq_xmx_probe probe.cpp)
target_compile_features(packed_iq_xmx_probe PRIVATE cxx_std_20)
target_include_directories(packed_iq_xmx_probe PRIVATE
  "${STRATA_ROOT}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../parallel-iq-dequant-v1/proposed/include")
target_compile_options(packed_iq_xmx_probe PRIVATE -fsycl -fsycl-id-queries-range=size_t -fno-fast-math -ffp-contract=off)
target_link_options(packed_iq_xmx_probe PRIVATE -fsycl -fsycl-id-queries-range=size_t)
target_link_libraries(packed_iq_xmx_probe PRIVATE MKL::MKL_SYCL::BLAS)
