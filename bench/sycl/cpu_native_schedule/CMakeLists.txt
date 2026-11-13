cmake_minimum_required(VERSION 3.22)
project(strata_cpu_native_schedule LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT IS_DIRECTORY "${STRATA_GGML_DIR}/ggml")
  message(FATAL_ERROR "Pass the pinned GGML checkout as STRATA_GGML_DIR")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${STRATA_GGML_DIR}/ggml" ggml EXCLUDE_FROM_ALL)
set(CPU "${STRATA_ROOT}/src/kernels/cpu")
# Fresh source compilation: do not borrow an unrecorded archive from an engine.
add_library(compact_reference_cpu STATIC
  ${CPU}/expert.cpp ${CPU}/pool.cpp ${CPU}/expert_layout.cpp ${CPU}/q2_avx2.cpp
  ${CPU}/native_expert.cpp ${CPU}/iq_avx512.cpp ${CPU}/iq_avx2.cpp
  ${CPU}/iq_single_avx2.cpp ${CPU}/iq2_pair_avx2.cpp ${CPU}/kq_avx2.cpp ${CPU}/compact_gu.cpp)
target_include_directories(compact_reference_cpu PUBLIC "${STRATA_ROOT}/include"
  PRIVATE "${STRATA_ROOT}/third_party/ggml")
target_compile_definitions(compact_reference_cpu PUBLIC
  STRATA_NATIVE_EXPERTS=1 STRATA_CPU_SERVICE_PROFILE=1)
target_compile_options(compact_reference_cpu PRIVATE -march=znver3 -mtune=znver3)
set_source_files_properties(${CPU}/expert.cpp ${CPU}/iq_avx512.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${CPU}/q2_avx2.cpp ${CPU}/iq_avx2.cpp
  ${CPU}/iq_single_avx2.cpp ${CPU}/iq2_pair_avx2.cpp ${CPU}/kq_avx2.cpp ${CPU}/compact_gu.cpp
  PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
find_package(Threads REQUIRED)
target_link_libraries(compact_reference_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
set(STRATA_CPU_NATIVE_FIXTURE "${CMAKE_CURRENT_LIST_DIR}/pool.cpp" CACHE FILEPATH
  "Committed CPU fixture; production CPU sources are always compiled freshly")
add_executable(cpu_native_schedule "${STRATA_CPU_NATIVE_FIXTURE}")
target_include_directories(cpu_native_schedule PRIVATE "${STRATA_GGML_DIR}/ggml/src")
target_compile_options(cpu_native_schedule PRIVATE -march=znver3 -mtune=znver3)
target_link_libraries(cpu_native_schedule PRIVATE compact_reference_cpu)
