cmake_minimum_required(VERSION 3.22)
project(strata_cpu_decode_mix LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Reuse the maintained production-source library and its pinned GGML resolver.
add_subdirectory(../cpu_roofline reference EXCLUDE_FROM_ALL)
add_executable(cpu_decode_mix main.cpp)
target_compile_options(cpu_decode_mix PRIVATE -march=znver3 -mtune=znver3)
target_link_libraries(cpu_decode_mix PRIVATE expert_cpu)
