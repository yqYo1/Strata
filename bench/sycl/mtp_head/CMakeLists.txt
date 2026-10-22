cmake_minimum_required(VERSION 3.24)
project(strata_mtp_head_calibration LANGUAGES CXX)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# Configure the unchanged production resolver with its recorded options. Build
# only this consumer; never overwrite the qualified engine's output directory.
add_subdirectory("${STRATA_ROOT}" production EXCLUDE_FROM_ALL)
add_executable(mtp_head main.cpp)
target_link_libraries(mtp_head PRIVATE strata_kernels)
target_compile_features(mtp_head PRIVATE cxx_std_20)
target_compile_options(mtp_head PRIVATE -fno-fast-math -ffp-contract=off)
