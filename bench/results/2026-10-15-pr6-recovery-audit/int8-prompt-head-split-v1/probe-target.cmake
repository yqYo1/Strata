# Main-only proposal: insert with the other manual SYCL component targets.
# No CTest registration and no default build dependency.
add_executable(sycl_int8_prompt_head_split_probe EXCLUDE_FROM_ALL tools/sycl/int8_prompt_head_split_probe.cpp)
target_link_libraries(sycl_int8_prompt_head_split_probe PRIVATE strata_kernels)
target_compile_options(sycl_int8_prompt_head_split_probe PRIVATE -fno-fast-math -ffp-contract=off)
