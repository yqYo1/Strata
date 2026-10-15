# Main-only proposal, after BOTH frozen Direct0/Direct1 full clean runs close.
# Manual target, no existing test invocation and no CTest registration.
add_executable(sycl_int8_prompt_head_split_probe_v2 EXCLUDE_FROM_ALL tools/sycl/int8_prompt_head_split_probe.cpp)
target_link_libraries(sycl_int8_prompt_head_split_probe_v2 PRIVATE strata_kernels)
target_compile_options(sycl_int8_prompt_head_split_probe_v2 PRIVATE -fno-fast-math -ffp-contract=off)
