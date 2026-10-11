# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Main may include this AFTER the pinned production targets exist, in an isolated
# qualification checkout only. No production CMake file was edited by the agent.
if(NOT TARGET strata_kernels OR NOT TARGET strata_core OR NOT TARGET strata_sycl)
  message(FATAL_ERROR "Component requires the actual production SYCL target closure")
endif()
add_executable(native_dequant_group_component
               "${CMAKE_CURRENT_LIST_DIR}/native_dequant_group_component.cpp")
target_link_libraries(native_dequant_group_component PRIVATE strata_kernels strata_core strata_warnings)
target_include_directories(native_dequant_group_component PRIVATE "${PROJECT_SOURCE_DIR}/third_party/main/ggml")
target_compile_options(native_dequant_group_component PRIVATE -fno-fast-math -ffp-contract=off)
