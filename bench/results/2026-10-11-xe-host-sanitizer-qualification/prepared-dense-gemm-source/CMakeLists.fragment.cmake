# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Insert into the existing STRATA_ENABLE_XE / STRATA_BUILD_TESTS block after strata_prefill is defined.
add_executable(prefill_dense_gemm_parity tools/test_prefill_dense_gemm.cpp)
target_link_libraries(prefill_dense_gemm_parity PRIVATE strata_prefill)
target_compile_options(prefill_dense_gemm_parity PRIVATE -fno-fast-math -ffp-contract=off)
if(STRATA_LICENSE STREQUAL "contrib")
  set(_strata_parity_license "contrib-llvm")
else()
  set(_strata_parity_license "${STRATA_LICENSE}")
endif()
target_compile_definitions(prefill_dense_gemm_parity PRIVATE
  STRATA_PARITY_LICENSE="${_strata_parity_license}")
if(TARGET onemath)
  target_compile_definitions(prefill_dense_gemm_parity PRIVATE STRATA_PARITY_ONEMATH=1)
else()
  target_compile_definitions(prefill_dense_gemm_parity PRIVATE STRATA_PARITY_ONEMATH=0)
endif()
add_test(NAME prefill_dense_gemm_parity COMMAND prefill_dense_gemm_parity)
set_tests_properties(prefill_dense_gemm_parity PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
