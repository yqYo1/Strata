// Main-authored component comparison; repository LICENSE applies.
// The shared oracle includes the checked variable-page diagnostic correction.
#include "qsa-colmajor-variants.hpp"
#include <string>
int qsa_page_fixture_page = 16;

namespace {
auto selected_attention = qsa_colmajor_functions()[0];
}
bool qualified_colmajor_attention(const uint16_t* query,
    const strata::kernels::QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
    int64_t cap, const strata::kernels::QsaShapes& shape, uint16_t* output, int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds& bounds, int64_t live_cells,
    const strata::kernels::QsaBf16PackedScratch& scratch, void* stream) {
  return selected_attention(query, pools, ids, steps, cap, shape, output,
      queries, bounds, live_cells, scratch, stream);
}
#define main qsa_colmajor_reference_main
#define qsa_bf16_prefix_batch qualified_colmajor_attention
#define STRATA_QSA_COMPARE_PREFIX33 1
#define STRATA_QSA_PACKED_GUARD 17
#include "numerical-reference.inc"
#undef STRATA_QSA_PACKED_GUARD
#undef STRATA_QSA_COMPARE_PREFIX33
#undef qsa_bf16_prefix_batch
#undef main

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const std::string page = argv[3], variant = argv[2];
  qsa_page_fixture_page = page == "1" ? 1 : page == "4" ? 4 : page == "7" ? 7 : page == "16" ? 16 : 0;
  const int index = variant == "default" ? 0 : variant == "control" ? 1 : variant == "colmajor" ? 2 : -1;
  if (!qsa_page_fixture_page || index < 0) return 2;
  selected_attention = qsa_colmajor_functions()[size_t(index)];
  std::cout << "colmajor_control variant=" << variant << " arithmetic=unchanged\n";
  const int status = qsa_colmajor_reference_main(2, argv);
  const auto routes = strata::kernels::colmajor::route_counts();
  if (!status) {
    // Includes the zero-query no-op's wrapper call, which must submit no GPU work.
    // There is no page-specific fallback in this candidate.
    if (index == 2) require(routes[0] > 0 && routes[1] == 0, "column-major numerical route");
    else require(routes[0] == 0 && routes[1] == 0, "control numerical route");
    std::cout << "colmajor_routes page=" << qsa_page_fixture_page
              << " fast=" << routes[0] << " generic=" << routes[1] << " status=PASS\n";
    std::cout << "colmajor_scope real_output_oracle=selected_arm padded_state=source_mapping_only\n";
  }
  return status;
}
