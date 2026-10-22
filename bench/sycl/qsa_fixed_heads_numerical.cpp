// Main-authored component comparison; repository LICENSE applies.
// The shared oracle includes the checked variable-page diagnostic correction.
#include "qsa-fixed_heads-variants.hpp"
#include <string>
int qsa_page_fixture_page = 16;

namespace {
auto selected_attention = qsa_fixed_heads_functions()[0];
}
bool qualified_fixed_heads_attention(const uint16_t* query,
    const strata::kernels::QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
    int64_t cap, const strata::kernels::QsaShapes& shape, uint16_t* output, int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds& bounds, int64_t live_cells,
    const strata::kernels::QsaBf16PackedScratch& scratch, void* stream) {
  return selected_attention(query, pools, ids, steps, cap, shape, output,
      queries, bounds, live_cells, scratch, stream);
}
#define main qsa_fixed_heads_reference_main
#define qsa_bf16_prefix_batch qualified_fixed_heads_attention
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
  const int index = variant == "default" ? 0 : variant == "control" ? 1 : variant == "fixed_heads" ? 2 : -1;
  if (!qsa_page_fixture_page || index < 0) return 2;
  selected_attention = qsa_fixed_heads_functions()[size_t(index)];
  std::cout << "fixed_heads_control variant=" << variant << " arithmetic=unchanged\n";
  const int status = qsa_fixed_heads_reference_main(2, argv);
  const auto routes = strata::kernels::fixed_heads::route_counts();
  if (!status) {
    // Includes the zero-query refusal's wrapper call, which must submit no GPU work.
    // There is no page-specific fallback in this candidate.
    if (index == 2) require(routes[0] > 0 && routes[1] == 0, "fixed-head numerical route");
    else require(routes[0] == 0 && routes[1] == 0, "control numerical route");
    std::cout << "fixed_heads_routes page=" << qsa_page_fixture_page
              << " fast=" << routes[0] << " generic=" << routes[1] << " status=PASS\n";
  }
  return status;
}
