// Main-authored comparison harness; repository LICENSE (MIT) applies.
// Reuse the maintained independent weighted-cell oracle, all boundary/NaN/tie
// cases and immutable-input/canary/drain checks; do not duplicate that fixture.
// One arm per process, selected before runtime initialization. No env writes.
#include "strata/kernels/qsa_radix_batched.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Selector = decltype(&strata::kernels::qsa_block_topk_prefill);
Selector qsa_fixture_select = nullptr;
}
namespace strata::kernels {
void qsa_block_topk_selected_fixture(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  if (!qsa_fixture_select) throw std::logic_error("fixture arm not selected");
  qsa_fixture_select(scores, steps, nq, blocks, cap, shape, ids, stream, active);
}
}

// The public API header is already included above. Only the dedicated-prefill
// fixture call is redirected; public original/ref checks stay unmodified.
#define qsa_block_topk_prefill qsa_block_topk_selected_fixture
#define main qsa_original_radix_fixture_main
#include "qsa_select_radix.cpp"
#undef main
#undef qsa_block_topk_prefill

int main(int argc, char **argv) {
  if (argc < 3 || std::string(argv[1]) != "--arm") {
    std::cerr << "usage: fixture --arm original|control|batched [original fixture options]\n";
    return 2;
  }
  const std::string arm = argv[2];
  if (arm == "original") qsa_fixture_select = strata::kernels::qsa_block_topk_prefill;
  else if (arm == "control") qsa_fixture_select = strata::kernels::qsa_block_topk_sg16_control;
  else if (arm == "batched") qsa_fixture_select = strata::kernels::qsa_block_topk_batched_prefill;
  else { std::cerr << "unknown fixture arm\n"; return 2; }
  const char *radix = std::getenv("STRATA_SYCL_SELECT_RADIX4");
  const char *cyclic = std::getenv("STRATA_SYCL_SELECT_CYCLIC");
  if (!radix || std::string(radix) != "1" ||
      (cyclic && std::string(cyclic) != "0")) {
    std::cerr << "fixture requires pre-exec radix4=1 and cyclic unset/0\n";
    return 2;
  }
  std::cout << "qsa_histogram_fixture_arm=" << arm << '\n';
  // argv[2] becomes the inherited fixture's unused program name.
  return qsa_original_radix_fixture_main(argc - 2, argv + 2);
}
