// Authored for Strata; repository LICENSE (MIT) applies.
// Test dispatch only: call the real current row-major product APIs.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace strata::kernels {
bool qsa_prefix_actual_entry(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  static const bool subgroup = [] {
    const char *value = std::getenv("STRATA_TEST_QSA_SOFTMAX");
    if (!value || (std::strcmp(value,"serial") && std::strcmp(value,"subgroup")))
      throw std::invalid_argument("test softmax must be serial or subgroup before exec");
    const bool result = std::strcmp(value,"subgroup") == 0;
    std::cerr << "strata QSA softmax fixture: layout=row_major softmax="
              << (result ? "subgroup" : "serial") << std::endl;
    return result;
  }();
  if (subgroup) return qsa_bf16_prefix_subgroup_batch(q,pools,ids,steps,cap,shape,
      out,queries,bounds,live_cells,scratch,stream);
  return qsa_bf16_prefix_optimized_batch(q,pools,ids,steps,cap,shape,
      out,queries,bounds,live_cells,scratch,stream);
}
} // namespace strata::kernels
