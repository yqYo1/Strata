// Authored for Strata; repository LICENSE (MIT) applies.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace strata::kernels {
bool qsa_bf16_prefix_kstage_batch(const uint16_t *,const QsaAttnPools &,
    const int32_t *,const int32_t *,int64_t,const QsaShapes &,uint16_t *,int64_t,
    const QsaPromptBf16Bounds &,int64_t,const QsaBf16PackedScratch &,void *,int);
bool qsa_kstage_actual_entry(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  static const int span = [] {
    const char *value=std::getenv("STRATA_TEST_QSA_KSTAGE");
    if (!value) throw std::invalid_argument("immutable test K schedule required");
    int result=0;
    if (!std::strcmp(value,"baseline")) result=1;
    if (!std::strcmp(value,"span4")) result=4;
    if (!std::strcmp(value,"row")) result=256;
    if (!result) throw std::invalid_argument("test K schedule must be baseline, span4 or row");
    std::cerr << "strata QSA K staging fixture: layout=row_major span=" << result << std::endl;
    return result;
  }();
  return qsa_bf16_prefix_kstage_batch(q,pools,ids,steps,cap,shape,out,queries,
      bounds,live_cells,scratch,stream,span);
}
} // namespace strata::kernels
