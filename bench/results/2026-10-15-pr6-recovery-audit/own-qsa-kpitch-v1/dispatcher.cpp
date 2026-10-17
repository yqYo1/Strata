// Authored for Strata; repository LICENSE (MIT) applies.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace strata::kernels {
bool qsa_bf16_prefix_pitch_batch(const uint16_t *,const QsaAttnPools &,
    const int32_t *,const int32_t *,int64_t,const QsaShapes &,uint16_t *,int64_t,
    const QsaPromptBf16Bounds &,int64_t,const QsaBf16PackedScratch &,void *,int);
bool qsa_pitch_actual_entry(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  static const int variant = [] {
    const char *value=std::getenv("STRATA_TEST_QSA_PITCH");
    if (!value) throw std::invalid_argument("immutable test pitch variant required");
    int result=-1;
    if (!std::strcmp(value,"baseline")) result=0;
    if (!std::strcmp(value,"k")) result=1;
    if (!std::strcmp(value,"v")) result=2;
    if (!std::strcmp(value,"both")) result=3;
    if (!std::strcmp(value,"wide")) result=4;
    if (result<0) throw std::invalid_argument("test pitch must be baseline,k,v,both,wide");
    std::cerr << "strata QSA pitch fixture: variant=" << result << std::endl;
    return result;
  }();
  return qsa_bf16_prefix_pitch_batch(q,pools,ids,steps,cap,shape,out,queries,
      bounds,live_cells,scratch,stream,variant);
}
} // namespace strata::kernels
