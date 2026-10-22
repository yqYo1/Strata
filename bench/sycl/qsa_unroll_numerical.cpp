// Main-authored static-indexing comparison; repository LICENSE (MIT) applies.
// Reuse the qualified host tensor oracle, bounds, alias and physical-capacity
// cases. Only this process's explicit function selector changes between arms.
#include "qsa-unroll-variants.hpp"
#include <string>

namespace {
auto qsa_unroll_selected=qsa_unroll_functions()[0];
}
bool qualified_unroll_attention(const uint16_t* query,
    const strata::kernels::QsaAttnPools& pools,const int32_t* ids,const int32_t* steps,
    int64_t cap,const strata::kernels::QsaShapes& shape,uint16_t* output,int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds& bounds,int64_t live_cells,
    const strata::kernels::QsaBf16PackedScratch& scratch,void* stream) {
  return qsa_unroll_selected(query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream);
}
#define main qsa_unroll_reference_fixture_main
#define qsa_bf16_prefix_batch qualified_unroll_attention
#define STRATA_QSA_COMPARE_PREFIX33 1
#define STRATA_QSA_PACKED_GUARD 17
#include "../../tests/sycl/qsa_bf16_prefix.cpp"
#undef STRATA_QSA_PACKED_GUARD
#undef STRATA_QSA_COMPARE_PREFIX33
#undef qsa_bf16_prefix_batch
#undef main

int main(int argc,char** argv) {
  if (argc!=3) return 2;
  const std::string variant=argv[2];
  const int index=variant=="default" ? 0 : variant=="control" ? 1 : variant=="unroll" ? 2 : -1;
  if (index<0) return 2;
  qsa_unroll_selected=qsa_unroll_functions()[size_t(index)];
  std::cout<<"unroll_control variant="<<variant<<" arithmetic=unchanged\n";
  return qsa_unroll_reference_fixture_main(2,argv);
}
