// Main-authored page-address comparison; repository LICENSE (MIT) applies.
// Reuse the qualified host tensor oracle, bounds, alias and physical-capacity
// cases. Only this process's explicit function selector changes between arms.
#include "qsa-page4-variants.hpp"
#include <string>
int qsa_page_fixture_page=16;

namespace {
auto qsa_page4_selected=qsa_page4_functions()[0];
}
bool qualified_page4_attention(const uint16_t* query,
    const strata::kernels::QsaAttnPools& pools,const int32_t* ids,const int32_t* steps,
    int64_t cap,const strata::kernels::QsaShapes& shape,uint16_t* output,int64_t queries,
    const strata::kernels::QsaPromptBf16Bounds& bounds,int64_t live_cells,
    const strata::kernels::QsaBf16PackedScratch& scratch,void* stream) {
  return qsa_page4_selected(query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch,stream);
}
#define main qsa_page4_reference_fixture_main
#define qsa_bf16_prefix_batch qualified_page4_attention
#define STRATA_QSA_COMPARE_PREFIX33 1
#define STRATA_QSA_PACKED_GUARD 17
#include "numerical-reference.inc"
#undef STRATA_QSA_PACKED_GUARD
#undef STRATA_QSA_COMPARE_PREFIX33
#undef qsa_bf16_prefix_batch
#undef main

int main(int argc,char** argv) {
  if (argc!=4) return 2;
  const std::string page=argv[3];
  qsa_page_fixture_page=page=="1" ? 1 : page=="4" ? 4 : page=="7" ? 7 : page=="16" ? 16 : 0;
  if (!qsa_page_fixture_page) return 2;
  const std::string variant=argv[2];
  const int index=variant=="default" ? 0 : variant=="control" ? 1 : variant=="page4" ? 2 : -1;
  if (index<0) return 2;
  qsa_page4_selected=qsa_page4_functions()[size_t(index)];
  std::cout<<"page4_control variant="<<variant<<" arithmetic=unchanged\n";
  const int status=qsa_page4_reference_fixture_main(2,argv);
  const auto routes=strata::kernels::page4::route_counts();
  if (!status) {
    if (index==2) {
      if (qsa_page_fixture_page==4) require(routes[0]>0 && routes[1]>0,"Page4 numerical fast andzeroqueryfallback routes");
      else require(routes[0]==0 && routes[1]>0,"generic numerical fallback route");
    } else require(routes[0]==0 && routes[1]==0,"control numerical route");
    std::cout<<"page4_routes page="<<qsa_page_fixture_page<<" fast="<<routes[0]<<" generic="<<routes[1]<<" status=PASS\n";
  }
  return status;
}
