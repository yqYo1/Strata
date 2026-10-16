#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
namespace strata_gather::kernels::cpu {
void iq256_gu_rows(int,const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int);
}
uint64_t hash(const std::vector<float>& v) {
 uint64_t h=14695981039346656037ull;
 for (size_t i=0;i<v.size()*4;++i) { h^=((const uint8_t*)v.data())[i];h*=1099511628211ull; }
 return h;
}
int main(int argc,char** argv) {
 try {
  if(argc!=2)throw std::runtime_error("usage: iq_gather_bench SHARD1");
  auto model=strata::GgufModel::open(argv[1]);std::set<int> seen;
  constexpr int E=8, repeats=5;
  std::cout<<"type,nt,gather,trial,microseconds,fnv1a64,unequal\n";
  for(int layer=0;layer<48;++layer) {
   size_t gs,us;auto pre="blk."+std::to_string(layer)+".";
   auto*g=model.find(pre+"ffn_gate_exps.weight",&gs);auto*u=model.find(pre+"ffn_up_exps.weight",&us);
   if(!g||!u||(g->type!=18&&g->type!=21)||!seen.insert(g->type).second)continue;
   if(g->shape.size()!=3||g->shape!=u->shape||g->type!=u->type||g->shape[2]<64||!model.in_bounds(*g,gs)||!model.in_bounds(*u,us))throw std::runtime_error("unexpected geometry");
   int n=g->shape[0], rows=g->shape[1], nb=n/256;size_t row=nb*(g->type==18?98:110),up=rows*row;
   std::vector<uint8_t> blobs(E*2*up+4096);
   for(int e=0;e<E;++e) {
    memcpy(blobs.data()+e*2*up,model.shard(gs).tensor_data(*g)+e*7*up,up);
    memcpy(blobs.data()+e*2*up+up,model.shard(us).tensor_data(*u)+e*7*up,up);
   }
   for(int nt:{1,2,3,4,8}) {
    std::vector<block_q8_K> acts(nt*nb);std::vector<const void*> ap(nt);
    uint32_t state=12345;
    for(int t=0;t<nt;++t) { ap[t]=acts.data()+t*nb;
     for(int b=0;b<nb;++b) { auto&x=acts[t*nb+b];x.d=.01f;
      for(int k=0;k<256;++k) { state=state*1664525u+1013904223u;x.qs[k]=(int)(state%255)-127; }
      for(int k=0;k<16;++k) {x.bsums[k]=0;for(int j=0;j<16;++j)x.bsums[k]+=x.qs[k*16+j];}
     }
    }
    std::vector<float> ref(E*nt*rows),got(ref.size());std::vector<float*> fp(nt);
    auto run=[&](bool gather,std::vector<float>& out) {
     auto fn=gather?strata_gather::kernels::cpu::iq256_gu_rows:strata::kernels::cpu::iq256_gu_rows;
     for(int e=0;e<E;++e) {for(int t=0;t<nt;++t)fp[t]=out.data()+(e*nt+t)*rows;fn(g->type,blobs.data()+e*2*up,row,up,n,ap.data(),nt,fp.data(),0,rows);}
    };
    run(false,ref);run(true,got);size_t unequal=0;
    for(size_t i=0;i<ref.size();++i) {if(!std::isfinite(ref[i])||!std::isfinite(got[i]))throw std::runtime_error("nonfinite result");unequal+=memcmp(&ref[i],&got[i],4)!=0;}
    if(unequal)throw std::runtime_error("gather changed output bits");
    for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k) {
     bool gather=(trial+k)%2;auto&out=gather?got:ref;run(gather,out);
     auto start=std::chrono::steady_clock::now();for(int it=0;it<repeats;++it)run(gather,out);
     auto stop=std::chrono::steady_clock::now();double us=std::chrono::duration<double,std::micro>(stop-start).count()/repeats;
     std::cout<<g->type<<','<<nt<<','<<gather<<','<<trial<<','<<us<<','<<hash(out)<<','<<unequal<<'\n'<<std::flush;
    }
   }
  }
  if(seen.size()!=2)throw std::runtime_error("missing IQ3 formats");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
