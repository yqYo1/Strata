#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#include "strata/kernels/cpu/iq_avx2.hpp"
uint64_t hash(const std::vector<float>&v){uint64_t h=14695981039346656037ull;for(size_t i=0;i<v.size()*4;++i){h^=((const uint8_t*)v.data())[i];h*=1099511628211ull;}return h;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: single-bench SHARD1 EXPERTS");
 ggml_cpu_init();const int E=std::stoi(argv[2]);if(E<1||E>64)throw std::runtime_error("bad expert count");
 auto model=strata::GgufModel::open(argv[1]);std::set<int> seen;
 std::cout<<"type,experts,variant,candidate,trial,microseconds,fnv1a64,unequal\n";
 for(int layer=0;layer<48;++layer){
  size_t gs,us;auto pre="blk."+std::to_string(layer)+".";
  auto*g=model.find(pre+"ffn_down_exps.weight",&gs);
  if(!g||g->type!=20||!seen.insert(g->type).second)continue;
  if(g->shape.size()!=3||g->shape[2]<512||!model.in_bounds(*g,gs))throw std::runtime_error("unexpected geometry");
  const int n=g->shape[0],rows=g->shape[1],nb=n/32;const size_t row=nb*sizeof(block_iq4_nl),up=rows*row;
  std::vector<uint8_t> blobs(E*up+4096);
  for(int e=0;e<E;++e)memcpy(blobs.data()+e*up,model.shard(gs).tensor_data(*g)+e*7*up,up);
  std::vector<block_q8_0> acts(nb);uint32_t state=12345;
  for(auto&x:acts){x.d=0x211f;for(int k=0;k<32;++k){state=state*1664525u+1013904223u;x.qs[k]=(int)(state%255)-127;}}
  auto ref_fn=ggml_get_type_traits_cpu((ggml_type)g->type)->vec_dot;
  const int variants[]={2};
  for(int vi=0;vi<1;++vi){
   std::vector<float> ref(E*rows),got(ref.size());
   auto run=[&](bool candidate,std::vector<float>&out){for(int e=0;e<E;++e){const auto*p=blobs.data()+e*up;auto*d=out.data()+e*rows;if(candidate){if(!strata::kernels::cpu::iq4nl256_single_down_rows(p,row,n,acts.data(),d,0,rows))throw std::runtime_error("candidate unavailable");}else for(int r=0;r<rows;++r)ref_fn(n,&d[r],0,p+r*row,0,acts.data(),0,1);}};
   run(false,ref);run(true,got);size_t unequal=0;for(size_t i=0;i<ref.size();++i){if(!std::isfinite(ref[i])||!std::isfinite(got[i]))throw std::runtime_error("nonfinite result");unequal+=memcmp(&ref[i],&got[i],4)!=0;}
   if(unequal)throw std::runtime_error("single-token kernel changed output bits");
   for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k){const bool candidate=(trial+k)%2;auto&out=candidate?got:ref;run(candidate,out);auto start=std::chrono::steady_clock::now();for(int it=0;it<3;++it)run(candidate,out);auto stop=std::chrono::steady_clock::now();const double us=std::chrono::duration<double,std::micro>(stop-start).count()/3;std::cout<<g->type<<','<<E<<','<<variants[vi]<<','<<candidate<<','<<trial<<','<<us<<','<<hash(out)<<','<<unequal<<'\n'<<std::flush;}
  }
 }
 if(seen.size()!=1)throw std::runtime_error("missing formats");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
