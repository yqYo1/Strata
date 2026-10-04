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
#define DECL(T,V) extern "C" void single_##T##_##V(int,float*,size_t,const void*,size_t,const void*,size_t,int);
DECL(21,0) DECL(21,10) DECL(21,11) DECL(21,12) DECL(21,14)
uint64_t hash(const std::vector<float>&v){uint64_t h=14695981039346656037ull;for(size_t i=0;i<v.size()*4;++i){h^=((const uint8_t*)v.data())[i];h*=1099511628211ull;}return h;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: single-bench SHARD1 EXPERTS");
 ggml_cpu_init();const int E=std::stoi(argv[2]);if(E<1||E>64)throw std::runtime_error("bad expert count");
 auto model=strata::GgufModel::open(argv[1]);std::set<int> seen;
 std::cout<<"type,experts,variant,candidate,trial,microseconds,fnv1a64,unequal\n";
 for(int layer=0;layer<48;++layer){
  size_t gs,us;auto pre="blk."+std::to_string(layer)+".";
  auto*g=model.find(pre+"ffn_gate_exps.weight",&gs);auto*u=model.find(pre+"ffn_up_exps.weight",&us);
  if(!g||!u||(g->type!=21)||!seen.insert(g->type).second)continue;
  if(g->shape.size()!=3||g->shape!=u->shape||g->type!=u->type||g->shape[2]<512||!model.in_bounds(*g,gs)||!model.in_bounds(*u,us))throw std::runtime_error("unexpected geometry");
  const int n=g->shape[0],rows=g->shape[1],nb=n/256;const size_t row=nb*sizeof(block_iq3_s),up=rows*row;
  std::vector<uint8_t> blobs(E*2*up+4096);
  for(int e=0;e<E;++e){memcpy(blobs.data()+e*2*up,model.shard(gs).tensor_data(*g)+e*7*up,up);memcpy(blobs.data()+e*2*up+up,model.shard(us).tensor_data(*u)+e*7*up,up);}
  std::vector<block_q8_K> acts(nb);uint32_t state=12345;
  for(auto&x:acts){x.d=.01f;for(int k=0;k<256;++k){state=state*1664525u+1013904223u;x.qs[k]=(int)(state%255)-127;}for(int k=0;k<16;++k){x.bsums[k]=0;for(int j=0;j<16;++j)x.bsums[k]+=x.qs[k*16+j];}}
  auto ref_fn=ggml_get_type_traits_cpu((ggml_type)g->type)->vec_dot;
  ggml_vec_dot_t candidates[]={single_21_0,single_21_10,single_21_11,single_21_12,single_21_14};
  const int variants[]={0,10,11,12,14};
  for(int vi=0;vi<5;++vi){auto cand_fn=candidates[vi];
   std::vector<float> ref(E*rows*3),got(ref.size());
   auto run=[&](bool candidate,std::vector<float>&out){auto fn=candidate?cand_fn:ref_fn;for(int e=0;e<E;++e)for(int r=0;r<rows;++r){float g1,u1;const auto*p=blobs.data()+e*2*up+r*row;fn(n,&g1,0,p,0,acts.data(),0,1);fn(n,&u1,0,p+up,0,acts.data(),0,1);auto*d=out.data()+3*(e*rows+r);d[0]=g1;d[1]=u1;d[2]=(g1/(1.f+std::exp(-g1)))*u1;}};
   run(false,ref);run(true,got);size_t unequal=0;for(size_t i=0;i<ref.size();++i){if(!std::isfinite(ref[i])||!std::isfinite(got[i]))throw std::runtime_error("nonfinite result");unequal+=memcmp(&ref[i],&got[i],4)!=0;}
   if(unequal)throw std::runtime_error("single-token kernel changed output bits");
   for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k){const bool candidate=(trial+k)%2;auto&out=candidate?got:ref;run(candidate,out);auto start=std::chrono::steady_clock::now();for(int it=0;it<3;++it)run(candidate,out);auto stop=std::chrono::steady_clock::now();const double us=std::chrono::duration<double,std::micro>(stop-start).count()/3;std::cout<<g->type<<','<<E<<','<<variants[vi]<<','<<candidate<<','<<trial<<','<<us<<','<<hash(out)<<','<<unequal<<'\n'<<std::flush;}
  }
 }
 if(seen.size()!=1)throw std::runtime_error("missing formats");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
