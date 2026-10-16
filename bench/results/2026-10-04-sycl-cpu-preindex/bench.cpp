#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "/tmp/strata-sycl-goal-index-format.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
namespace strata::kernels::cpu {
#define DECL(V) bool iq256_single_gu_rows_index_v##V(int,const uint8_t*,size_t,size_t,int,const void*,float*,int,int);
DECL(1) DECL(2) DECL(3) DECL(4) DECL(5)
#undef DECL
}
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t hash(const std::vector<float>&v){uint64_t h=14695981039346656037ull;for(size_t i=0;i<v.size()*4;++i){h^=((const uint8_t*)v.data())[i];h*=1099511628211ull;}return h;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: index-bench SHARD1 EXPERTS");
 ggml_cpu_init();const int E=std::stoi(argv[2]);if(E<1||E>64)throw std::runtime_error("bad experts");
 auto model=strata::GgufModel::open(argv[1]);std::set<int> seen;
 std::cout<<"type,experts,variant,candidate,trial,microseconds,fnv1a64,unequal\n";
 for(int layer=0;layer<48;++layer){
  size_t gs,us_;const auto pre="blk."+std::to_string(layer)+".";
  auto*g=model.find(pre+"ffn_gate_exps.weight",&gs);auto*u=model.find(pre+"ffn_up_exps.weight",&us_);
  if(!g||!u||(g->type!=21&&g->type!=22)||!seen.insert(g->type).second)continue;
  if(g->shape.size()!=3||g->shape!=u->shape||g->type!=u->type||g->shape[2]<512||!model.in_bounds(*g,gs)||!model.in_bounds(*u,us_))throw std::runtime_error("bad geometry");
  const int type=g->type,n=g->shape[0],rows=g->shape[1],nb=n/256;
  const size_t rawblock=type==21?sizeof(block_iq3_s):sizeof(block_iq2_s),packblock=type==21?sizeof(preindexed_iq3s):sizeof(preindexed_iq2s);
  const size_t rawrow=nb*rawblock,rawup=rows*rawrow,packrow=nb*packblock,packup=rows*packrow;
  std::vector<uint8_t> raw(E*2*rawup);
  for(int e=0;e<E;++e){memcpy(raw.data()+e*2*rawup,model.shard(gs).tensor_data(*g)+e*7*rawup,rawup);memcpy(raw.data()+e*2*rawup+rawup,model.shard(us_).tensor_data(*u)+e*7*rawup,rawup);}
  auto t0=Clock::now();std::vector<uint8_t> packed(E*2*packup);auto t1=Clock::now();
  const size_t blocks=size_t(E)*2*rows*nb;
  auto pack=[&]{if(type==21){
   auto*src=reinterpret_cast<const block_iq3_s*>(raw.data());auto*dst=reinterpret_cast<preindexed_iq3s*>(packed.data());
   for(size_t b=0;b<blocks;++b){dst[b].d=src[b].d;for(int k=0;k<64;++k)dst[b].indices[k]=src[b].qs[k]|(((src[b].qh[k/8]>>(k%8))&1)<<8);memcpy(dst[b].signs,src[b].signs,32);memcpy(dst[b].scales,src[b].scales,4);}
  }else{
   auto*src=reinterpret_cast<const block_iq2_s*>(raw.data());auto*dst=reinterpret_cast<preindexed_iq2s*>(packed.data());
   for(size_t b=0;b<blocks;++b){dst[b].d=src[b].d;for(int k=0;k<32;++k)dst[b].indices[k]=src[b].qs[k]|(((src[b].qh[k/4]>>(2*(k%4)))&3)<<8);memcpy(dst[b].signs,src[b].qs+32,32);memcpy(dst[b].scales,src[b].scales,8);}
  }};
  auto p0=Clock::now();pack();auto p1=Clock::now();std::vector<double> packing;
  for(int i=0;i<5;++i){auto a=Clock::now();pack();auto b=Clock::now();packing.push_back(us(a,b));}
  std::cerr<<"pack type="<<type<<" experts="<<E<<" raw_bytes="<<raw.size()<<" packed_bytes="<<packed.size()<<" allocation_us="<<us(t0,t1)<<" first_pack_us="<<us(p0,p1)<<" repeats_us=";
  for(auto x:packing)std::cerr<<x<<',';std::cerr<<'\n';
  std::vector<block_q8_K> acts(nb);uint32_t state=12345;
  for(auto&x:acts){x.d=.01f;for(int k=0;k<256;++k){state=state*1664525u+1013904223u;x.qs[k]=int(state%255)-127;}for(int k=0;k<16;++k){x.bsums[k]=0;for(int j=0;j<16;++j)x.bsums[k]+=x.qs[k*16+j];}}
  using namespace strata::kernels::cpu;using Fn=decltype(&iq256_single_gu_rows);
  Fn fns[]={iq256_single_gu_rows,iq256_single_gu_rows_index_v1,iq256_single_gu_rows_index_v2,iq256_single_gu_rows_index_v3,iq256_single_gu_rows_index_v4,iq256_single_gu_rows_index_v5};
  auto oracle=ggml_get_type_traits_cpu(ggml_type(type))->vec_dot;
  for(int v=1;v<=5;++v){
   std::vector<float> reference(E*rows),got(reference.size());
   auto run=[&](bool candidate,std::vector<float>&out){for(int e=0;e<E;++e){
    auto*blob=candidate?packed.data()+e*2*packup:raw.data()+e*2*rawup;
    if(!fns[candidate?v:0](type,blob,candidate?packrow:rawrow,candidate?packup:rawup,n,acts.data(),out.data()+e*rows,0,rows))throw std::runtime_error("unsupported candidate");
   }};
   run(false,reference);run(true,got);
   if(v==1)for(int e=0;e<E;++e)for(int r=0;r<rows;++r){auto*blob=raw.data()+e*2*rawup;float g1,u1;oracle(n,&g1,0,blob+r*rawrow,0,acts.data(),0,1);oracle(n,&u1,0,blob+rawup+r*rawrow,0,acts.data(),0,1);float result=(g1/(1.f+std::exp(-g1)))*u1;if(memcmp(&result,&reference[e*rows+r],4))throw std::runtime_error("baseline differs from GGML");}
   size_t unequal=0;for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])||!std::isfinite(reference[i]))throw std::runtime_error("nonfinite");unequal+=memcmp(&got[i],&reference[i],4)!=0;}
   if(unequal)throw std::runtime_error("preindex changed output bits");
   for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k){bool candidate=(trial+k)%2;auto&out=candidate?got:reference;run(candidate,out);auto a=Clock::now();for(int it=0;it<3;++it)run(candidate,out);auto b=Clock::now();std::cout<<type<<','<<E<<','<<v<<','<<candidate<<','<<trial<<','<<us(a,b)/3<<','<<hash(out)<<','<<unequal<<'\n'<<std::flush;}
  }
 }
 if(seen.size()!=2)throw std::runtime_error("missing types");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
