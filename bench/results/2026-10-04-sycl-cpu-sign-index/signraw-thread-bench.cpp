#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "/tmp/strata-sycl-goal-signindex-format.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#include <atomic>
#include <barrier>
#include <thread>
#include "strata/kernels/cpu/pool.hpp"
namespace strata::kernels::cpu {
#define DECL(V) bool iq256_single_gu_rows_signraw_v##V(int,const uint8_t*,size_t,size_t,int,const void*,float*,int,int);
DECL(1) DECL(2) DECL(3) DECL(4) DECL(5) DECL(6)
#undef DECL
}
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
uint64_t hash(const std::vector<float>&v){uint64_t h=14695981039346656037ull;for(size_t i=0;i<v.size()*4;++i){h^=((const uint8_t*)v.data())[i];h*=1099511628211ull;}return h;}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: index-bench SHARD1 EXPERTS THREADS");
 ggml_cpu_init();const int E=std::stoi(argv[2]);if(E<1||E>64)throw std::runtime_error("bad experts");
 const int threads=std::stoi(argv[3]);
 auto cores=strata::kernels::cpu::physical_cores(false);if(threads<1||threads>int(cores.size()))throw std::runtime_error("bad threads");
 const auto old_affinity=strata::kernels::cpu::pin_current_thread(cores[0]);if(old_affinity<0)throw std::runtime_error("pin failed");
 std::cerr<<"threads="<<threads<<" cores=";for(int i=0;i<threads;++i)std::cerr<<cores[i]<<',';std::cerr<<'\n';
 auto model=strata::GgufModel::open(argv[1]);std::set<int> seen;
 std::cout<<"type,experts,variant,candidate,trial,microseconds,fnv1a64,unequal\n";
 for(int layer=0;layer<48;++layer){
  size_t gs,us_;const auto pre="blk."+std::to_string(layer)+".";
  auto*g=model.find(pre+"ffn_gate_exps.weight",&gs);auto*u=model.find(pre+"ffn_up_exps.weight",&us_);
  if(!g||!u||g->type!=21||!seen.insert(g->type).second)continue;
  if(g->shape.size()!=3||g->shape!=u->shape||g->type!=u->type||g->shape[2]<512||!model.in_bounds(*g,gs)||!model.in_bounds(*u,us_))throw std::runtime_error("bad geometry");
  const int type=g->type,n=g->shape[0],rows=g->shape[1],nb=n/256;
  const size_t rawblock=sizeof(block_iq3_s),packblock=sizeof(block_iq3_s);
  const size_t rawrow=nb*rawblock,rawup=rows*rawrow,packrow=nb*packblock,packup=rows*packrow;
  std::vector<uint8_t> raw(E*2*rawup);
  for(int e=0;e<E;++e){memcpy(raw.data()+e*2*rawup,model.shard(gs).tensor_data(*g)+e*7*rawup,rawup);memcpy(raw.data()+e*2*rawup+rawup,model.shard(us_).tensor_data(*u)+e*7*rawup,rawup);}
  auto t0=Clock::now();std::vector<uint8_t> packed(E*2*packup);auto t1=Clock::now();
  const size_t blocks=size_t(E)*2*rows*nb;
  auto pack=[&]{memcpy(packed.data(),raw.data(),raw.size());};
  auto p0=Clock::now();pack();auto p1=Clock::now();std::vector<double> packing;
  for(int i=0;i<5;++i){auto a=Clock::now();pack();auto b=Clock::now();packing.push_back(us(a,b));}
  std::cerr<<"pack type="<<type<<" experts="<<E<<" raw_bytes="<<raw.size()<<" packed_bytes="<<packed.size()<<" allocation_us="<<us(t0,t1)<<" first_pack_us="<<us(p0,p1)<<" repeats_us=";
  for(auto x:packing)std::cerr<<x<<',';std::cerr<<'\n';
  std::vector<block_q8_K> acts(nb);uint32_t state=12345;
  for(auto&x:acts){x.d=.01f;for(int k=0;k<256;++k){state=state*1664525u+1013904223u;x.qs[k]=int(state%255)-127;}for(int k=0;k<16;++k){x.bsums[k]=0;for(int j=0;j<16;++j)x.bsums[k]+=x.qs[k*16+j];}}
  using namespace strata::kernels::cpu;using Fn=decltype(&iq256_single_gu_rows);
  Fn fns[]={iq256_single_gu_rows,iq256_single_gu_rows_signraw_v1,iq256_single_gu_rows_signraw_v2,iq256_single_gu_rows_signraw_v3};
  auto oracle=ggml_get_type_traits_cpu(ggml_type(type))->vec_dot;
  for(int v=1;v<=3;++v){
   std::vector<float> reference(E*rows),got(reference.size());
   std::barrier phase(threads);std::atomic<int> next{0};bool stopping=false,current_candidate=false;
   std::vector<float>* current_out=nullptr;const int chunks=(rows+63)/64;
   auto work=[&]{for(;;){int task=next.fetch_add(1,std::memory_order_relaxed);if(task>=E*chunks)break;
    int e=task/chunks,r0=(task%chunks)*64,r1=std::min(rows,r0+64);bool candidate=current_candidate;
    auto*blob=candidate?packed.data()+e*2*packup:raw.data()+e*2*rawup;
    if(!fns[candidate?v:0](type,blob,candidate?packrow:rawrow,candidate?packup:rawup,n,acts.data(),current_out->data()+e*rows,r0,r1))std::terminate();
   }};
   std::vector<std::thread> workers;
   for(int t=1;t<threads;++t)workers.emplace_back([&,t]{if(strata::kernels::cpu::pin_current_thread(cores[t])<0)std::terminate();for(;;){phase.arrive_and_wait();if(stopping)break;work();phase.arrive_and_wait();}});
   auto run=[&](bool candidate,std::vector<float>&out){current_candidate=candidate;current_out=&out;next=0;phase.arrive_and_wait();work();phase.arrive_and_wait();};
   run(false,reference);run(true,got);
   if(v==1)for(int e=0;e<E;++e)for(int r=0;r<rows;++r){auto*blob=raw.data()+e*2*rawup;float g1,u1;oracle(n,&g1,0,blob+r*rawrow,0,acts.data(),0,1);oracle(n,&u1,0,blob+rawup+r*rawrow,0,acts.data(),0,1);float result=(g1/(1.f+std::exp(-g1)))*u1;if(memcmp(&result,&reference[e*rows+r],4))throw std::runtime_error("baseline differs from GGML");}
   size_t unequal=0;for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])||!std::isfinite(reference[i]))throw std::runtime_error("nonfinite");unequal+=memcmp(&got[i],&reference[i],4)!=0;}
   if(unequal)throw std::runtime_error("preindex changed output bits");
   for(int trial=0;trial<10;++trial)for(int k=0;k<2;++k){bool candidate=(trial+k)%2;auto&out=candidate?got:reference;run(candidate,out);auto a=Clock::now();for(int it=0;it<5;++it)run(candidate,out);auto b=Clock::now();std::cout<<type<<','<<E<<','<<v<<','<<candidate<<','<<trial<<','<<us(a,b)/5<<','<<hash(out)<<','<<unequal<<'\n'<<std::flush;}
   stopping=true;phase.arrive_and_wait();for(auto& t:workers)t.join();
  }
 }
 strata::kernels::cpu::restore_thread_affinity(old_affinity);
 if(seen.size()!=1)throw std::runtime_error("missing types");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
