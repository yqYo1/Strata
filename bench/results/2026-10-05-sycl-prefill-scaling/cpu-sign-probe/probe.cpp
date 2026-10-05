#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <vector>
#define DECL(NS) namespace strata::kernels::NS { \
 void sign_bytes(uint32_t,uint8_t*); \
 void iq256_gu_rows(int,const uint8_t*,size_t,size_t,int,const void*const*,int,float*const*,int,int); \
 void iq256_rows(int,const uint8_t*,size_t,int,const void*const*,int,float*const*,int,int); }
DECL(cpu_original)
DECL(cpu_wide)
namespace orig=strata::kernels::cpu_original;
namespace wide=strata::kernels::cpu_wide;
static uint64_t floats_checked=0,signs_checked=0;
static void check(bool ok,const char* why){if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
static void equal(const float* a,const float* b,size_t n){
 for(size_t i=0;i<n;i++){check(std::isfinite(a[i])&&std::isfinite(b[i]),"non-finite CPU result");check(std::memcmp(a+i,b+i,4)==0,"FP32 bits differ");}floats_checked+=n;
}
static void signs(uint32_t m){uint8_t a[32],b[32];orig::sign_bytes(m,a);wide::sign_bytes(m,b);check(std::memcmp(a,b,32)==0,"sign bytes differ");for(int i=0;i<32;i++)check(b[i]==((m>>i&1)?255:1),"sign differs from scalar");++signs_checked;}
static void check_rows(int ty,const uint8_t* blob,size_t row,size_t up,int n,int rows,const void*const* act){
 for(int nt=1;nt<=8;nt++){
  std::vector<float>a(8*rows,0.125f),b(a);float* ap[8];float* bp[8];
  for(int t=0;t<8;t++){ap[t]=a.data()+t*rows;bp[t]=b.data()+t*rows;}
  orig::iq256_rows(ty,blob,row,n,act,nt,ap,2,rows-1);wide::iq256_rows(ty,blob,row,n,act,nt,bp,2,rows-1);equal(a.data(),b.data(),a.size());
  orig::iq256_gu_rows(ty,blob,row,up,n,act,nt,ap,2,rows-1);wide::iq256_gu_rows(ty,blob,row,up,n,act,nt,bp,2,rows-1);equal(a.data(),b.data(),a.size());
 }
}
static double milliseconds(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){
 check(argc==2,"provide GGUF shard");setvbuf(stdout,nullptr,_IONBF,0);
 cpu_set_t cpuset;CPU_ZERO(&cpuset);CPU_SET(2,&cpuset);check(sched_setaffinity(0,sizeof(cpuset),&cpuset)==0,"affinity");
 std::mt19937 rng(1011);signs(0);signs(~0u);for(int i=0;i<32;i++){signs(1u<<i);signs(~(1u<<i));}
 for(uint32_t i=0;i<65536;i++)signs(i|((i^0xffffu)<<16));for(int i=0;i<100000;i++)signs(rng());
 for(auto tb:std::vector<std::pair<int,int>>{{16,66},{17,74},{18,98},{21,110},{22,82},{23,136}}){
  for(int n:{256,768,2560})for(int offset:{0,1,3,7,15}){
   const int rows=19;size_t row=n/256*tb.second,up=rows*row;std::vector<uint8_t> data(2*up+offset+64);uint8_t*blob=data.data()+offset;
   for(size_t i=0;i<2*up;i++)blob[i]=rng();for(size_t i=0;i<2*up;i+=tb.second){uint16_t h=0x1000;std::memcpy(blob+i,&h,2);}
   std::vector<block_q8_K> actdata(8*n/256);const void* acts[8];
   for(int t=0;t<8;t++){acts[t]=actdata.data()+t*n/256;for(int j=0;j<n/256;j++){auto& q=actdata[t*n/256+j];q.d=.001f;for(int k=0;k<256;k++)q.qs[k]=(int)(rng()%255)-127;for(int k=0;k<16;k++){q.bsums[k]=0;for(int z=0;z<16;z++)q.bsums[k]+=q.qs[k*16+z];}}}
   check_rows(tb.first,blob,row,up,n,rows,acts);
  }
 }
 std::printf("{\"kind\":\"synthetic\",\"sign_patterns\":%llu,\"checked_floats\":%llu,\"all_bits_equal\":true}\n",(unsigned long long)signs_checked,(unsigned long long)floats_checked);
 ggml_cpu_init();strata::GgufFile file(argv[1]);
 auto tensor=[&](int l,const char* role)->const strata::TensorInfo*{for(const auto&t:file.tensors())if(t.name=="blk."+std::to_string(l)+".ffn_"+role+"_exps.weight")return &t;return nullptr;};
 std::normal_distribution<float> nd(0.f,1.f);int real_cases=0;std::vector<uint8_t> streaming;int dim=0,ff=0;size_t rowbytes=0,upoff=0;
 for(int l=0;l<48;l++){
  auto g=tensor(l,"gate"),u=tensor(l,"up");check(g&&u,"missing tensor");
  int n=g->shape[0],rows=g->shape[1];check(g->type==u->type,"gate/up types");if(g->type!=21&&g->type!=22){std::printf("{\"kind\":\"skip\",\"layer\":%d,\"type\":%u}\n",l,g->type);continue;}
  size_t row=ggml_row_size((ggml_type)g->type,n),up=row*rows;
  const auto* traits=ggml_get_type_traits_cpu((ggml_type)g->type);const auto* quant=ggml_get_type_traits_cpu(traits->vec_dot_type);size_t ab=ggml_row_size(traits->vec_dot_type,n);
  std::vector<uint8_t> actdata(8*ab);std::vector<float>x(8*n);const void*acts[8];for(float&v:x)v=nd(rng);for(int t=0;t<8;t++){acts[t]=actdata.data()+t*ab;quant->from_float(x.data()+t*n,actdata.data()+t*ab,n);}
  for(int e:{0,7,31}){
   std::vector<uint8_t> blob(2*up);std::memcpy(blob.data(),file.tensor_data(*g)+e*up,up);std::memcpy(blob.data()+up,file.tensor_data(*u)+e*up,up);check_rows(g->type,blob.data(),row,up,n,rows,acts);++real_cases;
  }
  if(streaming.empty()&&g->type==21){
   dim=n;ff=rows;rowbytes=row;upoff=up;streaming.resize(96*2*up);
   for(int e=0;e<96;e++){std::memcpy(streaming.data()+e*2*up,file.tensor_data(*g)+e*up,up);std::memcpy(streaming.data()+e*2*up+up,file.tensor_data(*u)+e*up,up);}
  }
 }
 std::printf("{\"kind\":\"real\",\"expert_cases\":%d,\"checked_floats\":%llu,\"all_bits_equal\":true}\n",real_cases,(unsigned long long)floats_checked);check(!streaming.empty(),"no IQ3_S dataset");
 const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);const auto* quant=ggml_get_type_traits_cpu(traits->vec_dot_type);size_t ab=ggml_row_size(traits->vec_dot_type,dim);
 std::vector<uint8_t> actdata(8*ab);std::vector<float>x(8*dim);const void*acts[8];for(float&v:x)v=nd(rng);for(int t=0;t<8;t++){acts[t]=actdata.data()+t*ab;quant->from_float(x.data()+t*dim,actdata.data()+t*ab,dim);}
 std::vector<float>output(8*ff);float*outs[8];for(int t=0;t<8;t++)outs[t]=output.data()+t*ff;
 using Fun=void(*)(int,const uint8_t*,size_t,size_t,int,const void*const*,int,float*const*,int,int);
 auto run=[&](Fun fn,int nt,int count,int repeats){double start=milliseconds();for(int z=0;z<repeats;z++)for(int e=0;e<count;e++)fn(21,streaming.data()+e*2*upoff,rowbytes,upoff,dim,acts,nt,outs,0,ff);return milliseconds()-start;};
 for(int nt:{1,2,3,4})for(int count:{1,96}){
  int repeats=count==1?128:2;run(orig::iq256_gu_rows,nt,count,1);run(wide::iq256_gu_rows,nt,count,1);
  for(int round=0;round<9;round++){
   double a,b;if(round%2==0){a=run(orig::iq256_gu_rows,nt,count,repeats);b=run(wide::iq256_gu_rows,nt,count,repeats);}else{b=run(wide::iq256_gu_rows,nt,count,repeats);a=run(orig::iq256_gu_rows,nt,count,repeats);}
   std::printf("{\"kind\":\"timing\",\"nt\":%d,\"experts\":%d,\"repeats\":%d,\"round\":%d,\"original_first\":%s,\"original_ms\":%.9f,\"wide_ms\":%.9f,\"speed_ratio\":%.9f}\n",nt,count,repeats,round,round%2==0?"true":"false",a,b,a/b);
  }
 }
 std::printf("{\"kind\":\"completed\",\"cpu\":2,\"streaming_bytes\":%zu,\"n_embd\":%d,\"n_ff\":%d,\"all_bits_equal\":true}\n",streaming.size(),dim,ff);return 0;
}
