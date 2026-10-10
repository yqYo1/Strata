#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
constexpr size_t H=2560,FF=640,GU=2*FF*H,D=H*FF,G=64;
struct Source{int layer,expert;};
void need(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
std::string js(const std::string& s){
 std::string o="\"";const char* hex="0123456789abcdef";
 for(unsigned char ch:s){if(ch=='"'||ch=='\\'){o+='\\';o+=char(ch);}
 else if(ch<32){o+="\\u00";o+=hex[ch>>4];o+=hex[ch&15];}else o+=char(ch);}return o+'"';
}

uint16_t rne(float f){
 const uint32_t b=std::bit_cast<uint32_t>(f),sign=(b>>16)&0x8000u;
 const uint32_t e=(b>>23)&255u,m=b&0x7fffffu;
 if(e==255)return uint16_t(sign|(m?0x7e00u:0x7c00u));
 if(e==0)return uint16_t(sign);
 int he=int(e)-127+15;
 if(he>=31)return uint16_t(sign|0x7c00u);
 if(he< -10)return uint16_t(sign);
 uint32_t v,rem,half;
 if(he<=0){
  const unsigned shift=unsigned(14-he);const uint32_t sig=m|0x800000u;
  v=sig>>shift;rem=sig&((1u<<shift)-1);half=1u<<(shift-1);
 }else{
  v=(uint32_t(he)<<10)|(m>>13);rem=m&8191u;half=4096u;
 }
 if(rem>half||(rem==half&&(v&1u)))++v;
 return uint16_t(sign|v);
}
struct Edge{uint32_t f;uint16_t h;};
constexpr std::array<Edge,24> EDGES{{
 {0x00000000,0x0000},{0x80000000,0x8000},{0x3f800000,0x3c00},{0xbf800000,0xbc00},
 {0x3f801000,0x3c00},{0x3f803000,0x3c02},{0xbf801000,0xbc00},{0xbf803000,0xbc02},
 {0x33800000,0x0001},{0xb3800000,0x8001},{0x33000000,0x0000},{0xb3000000,0x8000},
 {0x33c00000,0x0002},{0xb3c00000,0x8002},{0x38800000,0x0400},{0x387fe000,0x0400},
 {0x387fc000,0x03ff},{0x477fe000,0x7bff},{0x477ff000,0x7c00},{0xc77ff000,0xfc00},
 {0x7f800000,0x7c00},{0xff800000,0xfc00},{0x00000001,0x0000},{0x80000001,0x8000}}};
void host_edges(){
 for(size_t i=0;i<EDGES.size();++i){
  const float f=std::bit_cast<float>(EDGES[i].f);ggml_fp16_t h=0;ggml_fp32_to_fp16_row(&f,&h,1);
  need(rne(f)==EDGES[i].h,"independent encoder literal edge failure index="+std::to_string(i)+
       " actual_bits="+std::to_string(rne(f))+" expected_bits="+std::to_string(EDGES[i].h));
  need(uint16_t(h)==EDGES[i].h,"GGML conversion edge contract failure index="+std::to_string(i)+
       " actual_bits="+std::to_string(uint16_t(h))+" expected_bits="+std::to_string(EDGES[i].h));
 }
}

void guards(const std::vector<uint8_t>& b,size_t n){
 need(b.size()==n+2*G,"guarded readback extent");
 for(size_t i=0;i<G;++i){
  need(b[i]==0xa5,"front canary changed payload_bytes="+std::to_string(n)+" index="+std::to_string(i));
  need(b[G+n+i]==0xa5,"tail canary changed payload_bytes="+std::to_string(n)+" index="+std::to_string(i));
 }
}

void compare(const std::vector<uint8_t>& actual,const std::vector<uint16_t>& ref,
             size_t offset,size_t count,const Source& src,const char* name){
 need(actual.size()==count*2+2*G&&ref.size()==GU+D,"comparison geometry bug");
 for(size_t i=0;i<count;++i){
  uint16_t v;std::memcpy(&v,actual.data()+G+2*i,2);
  if(v!=ref[offset+i]){
   const size_t row=i/(offset?FF:H),col=i%(offset?FF:H);
   const std::string role=offset?"down":(row%2?"up":"gate");
   throw std::runtime_error(std::string("generic_vs_independent baseline mismatch stage=")+name+
    " layer="+std::to_string(src.layer)+" expert="+std::to_string(src.expert)+" role="+role+
    " row="+std::to_string(offset?row:row/2)+" col="+std::to_string(col)+
    " actual_bits="+std::to_string(v)+" expected_bits="+std::to_string(ref[offset+i]));
  }
 }
}

int main(){host_edges();uint32_t seed=17;size_t values=0,rejected=0;
for(size_t i=0;i<100000;++i){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
if(((seed>>23)&255)==255&&(seed&0x7fffff))continue;
float f=std::bit_cast<float>(seed);ggml_fp16_t h;ggml_fp32_to_fp16_row(&f,&h,1);
need(uint16_t(h)==rne(f),"random independent/host mismatch i="+std::to_string(i)+" fp32bits="+std::to_string(seed)+" GGMLbits="+std::to_string(uint16_t(h))+" RNEbits="+std::to_string(rne(f)));++values;}
for(size_t n: {GU,D}){
std::vector<uint8_t> a(2*n+2*G,0xa5);std::vector<uint16_t> ref(GU+D,0);size_t off=n==GU?0:GU;
std::memset(a.data()+G,0,2*n);guards(a,2*n);compare(a,ref,off,n,Source{47,511},"host");
for(size_t index:{size_t(0),n-1}){a[G+2*index]=1;try{compare(a,ref,off,n,Source{47,511},"host");throw std::logic_error("uncaught bit change");}catch(const std::runtime_error& e){need(std::string(e.what()).find("generic_vs_independent")!=std::string::npos,"wrong mismatch gate");++rejected;}a[G+2*index]=0;}
for(size_t index:{size_t(0),G-1,G+2*n,G+2*n+G-1}){a[index]=0;try{guards(a,2*n);throw std::logic_error("uncaught guard change");}catch(const std::runtime_error& e){need(std::string(e.what()).find("canary")!=std::string::npos,"wrong guard gate");++rejected;}a[index]=0xa5;}}
std::cout<<"{\"passed\":true,\"literal_edges\":24,\"random_half_values\":"<<values<<",\"injected_changes_rejected\":"<<rejected<<"}\n";
}
