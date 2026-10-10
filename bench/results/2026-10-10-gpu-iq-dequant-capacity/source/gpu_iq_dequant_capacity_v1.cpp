// Actual-data isolated generic GPU IQ dequant control; no engine changes.
// All performance rows withheld until every selected source/oracle gate passes.
#include "strata/artifact/native_role_plan.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>
namespace c=strata::kernels::cpu;
namespace k=strata::kernels;
using Clock=std::chrono::steady_clock;
constexpr size_t H=2560,FF=640,GU=2*FF*H,D=H*FF,MAXBLOB=2662400,G=64;
constexpr unsigned SAMPLES=7;
constexpr size_t MAX_EXPLICIT_BUFFER_BYTES=32*(MAXBLOB+2*G)*2+8*(GU+D)+MAXBLOB+2*G+(8ull<<20);
static_assert(MAX_EXPLICIT_BUFFER_BYTES<(512ull<<20));
constexpr std::array<std::array<int,2>,7> PAIRS{{{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}}};
constexpr std::array<unsigned,7> LAYER_COUNTS{14,3,9,1,15,5,1};
constexpr const char* ENVS[]={"ONEAPI_DEVICE_SELECTOR","SYCL_DEVICE_FILTER","SYCL_CACHE_PERSISTENT",
 "SYCL_PROGRAM_COMPILE_OPTIONS","SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
 "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE","SYCL_PI_LEVEL_ZERO_DISABLE_COPY_OFFLOAD",
 "UR_L0_USE_IMMEDIATE_COMMANDLISTS","UR_L0_USE_COPY_ENGINE","UR_L0_USE_DIRECT_SUBMISSION",
 "UR_L0_ENABLE_COPY_OFFLOAD","UR_L0_USE_COPY_OFFLOAD","UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS",
 "UR_L0_ENABLE_ZESINIT","ZE_AFFINITY_MASK","ZE_FLAT_DEVICE_HIERARCHY","LD_PRELOAD","LD_AUDIT"};
static_assert(sizeof(ggml_fp16_t)==2 && sizeof(sycl::half)==2);
void need(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
std::string js(const std::string& s){
 std::string o="\"";const char* hex="0123456789abcdef";
 for(unsigned char ch:s){if(ch=='"'||ch=='\\'){o+='\\';o+=char(ch);}
 else if(ch<32){o+="\\u00";o+=hex[ch>>4];o+=hex[ch&15];}else o+=char(ch);}return o+'"';
}
uint64_t decimal(const char* p){
 const std::string_view s(p);uint64_t v=0;
 need(!s.empty()&&s.size()<=20&&(s.size()==1||s.front()!='0'),"strict decimal width");
 auto r=std::from_chars(s.data(),s.data()+s.size(),v);
 need(r.ec==std::errc{}&&r.ptr==s.data()+s.size(),"decimal syntax/overflow");return v;
}
uint64_t fnv(const void* p,size_t n){
 uint64_t v=14695981039346656037ull;auto b=static_cast<const uint8_t*>(p);
 for(size_t i=0;i<n;++i)v=(v^b[i])*1099511628211ull;return v;
}
uint64_t elapsed(Clock::time_point a,Clock::time_point b){
 auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
 need(n>=0,"host steady clock reversed");return uint64_t(n);
}
void path_check(const char* p){need(*p&&std::strlen(p)<=4096,"path width");}
struct FileState{
 uint64_t dev,ino,bytes;int64_t mt_s,mt_n,ct_s,ct_n;
 bool operator==(const FileState&)const=default;
};
FileState state(const std::string& path){
 struct stat s{};need(::stat(path.c_str(),&s)==0&&S_ISREG(s.st_mode)&&s.st_size>0,"regular source stat failed: "+path);
 return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),
 s.st_mtim.tv_sec,s.st_mtim.tv_nsec,s.st_ctim.tv_sec,s.st_ctim.tv_nsec};
}
std::string state_json(const FileState& s){
 std::ostringstream o;o<<"{\"dev\":"<<s.dev<<",\"ino\":"<<s.ino<<",\"bytes\":"<<s.bytes
 <<",\"mtime\":["<<s.mt_s<<','<<s.mt_n<<"],\"ctime\":["<<s.ct_s<<','<<s.ct_n<<"]}";return o.str();
}
uint64_t manifest_hash(const std::string& path){
 std::ifstream f(path,std::ios::binary);need(bool(f),"manifest open");
 std::array<char,65537> b{};f.read(b.data(),b.size());auto n=f.gcount();
 need(n>0&&n<=65536&&f.eof(),"manifest byte bound/read");return fnv(b.data(),size_t(n));
}
// Independent binary32 -> binary16 RNE bit encoder, not a host half constructor.
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
[[noreturn]] void fatal(const char* why){
 std::fprintf(stderr,"{\"success\":false,\"fatal_no_unwind\":true,\"error\":\"%s\",\"no_timing_rows_valid\":true}\n",why);
 std::fflush(stderr);std::_Exit(86);
}
void fault_detail(const char* what)noexcept{
 // Fixed log budget, no allocation or runtime calls in the async handler.
 std::fputs("{\"success\":false,\"first_queue_exception\":\"",stderr);
 const char* hex="0123456789abcdef";size_t i=0;
 for(;i<4096&&what[i];++i){const auto ch=static_cast<unsigned char>(what[i]);
  if(ch=='"'||ch=='\\'){std::fputc('\\',stderr);std::fputc(ch,stderr);}
  else if(ch<32)std::fprintf(stderr,"\\u00%c%c",hex[ch>>4],hex[ch&15]);
  else std::fputc(ch,stderr);
 }
 std::fprintf(stderr,"\",\"detail_truncated_at_4096\":%s,\"no_timing_rows_valid\":true}\n",i==4096?"true":"false");
}
struct Session{
 std::atomic<bool> async_bad{false};
 sycl::queue q;
 Session(const sycl::device& d):q(d,[this](sycl::exception_list list)noexcept{
  if(list.begin()!=list.end()){
   async_bad.store(true,std::memory_order_release);
   try{std::rethrow_exception(*list.begin());}
   catch(const std::exception& e){fault_detail(e.what());}
   catch(...){fault_detail("nonstandard asynchronous exception");}
  }
 },sycl::property_list{sycl::property::queue::in_order{}}){
  need(q.has_property<sycl::property::queue::in_order>(),"queue not in order");
 }
 void drain()noexcept{
  try{q.wait_and_throw();if(async_bad.load(std::memory_order_acquire))fatal("asynchronous queue fault");}
  catch(const std::exception& e){fault_detail(e.what());fatal("queue drain failed; completion unknown");}
  catch(...){fatal("queue drain failed; completion unknown");}
 }
};
struct Drain{Session& s;~Drain(){s.drain();}};
struct DeviceBuffer{
 Session& s;uint8_t* base=nullptr;size_t n;
 DeviceBuffer(Session& ss,size_t bytes):s(ss),n(bytes){
  need(bytes<=MAXBLOB||bytes==GU*2||bytes==D*2||bytes<4096,"device buffer bound");
  const size_t allocated=(n+2*G+63)/64*64;
  base=static_cast<uint8_t*>(sycl::aligned_alloc_device(64,allocated,s.q));
  need(base!=nullptr,"USM allocation failed");
 }
 ~DeviceBuffer(){s.drain();try{sycl::free(base,s.q);}catch(...){fatal("USM free failed after drain");}}
 DeviceBuffer(const DeviceBuffer&)=delete;
 uint8_t* data(){return base+G;}
 void initialize(){
  s.q.memset(base,0xa5,n+2*G);
  s.q.memset(data(),0xff,n);
 }
};
void guards(const std::vector<uint8_t>& b,size_t n){
 need(b.size()==n+2*G,"guarded readback extent");
 for(size_t i=0;i<G;++i){
  need(b[i]==0xa5,"front canary changed payload_bytes="+std::to_string(n)+" index="+std::to_string(i));
  need(b[G+n+i]==0xa5,"tail canary changed payload_bytes="+std::to_string(n)+" index="+std::to_string(i));
 }
}
std::vector<uint8_t> readback(DeviceBuffer& b){
 std::vector<uint8_t> h(b.n+2*G);Drain drain{b.s};
 b.s.q.memcpy(h.data(),b.base,h.size());b.s.drain();guards(h,b.n);return h;
}
struct Source{
 int layer,expert;c::NativeFmt fmt;
 std::vector<uint8_t> packed;uint64_t hash;
};
std::vector<uint16_t> expected(const Source& source){
 std::vector<uint16_t> out(GU+D);
 size_t written=0;
 std::array<float,H> row{};
 std::array<ggml_fp16_t,H> half{};
 for(int role=0;role<3;++role){
  const int type=role<2?source.fmt.gu_type:source.fmt.d_type;
  const auto* tr=ggml_get_type_traits(static_cast<ggml_type>(type));
  need(tr&&tr->to_float,"GGML FP32 row decoder missing");
  const size_t columns=role<2?H:FF,rows=role<2?FF:H;
  const size_t rb=role<2?source.fmt.gu_row:source.fmt.d_row;
  const size_t offset=role==0?0:role==1?source.fmt.up_off:source.fmt.down_off;
  need(tr->blck_size>0&&columns%size_t(tr->blck_size)==0&&
       columns/size_t(tr->blck_size)*tr->type_size==rb,"GGML/oracle row-geometry metadata bug");
  need(offset<=source.fmt.bytes&&rows*rb<=source.fmt.bytes-offset,"oracle packed role extent bug");
  for(size_t r=0;r<rows;++r){
   row.fill(std::numeric_limits<float>::quiet_NaN());
   tr->to_float(source.packed.data()+G+offset+r*rb,row.data(),int64_t(columns));
   ggml_fp32_to_fp16_row(row.data(),half.data(),int64_t(columns));
   for(size_t col=0;col<columns;++col){
    if(!std::isfinite(row[col]))throw std::runtime_error("GGML decoded nonfinite/incomplete row");
    const uint16_t independent=rne(row[col]);
    if(uint16_t(half[col])!=independent)throw std::runtime_error("GGML/independent half conversion mismatch role="+std::to_string(role)+
         " row="+std::to_string(r)+" col="+std::to_string(col)+
         " GGML_bits="+std::to_string(uint16_t(half[col]))+" independent_bits="+std::to_string(independent));
    if((independent&0x7c00)==0x7c00)throw std::runtime_error("actual-weight FP16 oracle nonfinite");
    const size_t index=role<2?(2*r+size_t(role))*H+col:GU+r*FF+col;
    if(index>=out.size())throw std::runtime_error("oracle output index metadata bug");
    out[index]=independent;++written;
   }
  }
 }
 need(written==GU+D,"oracle complete-output count mismatch");return out;
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
void launch(Session& s,const Source& source,DeviceBuffer& packed,DeviceBuffer& gu,DeviceBuffer& down){
 k::iq_dequant_gu_f16(source.fmt.gu_type,packed.data(),packed.data()+source.fmt.up_off,FF,H,
                     reinterpret_cast<uint16_t*>(gu.data()),&s.q);
 k::iq_dequant_f16(source.fmt.d_type,packed.data()+source.fmt.down_off,H*FF,
                  reinterpret_cast<uint16_t*>(down.data()),&s.q);
}
void verify(Session& s,const Source& source,DeviceBuffer& gu,DeviceBuffer& down,const char* stage){
 const auto ref=expected(source);
 auto g=readback(gu),d=readback(down);
 compare(g,ref,0,GU,source,stage);compare(d,ref,GU,D,source,stage);
 need(!s.async_bad.load(),"asynchronous fault after readback");
}
void device_edges(Session& s){
 std::vector<float> input;for(auto e:EDGES)input.push_back(std::bit_cast<float>(e.f));
 DeviceBuffer in(s,input.size()*sizeof(float)),out(s,2*input.size()*sizeof(uint16_t));
 Drain drain{s};in.initialize();out.initialize();
 s.q.memcpy(in.data(),input.data(),input.size()*sizeof(float));s.drain();
 const auto* f=reinterpret_cast<const float*>(in.data());
 auto* h=reinterpret_cast<sycl::half*>(out.data());
 const size_t n=input.size();
 s.q.parallel_for(sycl::range<1>(n/8),[=](sycl::id<1> id){
  sycl::vec<float,8> v;
  for(int j=0;j<8;++j)v[j]=f[id[0]*8+size_t(j)];
  const auto converted=v.convert<sycl::half,sycl::rounding_mode::automatic>();
  for(int j=0;j<8;++j){
   const size_t i=id[0]*8+size_t(j);h[i]=sycl::half(f[i]);h[n+i]=converted[j];
  }
 });s.drain();
 auto bits=readback(out);auto source=readback(in);
 need(std::memcmp(source.data()+G,input.data(),input.size()*sizeof(float))==0,"edge source changed");
 for(size_t i=0;i<2*n;++i){uint16_t actual;std::memcpy(&actual,bits.data()+G+2*i,2);
  need(actual==EDGES[i%n].h,"device conversion edge baseline failure form="+
       std::string(i<n?"scalar":"vec8_automatic")+" index="+std::to_string(i%n)+
       " actual_bits="+std::to_string(actual)+" expected_bits="+std::to_string(EDGES[i%n].h));}
}
struct PairResult{
 int gu,down,layer;size_t selected,blob_bytes;uint64_t first_use_ns=0,warm_ns=0;
 std::array<uint64_t,SAMPLES> wall{},submit{};
 std::vector<std::string> identities;
 uint64_t compared=0,end_compared=0,source_bytes_checked=0,launches=0;
};
struct Config{std::string pack,primary,mode;unsigned batch,gpu;uint64_t seed;};
Config parse(int argc,char** argv){
 need(argc==7,"usage: probe PACK PRIMARY qualify|timing BATCH16_OR32 SEED GPU_INDEX");
 path_check(argv[1]);path_check(argv[2]);need(std::strlen(argv[3])<=8,"mode width");
 Config x{argv[1],argv[2],argv[3],0,0,0};
 need(x.mode=="qualify"||x.mode=="timing","mode");
 const auto batch=decimal(argv[4]),seed=decimal(argv[5]),gpu=decimal(argv[6]);
 need(batch==16||batch==32,"batch must be 16 or 32");need(gpu<=15,"GPU index bound");
 x.batch=unsigned(batch);x.seed=seed;x.gpu=unsigned(gpu);return x;
}
int main(int argc,char** argv){
 std::string capability="null";
 try{
  const auto cfg=parse(argc,argv); // no runtime/queue/GGML allocations before full parse
  need(sizeof(void*)==8&&std::endian::native==std::endian::little,"64-bit little-endian host required");
  for(const char* name:ENVS){const char* value=std::getenv(name);
   need(!value||std::strlen(value)<=4096,"recorded environment width exceeded");}
  need(!std::getenv("LD_PRELOAD")&&!std::getenv("LD_AUDIT"),"preload/audit forbidden");
  host_edges(); // independent literal/GGML conversion gates, before GPU use
  std::map<std::string,FileState> files;
  const std::string manifest=cfg.pack+"/native_experts.txt";path_check(manifest.c_str());
  files.emplace(manifest,state(manifest));const auto manifest_fnv=manifest_hash(manifest);
  files.emplace(cfg.primary,state(cfg.primary));
  need(files.at(cfg.primary).bytes==54817524224ull,"current primary shard exact size mismatch");
  std::string error;
  if(!c::expert_layout_load(cfg.pack,48,512,error))throw std::runtime_error("layout: "+error);
  const auto& layout=c::expert_layout();
  need(layout.native&&layout.n_layers==48&&layout.n_expert==512&&layout.fmt.size()==48&&layout.max_blob==MAXBLOB,
       "current native geometry");
  std::array<int,7> layers{};layers.fill(-1);std::array<unsigned,7> counts{};
  for(size_t l=0;l<48;++l){
   const auto& f=layout.fmt[l];size_t p=0;
   while(p<PAIRS.size()&&(f.gu_type!=PAIRS[p][0]||f.d_type!=PAIRS[p][1]))++p;
   need(p<PAIRS.size(),"unexpected actual pair");
   need(f.n_embd==H&&f.n_ff==FF&&f.bytes<=MAXBLOB&&f.up_off==f.gu_row*FF&&
        f.down_off==2*f.up_off&&f.bytes==f.down_off+f.d_row*H,"actual layout shape/offset");
   need(k::native_expert_supported(f.gu_type,f.d_type,H,FF),"production role eligibility");
   ++counts[p];if(layers[p]<0)layers[p]=int(l);
  }
  need(counts==LAYER_COUNTS,"current seven-pair layer census mismatch");
  strata::NativeRolePlan plan;
  if(!plan.open(cfg.primary,layout,error))throw std::runtime_error("NativeRolePlan: "+error);
  need(plan.file_count()==1,"current primary-only native role fixture expected");
  for(int l=0;l<48;++l)for(int role=0;role<3;++role){
   const auto* e=plan.extent(l,role);need(e,"missing role extent");path_check(e->path.c_str());
   need(e->path==cfg.primary,"current census roles must use admitted primary shard");
   if(!files.count(e->path))files.emplace(e->path,state(e->path));
  }
  for(const auto& [path,before]:files)need(state(path)==before,"source changed during plan open");
  const auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);
  need(cfg.gpu<devices.size(),"GPU index unavailable");const auto dev=devices[cfg.gpu];
  need(dev.get_info<sycl::info::device::vendor_id>()==0x8086&&dev.has(sycl::aspect::fp16)&&
       dev.has(sycl::aspect::usm_device_allocations),"Intel GPU/fp16/device-USM admission");
  Session session(dev);
  need(session.q.get_backend()==sycl::backend::ext_oneapi_level_zero,"Level Zero backend required");
  {
   std::ostringstream o;o<<"{\"device\":"<<js(dev.get_info<sycl::info::device::name>())
    <<",\"driver\":"<<js(dev.get_info<sycl::info::device::driver_version>())
    <<",\"backend\":\"level_zero\",\"half_fp_config\":[";
   const auto fp=dev.get_info<sycl::info::device::half_fp_config>();
   for(size_t i=0;i<fp.size();++i){if(i)o<<',';o<<static_cast<int>(fp[i]);}
   o<<"]}";capability=o.str();
  }
  if(cfg.mode=="qualify")device_edges(session);
  std::vector<PairResult> receipts;receipts.reserve(7);
  for(size_t pair=0;pair<PAIRS.size();++pair){
   const int layer=layers[pair];const auto fmt=layout.fmt[size_t(layer)];
   const unsigned selected=cfg.mode=="qualify"?1:cfg.batch;
   std::vector<Source> sources;sources.reserve(selected);
   PairResult r{fmt.gu_type,fmt.d_type,layer,selected,fmt.bytes};
   for(unsigned j=0;j<selected;++j){
    const int expert=int(((cfg.seed%512)+pair*53+size_t(j)*17)%512);
    Source source{layer,expert,fmt,std::vector<uint8_t>(fmt.bytes+2*G,0xa5),0};
    if(!plan.copy_blob(layer,expert,source.packed.data()+G,fmt.bytes,error))
     throw std::runtime_error("NativeRolePlan copy: "+error);
    guards(source.packed,fmt.bytes);source.hash=fnv(source.packed.data()+G,fmt.bytes);
    std::ostringstream id;id<<"{\"layer\":"<<layer<<",\"expert\":"<<expert
     <<",\"packed_bytes\":"<<fmt.bytes<<",\"packed_fnv1a64\":"<<source.hash<<",\"roles\":[";
    for(int role=0;role<3;++role){if(role)id<<',';const auto* e=plan.extent(layer,role);
     need(e&&e->bytes_per_expert<=UINT64_MAX/512,"role expert stride overflow");
     const uint64_t delta=uint64_t(expert)*e->bytes_per_expert;
     need(e->offset<=UINT64_MAX-delta,"role source offset overflow");
     id<<"{\"path\":"<<js(e->path)<<",\"tensor\":"<<js(e->tensor)
       <<",\"offset\":"<<e->offset+delta<<",\"bytes\":"<<e->bytes_per_expert<<"}";}
    id<<"]}";r.identities.push_back(id.str());sources.push_back(std::move(source));
   }
   // All host copy sources exist before this drain guard and device staging.
   std::vector<std::unique_ptr<DeviceBuffer>> packed;packed.reserve(selected);
   std::array<std::unique_ptr<DeviceBuffer>,2> gu,down;
   Drain pair_drain{session}; // destroyed BEFORE host sources on any unwind
   for(unsigned j=0;j<selected;++j){
    packed.push_back(std::make_unique<DeviceBuffer>(session,fmt.bytes));
    session.q.memcpy(packed.back()->base,sources[j].packed.data(),sources[j].packed.size());
   }
   for(unsigned slot=0;slot<2;++slot){
    gu[slot]=std::make_unique<DeviceBuffer>(session,GU*2);
    down[slot]=std::make_unique<DeviceBuffer>(session,D*2);
    gu[slot]->initialize();down[slot]->initialize();
   }
   session.drain();
   if(cfg.mode=="qualify"){
    launch(session,sources[0],*packed[0],*gu[0],*down[0]);session.drain();r.launches+=2;
    verify(session,sources[0],*gu[0],*down[0],"qualifier");r.compared+=GU+D;
   }else{
    // First use / JIT and one full warm batch are excluded from sample clocks.
    auto first=Clock::now();launch(session,sources[0],*packed[0],*gu[0],*down[0]);
    session.drain();r.first_use_ns=elapsed(first,Clock::now());r.launches+=2;
    auto warm=Clock::now();
    for(unsigned j=0;j<selected;++j)launch(session,sources[j],*packed[j],*gu[j%2],*down[j%2]);
    session.drain();r.warm_ns=elapsed(warm,Clock::now());r.launches+=2*selected;
    for(unsigned sample=0;sample<SAMPLES;++sample){
     // Fixed predeclared distinct source order and production DQ=2 ring.
     const auto begin=Clock::now();
     for(unsigned j=0;j<selected;++j)launch(session,sources[j],*packed[j],*gu[j%2],*down[j%2]);
     const auto submitted=Clock::now();session.drain();const auto end=Clock::now();
     r.submit[sample]=elapsed(begin,submitted);r.wall[sample]=elapsed(begin,end);
     need(r.wall[sample]>0&&r.submit[sample]<=r.wall[sample],"sample host interval validity");
     r.launches+=2*selected;
     // Both final slot occupants get full independent oracle checks outside timing.
     for(unsigned j=selected-2;j<selected;++j){
      verify(session,sources[j],*gu[j%2],*down[j%2],"timed_end_slot");
      r.end_compared+=GU+D;
     }
    }
    // Every distinct selected source is independently qualified, untimed.
    // This is planned validation, not fault replay; any earlier fault exits.
    for(unsigned j=0;j<selected;++j){
     gu[j%2]->initialize();down[j%2]->initialize();
     launch(session,sources[j],*packed[j],*gu[j%2],*down[j%2]);session.drain();r.launches+=2;
     verify(session,sources[j],*gu[j%2],*down[j%2],"all_selected_sources");r.compared+=GU+D;
    }
   }
   for(unsigned j=0;j<selected;++j){
    const auto check=readback(*packed[j]);
    need(check==sources[j].packed,"full device packed source changed");
    guards(sources[j].packed,fmt.bytes);
    need(fnv(sources[j].packed.data()+G,fmt.bytes)==sources[j].hash,"host packed source changed");
    r.source_bytes_checked+=fmt.bytes;
   }
   // Include the allocated but unused qualifier slot's front/tail canaries.
   for(unsigned slot=0;slot<2;++slot){(void)readback(*gu[slot]);(void)readback(*down[slot]);}
   const unsigned expected_launches=cfg.mode=="qualify"?2:2+18*selected;
   need(r.launches==expected_launches&&r.compared==uint64_t(selected)*(GU+D)&&
       r.source_bytes_checked==uint64_t(selected)*fmt.bytes,"pair coverage reconciliation");
   need(r.end_compared==(cfg.mode=="qualify"?0:uint64_t(SAMPLES)*2*(GU+D)),"end slot comparison coverage");
   receipts.push_back(std::move(r));
  }
  session.drain();plan.close();
  for(const auto& [path,before]:files)need(state(path)==before,"source/manifest changed during run");
  need(manifest_hash(manifest)==manifest_fnv&&receipts.size()==7,"manifest/coverage final check");
  std::cout<<"{\"success\":true,\"mode\":"<<js(cfg.mode)
   <<",\"timing_rows_complete\":"<<(cfg.mode=="timing"?"true":"false")
   <<",\"scope\":\"actual current-pack blobs; isolated generic IQ dequant, not selected production route or model capacity\","
   <<"\"candidate_present\":false,\"generic_vs_independent_all_bits\":true,"
   <<"\"all_allocated_source_and_output_guards_checked\":true,"
   <<"\"host_ggml_vs_independent_edges\":true,\"device_edges_qualified\":"
   <<(cfg.mode=="qualify"?"true":"false")<<",\"qualification_process_required_before_timing\":true,"
   <<"\"H\":2560,\"FF\":640,\"NE\":512,\"DQ\":2,\"workgroup_size\":32,"
   <<"\"max_explicit_payload_buffer_bytes\":"<<MAX_EXPLICIT_BUFFER_BYTES<<','
   <<"\"compiler_policy\":\"IntelLLVM precise; AOT intel_gpu_bmg_g21; default SG32\","
   <<"\"device\":"<<js(dev.get_info<sycl::info::device::name>())
   <<",\"driver\":"<<js(dev.get_info<sycl::info::device::driver_version>())
   <<",\"backend\":\"level_zero\",\"half_fp_config\":[";
  auto fp=dev.get_info<sycl::info::device::half_fp_config>();
  for(size_t i=0;i<fp.size();++i){if(i)std::cout<<',';std::cout<<static_cast<int>(fp[i]);}
  std::cout<<"],\"queue_in_order\":true,\"profiling_markers\":false,"
   <<"\"kernel_device_service_unavailable_void_wrappers\":true,"
   <<"\"wall_scope\":\"host submit through one wait_and_throw per batch; not exclusive kernel service\","
   <<"\"seed\":"<<cfg.seed<<",\"configured_batch\":"<<cfg.batch<<",\"samples\":"
   <<(cfg.mode=="timing"?SAMPLES:0)<<",\"manifest_fnv1a64\":"<<manifest_fnv<<",\"runtime_environment\":{";
  for(size_t i=0;i<std::size(ENVS);++i){if(i)std::cout<<',';const char* value=std::getenv(ENVS[i]);
   std::cout<<js(ENVS[i])<<':'<<(value?js(value):"null");}
  std::cout<<"},\"unchanged_source_stats\":[";
  size_t si=0;for(const auto& [path,before]:files){if(si++)std::cout<<',';
   std::cout<<"{\"path\":"<<js(path)<<",\"stat\":"<<state_json(before)<<"}";}
  std::cout<<"],\"pairs\":[";
  for(size_t p=0;p<receipts.size();++p){
   if(p)std::cout<<',';const auto& r=receipts[p];
   std::cout<<"{\"gu_type\":"<<r.gu<<",\"down_type\":"<<r.down<<",\"layer\":"<<r.layer
    <<",\"selected_distinct_experts\":"<<r.selected<<",\"packed_bytes_each\":"<<r.blob_bytes
    <<",\"output_bytes_each\":"<<2*(GU+D)<<",\"values_compared_all_selected\":"<<r.compared
    <<",\"values_compared_timed_end_slots\":"<<r.end_compared
    <<",\"device_packed_payload_bytes_fully_checked\":"<<r.source_bytes_checked
    <<",\"wrapper_launches_total\":"<<r.launches<<",\"selected\":[";
   for(size_t j=0;j<r.identities.size();++j){if(j)std::cout<<',';std::cout<<r.identities[j];}
   std::cout<<']';
   if(cfg.mode=="timing"){
    std::cout<<",\"first_use_wall_ns\":"<<r.first_use_ns<<",\"warm_batch_wall_ns\":"<<r.warm_ns<<",\"individual_samples\":[";
    for(unsigned sample=0;sample<SAMPLES;++sample){if(sample)std::cout<<',';
     const uint64_t input=r.selected*r.blob_bytes,output=r.selected*2*(GU+D);
     std::cout<<"{\"sample\":"<<sample<<",\"experts\":"<<r.selected<<",\"wrapper_launches\":"<<2*r.selected
      <<",\"host_submit_ns\":"<<r.submit[sample]<<",\"host_submit_completion_wall_ns\":"<<r.wall[sample]
      <<",\"logical_packed_input_bytes\":"<<input<<",\"logical_FP16_output_bytes\":"<<output
      <<",\"logical_input_plus_output_GBps\":"<<double(input+output)/double(r.wall[sample])<<"}";}
    std::cout<<']';
   }
   std::cout<<'}';
  }
  std::cout<<"]}\n";need(bool(std::cout),"receipt output failed");return 0;
 }catch(const std::exception& e){
  std::cerr<<"{\"success\":false,\"no_timing_rows_valid\":true,\"capability\":"<<capability
           <<",\"error\":"<<js(e.what())<<"}\n";return 1;
 }catch(...){std::cerr<<"{\"success\":false,\"no_timing_rows_valid\":true,\"error\":\"unknown exception\"}\n";return 1;}
}
