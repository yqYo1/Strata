// Expression provenance: immutable gdn-gate-consumers-cd353f30 snapshot,
// kernels.dp.cpp SHA256 cc5254f688b9138fc71800884c853211771067c0f07a0a44a237fd45b59b4bfa.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
namespace {
constexpr size_t HV=48,HK=16,S=128,MaxT=8192,N=MaxT*HV,G=16;
constexpr uint32_t Canary=0x7fc12345U;
static_assert((2*N+8*N+96+22*G)*sizeof(float)<16*1024*1024);
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
uint32_t bits(float x){return std::bit_cast<uint32_t>(x);}
struct Errors {std::mutex mutex;std::vector<std::string> messages;std::atomic<bool> failed{false},overflow{false};
 void collect(sycl::exception_list es)noexcept{failed.store(true);try{std::lock_guard<std::mutex> lock(mutex);for(auto e:es){try{std::rethrow_exception(e);}catch(const std::exception& x){if(messages.size()<8){const char* m=x.what();size_t n=0;while(n<1024&&m[n])++n;messages.emplace_back(m,n);}else overflow.store(true);}catch(...){overflow.store(true);}}}catch(...){overflow.store(true);}}
 void check(){require(!failed.load()&&!overflow.load(),"asynchronous SYCL error");}
 void print(){std::lock_guard<std::mutex> lock(mutex);for(const auto& s:messages)std::cerr<<"ASYNC,"<<s<<'\n';if(overflow)std::cerr<<"ASYNC,overflow_or_nonstandard_exception\n";}
};

void ze_ok(ze_result_t result,const char* operation){
 if(result!=ZE_RESULT_SUCCESS)throw std::runtime_error(std::string("identity query failed:")+operation+":"+std::to_string(static_cast<uint32_t>(result)));
}
// Sanitized bounded text only; typed identity fields are the assertion.
std::string identity_text(const char* text,size_t capacity){
 require(text!=nullptr,"null identity text");size_t n=0;while(n<capacity&&text[n])++n;
 require(n<capacity,"unterminated/oversized identity text");std::string out;out.reserve(n);
 for(size_t i=0;i<n;++i){unsigned char c=static_cast<unsigned char>(text[i]);out+=(c>=32&&c<=126&&c!=',')?char(c):'_';}return out;
}
void exact_identity(sycl::queue& q){
 const auto selected=q.get_device();
 const ze_device_handle_t device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(selected);
 const ze_driver_handle_t driver=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(selected.get_platform());
 require(device&&driver,"null selected native handle");
 ze_device_properties_t properties{};properties.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;properties.pNext=nullptr;
 ze_ok(zeDeviceGetProperties(device,&properties),"zeDeviceGetProperties");
 require(properties.vendorId==0x8086&&properties.deviceId==0xe20c,"selected native vendor/device mismatch");
 require((properties.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE)==0,"selected native device is a subdevice");
 ze_driver_properties_t driver_properties{};driver_properties.stype=ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;driver_properties.pNext=nullptr;
 ze_ok(zeDriverGetProperties(driver,&driver_properties),"zeDriverGetProperties");
 require(driver_properties.driverVersion!=0,"invalid driver version");
 uint32_t count=0;ze_ok(zeDriverGetExtensionProperties(driver,&count,nullptr),"extension count");
 require(count>0&&count<=256,"driver extension count outside finite bound");
 std::array<ze_driver_extension_properties_t,256> extensions{};const uint32_t original=count;
 ze_ok(zeDriverGetExtensionProperties(driver,&count,extensions.data()),"extension properties");
 require(count>0&&count<=original,"driver extension list count changed");
 bool advertised=false;uint32_t version=0;
 for(uint32_t i=0;i<count;++i){
  require(std::memchr(extensions[i].name,0,sizeof(extensions[i].name))!=nullptr,"unterminated extension name");
  if(std::strcmp(extensions[i].name,ZE_PCI_PROPERTIES_EXT_NAME)==0){require(!advertised,"duplicate PCI extension");advertised=true;version=extensions[i].version;}
 }
 require(advertised&&version>=ZE_PCI_PROPERTIES_EXT_VERSION_1_0,"PCI extension unsupported/version too old");
 ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;pci.pNext=nullptr;
 // This is a public loader entry point. Installed driver26.31.39395.14
 // advertises PCI1.0 but rejects this name in its driver-function lookup.
 ze_ok(zeDevicePciGetPropertiesExt(device,&pci),"zeDevicePciGetPropertiesExt");
 require(pci.address.domain==0&&pci.address.bus==5&&pci.address.device==0&&pci.address.function==0,"selected PCI BDF mismatch");
 std::cout<<"IDENTITY,backend,LevelZero,vendor,"<<properties.vendorId<<",device,"<<properties.deviceId
 <<",domain,"<<pci.address.domain<<",bus,"<<pci.address.bus<<",pci_device,"<<pci.address.device<<",function,"<<pci.address.function
 <<",root,1,flags,"<<static_cast<uint32_t>(properties.flags)<<",driver_version,"<<driver_properties.driverVersion
 <<",pci_extension_version,"<<version<<",name,"<<identity_text(properties.name,sizeof(properties.name))<<",passed,1\n";
 for(const char* key:{"ZE_FLAT_DEVICE_HIERARCHY","ZE_AFFINITY_MASK","ONEAPI_DEVICE_SELECTOR"}){
  const char* value=std::getenv(key);std::cout<<"IDENTITY_ENV,"<<key<<','<<(value?identity_text(value,512):"unset")<<'\n';
 }
 std::cout.flush();require(bool(std::cout),"identity output failed");
}

struct Buffer {sycl::queue* q;float* base=nullptr;size_t n;
 Buffer(sycl::queue& queue,size_t count):q(&queue),n(count){base=sycl::malloc_shared<float>(n+2*G,queue);require(base,"USM allocation failed");reset();}
 Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
 ~Buffer(){if(base)sycl::free(base,*q);}
 float* data(){return base+G;}
 void reset(){std::fill(base,base+n+2*G,std::bit_cast<float>(Canary));}
 void guards(size_t active){require(active<=n,"active bound");for(size_t i=0;i<G;++i)require(bits(base[i])==Canary,"leading guard changed");for(size_t i=G+active;i<n+2*G;++i)require(bits(base[i])==Canary,"inactive tail/trailing guard changed");}
};
// On ANY submitted-stage error, no stack unwind/free is attempted: the owner
// supervises this process session. A successful wait_and_throw is the only drain.
template<class F> void stage(sycl::queue& q,Errors& errors,const char* name,F action){
 std::cout<<"STAGE,"<<name<<'\n';std::cout.flush();require(bool(std::cout),"progress output failed");
 try{action();q.wait_and_throw();errors.check();}catch(const std::exception& e){std::cerr<<"SUBMITTED_STAGE_FAIL,"<<name<<','<<e.what()<<'\n';errors.print();std::cerr.flush();std::_Exit(2);}catch(...){std::cerr<<"SUBMITTED_STAGE_FAIL,"<<name<<",unknown\n";std::cerr.flush();std::_Exit(2);}
}
class BaselineProducer;class CandidateProducer;class ScalarReader;
template<int SG> class GroupReader;
void producer(sycl::queue& q,size_t T,const float* ab,const float* dt,const float* a,float* gate,float* beta,float* factor){
 if(!T)return;size_t active=T*HV,global=(active+127)/128*128;
 if(!factor)q.parallel_for<BaselineProducer>(sycl::nd_range<1>{global,128},[=](sycl::nd_item<1> item){
  size_t i=item.get_global_linear_id();if(i>=active)return;size_t t=i/HV,h=i%HV;
  const float v=ab[t*2*HV+h]+dt[h];
  gate[i]=(v>20.0f?v:sycl::log1p(sycl::native::exp(v)))*a[h];
  beta[i]=1.0f/(1.0f+sycl::native::exp(-ab[t*2*HV+HV+h]));
 });
 else q.parallel_for<CandidateProducer>(sycl::nd_range<1>{global,128},[=](sycl::nd_item<1> item){
  size_t i=item.get_global_linear_id();if(i>=active)return;size_t t=i/HV,h=i%HV;
  const float v=ab[t*2*HV+h]+dt[h];
  const float loggate=(v>20.0f?v:sycl::log1p(sycl::native::exp(v)))*a[h];
  gate[i]=loggate;
  beta[i]=1.0f/(1.0f+sycl::native::exp(-ab[t*2*HV+HV+h]));
  factor[i]=sycl::native::exp(loggate);
 });
}
template<int SG>void group_reader(sycl::queue& q,size_t active,const float* log,float* factor){
 if(!active)return;size_t global=(active+127)/128*128;
 q.parallel_for<GroupReader<SG>>(sycl::nd_range<1>{global,128},[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
  auto sg=item.get_sub_group();size_t i=item.get_group_linear_id()*128+sg.get_group_linear_id()*SG+sg.get_local_linear_id();
  if(i<active)factor[i]=sycl::native::exp(log[i]);
 });
}
const char* domain(size_t i,bool specials){unsigned k=(i%HV)%16;if(specials&&k>=14)return "artificial_nonfinite";if(k>=8)return "artificial_finite_boundary";return "finite_synthetic_domain_unverified";}
}
int main(int argc,char** argv){try{
 require(argc==1||(argc==2&&std::string(argv[1])=="--include-artificial-nonfinite"),"unknown argument");bool specials=argc==2;
 Errors errors;
 // Selector/environment is recorded by owner; no fallback to CPU/OpenCL.
 sycl::queue q(sycl::gpu_selector_v,[&](sycl::exception_list es){errors.collect(es);},{sycl::property::queue::in_order{}});
 require(q.get_backend()==sycl::backend::ext_oneapi_level_zero&&q.get_device().is_gpu(),"Level Zero GPU required");
 require(q.is_in_order(),"in-order queue required");
 exact_identity(q);
 auto device_name=q.get_device().get_info<sycl::info::device::name>();
 auto vendor=q.get_device().get_info<sycl::info::device::vendor_id>();
 require(vendor==0x8086&&device_name.find("B570")!=std::string::npos,"Intel Arc B570 required");
 require(q.get_device().has(sycl::aspect::usm_shared_allocations),"shared USM required");
 auto sizes=q.get_device().get_info<sycl::info::device::sub_group_sizes>();
 require(std::find(sizes.begin(),sizes.end(),16)!=sizes.end()&&std::find(sizes.begin(),sizes.end(),32)!=sizes.end(),"SG16/32 required");
 require(q.get_device().get_info<sycl::info::device::max_work_group_size>()>=128,"WG128 required");
 std::cout<<"SCOPE,necessary_bitwise_gate_screen_not_recurrence_model_or_performance\nGEOMETRY,HV48,HK16,S128,MaxT8192,device_allocations_below16MiB\n";
 std::cout<<"DEVICE,"<<device_name<<",vendor,"<<vendor<<",driver,"<<q.get_device().get_info<sycl::info::device::driver_version>()<<'\n';std::cout.flush();require(bool(std::cout),"device output failed");
 Buffer ab(q,2*N),dt(q,HV),a(q,HV),bl(q,N),bb(q,N),cl(q,N),cb(q,N),cf(q,N),scalar(q,N),sg16(q,N),sg32(q,N);
 std::array<float,16> values{-8.f,-1.f,-0.00001f,0.f,1.f,19.9999980926513671875f,20.f,20.0000019073486328125f,80.f,87.33655f,103.27893f,104.f,-80.f,-104.f,0.f,0.f};
 if(specials){values[14]=std::numeric_limits<float>::infinity();values[15]=std::bit_cast<float>(0x7fc00001U);}
 for(size_t h=0;h<HV;++h){dt.data()[h]=0.f;a.data()[h]=h%16==3?-0.000001f:-1.f;}
 for(size_t t=0;t<MaxT;++t)for(size_t h=0;h<HV;++h){ab.data()[t*2*HV+h]=values[h%16];ab.data()[t*2*HV+HV+h]=float(int((t+h)%17)-8)*0.25f;}
 std::vector<uint32_t> immutable(2*N);for(size_t i=0;i<2*N;++i)immutable[i]=bits(ab.data()[i]);
 const std::array<size_t,15> lengths{0,1,3,5,15,16,17,127,128,129,8191,8192,3,8192,8192};uint64_t compared=0,cases=0,full_hash=0,full_repeats=0;bool full_seen=false;
 for(size_t T:lengths){require(T<=MaxT,"length bound");size_t active=T*HV;
  for(Buffer* b:{&bl,&bb,&cl,&cb,&cf,&scalar,&sg16,&sg32})b->reset();
  std::cout<<"CASE,"<<cases<<",T,"<<T<<",specials,"<<specials<<'\n';
  stage(q,errors,"baseline_producer",[&]{producer(q,T,ab.data(),dt.data(),a.data(),bl.data(),bb.data(),nullptr);});
  stage(q,errors,"candidate_producer",[&]{producer(q,T,ab.data(),dt.data(),a.data(),cl.data(),cb.data(),cf.data());});
  stage(q,errors,"scalar_reader",[&]{if(active)q.parallel_for<ScalarReader>(sycl::range<1>(active),[=,log=bl.data(),out=scalar.data()](sycl::id<1> i){out[i[0]]=sycl::native::exp(log[i[0]]);});});
  stage(q,errors,"SG16_reader",[&]{group_reader<16>(q,active,bl.data(),sg16.data());});
  stage(q,errors,"SG32_reader",[&]{group_reader<32>(q,active,bl.data(),sg32.data());});
  for(size_t i=0;i<active;++i){
   const std::array<std::pair<const char*,std::pair<float,float>>,5> checks{{{"loggate",{bl.data()[i],cl.data()[i]}},{"beta",{bb.data()[i],cb.data()[i]}},{"factor_scalar",{scalar.data()[i],cf.data()[i]}},{"factor_SG16",{sg16.data()[i],cf.data()[i]}},{"factor_SG32",{sg32.data()[i],cf.data()[i]}}}};
   for(const auto& c:checks){++compared;if(bits(c.second.first)!=bits(c.second.second)){std::cerr<<"FIRST_DIFFERENCE,"<<cases<<','<<T<<','<<i<<','<<c.first<<','<<bits(c.second.first)<<','<<bits(c.second.second)<<','<<domain(i,specials)<<'\n';throw std::runtime_error("bitwise mismatch after comparisons="+std::to_string(compared)+" completed_cases="+std::to_string(cases));}}
  }
  if(T==MaxT){++full_repeats;uint64_t hash=14695981039346656037ULL;for(size_t i=0;i<active;++i)for(Buffer* b:{&bl,&bb,&cl,&cb,&cf,&scalar,&sg16,&sg32}){uint32_t word=bits(b->data()[i]);for(unsigned k=0;k<4;++k){hash^=(word>>(8*k))&255U;hash*=1099511628211ULL;}}
   if(full_seen)require(hash==full_hash,"large-prefix repeat/reset hash changed");else{full_hash=hash;full_seen=true;}
   std::cout<<"FULL_REPEAT_HASH,"<<hash<<'\n';
  }
  for(Buffer* b:{&bl,&bb,&cl,&cb,&cf,&scalar,&sg16,&sg32})b->guards(active);
  ab.guards(2*N);dt.guards(HV);a.guards(HV);
  for(size_t i=0;i<2*N;++i)require(bits(ab.data()[i])==immutable[i],"input changed");
  for(size_t h=0;h<HV;++h){require(bits(dt.data()[h])==bits(0.f),"dt changed");require(bits(a.data()[h])==bits(h%16==3?-0.000001f:-1.f),"ssm_a changed");}
  ++cases;std::cout<<"CASE_PASS,"<<cases<<','<<active<<'\n';std::cout.flush();require(bool(std::cout),"output failure");
 }
 require(cases==15&&compared==7970640&&full_seen&&full_repeats==3,"terminal count mismatch");
 std::cout<<"TERMINAL,pass,cases,"<<cases<<",comparisons,"<<compared<<",necessary_screen_only,no_state_no_norm_no_model_no_timing\n";
 std::cout.flush();std::cerr.flush();require(bool(std::cout)&&bool(std::cerr),"final output failure");return 0;
 }catch(const std::exception& e){std::cerr<<"TERMINAL,fail,"<<e.what()<<'\n';std::cerr.flush();return 1;}}
