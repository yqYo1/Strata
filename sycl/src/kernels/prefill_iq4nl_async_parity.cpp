// Completed host-origin async fault only; no device-fault/recovery/model claims.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include "../prefill/iq4nl_dequant.hpp"
#include "../prefill/iq4nl_async_boundary.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>
using namespace strata::prefill::detail;
namespace {
void require(bool b,const char* m) {if(!b)throw std::runtime_error(m);}
void progress(const char* s) {std::printf("STAGE,%s\n",s);if(std::fflush(stdout))std::_Exit(2);}
void identity(sycl::queue& q) {
 require(q.is_in_order()&&q.get_backend()==sycl::backend::ext_oneapi_level_zero,"LevelZero in-order required");
 auto d=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
 ze_device_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
 require(zeDeviceGetProperties(d,&p)==ZE_RESULT_SUCCESS,"identity query");
 require(p.vendorId==0x8086&&p.deviceId==0xe20c&&!(p.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),"B570 root required");
 ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
 require(zeDevicePciGetPropertiesExt(d,&pci)==ZE_RESULT_SUCCESS,"PCI query");
 require(pci.address.domain==0&&pci.address.bus==5&&pci.address.device==0&&pci.address.function==0,"PCI0000:05:00.0 required");
 require(q.get_device().has(sycl::aspect::fp16),"fp16 required");
 progress("IDENTITY_8086_e20c_0000:05:00.0_root");
}
void run(bool fault) {
 // Production queue construction and its rethrowing dpct async handler.
 auto* qp=dpct::get_current_device().create_queue(true);auto& q=*qp;
 identity(q); // before USM allocation
 std::atomic<unsigned> marker{0}; // owned until queue destruction
 std::array<unsigned char,160> input{}; // eight18-byte IQ4NL blocks with8-byte guards
 input.fill(0x6d);
 for(unsigned block=0;block<8;++block) {
  auto* packed=input.data()+8+18*block;
  packed[0]=0;packed[1]=0x3c; // exact positive unit scale
  std::memset(packed+2,0,16); // code0 is exactly-127
 }
 // Independent literal binary16 encoding of-127 is0xd7f0.
 std::array<uint16_t,288> output{};output.fill(0x55aa);
 auto* src=sycl::malloc_device<unsigned char>(input.size(),q);
 auto* dst=sycl::malloc_device<uint16_t>(output.size(),q);
 require(src&&dst,"allocation");
 bool completed=false;std::string err;unsigned calls=0,callbacks=0,retries=0;
 try {
  progress("upload_input");q.memcpy(src,input.data(),input.size());
  progress("upload_output");q.memcpy(dst,output.data(),sizeof(output));
  progress("private_submit");strata::kernels::iq_dequant_f16_prefill_iq4nl(20,src+8,256,dst+16,&q);++calls;
  if(fault) {progress("host_task_submit");q.submit([&](sycl::handler& h){h.host_task([&marker]{marker.fetch_add(1);throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),"IQ4NL_HOST_TASK_ONE_SHOT");});});}
  progress("existing_completion_wait");try {q.wait();completed=true;}catch(...){iq4nl_unknown_completion("fixture_wait");}
  const bool ok=iq4nl_completed_boundary(q,err);
  if(ok)++callbacks;
  require(ok==!fault,"boundary result");require(callbacks==unsigned(!fault)&&retries==0&&calls==1,"counts");
  require(marker.load()==unsigned(fault),"one-shot marker");
  if(fault)require(err.find("IQ4NL_HOST_TASK_ONE_SHOT")!=std::string::npos,"concrete error");
  progress("later_drain");completed=false;q.wait_and_throw();completed=true; // same injected error must already be consumed
  progress("readback");completed=false;q.memcpy(output.data(),dst,sizeof(output));q.wait();completed=true;q.throw_asynchronous();
  for(unsigned i=0;i<output.size();++i)require(output[i]==(i>=16&&i<272?0xd7f0:0x55aa),"output or canary");
  progress("input_readback");completed=false;std::array<unsigned char,160> check{};q.memcpy(check.data(),src,check.size());q.wait();completed=true;q.throw_asynchronous();require(check==input,"input changed");
  std::printf("CASE,%s,completed_wait,1,boundary_ok,%u,callback,%u,generic_retry,%u,private_calls,%u,host_task,%u,later_drain,normal\n",fault?"fault":"control",unsigned(ok),callbacks,retries,calls,marker.load());std::fflush(stdout);
 }catch(...){if(!completed)iq4nl_unknown_completion("fixture_submission");throw;}
 progress("free_buffers");sycl::free(src,q);sycl::free(dst,q);
 progress("queue_teardown");dpct::get_current_device().destroy_queue(qp);
}
}
int main(int argc,char** argv) {
 if(argc!=2||std::strcmp(argv[1],"--run")!=0){std::fprintf(stderr,"usage: prefill_iq4nl_async_parity --run\n");return 2;}
 try {run(false);run(true);std::puts("TERMINAL,pass,host_origin_async_boundary_only,2_cases,production_unintegrated,device_fault_unqualified,adoption_false");return 0;}
 catch(const std::exception& e){std::fprintf(stderr,"FAIL,%s\n",e.what());return 1;}
}
