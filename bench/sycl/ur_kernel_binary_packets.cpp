// Main-authored host-only adversarial packets. Repository LICENSE.
// Stub every UR/Level Zero entrypoint; do not link either native runtime.
#include "ur_kernel_binary_subscriber.cpp"

namespace {
unsigned queries = 0;
const char* mode() { const char* p = std::getenv("STRATA_NATIVE_PACKET_MODE"); return p ? p : ""; }
bool is(const char* p) { return !std::strcmp(mode(), p); }
template<class T> T handle(uintptr_t value) { return reinterpret_cast<T>(value); }
template<class T> ur_result_t answer(size_t size, void* out, T value) {
  if (!out || size != sizeof(value)) return UR_RESULT_ERROR_INVALID_SIZE;
  std::memcpy(out, &value, size); return UR_RESULT_SUCCESS;
}
void unlocked() {
  require(pthread_mutex_trylock(&state_mutex) == 0, "state mutex held across native query");
  require(pthread_mutex_unlock(&state_mutex) == 0, "host state unlock");
}
void send(ur_function_t id, const char* name, void* args, ur_result_t result = UR_RESULT_SUCCESS) {
  xpti::function_with_args_t packet{uint32_t(id), name, args, &result, nullptr};
  callback(xpti::trace_function_with_args_end, nullptr, nullptr, 100 + queries, &packet);
}
void create(unsigned arm, uintptr_t address) {
  ur_kernel_handle_t kernel = handle<ur_kernel_handle_t>(address);
  auto* result = &kernel;
  const char* name = arm < 3 ? arms[arm].name : "_ZTSunselected";
  ur_kernel_create_params_t p{};p.ppKernelName = &name;p.pphKernel = &result;
  send(UR_FUNCTION_KERNEL_CREATE, is("wrong-api") ? "wrongAPI" : "urKernelCreate", &p);
}
void enqueue(unsigned arm, bool old_path = false) {
  ur_queue_handle_t queue = handle<ur_queue_handle_t>(0x1000);
  auto kernel = handle<ur_kernel_handle_t>(0x100 + arm);
  uint32_t dimension = 1;
  size_t global = is("bad-geometry") ? 127 : 256, local = 128;
  const size_t* global_ptr = &global;
  const size_t* local_ptr = &local;
  const auto status = is("failed-launch") ? UR_RESULT_ERROR_INVALID_VALUE : UR_RESULT_SUCCESS;
  if (old_path) {
    ur_enqueue_kernel_launch_params_t p{};p.phQueue=&queue;p.phKernel=&kernel;p.pworkDim=&dimension;
    p.ppGlobalWorkSize=&global_ptr;p.ppLocalWorkSize=&local_ptr;
    send(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH,"urEnqueueKernelLaunch",&p,status);
  } else {
    ur_enqueue_kernel_launch_with_args_exp_params_t p{};p.phQueue=&queue;p.phKernel=&kernel;p.pworkDim=&dimension;
    p.ppGlobalWorkSize=&global_ptr;p.ppLocalWorkSize=&local_ptr;
    send(UR_FUNCTION_ENQUEUE_KERNEL_LAUNCH_WITH_ARGS_EXP,"urEnqueueKernelLaunchWithArgsExp",&p,status);
  }
}
} // namespace

extern "C" ur_result_t UR_APICALL urQueueGetInfo(ur_queue_handle_t, ur_queue_info_t key,
    size_t n, void* out, size_t*) {
  require(key==UR_QUEUE_INFO_DEVICE,"mock queue property");
  return answer(n,out,handle<ur_device_handle_t>(0x10));
}
extern "C" ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t, ur_kernel_info_t key,
    size_t n, void* out, size_t*) {
  require(key==UR_KERNEL_INFO_PROGRAM,"mock kernel property");
  return answer(n,out,handle<ur_program_handle_t>(0x20));
}
extern "C" ur_result_t UR_APICALL urProgramGetInfo(ur_program_handle_t, ur_program_info_t key,
    size_t n, void* out, size_t*) {
  if(key==UR_PROGRAM_INFO_NUM_DEVICES)return answer(n,out,uint32_t(is("multi-device")?2:1));
  require(key==UR_PROGRAM_INFO_DEVICES,"mock program property");
  return answer(n,out,handle<ur_device_handle_t>(is("other-device")?0x11:0x10));
}
extern "C" ur_result_t UR_APICALL urDeviceGetInfo(ur_device_handle_t, ur_device_info_t key,
    size_t n, void* out, size_t*) {
  require(key==UR_DEVICE_INFO_PLATFORM,"mock device property");
  return answer(n,out,handle<ur_platform_handle_t>(0x30));
}
extern "C" ur_result_t UR_APICALL urPlatformGetInfo(ur_platform_handle_t, ur_platform_info_t key,
    size_t n, void* out, size_t*) {
  require(key==UR_PLATFORM_INFO_BACKEND,"mock platform property");
  return answer(n,out,is("wrong-backend")?UR_BACKEND_OPENCL:UR_BACKEND_LEVEL_ZERO);
}
extern "C" ur_result_t UR_APICALL urDeviceGetNativeHandle(ur_device_handle_t, ur_native_handle_t* out) {
  *out=0x40;return UR_RESULT_SUCCESS;
}
extern "C" ur_result_t UR_APICALL urPlatformGetNativeHandle(ur_platform_handle_t, ur_native_handle_t* out) {
  *out=0x50;return UR_RESULT_SUCCESS;
}
extern "C" ur_result_t UR_APICALL urProgramGetNativeHandle(ur_program_handle_t, ur_native_handle_t* out) {
  unlocked();*out=0x60;return UR_RESULT_SUCCESS;
}
extern "C" ur_result_t UR_APICALL urKernelGetNativeHandle(ur_kernel_handle_t kernel, ur_native_handle_t* out) {
  unlocked();++queries;
  // A recursive notification must return before locking or inspecting raw.
  callback(xpti::trace_function_with_args_end,nullptr,nullptr,999,nullptr);
  *out=uintptr_t(kernel)+0x100+(is("native-change")&&queries>3?0x100:0);
  return UR_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeDeviceGetProperties(ze_device_handle_t, ze_device_properties_t* p) {
  p->type=ZE_DEVICE_TYPE_GPU;p->vendorId=0x8086;p->deviceId=is("wrong-gpu")?0:0xe20c;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeDriverGetExtensionProperties(ze_driver_handle_t,
    uint32_t* count, ze_driver_extension_properties_t* p) {
  if(!p){*count=is("extension-overflow")?257:1;return ZE_RESULT_SUCCESS;}
  require(*count==1,"mock extension capacity");
  std::strcpy(p[0].name,is("no-extension")?"different_extension":ZE_GET_KERNEL_BINARY_EXP_NAME);
  p[0].version=is("old-extension")?0:ZE_KERNEL_GET_BINARY_EXP_VERSION_1_0;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeKernelGetName(ze_kernel_handle_t kernel,size_t* n,char* out) {
  const uintptr_t index=uintptr_t(kernel)-0x200;require(index<3,"mock kernel address");
  const char* name=is("wrong-name")?"_ZTSwrong":arms[index].name;
  require(out && *n>std::strlen(name),"mock name buffer");
  std::strcpy(out,name);*n=std::strlen(name)+1;return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeKernelGetProperties(ze_kernel_handle_t,ze_kernel_properties_t* p) {
  p->numKernelArgs=is("wrong-arguments")?17:18;p->requiredSubgroupSize=16;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeKernelGetBinaryExp(ze_kernel_handle_t,size_t* n,uint8_t* out) {
  unlocked();
  if(is("unsupported"))return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
  if(!out){*n=is("oversize")?binary_limit+1:(is("empty-binary")?0:64);return ZE_RESULT_SUCCESS;}
  require(*n==64,"mock binary capacity");
  for(unsigned i=0;i<64;++i)out[i]=uint8_t(i);
  if(is("size-change"))*n=63;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL zeModuleGetNativeBinary(ze_module_handle_t,size_t* n,uint8_t* out) {
  unlocked();
  if(!out){*n=128;return ZE_RESULT_SUCCESS;}
  require(*n==128,"mock module capacity");std::memset(out,0,128);
  std::memcpy(out,"\x7f" "ELF",4);return ZE_RESULT_SUCCESS;
}

int main() {
  require(*mode(),"host packet mode required");
  if(is("null-packet"))callback(xpti::trace_function_with_args_end,nullptr,nullptr,0,nullptr);
  if(is("ignored")){
    send(UR_FUNCTION_LOADER_TEAR_DOWN,"urLoaderTearDown",nullptr);
    create(3,0x103);enqueue(3);require(queries==0,"unselected query");return 0;
  }
  for(unsigned arm=0;arm<3;++arm)create(arm,0x100+arm);
  if(is("recreated"))create(0,0x777);
  for(unsigned arm=0;arm<3;++arm)enqueue(arm);
  if(is("failed-launch")){require(queries==0,"failed launch queried");return 0;}
  for(unsigned arm=0;arm<3;++arm)enqueue(arm,true);
  require(queries==6,"exact successful native query count");
  for(const auto& arm:arms)require(arm.captured && arm.launches==2 && !arm.extracting,
      "final host capture census");
  std::puts("host_packets=PASS native_runtime_linked=0 GPU_enqueues=0");
}
