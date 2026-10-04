#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
void check(ze_result_t r){ if(r!=ZE_RESULT_SUCCESS){std::printf("error=%x\n",unsigned(r));throw std::runtime_error("native query failed");}}
int main(){
 sycl::device d{sycl::gpu_selector_v}; sycl::queue q(d);
 auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
 ze_device_usablemem_size_ext_properties_t usable{ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES};
 ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,&usable};
 check(zeDeviceGetProperties(native,&props));
 std::printf("device=%s sycltotal=%llu usable=%llu\n",props.name,(unsigned long long)d.get_info<sycl::info::device::global_mem_size>(),(unsigned long long)usable.currUsableMemSize);
 check(zesInit(0)); uint32_t count=0;check(zesDriverGet(&count,nullptr));std::vector<zes_driver_handle_t> drivers(count);check(zesDriverGet(&count,drivers.data()));
 for(auto driver:drivers){zes_uuid_t uuid{};std::memcpy(uuid.id,props.uuid.id,sizeof(uuid.id));zes_device_handle_t sd=nullptr;ze_bool_t sub=0;uint32_t id=0;auto r=zesDriverGetDeviceByUuidExp(driver,uuid,&sd,&sub,&id);std::printf("match=%x sub=%u\n",unsigned(r),sub);if(r!=ZE_RESULT_SUCCESS)continue;
 uint32_t n=0;check(zesDeviceEnumMemoryModules(sd,&n,nullptr));std::vector<zes_mem_handle_t> modules(n);check(zesDeviceEnumMemoryModules(sd,&n,modules.data()));
 auto report=[&]{for(auto m:modules){zes_mem_properties_t p{ZES_STRUCTURE_TYPE_MEM_PROPERTIES};zes_mem_state_t s{ZES_STRUCTURE_TYPE_MEM_STATE};check(zesMemoryGetProperties(m,&p));check(zesMemoryGetState(m,&s));std::printf("location=%u sub=%u physical=%llu total=%llu free=%llu\n",p.location,p.onSubdevice,(unsigned long long)p.physicalSize,(unsigned long long)s.size,(unsigned long long)s.free);}};
 report();auto* p=sycl::malloc_device(64*1024*1024,q);q.memset(p,1,64*1024*1024).wait_and_throw();report();sycl::free(p,q);report();}
}
