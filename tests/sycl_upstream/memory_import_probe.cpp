// Opt-in driver qualification. A retained mapping deliberately returns failure.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
int main() try {
    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
    const size_t page=size_t(sysconf(_SC_PAGESIZE)), bytes=page*2;
    void* storage=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    require(storage!=MAP_FAILED,"mmap");
    std::cout<<"driver="<<queue.get_device().get_info<sycl::info::device::driver_version>()<<'\n';
    for (int run=0;run<2;++run) {
        ze_external_memmap_sysmem_ext_desc_t ext{ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC,nullptr,storage,bytes};
        ze_host_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,&ext,0};
        void* imported=nullptr;
        require(zeMemAllocHost(context,&desc,bytes,page,&imported)==ZE_RESULT_SUCCESS,"native import");
        require(imported==storage,"native import changed pointer");
        ze_memory_allocation_properties_t before{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
        require(zeMemGetAllocProperties(context,storage,&before,nullptr)==ZE_RESULT_SUCCESS,"query import");
        auto* p=static_cast<uint32_t*>(storage);
        for(size_t i=0;i<bytes/4;++i)p[i]=uint32_t(run*100+i);
        queue.parallel_for(sycl::range<1>(bytes/4),[=](sycl::id<1> i){p[i]+=7;}).wait_and_throw();
        for(size_t i=0;i<bytes/4;++i)require(p[i]==run*100+i+7,"native mapped kernel result");
        auto release=zeMemFree(context,storage);
        ze_memory_allocation_properties_t after{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
        require(zeMemGetAllocProperties(context,storage,&after,nullptr)==ZE_RESULT_SUCCESS,"query release");
        std::cout<<"run="<<run<<" imported_type="<<before.type<<" release_result="<<release
                 <<" released_type="<<after.type<<" exact_values="<<bytes/4<<'\n';
        // A buggy driver still owns this mapping. Keep its system storage valid
        // until process exit instead of unmapping a still-imported allocation.
        if(release!=ZE_RESULT_SUCCESS || after.type!=ZE_MEMORY_TYPE_UNKNOWN) return 1;
    }
    munmap(storage,bytes);
    std::cout<<"PASS native_import_release_and_reimport=1\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
