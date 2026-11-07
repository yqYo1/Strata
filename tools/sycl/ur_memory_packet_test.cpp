// Main-authored CPU-only ABI packet check; repository LICENSE applies.
// It submits no UR operation and does not initialize the loader or a device.
#include "ur_memory_subscriber.cpp"

int main() {
    ur_result_t ok=UR_RESULT_SUCCESS, failed=UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    auto emit=[&](ur_function_t id,const char* name,void* args,ur_result_t& result) {
        xpti::function_with_args_t call{static_cast<uint32_t>(id),name,args,&result,nullptr};
        callback(static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end),nullptr,nullptr,17,&call);
    };
    void* pointer=reinterpret_cast<void*>(0x1000);void** output=&pointer;size_t size=256;
    ur_usm_device_alloc_params_t device{};device.psize=&size;device.pppMem=&output;
    ur_usm_host_alloc_params_t host{};host.psize=&size;host.pppMem=&output;
    ur_usm_shared_alloc_params_t shared{};shared.psize=&size;shared.pppMem=&output;
    emit(UR_FUNCTION_USM_DEVICE_ALLOC,"urUSMDeviceAlloc",&device,ok);
    emit(UR_FUNCTION_USM_HOST_ALLOC,"urUSMHostAlloc",&host,ok);
    emit(UR_FUNCTION_USM_SHARED_ALLOC,"urUSMSharedAlloc",&shared,ok);
    // If a failed output is accidentally dereferenced, this packet faults.
    device.pppMem=reinterpret_cast<void***>(1);
    emit(UR_FUNCTION_USM_DEVICE_ALLOC,"urUSMDeviceAlloc",&device,failed);
    ur_usm_free_params_t free{};free.ppMem=&pointer;
    emit(UR_FUNCTION_USM_FREE,"urUSMFree",&free,ok);
    ur_mem_handle_t buffer=reinterpret_cast<ur_mem_handle_t>(0x2000),alias=reinterpret_cast<ur_mem_handle_t>(0x3000);
    ur_mem_handle_t* buffer_out=&buffer;ur_mem_handle_t* alias_out=&alias;size=512;
    ur_mem_buffer_create_params_t create{};create.psize=&size;create.pphBuffer=&buffer_out;
    emit(UR_FUNCTION_MEM_BUFFER_CREATE,"urMemBufferCreate",&create,ok);
    ur_mem_retain_params_t retain{};retain.phMem=&buffer;
    emit(UR_FUNCTION_MEM_RETAIN,"urMemRetain",&retain,ok);
    ur_mem_buffer_partition_params_t part{};part.phBuffer=&buffer;part.pphMem=&alias_out;
    emit(UR_FUNCTION_MEM_BUFFER_PARTITION,"urMemBufferPartition",&part,ok);
    ur_mem_release_params_t release{};release.phMem=&alias;
    emit(UR_FUNCTION_MEM_RELEASE,"urMemRelease",&release,ok);
    release.phMem=&buffer;emit(UR_FUNCTION_MEM_RELEASE,"urMemRelease",&release,ok);
    emit(static_cast<ur_function_t>(0),"urUSMFutureAllocation",&size,ok);
    emit(UR_FUNCTION_USM_DEVICE_ALLOC,"urUSMDeviceAlloc",nullptr,ok);
    return records==12 && invalid==1 && unsupported==1 && lost==0 && active==0 &&
           pointer==reinterpret_cast<void*>(0x1000) && size==512 ? 0 : 1;
}
