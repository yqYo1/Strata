// Main-authored; repository LICENSE. Metadata only, no inference or submission.
// Native-handle contracts: installed oneAPI2026.2 Level Zero backend/header.
#include "strata/sycl/runtime.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace {
void gate(const char* name) {
    std::printf("{\"kind\":\"gate\",\"name\":\"%s\"}\n", name);
    char input[16];
    if (!std::fgets(input,sizeof(input),stdin) || std::strcmp(input,"continue\n"))
        throw std::runtime_error("queue metadata gate refused");
}
void checked(ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("Level Zero metadata query failed");
}
void observe(const char* name, const sycl::queue& queue) {
    std::fprintf(stderr,"queue metadata: before native handle query for %s\n",name);
    const auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
    uint32_t ordinal=UINT32_MAX,index=UINT32_MAX;
    const bool immediate=std::holds_alternative<ze_command_list_handle_t>(native);
    if (immediate) {
        const auto list=std::get<ze_command_list_handle_t>(native);
        checked(zeCommandListGetOrdinal(list,&ordinal));
        checked(zeCommandListImmediateGetIndex(list,&index));
    } else {
        const auto handle=std::get<ze_command_queue_handle_t>(native);
        checked(zeCommandQueueGetOrdinal(handle,&ordinal));
        checked(zeCommandQueueGetIndex(handle,&index));
    }
    std::printf("{\"kind\":\"queue\",\"name\":\"%s\",\"ordinal\":%u,\"index\":%u,"
                "\"immediate\":%s,\"in_order\":%s,\"profiling\":%s}\n",name,ordinal,index,
        immediate?"true":"false",queue.is_in_order()?"true":"false",
        queue.has_property<sycl::property::queue::enable_profiling>()?"true":"false");
}
}
int main() {
    std::setvbuf(stdout,nullptr,_IOLBF,0); std::setvbuf(stderr,nullptr,_IOLBF,0);
    try {
        gate("before-runtime");
        std::fprintf(stderr,"queue metadata: before production Runtime construction\n");
        auto runtime=strata::sycl_backend::runtime_for();
        auto profiled=runtime->make_queue(true);
        auto ordinary=runtime->make_queue(false);
        const auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(runtime->device());
        ze_pci_ext_properties_t pci{}; pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        checked(zeDevicePciGetPropertiesExt(device,&pci));
        if (pci.address.domain || pci.address.bus!=5 || pci.address.device || pci.address.function)
            throw std::runtime_error("expected B570 PCI0000:05:00.0");
        gate("runtime-ready");
        observe("compute",runtime->compute()); observe("transfer",runtime->transfer());
        observe("make_queue_profiled",profiled); observe("make_queue_unprofiled",ordinary);
        runtime->check();
        gate("before-exit");
        std::puts("{\"kind\":\"done\",\"application_work_submitted\":false}");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"queue metadata failure: %s\n",error.what()); return 2;
    }
}
