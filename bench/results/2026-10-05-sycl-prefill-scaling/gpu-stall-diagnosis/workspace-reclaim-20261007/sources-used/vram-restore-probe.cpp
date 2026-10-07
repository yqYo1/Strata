// Finite external ownership is required. Allocation-pressure test, no model.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vm=sycl::ext::oneapi::experimental;
constexpr size_t MiB=1024*1024, segment=8*MiB;

void memory(const char* stage) {
    std::printf("stage %s\n",stage);
    for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fdinfo")) {
        std::ifstream file(entry.path());
        std::string text((std::istreambuf_iterator<char>(file)),{});
        if (text.find("drm-pdev:\t0000:05:00.0\n")==std::string::npos) continue;
        for (const auto* key : {"drm-total-vram0:","drm-resident-vram0:","drm-total-gtt:"}) {
            const auto at=text.find(key);
            if (at!=std::string::npos) std::printf("%s\n",text.substr(at,text.find('\n',at)-at).c_str());
        }
    }
    std::fflush(stdout);
}

struct Physical {
    sycl::queue &queue;
    uintptr_t address=0;
    size_t bytes;
    std::vector<std::unique_ptr<vm::physical_mem>> blocks;
    explicit Physical(sycl::queue &q,size_t size):queue(q),bytes(size) {
        if (size%vm::get_mem_granularity(q.get_context(),vm::granularity_mode::minimum) ||
            segment%vm::get_mem_granularity(q.get_device(),q.get_context()))
            throw std::runtime_error("VMM granularity does not divide fixed sizes");
        address=vm::reserve_virtual_mem(bytes,q.get_context());
    }
    void release() {
        queue.wait_and_throw();
        while (!blocks.empty()) {
            vm::unmap(reinterpret_cast<void*>(address+(blocks.size()-1)*segment),segment,queue.get_context());
            blocks.pop_back();
        }
    }
    bool grow() {
        while (blocks.size()<bytes/segment) {
            try {
                auto block=std::make_unique<vm::physical_mem>(queue,segment);
                const auto expected=address+blocks.size()*segment;
                if (block->map(expected,segment,vm::address_access_mode::read_write)!=reinterpret_cast<void*>(expected))
                    throw std::runtime_error("physical map address mismatch");
                blocks.push_back(std::move(block));
            } catch (const sycl::exception &error) {
                std::printf("physical allocation failure after %zu bytes: %s\n",blocks.size()*segment,error.what());
                std::fflush(stdout);
                return false;
            }
        }
        return true;
    }
    void check() {
        for (size_t i=0;i<blocks.size();++i) {
            const uint32_t input=uint32_t(i)*1664525u+1013904223u;
            uint32_t output=0;
            auto *pointer=reinterpret_cast<void*>(address+i*segment);
            queue.memcpy(pointer,&input,sizeof input).wait_and_throw();
            queue.memcpy(&output,pointer,sizeof output).wait_and_throw();
            if (output!=input) throw std::runtime_error("restored virtual word mismatch");
        }
    }
    ~Physical() {
        release();
        if (address) vm::free_virtual_mem(address,bytes,queue.get_context());
    }
};

int main(int argc,char** argv) try {
    if (argc!=2) throw std::runtime_error("pass either single or split for the temporary allocation");
    const bool split=std::string(argv[1])=="split";
    if (!split && std::string(argv[1])!="single") throw std::runtime_error("invalid temporary allocation mode");
    std::optional<sycl::device> device;
    for (const auto &candidate : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (candidate.get_backend()==sycl::backend::ext_oneapi_level_zero &&
            candidate.get_info<sycl::ext::intel::info::device::pci_address>()=="0000:05:00.0") device=candidate;
    if (!device || !device->has(sycl::aspect::ext_oneapi_virtual_mem))
        throw std::runtime_error("B570 virtual memory device unavailable");
    sycl::queue queue(*device,[](sycl::exception_list errors){for(auto error:errors)std::rethrow_exception(error);},
                      sycl::property::queue::in_order{});
    std::printf("device 0000:05:00.0: %s\n",device->get_info<sycl::info::device::name>().c_str());
    memory("empty");
    std::vector<void*> anchors,temporary;
    auto allocate=[&](size_t size,std::vector<void*> &list) {
        auto *p=sycl::malloc_device(size,queue);
        if (!p) throw std::runtime_error("USM allocation returned null");
        list.push_back(p);
    };
    for (unsigned i=0;i<138;++i) allocate(64*MiB,anchors); // 8,832 MiB retained.
    memory("anchors-ready");
    bool restored=false;
    {
        Physical physical(queue,1280*MiB); // Main-cache plus draft-weight budget.
        if (!physical.grow()) throw std::runtime_error("initial physical budget does not fit");
        physical.check();memory("initial-physical-ready");
        physical.release();memory("physical-released");
        if (split) {
            for (unsigned i=0;i<20;++i) allocate(64*MiB,temporary);
            allocate(20*MiB,temporary);
        } else allocate(1300*MiB,temporary); // Exact full-layer temporary cache size.
        memory("temporary-ready");queue.wait_and_throw();
        for (void *p:temporary) sycl::free(p,queue);
        temporary.clear();memory("temporary-freed");
        restored=physical.grow();memory("physical-restored-or-failed");
        if (restored) physical.check();
    }
    memory("physical-cleaned");
    queue.wait_and_throw();
    for (void *p:anchors) sycl::free(p,queue);
    memory("anchors-freed");
    std::printf("{\"restored\":%s,\"anchors_bytes\":%zu,\"physical_bytes\":%zu,\"temporary_bytes\":%zu,\"split\":%s}\n",
                restored?"true":"false",8832*MiB,1280*MiB,1300*MiB,split?"true":"false");
    // Failed restoration is an observed outcome, with orderly cleanup.
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr,"probe error: %s\n",error.what());
    return 2;
}
