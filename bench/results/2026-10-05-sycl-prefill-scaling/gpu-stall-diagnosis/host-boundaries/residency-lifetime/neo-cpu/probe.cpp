
#include <atomic>
#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
using ze_result_t=int;
using ze_physical_mem_handle_t=void*;
constexpr int ZE_RESULT_SUCCESS=0;
namespace NEO {
using TaskCountType=uint32_t;
struct CommandStreamReceiver;
struct GraphicsAllocation {
    static constexpr uint32_t objectNotUsed=0, objectNotResident=~0u;
    GraphicsAllocation* parentAllocation=nullptr;
    struct Usage { uint32_t taskCount=0; };
    std::vector<Usage> usageInfos=std::vector<Usage>(1);
    std::atomic<uint32_t> hostPtrTaskCountAssignment{1};
    std::atomic<uint64_t> residencyContainerStamp{0};
    // The simulated virtual address is deliberately identical for all objects.
    uintptr_t gpuAddress=0x100000;
#include "allocation-methods.inc"
    void updateTaskCount(uint32_t n,uint32_t i) { usageInfos.at(i).taskCount=n; }
    void prepareHostPtrForResidency(CommandStreamReceiver*);
};
using ResidencyContainer=std::vector<GraphicsAllocation*>;
struct CommandStreamReceiver {
    struct OsContext { uint32_t getContextId() { return 0; } } context;
    OsContext& getOsContext() { return context; }
    uint32_t peekTaskCount() { return 2; }
    void makeResident(GraphicsAllocation& a) { assert(a.gpuAddress==0x100000); }
};
struct CommandContainer {
    uint64_t residencyContainerStamp;
    ResidencyContainer residencyContainer;
    explicit CommandContainer(uint64_t n):residencyContainerStamp(n) {}
    void addToResidencyContainer(GraphicsAllocation*);
};
struct PhysicalMemoryAllocation { GraphicsAllocation* allocation; };
struct MemoryManager {
    std::mutex mutex;
    std::map<void*,PhysicalMemoryAllocation*> allocations;
    auto lockPhysicalMemoryAllocationMap() { return std::unique_lock(mutex); }
    auto& getPhysicalMemoryAllocationMap() { return allocations; }
    void freeGraphicsMemoryImpl(GraphicsAllocation* a) { delete a; }
};
}
namespace L0 {
struct Driver {
    NEO::MemoryManager manager;
    NEO::MemoryManager* getMemoryManager() { return &manager; }
};
struct Context {
    Driver* driverHandle;
    ze_result_t destroyPhysicalMem(ze_physical_mem_handle_t);
};
struct CommandQueue {
    NEO::CommandStreamReceiver* csr;
    void makeResidentForResidencyContainer(const NEO::ResidencyContainer&);
};
}
#include "backend-functions.inc"
struct Kernel {
    struct { NEO::ResidencyContainer argumentsResidencyContainer{nullptr}; } privateState;
    void setArg(NEO::GraphicsAllocation* allocation) {
        constexpr uint32_t argIndex=0;
#include "kernel-assignment.inc"
    }
    const auto& getArgumentsResidencyContainer() { return privateState.argumentsResidencyContainer; }
};
struct CommandList {
    NEO::CommandContainer commandContainer;
    explicit CommandList(uint64_t n):commandContainer(n) {}
#include "add-residency.inc"
    void record(Kernel* kernel) {
#include "record-argument-residency.inc"
    }
};
int main(int argc,char** argv) {
    assert(argc==2); std::string mode=argv[1];
    bool retire=mode=="retire-and-recreate", keepMemory=mode=="keep-physical-memory";
    assert(retire || keepMemory || mode=="retain-old-list" || mode=="update-kernel-only");
    L0::Driver driver; L0::Context context{&driver};
    NEO::CommandStreamReceiver csr; L0::CommandQueue queue{&csr};
    auto allocate=[&] {
        auto* node=new NEO::PhysicalMemoryAllocation{new NEO::GraphicsAllocation};
        driver.manager.allocations.emplace(node,node); return node;
    };
    auto* physical=allocate(); Kernel kernel; kernel.setArg(physical->allocation);
    auto list=std::make_unique<CommandList>(1); list->record(&kernel);
    queue.makeResidentForResidencyContainer(list->commandContainer.residencyContainer);
    for(uint64_t cycle=2;cycle<=4;++cycle) {
        if(retire) list.reset();
        auto oldAddress=physical->allocation->gpuAddress;
        if(!keepMemory) {
            assert(context.destroyPhysicalMem(physical)==ZE_RESULT_SUCCESS);
            physical=allocate(); assert(physical->allocation->gpuAddress==oldAddress);
        }
        // Refreshing the kernel does not change an already recorded list.
        if(retire || mode=="update-kernel-only") kernel.setArg(physical->allocation);
        if(retire) { list=std::make_unique<CommandList>(cycle); list->record(&kernel); }
        std::printf("replay %llu, same simulated VA\n",(unsigned long long)cycle);
        std::fflush(stdout);
        queue.makeResidentForResidencyContainer(list->commandContainer.residencyContainer);
    }
    list.reset(); assert(context.destroyPhysicalMem(physical)==ZE_RESULT_SUCCESS);
    assert(driver.manager.allocations.empty());
    std::printf("PASS %s\n",mode.c_str());
}
