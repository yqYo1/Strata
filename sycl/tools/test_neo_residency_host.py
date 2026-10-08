"""Exercise installed-tag NEO residency fragments with CPU stand-ins.

This is a controlled lifetime mechanism test, not a GPU reproduction or an
identification of the object involved in Strata's Oct 7 crash. Hardware, mapping,
kernel argument setup and command-list execution are modeled explicitly.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from test_shutdown_spin_host import function


TAG = '26.31.39395.14'
PROGRAM = r'''
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
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neo-source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    sources={}
    def source(path):
        text=subprocess.check_output(['git','show',TAG+':'+path],cwd=args.neo_source,text=True)
        sources[path]=hashlib.sha256(text.encode()).hexdigest(); return text
    allocation=source('shared/source/memory_manager/graphics_allocation.h')
    methods=[]
    for signature in ['TaskCountType getTaskCount(', 'uint32_t getHostPtrTaskCountAssignment(',
                      'void decrementHostPtrTaskCountAssignment(', 'uint64_t getResidencyContainerStamp(',
                      'void setResidencyContainerStamp(']:
        methods.append(function(allocation,signature))
    (args.output/'allocation-methods.inc').write_text('\n'.join(methods)+'\n')
    functions=[]
    for path,signature in [
        ('shared/source/memory_manager/graphics_allocation.cpp','void GraphicsAllocation::prepareHostPtrForResidency('),
        ('shared/source/command_container/cmdcontainer.cpp','void CommandContainer::addToResidencyContainer('),
        ('level_zero/core/source/context/context.cpp','ze_result_t Context::destroyPhysicalMem('),
        ('level_zero/core/source/cmdqueue/cmdqueue.cpp','void CommandQueue::makeResidentForResidencyContainer(')]:
        namespace='NEO' if path.startswith('shared/') else 'L0'
        functions.append('namespace '+namespace+' {\n'+function(source(path),signature)+'\n}')
    (args.output/'backend-functions.inc').write_text('\n'.join(functions)+'\n')
    kernel=source('level_zero/core/source/kernel/kernel_imp.cpp')
    assignment='    privateState.argumentsResidencyContainer[argIndex] = allocation;'
    assert assignment in function(kernel,'ze_result_t KernelImp::setArgBufferWithAlloc(')
    (args.output/'kernel-assignment.inc').write_text(assignment+'\n')
    list_header=source('level_zero/core/source/cmdlist/cmdlist_hw.h')
    add='template <typename Container>\n'+function(list_header,'void addResidency(const Container &allocs)')
    (args.output/'add-residency.inc').write_text(add+'\n')
    recording=source('level_zero/core/source/cmdlist/cmdlist_hw_xehp_and_later.inl')
    fragment='            auto &argumentsResidencyContainer = kernel->getArgumentsResidencyContainer();\n            this->addResidency(argumentsResidencyContainer);'
    assert fragment in recording
    (args.output/'record-argument-residency.inc').write_text(fragment+'\n')
    (args.output/'probe.cpp').write_text(PROGRAM)
    record=dict(scope=__doc__,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                tag=TAG,tag_commit=subprocess.check_output(['git','rev-parse',TAG+'^{commit}'],cwd=args.neo_source,text=True).strip(),
                source_sha256=sources,runs=[],passed=False)
    command=['g++','-std=c++20','-O1','-g','-fno-omit-frame-pointer','-fno-pie','-no-pie',
             '-fsanitize=address,undefined',str(args.output/'probe.cpp'),'-o',str(args.output/'probe')]
    build=subprocess.run(command,capture_output=True,text=True,timeout=30)
    (args.output/'build.stderr').write_text(build.stderr); assert build.returncode==0,build.stderr
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=0',
             UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    for mode in ['retain-old-list','update-kernel-only','retire-and-recreate','keep-physical-memory']:
        run=subprocess.run([str(args.output/'probe'),mode],capture_output=True,text=True,env=env,timeout=5)
        (args.output/(mode+'.stdout')).write_text(run.stdout)
        (args.output/(mode+'.stderr')).write_text(run.stderr)
        expect_uaf=mode in ['retain-old-list','update-kernel-only']
        passed=(run.returncode!=0 and 'AddressSanitizer: heap-use-after-free' in run.stderr) if expect_uaf else (run.returncode==0 and 'PASS' in run.stdout)
        record['runs'].append(dict(mode=mode,exit_code=run.returncode,expected_uaf=expect_uaf,passed=passed))
        (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        assert passed,(mode,run.stdout,run.stderr)
    record.update(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'runs':record['runs']},indent=2))


if __name__=='__main__': main()
