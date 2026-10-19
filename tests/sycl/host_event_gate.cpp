#include "host_event_gate.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <cstdio>
#include <string>

namespace {
using strata::test::event_need;
using Clock=std::chrono::steady_clock;
long long micros(Clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-begin).count();
}
void require_pending(const sycl::event& event) {
    event_need(event.get_info<sycl::info::event::command_execution_status>()!=
        sycl::info::event_command_status::complete,"dependent native gate work completed before release");
}
}

int main(int argc,char** argv) {
    try {
        event_need(argc==2,"usage: sycl_host_event_gate_test import|copy|graph|copy-query");
        const std::string mode=argv[1];
        event_need(mode=="import" || mode=="copy" || mode=="graph" || mode=="copy-query","unknown native event gate case");
        auto runtime=strata::sycl_backend::runtime_for();
        if(mode=="import") {
            // First qualification creates/imports only: no dependent command,
            // command graph, USM allocation or GPU kernel is submitted.
            strata::test::HostEventGate gate(runtime,{});
            long long import_us=0;(void)gate.import(import_us);
            const auto begin=Clock::now();
            do {
                event_need(gate.held(),"native import gate expired during observations");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while(micros(begin)<50000);
            const auto pending_us=micros(begin);
            event_need(gate.release(),"native import gate released through deadline");
            std::printf("event_gate_case=import import_us=%lld submit_us=0 pending_us=%lld release=manual payload=PASS status=PASS\n",
                import_us,pending_us);
            return 0;
        }
        auto copy=runtime->make_queue(true), compute=runtime->make_queue(true), probe=runtime->make_queue(false);
        using strata::sycl_backend::Allocation;
        using strata::sycl_backend::MemoryKind;
        // All host/device bytes outlive the gate and every dependent command.
        Allocation host(runtime,3*sizeof(uint32_t),MemoryKind::Host);
        Allocation gpu(runtime,2*sizeof(uint32_t),MemoryKind::Device);
        auto* input=host.as<uint32_t>();auto* resident=input+1;auto* seen=input+2;
        auto* stage=gpu.as<uint32_t>();auto* output=stage+1;
        *input=0x12345678u;*resident=0;*seen=0;
        runtime->wait(copy.memset(gpu.data(),0xe7,gpu.size()));
        auto kernel=[=] {
            sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space> word(*output);
            word.store(*stage ^ *resident);
        };
        namespace exp=sycl::ext::oneapi::experimental;
        exp::command_graph graph(runtime->context(),runtime->device());
        graph.begin_recording(compute);compute.single_task(kernel);graph.end_recording(compute);
        auto executable=graph.finalize();
        // Prewarm the exact command graph and probe before arming the timer.
        runtime->wait(compute.submit([&](sycl::handler& h) {h.ext_oneapi_graph(executable);}));
        auto inspect=[&] {
            runtime->wait(probe.single_task([=] {
                sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space> word(*output);
                *seen=word.load();
            }));
        };
        inspect();runtime->wait(copy.memset(gpu.data(),0xe7,gpu.size()));
        strata::test::HostEventGate gate(runtime,{copy,compute,probe});
        long long import_us=0,submit_us=0;
        const auto& imported=gate.import(import_us);
        std::optional<sycl::event> done;
        if(mode!="import") {
            const auto begin=Clock::now();
            copy.ext_oneapi_submit_barrier({imported});
            const auto copied=copy.memcpy(stage,input,sizeof(uint32_t));
            // Resident work precedes the cross-queue wait on the same compute
            // queue; the independent probe must also progress while held.
            const auto resident_done=compute.single_task([=] {*resident=0x3cu;});
            compute.ext_oneapi_submit_barrier({copied});
            if(mode=="graph") done=compute.submit([&](sycl::handler& h) {h.ext_oneapi_graph(executable);});
            else done=compute.single_task(kernel);
            submit_us=micros(begin);
            event_need(submit_us<=100000 && gate.held(),"native dependent submit exceeded100ms or timer expired");
            if(mode=="copy-query") {
                // Diagnostic only: query the earlier event without reading its
                // non-atomic host payload or changing the production queue flags.
                const auto handle_begin=Clock::now();
                const auto native_resident=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(resident_done);
                const auto native_gate=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(imported);
                const auto handle_us=micros(handle_begin);
                event_need(native_resident!=native_gate && handle_us<=100000 && gate.held(),
                    "resident native handle materialization failed or exceeded100ms");
                const auto query_begin=Clock::now();
                bool ready=false;int queries=0;
                for(;queries<20 && gate.held();) {
                    ++queries;
                    const auto status=zeEventQueryStatus(native_resident);
                    if(status==ZE_RESULT_SUCCESS) {ready=true;break;}
                    strata::test::ze_check(status==ZE_RESULT_NOT_READY ? ZE_RESULT_SUCCESS : status,
                        "resident native status query");
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                const auto query_us=micros(query_begin);
                event_need(query_us<=200000 && gate.held(),"native status query exceeded its bounded held window");
                // An unready event is released BEFORE waiting. A ready event is
                // waited with the gate still held, separating wait latency from
                // the already-observed native completion. Neither branch admits
                // the failed ordering gate or model execution.
                const bool release_before_wait=!ready;
                bool manual=false;
                if(release_before_wait) manual=gate.release();
                const bool held_before_wait=gate.held();
                const auto wait_begin=Clock::now();runtime->wait(resident_done);
                const auto wait_us=micros(wait_begin);
                const bool held_after_wait=gate.held();
                if(!release_before_wait) manual=gate.release();
                runtime->wait(*done);inspect();
                event_need(*resident==0x3cu && *seen==(*input ^ *resident),"native query diagnostic payload differs");
                std::printf("event_gate_query native_ready=%d queries=%d import_us=%lld submit_us=%lld handle_us=%lld query_us=%lld wait_us=%lld "
                            "held_before_wait=%d held_after_wait=%d released_before_wait=%d marker=%u "
                            "release=%s payload=PASS status=PASS\n",int(ready),queries,import_us,submit_us,handle_us,query_us,wait_us,
                            int(held_before_wait),int(held_after_wait),int(release_before_wait),*resident,manual?"manual":"fallback");
                return 0;
            }
            const auto wait_begin=Clock::now();
            runtime->wait(resident_done);
            const auto wait_us=micros(wait_begin);
            const bool still_held=gate.held();
            std::printf("event_gate_resident value=%u held=%d wait_us=%lld submit_us=%lld\n",
                *resident,int(still_held),wait_us,submit_us);
            std::fflush(stdout);
            event_need(*resident==0x3cu && still_held,"native gate blocked independent resident progress");
        }
        const auto begin=Clock::now();
        do {
            event_need(gate.held(),"native gate expired during pending observations");
            if(done) require_pending(*done);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while(micros(begin)<50000);
        if(done) {
            inspect();event_need(*seen==0xe7e7e7e7u,"native gated output changed before release");
        }
        const auto pending_us=micros(begin);
        event_need(gate.release(),"native gate released through deadline rather than explicit request");
        if(done) {
            runtime->wait(*done);inspect();
            event_need(*seen==(*input ^ *resident),"native gate payload/resident result differs");
        }
        std::printf("event_gate_case=%s import_us=%lld submit_us=%lld pending_us=%lld release=manual payload=PASS status=PASS\n",
            mode.c_str(),import_us,submit_us,pending_us);
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"native event gate: %s\n",error.what());return 1;
    }
}
