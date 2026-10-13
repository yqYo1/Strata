// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/kernels/cpu/service_budget.hpp"
#include <cassert>
#include <initializer_list>
#include <thread>
#include <stdexcept>
using namespace strata::kernels::cpu;
namespace {
int cpu_reads=0,wall_reads=0;
bool fake_cpu(uint64_t& n) { n=(uint64_t)++cpu_reads*10; return true; }
bool failed_cpu(uint64_t& n) { ++cpu_reads; n=0; return false; }
uint64_t fake_wall() { return (uint64_t)++wall_reads*100; }
}
int main() {
    // Off must call neither injected clock, including thousands of hot-path scopes.
    for (int i=0;i<10000;++i) { ServiceClockScope off(nullptr,fake_cpu,fake_wall); }
    assert(cpu_reads==0 && wall_reads==0);
    ServiceCounter exact;
    { ServiceClockScope scope(&exact,fake_cpu,fake_wall); }
    assert(exact.complete() && exact.calls==1 && exact.reads==2 && exact.cpu_ns==10 && exact.wall_ns==100);
    assert(service_layer_complete(exact,1,1,true));
    assert(!service_layer_complete(exact,1,1,false));
    assert(!service_layer_complete(exact,1,2,true));
    assert(!service_layer_complete(exact,1,-1,true));
    uint64_t activation_rows=0;
    assert(service_activation_rows_add(activation_rows,1,8));
    assert(service_activation_rows_add(activation_rows,2,8));
    // Two coalesced jobs carry three rows: quantizer calls match rows, not jobs.
    ServiceCounter coalesced_quant;
    for (uint64_t row=0;row<activation_rows;++row) coalesced_quant.record(0,1,0,2,true);
    assert(coalesced_quant.complete() && coalesced_quant.calls==3 && coalesced_quant.calls!=2);
    for (int nt : {-1,0,9}) {
        assert(!service_activation_rows_add(activation_rows,nt,8) && activation_rows==3);
    }
    activation_rows=UINT64_MAX;
    assert(!service_activation_rows_add(activation_rows,1,8) && activation_rows==UINT64_MAX);
    ServiceCounter partial;
    { ServiceClockScope scope(&partial,failed_cpu,fake_wall); }
    assert(partial.calls==1 && partial.reads==2 && partial.invalid && !partial.complete());
    ServiceCounter early;
    try { ServiceClockScope scope(&early,fake_cpu,fake_wall); throw std::runtime_error("fixture"); }
    catch (const std::runtime_error&) {}
    assert(early.calls==1 && early.complete());
    assert(!service_layer_complete(early,0,1,false));
    ServiceCounter reverse;
    reverse.record(20,10,100,200,true); assert(reverse.invalid && !reverse.complete());
    ServiceCounter aggregate=exact;
    aggregate.merge(exact); assert(aggregate.calls==2 && aggregate.cpu_ns==20 && aggregate.wall_ns==200);
    aggregate.merge(partial); assert(!aggregate.complete() && aggregate.calls==3);
    ServiceCounter overflow;
    overflow.cpu_ns=UINT64_MAX;
    overflow.record(0,1,0,1,true); assert(overflow.overflow && overflow.cpu_ns==UINT64_MAX && !overflow.complete());
    ServiceCounter count_overflow;
    count_overflow.calls=UINT64_MAX; count_overflow.record(0,0,0,0,true);
    assert(count_overflow.overflow && count_overflow.calls==UINT64_MAX);
    ServiceCounter merge_overflow;
    merge_overflow.wall_ns=UINT64_MAX; merge_overflow.merge(exact);
    assert(merge_overflow.overflow && merge_overflow.wall_ns==UINT64_MAX && !merge_overflow.complete());
    ServiceCounter read_overflow;
    read_overflow.reads=UINT64_MAX; read_overflow.record(0,0,0,0,true);
    assert(read_overflow.overflow && read_overflow.reads==UINT64_MAX);
    ServiceCounter read_missing=exact; read_missing.reads=1; assert(!read_missing.complete());
    aggregate=ServiceCounter{}; assert(aggregate.complete() && aggregate.calls==0 && aggregate.cpu_ns==0);
    NativeServiceBudget budget;
    budget.planned_tasks[0]=3;
    for (auto& c:budget.phase[0]) c.record(0,1,0,2,true);
    auto work=[&](int id) {
        budget.task(id,0);
        ServiceClockScope scope(&budget.slot(id)->call[0]);
        volatile uint64_t sum=0;
        for (uint64_t i=0;i<100000;++i) sum=sum+i;
        (void)sum;
    };
    std::thread a(work,0),b(work,1);
    work(-1); a.join(); b.join(); // Merge only after the existing consumer's completion boundary.
    uint64_t ns=0; const bool available=service_cpu_now(ns);
    assert(budget.complete(2)==available);
    ServiceCounter workers;
    workers.merge(budget.slot(0)->call[0]); workers.merge(budget.slot(1)->call[0]);
    assert(workers.calls==2 && workers.reads==4 && budget.slot(-1)->call[0].calls==1);
    if (available) assert(workers.complete()); else assert(workers.invalid && !workers.complete());
    assert(!budget.complete(NativeServiceBudget::kWorkers+1));
    assert(budget.slot(-2)==nullptr && budget.slot(NativeServiceBudget::kWorkers)==nullptr);
    budget.planned_tasks[0]=4; assert(!budget.complete(2));
    budget=NativeServiceBudget{}; assert(budget.complete(2));
    budget.slots[0].tasks[0]=UINT64_MAX; budget.task(-1,0); assert(!budget.complete(2));
    budget=NativeServiceBudget{};
    budget.slots[0].tasks[0]=UINT64_MAX; budget.slots[1].tasks[0]=1;
    assert(!budget.complete(2));
    return 0;
}
