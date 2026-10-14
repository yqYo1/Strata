// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// UNCOMPILED/UNRUN. Adaptation of archived route/service fixtures using actual helpers.
// Main after application: c++ -std=c++17 -pthread -Iinclude PATH/cpu_fixture.cpp -o PRIVATE/fixture
#include "strata/core/decode_route_ledger.hpp"
#include <cassert>
#include <memory>
#include <string>

using namespace strata::core;
using namespace strata::kernels::cpu;
namespace {
int cpu_reads=0,wall_reads=0;
bool fake_cpu(uint64_t& ns) { ns=(uint64_t)++cpu_reads*10; return true; }
bool failed_cpu(uint64_t& ns) { ++cpu_reads; ns=0; return false; }
uint64_t fake_wall() { return (uint64_t)++wall_reads*100; }
const uint8_t dummy_blob=0;
NativeFmt format(int layer) {
    NativeFmt f;
    f.gu_type=layer ? 20 : 21; f.d_type=layer ? 20 : 42;
    f.gu_act=f.d_act=8; f.n_embd=2560; f.n_ff=640;
    f.gu_row=f.d_row=(size_t)layer+1;
    f.up_off=640*f.gu_row; f.down_off=2*f.up_off; f.bytes=f.down_off+2560*f.d_row;
    f.act_bytes=f.h_bytes=8;
    return f;
}
std::unique_ptr<DecodeRouteLedger> fresh(int layers=2) {
    auto value=std::make_unique<DecodeRouteLedger>();
    value->layers=layers; value->k=2; value->experts=16; value->workers=1; value->host_works=true;
    value->service.worker_limit=1;
    value->sequence=7; value->prompt_hash=123; value->prompt_tokens=32; value->max_new=7; value->context=256;
    for (int l=0;l<layers && l<64;++l) value->fmt[l]=format(l&1);
    value->before.valid=value->after.valid=true;
    return value;
}
void pool_dispatch(DecodeRouteLedger& value,const NativeFmt& f,int nt) {
    auto& service=value.service;
    service.actual_t=nt;
    // Actual flattened-row split of one job: even experts < participants has no
    // empty task here. This is a deterministic task receipt, not a kernel implementation.
    auto phase=[&](int ph) {
        const int tasks=6; const int64_t rows=ph==0?640:2560;
        service.phase_begin(ph,tasks,rows);
        for (int task=0;task<tasks;++task) {
            const int id=task%2-1;
            const int64_t begin=rows*task/tasks,end=rows*(task+1)/tasks;
            assert(end>begin);
            service.task(id,ph,end-begin);
            const int role=ph==0 ? (f.gu_type==42?3:0) : (f.d_type==42?5:2);
            auto* c=service.counter(id,role,nt); assert(c);
            c->record(0,1,0,2,true);
            service.task_done(id,ph);
        }
        service.phase_end(ph);
    };
    phase(0);
    for (int row=0;row<nt;++row) {
        auto* c=service.counter(-1,f.d_type==42?4:1,nt); assert(c);
        c->record(0,1,0,2,true);
    }
    phase(1);
}
void window(DecodeRouteLedger& value,int T,int accepted,int emit,bool terminal,bool all_gpu=false,int omit_layer=-1) {
    value.begin_window(T);
    int32_t ids[8]{},kind[8]{};
    for (int row=0;row<T;++row) { ids[2*row]=0; ids[2*row+1]=1; kind[2*row]=all_gpu?0:-1; kind[2*row+1]=0; }
    ExpertJobMulti job; job.blob=&dummy_blob; job.nt=T;
    for (int l=0;l<value.layers;++l) {
        if (l==omit_layer) continue;
        value.attempt_dispatch(l,T,2);
        if (!all_gpu) pool_dispatch(value,value.fmt[l],T);
        value.record(l,T,ids,kind,2*T,all_gpu?nullptr:&job,all_gpu?0:1,1,value.fmt[l],0);
        // Independent fake source counters, incremented at the fake successful call.
        value.after.count[1]+=all_gpu?0:1;
        value.after.count[2]+=all_gpu?0:T;
        value.after.count[3]+=all_gpu?2*T:T;
        value.after.count[5]+=all_gpu?0:(int64_t)value.fmt[l].bytes;
    }
    value.verified(T); ++value.after.count[0];
    value.committed(accepted,accepted+1); value.emitted(emit,terminal);
}
std::unique_ptr<DecodeRouteLedger> full() {
    auto value=fresh();
    window(*value,1,0,1,false);
    window(*value,4,2,3,false);
    window(*value,4,0,1,false);
    window(*value,4,3,2,true);
    value->wait_commit_ok=true;
    return value;
}
bool complete(const DecodeRouteLedger& value) { return value.complete(true,7,9,5); }
}

int main() {
    assert(decode_joined_requested("1","1"));
    const char* no_values[]={nullptr,"","0","01","true","1 "};
    for (const char* no:no_values) {
        assert(!decode_joined_requested(no,"1")); assert(!decode_joined_requested("1",no));
    }
    assert(decode_route_ledger_eligible(true,0,1,true,0,0,4,false));
    assert(!decode_route_ledger_eligible(true,0,1,false,0,0,4,false));
    assert(!decode_route_ledger_eligible(true,0,1,true,0,0,4,true));
    assert(!decode_route_ledger_eligible(true,1,1,true,0,0,4,false));
    assert(!decode_route_ledger_eligible(true,0,2,true,0,0,4,false));
    assert(!decode_route_ledger_eligible(true,0,1,true,1,0,4,false));
    assert(!decode_route_ledger_eligible(true,0,1,true,0,1,4,false));
    for (int T=1;T<=4;++T) assert(decode_route_ledger_dispatch_eligible(true,T,0,false));
    assert(!decode_route_ledger_dispatch_eligible(true,5,0,false));
    assert(!decode_route_ledger_dispatch_eligible(true,4,1,false));
    assert(!decode_route_ledger_dispatch_eligible(true,4,0,true));
    // Reused archive fixture: off means no injected clocks even across many calls.
    for (int i=0;i<10000;++i) { ServiceClockScope off(nullptr,fake_cpu,fake_wall); }
    assert(cpu_reads==0 && wall_reads==0);
    ServiceCounter measured;
    { ServiceClockScope clock(&measured,fake_cpu,fake_wall); }
    assert(measured.complete() && measured.calls==1 && measured.cpu_ns==10 && measured.wall_ns==100);
    ServiceCounter unavailable;
    { ServiceClockScope clock(&unavailable,failed_cpu,fake_wall); }
    assert(unavailable.invalid && !unavailable.complete() && unavailable.reads==2);
    ServiceCounter reverse; reverse.record(20,10,100,200,true); assert(!reverse.complete());
    ServiceCounter counter_overflow; counter_overflow.cpu_ns=UINT64_MAX;
    counter_overflow.record(0,1,0,1,true); assert(counter_overflow.overflow && !counter_overflow.complete());

    auto value=full();
    assert(value->W==4 && value->R==13 && value->D==9 && value->A==5 && value->C==9 && value->E==7);
    assert(value->D-value->A==4 && value->C-value->E==2 && value->terminal_unemitted);
    assert(value->cpu_jobs==8 && value->cpu_rows==26 && value->route_entries==52);
    assert(value->layer[0].completed[0]==1 && value->layer[0].completed[3]==3);
    assert(value->layer[0].verified_rows[0]+value->layer[0].verified_rows[3]==13);
    assert(value->layer[0].completed[0]+value->layer[0].completed[3]!=value->E);
    assert(value->rows[0][3][0][3].groups==3 && value->rows[0][3][0][3].entries==12);
    assert(value->fmt[0].gu_type==21 && value->fmt[0].d_type==42 && value->fmt[1].gu_type==20);
    assert(complete(*value));
    auto variable=fresh();
    window(*variable,1,0,1,false); window(*variable,2,1,2,false); window(*variable,3,1,2,true);
    variable->wait_commit_ok=true;
    assert(variable->complete(true,5,3,2) && variable->by_t[1].W==1 && variable->by_t[2].W==1);
    value->wait_commit_ok=false; assert(!complete(*value)); value->wait_commit_ok=true;
    assert(!value->complete(false,7,9,5)); // cancel/error is never normal completion
    value->pool_bound=true; assert(!complete(*value)); value->pool_bound=false;
    --value->service.slots[0].finished[0]; assert(!complete(*value)); ++value->service.slots[0].finished[0];
    ++value->layer[0].cpu_rows[3]; assert(!complete(*value)); --value->layer[0].cpu_rows[3];
    ++value->after.count[1]; assert(!complete(*value)); --value->after.count[1];
    value->service.slots[0].call[0].invalid=true; assert(!complete(*value)); value->service.slots[0].call[0].invalid=false;
    value->overflow=true; assert(!complete(*value)); value->overflow=false;
    assert(complete(*value));

    auto bytes=std::make_unique<char[]>(DecodeLedgerOutput::kCap);
    DecodeLedgerOutput output{bytes.get(),DecodeLedgerOutput::kCap};
    value->receipt(output,"length",7,9,5,12.5);
    const std::string text(bytes.get(),output.used);
    assert(output.bounded && output.used<=DecodeLedgerOutput::kCap);
    assert(text.find("request=7")!=std::string::npos && text.find("complete=1")!=std::string::npos);
    DecodeLedgerOutput small{bytes.get(),DecodeLedgerOutput::kFooter+1};
    value->receipt(small,"length",7,9,5,12.5);
    const std::string truncated(bytes.get(),small.used);
    assert(!small.bounded && small.omitted && small.used<=DecodeLedgerOutput::kFooter+1 && truncated.find("complete=0")!=std::string::npos);

    auto zero=fresh(); window(*zero,1,0,1,true,true); zero->wait_commit_ok=true;
    assert(zero->layer[0].zero_jobs[0]==1 && zero->service.planned_tasks[0]==0);
    assert(zero->complete(true,1,0,0));
    auto missing=fresh(); window(*missing,1,0,1,true,false,1); missing->wait_commit_ok=true;
    assert(missing->invalid && !missing->complete(true,1,0,0));
    auto prefix=fresh(); window(*prefix,1,0,1,false); prefix->begin_window(4);
    assert(prefix->attempted_windows==2 && prefix->W==1 && !prefix->complete(true,1,0,0));
    auto failed_commit=fresh();
    failed_commit->begin_window(1);
    for (int l=0;l<2;++l) {
        int32_t ids[]={0,1},kind[]={0,0}; failed_commit->attempt_dispatch(l,1,2);
        failed_commit->record(l,1,ids,kind,2,nullptr,0,0,failed_commit->fmt[l],0);
    }
    failed_commit->verified(1);
    assert(failed_commit->verified_windows==1 && failed_commit->W==0 && failed_commit->window_active);
    assert(!failed_commit->complete(true,0,0,0));

    auto mixed=fresh(); mixed->k=4; mixed->begin_window(4);
    int32_t mixed_ids[]={0,1,2,3, 0,1,2,4, 0,1,5,4, 0,6,5,4};
    int32_t mixed_kind[]={-1,-1,0,1, -1,-1,0,0, -1,-1,-1,0, -1,1,-1,0};
    ExpertJobMulti mixed_jobs[3];
    for (int j=0;j<3;++j) { mixed_jobs[j].blob=&dummy_blob; mixed_jobs[j].nt=4-j; }
    for (int l=0;l<2;++l) {
        mixed->attempt_dispatch(l,4,4);
        mixed->record(l,4,mixed_ids,mixed_kind,16,mixed_jobs,3,3,mixed->fmt[l],l);
    }
    assert(!mixed->invalid && mixed->layer[0].cpu_jobs[3]==3 && mixed->layer[0].cpu_rows[3]==9);
    assert(mixed->rows[0][3][0][3].groups==1 && mixed->rows[0][3][0][2].groups==1 && mixed->rows[0][3][0][1].groups==1);
    assert(mixed->rows[0][3][2][0].groups==2 && mixed->rows[1][3][3][0].groups==2);

    auto malformed=fresh(); malformed->begin_window(5); assert(malformed->invalid);
    auto too_many=fresh(65); too_many->begin_window(1); assert(too_many->invalid);
    auto bad=fresh(); bad->begin_window(4); bad->attempt_dispatch(0,4,2);
    int32_t ids[]={0,1,0,1,0,1,0,1},kind[]={-1,0,-1,0,-1,0,-1,0};
    ExpertJobMulti job; job.blob=&dummy_blob; job.nt=5;
    bad->record(0,4,ids,kind,8,&job,1,1,bad->fmt[0],0); assert(bad->invalid && bad->layer[0].completed[3]==0);
    auto owners=fresh(); owners->begin_window(4); owners->attempt_dispatch(0,4,2); kind[0]=2; job.nt=4;
    owners->record(0,4,ids,kind,8,&job,1,1,owners->fmt[0],0); assert(owners->invalid);
    kind[0]=-1;
    auto formats=fresh(); formats->begin_window(4); formats->attempt_dispatch(0,4,2);
    auto f=formats->fmt[0]; ++f.d_type;
    formats->record(0,4,ids,kind,8,&job,1,1,f,0); assert(formats->invalid);
    auto bounds=fresh(); bounds->begin_window(4); bounds->attempt_dispatch(64,4,2); assert(bounds->invalid);
    auto maximum=fresh(64); maximum->begin_window(1); assert(maximum->all_layers()==UINT64_MAX);
    auto duplicate=fresh(); duplicate->begin_window(4); duplicate->attempt_dispatch(0,4,2); ids[1]=0;
    duplicate->record(0,4,ids,kind,8,&job,1,1,duplicate->fmt[0],0); assert(duplicate->invalid);
    auto overflow=fresh(); uint64_t sum=UINT64_MAX; overflow->add(sum,1); assert(overflow->overflow && sum==UINT64_MAX);
    auto worker=fresh(); worker->service.actual_t=4;
    assert(!worker->service.counter(128,0,1) && worker->service.invalid.load());
    auto maximum_worker=fresh(); maximum_worker->workers=128; maximum_worker->service.worker_limit=128; maximum_worker->service.actual_t=4;
    assert(maximum_worker->service.counter(127,0,4));
    auto nt=fresh(); nt->service.actual_t=2;
    assert(!nt->service.counter(0,0,3) && nt->service.invalid.load());
    // Fixture source contains no model weights or real kernel execution.
    return 0;
}
