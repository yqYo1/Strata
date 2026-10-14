// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <chrono>
#include <cstdint>
#include <ctime>

namespace strata::kernels::cpu {
// Integer receipts retain failed reads, count every attempted interval, and never wrap.
struct ServiceCounter {
    uint64_t calls=0, cpu_ns=0, wall_ns=0, reads=0;
    bool invalid=false, overflow=false;
    void add(uint64_t& dst,uint64_t value) {
        if (value>UINT64_MAX-dst) { dst=UINT64_MAX; overflow=true; }
        else dst+=value;
    }
    void merge(const ServiceCounter& b) {
        add(calls,b.calls); add(cpu_ns,b.cpu_ns); add(wall_ns,b.wall_ns); add(reads,b.reads);
        invalid=invalid||b.invalid; overflow=overflow||b.overflow;
    }
    void record(uint64_t c0,uint64_t c1,uint64_t w0,uint64_t w1,bool ok) {
        add(calls,1); add(reads,2);
        if (!ok || c1<c0 || w1<w0) { invalid=true; return; }
        add(cpu_ns,c1-c0); add(wall_ns,w1-w0);
    }
    bool complete() const { return !invalid && !overflow && calls<=UINT64_MAX/2 && reads==2*calls; }
};
inline bool service_layer_complete(const ServiceCounter& c,uint64_t completed,int64_t emitted,bool normal) {
    return normal && emitted>=0 && c.complete() && completed==(uint64_t)emitted && c.calls==completed;
}
// Coalesced expert jobs may contain several activation rows even for dispatch NT=1.
inline bool service_activation_rows_add(uint64_t& rows,int nt,int limit) {
    if (nt<1 || nt>limit || (uint64_t)nt>UINT64_MAX-rows) return false;
    rows+=(uint64_t)nt; return true;
}
inline bool service_cpu_now(uint64_t& ns) {
#if defined(__linux__)
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)!=0 || t.tv_sec<0 || t.tv_nsec<0 ||
        t.tv_nsec>=1000000000 || (uint64_t)t.tv_sec>(UINT64_MAX-(uint64_t)t.tv_nsec)/1000000000) return false;
    ns=(uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec; return true;
#else
    (void)ns; return false;
#endif
}
inline uint64_t service_wall_now() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
class ServiceClockScope {
    ServiceCounter* counter_;
    bool (*cpu_now_)(uint64_t&);
    uint64_t (*wall_now_)();
    uint64_t cpu_=0,wall_=0;
    bool ok_=false;
public:
    explicit ServiceClockScope(ServiceCounter* counter,bool (*cpu_now)(uint64_t&)=service_cpu_now,
                               uint64_t (*wall_now)()=service_wall_now)
        :counter_(counter),cpu_now_(cpu_now),wall_now_(wall_now) {
        if (counter_) { wall_=wall_now_(); ok_=cpu_now_(cpu_); }
    }
    ~ServiceClockScope() {
        if (counter_) { uint64_t cpu=0; const bool ok=cpu_now_(cpu);
            const uint64_t wall=wall_now_(); counter_->record(cpu_,cpu,wall_,wall,ok_&&ok); }
    }
    ServiceClockScope(const ServiceClockScope&)=delete;
    ServiceClockScope& operator=(const ServiceClockScope&)=delete;
};
struct NativeServiceBudget {
    static constexpr int kWorkers=128;
    // Each slot has exactly one writer. Slot 0 is the inline issuer; worker id i owns i+1.
    struct alignas(64) Slot { ServiceCounter call[6]; uint64_t tasks[2]{}; bool overflow=false; };
    Slot slots[kWorkers+1]{};
    ServiceCounter phase[2][3]; // parked-before, publish-through-done, done-through-parked
    bool invalid=false;
    uint64_t planned_tasks[2]{};
    Slot* slot(int id) { return id>=-1 && id<kWorkers ? &slots[id+1] : nullptr; }
    void task(int id,int phase_id) {
        // Pool callers pass 0 or 1; reject out-of-envelope inputs before indexing.
        if (phase_id<0 || phase_id>1) return;
        if (auto* s=slot(id)) {
            if (s->tasks[phase_id]==UINT64_MAX) s->overflow=true; else ++s->tasks[phase_id];
        }
    }
    bool complete(int workers) const {
        if (invalid || workers<0 || workers>kWorkers) return false;
        for (int i=0;i<=workers;++i) { if (slots[i].overflow) return false;
            for (const auto& c:slots[i].call) if (!c.complete()) return false;
            for (int ph=0;ph<2;++ph) {
                const int native=ph==0?0:2,fallback=ph==0?3:5;
                const uint64_t a=slots[i].call[native].calls,b=slots[i].call[fallback].calls;
                if (b>UINT64_MAX-a || a+b<slots[i].tasks[ph]) return false;
            }
        }
        for (int ph=0;ph<2;++ph) {
            uint64_t tasks=0;
            for (int i=0;i<=workers;++i) {
                if (slots[i].tasks[ph]>UINT64_MAX-tasks) return false;
                tasks+=slots[i].tasks[ph];
            }
            if (tasks!=planned_tasks[ph] || (tasks>0 && phase[ph][0].calls==0) ||
                phase[ph][0].calls!=phase[ph][1].calls || phase[ph][1].calls!=phase[ph][2].calls) return false;
        }
        for (const auto& row:phase) for (const auto& c:row) if (!c.complete()) return false;
        return true;
    }
};
static_assert(sizeof(NativeServiceBudget)<=65536,"native service storage budget");
}
