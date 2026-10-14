// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <chrono>
#include <atomic>
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
// Coverage uses completed verifier windows and rows, never emitted-token count.
inline bool service_layer_complete(uint64_t calls,uint64_t rows,uint64_t W,uint64_t R,bool normal) {
    return normal && calls==W && rows==R;
}
// Job activation-row sums use actual verifier T and remain separate from job count.
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
// Reused per-thread CPU clocks; existing pool wall phases are snapshotted elsewhere.
struct NativeServiceBudget {
    static constexpr int kWorkers=128;
    struct alignas(64) Slot {
        ServiceCounter call[6];
        uint64_t tasks[2]{}, finished[2]{}, nonempty[2]{}, rows[2]{}, finished_rows[2]{};
        bool overflow=false;
        void add(uint64_t& dst,uint64_t value) {
            if (value>UINT64_MAX-dst) { dst=UINT64_MAX; overflow=true; } else dst+=value;
        }
    };
    Slot slots[kWorkers+1]{};
    std::atomic<bool> invalid{false}; // Written by workers only on an invalid envelope.
    bool overflow=false;
    int actual_t=0,worker_limit=0;
    uint64_t planned_tasks[2]{}, planned_rows[2]{}, phases[2]{}, phases_done[2]{}, done_boundaries[2]{}, park_boundaries[2]{};
    void add(uint64_t& dst,uint64_t value) {
        if (value>UINT64_MAX-dst) { dst=UINT64_MAX; overflow=true; } else dst+=value;
    }
    Slot* slot(int id) {
        return worker_limit>=1 && worker_limit<=kWorkers && id>=-1 && id<worker_limit ? &slots[id+1] : nullptr;
    }
    ServiceCounter* counter(int id,int role,int nt) {
        auto* s=slot(id);
        if (!s || role<0 || role>=6 || actual_t<1 || actual_t>4 || nt<1 || nt>actual_t) {
            invalid.store(true,std::memory_order_relaxed); return nullptr;
        }
        return &s->call[role];
    }
    void phase_begin(int ph,int tasks,int64_t rows) {
        if (ph<0 || ph>1 || tasks<1 || rows<1) { invalid.store(true); return; }
        add(phases[ph],1); add(planned_tasks[ph],(uint64_t)tasks); add(planned_rows[ph],(uint64_t)rows);
    }
    void phase_end(int ph,bool exact=true) {
        if (ph<0 || ph>1) { invalid.store(true); return; }
        add(park_boundaries[ph],1); // called only after existing wait_parked returned
        if (!exact) { invalid.store(true); return; }
        add(done_boundaries[ph],1); add(phases_done[ph],1);
    }
    void task(int id,int ph,int64_t rows) {
        auto* s=slot(id);
        if (!s || ph<0 || ph>1 || rows<0) { invalid.store(true); return; }
        s->add(s->tasks[ph],1); s->add(s->rows[ph],(uint64_t)rows);
        if (rows) s->add(s->nonempty[ph],1);
    }
    void task_done(int id,int ph,int64_t rows) {
        auto* s=slot(id);
        if (!s || ph<0 || ph>1 || rows<0) { invalid.store(true); return; }
        s->add(s->finished[ph],1); s->add(s->finished_rows[ph],(uint64_t)rows);
    }
    bool complete(int workers) const {
        if (invalid.load() || overflow || workers<1 || workers>kWorkers || workers!=worker_limit) return false;
        uint64_t tasks[2]{}, rows[2]{};
        for (int i=0;i<=workers;++i) {
            const auto& s=slots[i];
            if (s.overflow) return false;
            for (const auto& c:s.call) if (!c.complete()) return false;
            for (int ph=0;ph<2;++ph) {
                const int a=ph==0?0:2,b=ph==0?3:5;
                if (s.finished[ph]!=s.tasks[ph] || s.finished_rows[ph]!=s.rows[ph] || s.nonempty[ph]>s.tasks[ph] ||
                    s.call[b].calls>UINT64_MAX-s.call[a].calls ||
                    s.call[a].calls+s.call[b].calls<s.nonempty[ph] ||
                    s.tasks[ph]>UINT64_MAX-tasks[ph] || s.rows[ph]>UINT64_MAX-rows[ph]) return false;
                tasks[ph]+=s.tasks[ph]; rows[ph]+=s.rows[ph];
            }
        }
        for (int ph=0;ph<2;++ph)
            if (tasks[ph]!=planned_tasks[ph] || rows[ph]!=planned_rows[ph] ||
                phases[ph]!=phases_done[ph] || phases[ph]!=done_boundaries[ph] || phases[ph]!=park_boundaries[ph] || (tasks[ph] && !phases[ph])) return false;
        return true;
    }
};
static_assert(sizeof(NativeServiceBudget)<=65536,"native service storage budget");
}
