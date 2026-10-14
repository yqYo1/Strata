#!/usr/bin/env python3
"""Artifact transform of the archived CPU clock helper; no product writes/tests."""
from pathlib import Path
p=Path(__file__).resolve().parent
text=(p/'archive-service_budget.hpp').read_text()
text=text.replace('#include <chrono>', '#include <chrono>\n#include <atomic>',1)
text=text.replace('// Coalesced expert jobs may contain several activation rows even for dispatch NT=1.',
                  '// Job activation-row sums use actual verifier T and remain separate from job count.')
begin=text.index('inline bool service_layer_complete(')
end=text.index('// Job activation-row sums',begin)
text=text[:begin]+'''// Coverage uses completed verifier windows and rows, never emitted-token count.
inline bool service_layer_complete(uint64_t calls,uint64_t rows,uint64_t W,uint64_t R,bool normal) {
    return normal && calls==W && rows==R;
}
'''+text[end:]
begin=text.index('struct NativeServiceBudget {')
text=text[:begin]+'''// Reused per-thread CPU clocks; existing pool wall phases are snapshotted elsewhere.
struct NativeServiceBudget {
    static constexpr int kWorkers=128;
    struct alignas(64) Slot {
        ServiceCounter call[6];
        uint64_t tasks[2]{}, finished[2]{}, nonempty[2]{}, rows[2]{};
        bool overflow=false;
        void add(uint64_t& dst,uint64_t value) {
            if (value>UINT64_MAX-dst) { dst=UINT64_MAX; overflow=true; } else dst+=value;
        }
    };
    Slot slots[kWorkers+1]{};
    std::atomic<bool> invalid{false}; // Written by workers only on an invalid envelope.
    bool overflow=false;
    int actual_t=0,worker_limit=0;
    uint64_t planned_tasks[2]{}, planned_rows[2]{}, phases[2]{}, phases_done[2]{};
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
    void phase_end(int ph) {
        if (ph<0 || ph>1) { invalid.store(true); return; }
        add(phases_done[ph],1);
    }
    void task(int id,int ph,int64_t rows) {
        auto* s=slot(id);
        if (!s || ph<0 || ph>1 || rows<0) { invalid.store(true); return; }
        s->add(s->tasks[ph],1); s->add(s->rows[ph],(uint64_t)rows);
        if (rows) s->add(s->nonempty[ph],1);
    }
    void task_done(int id,int ph) {
        auto* s=slot(id);
        if (!s || ph<0 || ph>1) { invalid.store(true); return; }
        s->add(s->finished[ph],1);
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
                if (s.finished[ph]!=s.tasks[ph] || s.nonempty[ph]>s.tasks[ph] ||
                    s.call[b].calls>UINT64_MAX-s.call[a].calls ||
                    s.call[a].calls+s.call[b].calls<s.nonempty[ph] ||
                    s.tasks[ph]>UINT64_MAX-tasks[ph] || s.rows[ph]>UINT64_MAX-rows[ph]) return false;
                tasks[ph]+=s.tasks[ph]; rows[ph]+=s.rows[ph];
            }
        }
        for (int ph=0;ph<2;++ph)
            if (tasks[ph]!=planned_tasks[ph] || rows[ph]!=planned_rows[ph] ||
                phases[ph]!=phases_done[ph] || (tasks[ph] && !phases[ph])) return false;
        return true;
    }
};
static_assert(sizeof(NativeServiceBudget)<=65536,"native service storage budget");
}
'''
(p/'adapted_service_budget.hpp').write_text(text)
