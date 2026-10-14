// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// MTP4 adaptation of archived DecodeRouteLedger + NativeServiceBudget.
#pragma once
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/service_budget.hpp"
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::core {
inline bool decode_joined_requested(const char* route,const char* service) {
    return route && service && !std::strcmp(route,"1") && !std::strcmp(service,"1");
}
inline bool decode_route_ledger_eligible(bool ordinary,int batch,int stages,bool mtp,
                                        int suffix,int lookup,int capacity,bool split) {
    return ordinary && batch==0 && stages==1 && mtp && suffix==0 && lookup==0 && capacity==4 && !split;
}
inline bool decode_route_ledger_dispatch_eligible(bool native,int64_t T,int remote,bool peer) {
    return native && T>=1 && T<=4 && remote==0 && !peer;
}
struct DecodeLedgerSnapshot {
    static constexpr int kReal=17,kCount=6;
    double real[kReal]{};
    int64_t count[kCount]{};
    bool valid=false;
};
struct DecodeLedgerSources {
    const double* real[14]{}; // existing verifier/dispatch/pool/window fields
    const int64_t* count[6]{};
};
struct DecodeLedgerOutput {
    static constexpr size_t kCap=256*1024,kFooter=1024;
    char* bytes;
    size_t capacity,used=0;
    uint64_t omitted=0;
    bool bounded=true;
    bool append(bool footer,const char* format,...) {
        if (!bytes) { bounded=false; return false; }
        char line[768];
        va_list args; va_start(args,format);
        const int n=std::vsnprintf(line,sizeof line,format,args); va_end(args);
        const size_t limit=footer ? capacity : capacity>=kFooter ? capacity-kFooter : 0;
        if (n<0 || (size_t)n>=sizeof line || (size_t)n>limit || used>limit-(size_t)n) {
            bounded=false;
            if (omitted!=UINT64_MAX) ++omitted;
            return false;
        }
        std::memcpy(bytes+used,line,(size_t)n); used+=(size_t)n; return true;
    }
};

class DecodeRouteLedger {
public:
    static constexpr int kLayers=64,kT=4,kOwners=4,kEntries=128;
    struct Bucket { uint64_t groups=0,entries=0,gu=0,down=0,blob=0; };
    Bucket rows[kLayers][kT][kOwners][kT]{};
    strata::kernels::cpu::NativeFmt fmt[kLayers]{};
    strata::kernels::cpu::NativeServiceBudget service;
    struct Layer { uint64_t attempted[kT]{},completed[kT]{},verified_rows[kT]{},
                            cpu_jobs[kT]{},cpu_rows[kT]{},entries[kT]{},zero_jobs[kT]{}; } layer[kLayers];
    struct Window { uint64_t attempted=0,verified=0,W=0,R=0,D=0,A=0,C=0,E=0; } by_t[kT];
    DecodeLedgerSnapshot before,after;
    int64_t layers=0,k=0,experts=0;
    int workers=0;
    bool host_works=false,active=true,invalid=false,overflow=false;
    bool window_active=false,window_verified=false,window_committed=false,pool_bound=false;
    bool wait_commit_ok=false,context_limit=false,terminal_unemitted=false,terminal_seen=false;
    int current_t=0,current_a=0;
    uint64_t sequence=0,prompt_hash=0;
    int64_t prompt_tokens=0,max_new=0,context=0;
    uint64_t attempted_windows=0,verified_windows=0,verified_rows=0,dispatch_attempts=0;
    uint64_t W=0,R=0,D=0,A=0,C=0,E=0,cpu_jobs=0,cpu_rows=0,route_entries=0,expected_phases=0;
    uint64_t attempted_mask=0,completed_mask=0;
    void add(uint64_t& value,uint64_t n) {
        if (n>UINT64_MAX-value) { value=UINT64_MAX; overflow=true; } else value+=n;
    }
    uint64_t all_layers() const { return layers<1 || layers>64 ? 0 : layers==64 ? UINT64_MAX : (uint64_t{1}<<layers)-1; }
    static bool valid_format(const strata::kernels::cpu::NativeFmt& f) {
        return f.gu_type>=0 && f.gu_type<=255 && f.d_type>=0 && f.d_type<=255 &&
            f.gu_act>=0 && f.gu_act<=255 && f.d_act>=0 && f.d_act<=255 &&
            f.n_embd==2560 && f.n_ff==640 && f.gu_row && f.d_row &&
            f.up_off && f.up_off<f.down_off && f.down_off<f.bytes &&
            f.act_bytes && f.act_bytes<=strata::kernels::cpu::kNativeActBytes &&
            f.h_bytes && f.h_bytes<=strata::kernels::cpu::kNativeHBytes;
    }
    static bool same_format(const strata::kernels::cpu::NativeFmt& a,const strata::kernels::cpu::NativeFmt& b) {
        return a.gu_type==b.gu_type && a.d_type==b.d_type && a.gu_act==b.gu_act && a.d_act==b.d_act &&
            a.n_embd==b.n_embd && a.n_ff==b.n_ff && a.gu_row==b.gu_row && a.d_row==b.d_row &&
            a.up_off==b.up_off && a.down_off==b.down_off && a.bytes==b.bytes &&
            a.act_bytes==b.act_bytes && a.h_bytes==b.h_bytes;
    }
    bool envelope() const { return layers>=1 && layers<=64 && k>=1 && k<=32 && experts>=1 && experts<=32767 && workers>=1 && workers<=128; }
    void begin_window(int T) {
        if (!active) return;
        add(attempted_windows,1);
        if (!envelope() || window_active || T<1 || T>4) { invalid=true; return; }
        current_t=T; current_a=0; attempted_mask=completed_mask=0;
        window_active=true; window_verified=window_committed=false;
        add(by_t[T-1].attempted,1);
    }
    void attempt_dispatch(int64_t l,int64_t T,int64_t actual_k) {
        if (!active) return;
        add(dispatch_attempts,1);
        if (!envelope() || !window_active || l<0 || l>=layers || T<1 || T>4 || T!=current_t || actual_k!=k) {
            invalid=true; return;
        }
        const uint64_t bit=uint64_t{1}<<l;
        if (attempted_mask&bit) invalid=true;
        attempted_mask|=bit; add(layer[l].attempted[T-1],1);
    }
    void record(int64_t l,int64_t T,const int32_t* ids,const int32_t* kind,int64_t n,
                const strata::kernels::cpu::ExpertJobMulti* jobs,int count,size_t jobs_capacity,
                const strata::kernels::cpu::NativeFmt& f,int pcie_mode) {
        if (!active) return;
        if (!envelope() || !window_active || T<1 || T>4 || T!=current_t || l<0 || l>=layers ||
            n<1 || n>kEntries || n!=T*k || count<0 || count>n || (size_t)count>jobs_capacity ||
            !ids || !kind || (count && !jobs) || pcie_mode<0 || pcie_mode>2 ||
            !valid_format(f) || !same_format(fmt[l],f)) { invalid=true; return; }
        const uint64_t bit=uint64_t{1}<<l;
        if (!(attempted_mask&bit) || (completed_mask&bit)) { invalid=true; return; }
        struct Group { int owner,nt; } groups[kEntries];
        int ng=0,jcpu=0; uint64_t cpu_entry_count=0;
        for (int64_t i=0;i<n;++i) {
            if (ids[i]<0 || ids[i]>=experts || kind[i]<-1 || kind[i]>1) { invalid=true; return; }
            for (int64_t j=(i/k)*k;j<i;++j) if (ids[j]==ids[i]) { invalid=true; return; }
        }
        for (int64_t i=0;i<n;++i) {
            bool first=true; for (int64_t j=0;j<i;++j) if (ids[j]==ids[i]) first=false;
            if (!first) continue;
            int nt=0; for (int64_t j=i;j<n;++j) if (ids[j]==ids[i]) {
                if (kind[j]!=kind[i]) { invalid=true; return; } ++nt;
            }
            if (nt<1 || nt>T) { invalid=true; return; }
            const int owner=kind[i]==-1 ? 0 : kind[i]==0 ? 1 : pcie_mode ? 3 : 2;
            if (owner==0) {
                if (jcpu>=count || jobs[jcpu].nt<1 || jobs[jcpu].nt>T || jobs[jcpu].nt!=nt || !jobs[jcpu].blob) {
                    invalid=true; return;
                }
                cpu_entry_count+=(uint64_t)nt; ++jcpu;
            }
            groups[ng++]={owner,nt};
        }
        if (jcpu!=count) { invalid=true; return; }
        for (int i=0;i<ng;++i) {
            auto& b=rows[l][T-1][groups[i].owner][groups[i].nt-1];
            add(b.groups,1); add(b.entries,(uint64_t)groups[i].nt);
            add(b.gu,f.down_off); add(b.down,f.bytes-f.down_off); add(b.blob,f.bytes);
        }
        auto& L=layer[l]; const int t=(int)T-1;
        add(L.completed[t],1); add(L.verified_rows[t],(uint64_t)T);
        add(L.cpu_jobs[t],(uint64_t)count); add(L.cpu_rows[t],cpu_entry_count); add(L.entries[t],(uint64_t)n);
        if (!count) add(L.zero_jobs[t],1);
        add(cpu_jobs,(uint64_t)count); add(cpu_rows,cpu_entry_count); add(route_entries,(uint64_t)n);
        if (count) add(expected_phases,((uint64_t)count+strata::kernels::cpu::ExpertPool::kMaxSplitMulti-1)/strata::kernels::cpu::ExpertPool::kMaxSplitMulti);
        completed_mask|=bit;
    }
    void verified(int T) {
        if (!active) return;
        if (!envelope() || !window_active || window_verified || T!=current_t || T<1 || T>4) { invalid=true; return; }
        if (attempted_mask!=all_layers() || completed_mask!=all_layers()) invalid=true;
        window_verified=true; add(verified_windows,1); add(verified_rows,(uint64_t)T); add(by_t[T-1].verified,1);
    }
    void committed(int accepted,int keep) {
        if (!active) return;
        if (!window_active || !window_verified || window_committed || accepted<0 || accepted>=current_t || keep!=accepted+1) {
            invalid=true; return;
        }
        current_a=accepted; window_committed=true;
        auto& b=by_t[current_t-1];
        add(W,1); add(R,current_t); add(D,current_t-1); add(A,accepted); add(C,keep);
        add(b.W,1); add(b.R,current_t); add(b.D,current_t-1); add(b.A,accepted); add(b.C,keep);
    }
    void emitted(int64_t actual,bool terminal) {
        if (!active) return;
        if (!window_active || !window_committed || actual<0 || actual>current_a+1) { invalid=true; return; }
        if (terminal) terminal_seen=true;
        if (actual<current_a+1) { terminal_unemitted=true; if (!terminal) invalid=true; }
        add(E,(uint64_t)actual); add(by_t[current_t-1].E,(uint64_t)actual); window_active=false;
    }
    bool snapshot_values_safe() const {
        if (!before.valid || !after.valid) return false;
        for (int i=0;i<17;++i)
            if (!std::isfinite(before.real[i]) || !std::isfinite(after.real[i]) || before.real[i]<0 || after.real[i]<0) return false;
        for (int i=0;i<6;++i) if (before.count[i]<0 || after.count[i]<0) return false;
        return true;
    }
    bool complete(bool normal,int64_t emitted_done,int64_t offered_done,int64_t accepted_done) const {
        if (!snapshot_values_safe()) return false;
        for (int i=0;i<17;++i) if (after.real[i]<before.real[i]) return false;
        if (!normal || !active || invalid || overflow || !envelope() || context_limit || window_active || pool_bound || !wait_commit_ok || !terminal_seen ||
            emitted_done<0 || offered_done<0 || accepted_done<0 || E!=(uint64_t)emitted_done ||
            D!=(uint64_t)offered_done || A!=(uint64_t)accepted_done || W==0 || W!=attempted_windows || W!=verified_windows ||
            R!=verified_rows || R<W || D!=R-W || A>D || W>UINT64_MAX-A || C!=W+A || E>C ||
            !before.valid || !after.valid || after.count[0]<before.count[0] ||
            (uint64_t)(after.count[0]-before.count[0])!=verified_windows || !service.complete(workers)) return false;
        uint64_t totals[8]{};
        for (int t=0;t<4;++t) {
            const auto& b=by_t[t];
            if (b.W!=b.attempted || b.W!=b.verified || b.W>UINT64_MAX/(uint64_t)(t+1) ||
                b.R!=b.W*(uint64_t)(t+1) || b.D!=b.R-b.W || b.A>b.D ||
                b.W>UINT64_MAX-b.A || b.C!=b.W+b.A || b.E>b.C) return false;
            const uint64_t values[]={b.attempted,b.verified,b.W,b.R,b.D,b.A,b.C,b.E};
            for (int i=0;i<8;++i) { if (values[i]>UINT64_MAX-totals[i]) return false; totals[i]+=values[i]; }
        }
        const uint64_t expected[]={attempted_windows,verified_windows,W,R,D,A,C,E};
        for (int i=0;i<8;++i) if (totals[i]!=expected[i]) return false;
        if (W>UINT64_MAX/(uint64_t)layers || dispatch_attempts!=W*(uint64_t)layers) return false;
        uint64_t jtotal=0,rtotal=0,etotal=0,vram=0,pcie=0,blobs=0;
        for (int64_t l=0;l<layers;++l) for (int t=0;t<4;++t) {
            const auto& L=layer[l]; const auto& b=by_t[t];
            if (L.attempted[t]!=L.completed[t] || L.completed[t]!=b.W ||
                b.W>UINT64_MAX/(uint64_t)(t+1) || L.verified_rows[t]!=b.W*(uint64_t)(t+1) ||
                L.verified_rows[t]>UINT64_MAX/(uint64_t)k || L.entries[t]!=L.verified_rows[t]*(uint64_t)k) return false;
            uint64_t jobs=0,cr=0,entries=0;
            for (int owner=0;owner<4;++owner) for (int nt=0;nt<4;++nt) {
                const auto& row=rows[l][t][owner][nt];
                if (row.groups>UINT64_MAX/(uint64_t)(nt+1) || row.entries!=row.groups*(uint64_t)(nt+1) ||
                    (nt>t && row.groups)) return false;
                if (row.entries>UINT64_MAX-entries) return false; entries+=row.entries;
                if (owner==0) {
                    if (row.groups>UINT64_MAX-jobs || row.entries>UINT64_MAX-cr || row.blob>UINT64_MAX-blobs) return false;
                    jobs+=row.groups; cr+=row.entries; blobs+=row.blob;
                } else if (owner==1) {
                    if (row.entries>UINT64_MAX-vram) return false; vram+=row.entries;
                } else {
                    if (row.groups>UINT64_MAX-pcie) return false; pcie+=row.groups;
                }
            }
            if (jobs!=L.cpu_jobs[t] || cr!=L.cpu_rows[t] || entries!=L.entries[t] ||
                jobs>UINT64_MAX-jtotal || cr>UINT64_MAX-rtotal || entries>UINT64_MAX-etotal) return false;
            jtotal+=jobs; rtotal+=cr; etotal+=entries;
        }
        if (jtotal!=cpu_jobs || rtotal!=cpu_rows || etotal!=route_entries ||
            cpu_jobs>UINT64_MAX/2560 || expected_phases>UINT64_MAX/(3*(uint64_t)(workers+host_works))) return false;
        const uint64_t expected_count[]={verified_windows,cpu_jobs,cpu_rows,vram,pcie,blobs};
        for (int i=0;i<6;++i)
            if (before.count[i]<0 || after.count[i]<before.count[i] ||
                (uint64_t)(after.count[i]-before.count[i])!=expected_count[i]) return false;
        uint64_t quant=service.slots[0].call[1].calls;
        if (service.slots[0].call[4].calls>UINT64_MAX-quant) return false;
        quant+=service.slots[0].call[4].calls;
        if (quant!=cpu_rows) return false;
        for (int ph=0;ph<2;++ph)
            if (service.phases[ph]!=expected_phases || service.planned_tasks[ph]!=expected_phases*3*(uint64_t)(workers+host_works) ||
                service.planned_rows[ph]!=cpu_jobs*(uint64_t)(ph==0?640:2560)) return false;
        return true;
    }
    void receipt(DecodeLedgerOutput& out,const char* status,int64_t emitted_done,int64_t offered_done,int64_t accepted_done,double decode_ms) const {
        const bool normal=status && (!std::strcmp(status,"length") || !std::strcmp(status,"stop"));
        const bool coverage=complete(normal,emitted_done,offered_done,accepted_done) &&
            (std::strcmp(status,"length")!=0 || (max_new>0 && E==(uint64_t)max_new));
        out.append(false,"strata decode joined begin schema=mtp4-route-service-v1 request=%llu prompt_fnv64=%016llx prompt=%lld max_new=%lld context=%lld layers=%lld k=%lld workers=%d host=%d status=%.32s decode_ms=%.6f\n",
            (unsigned long long)sequence,(unsigned long long)prompt_hash,(long long)prompt_tokens,(long long)max_new,(long long)context,
            (long long)layers,(long long)k,workers,host_works,status?status:"unknown",decode_ms);
        out.append(false,"coverage attempted=%llu verified=%llu run_rows=%llu W=%llu R=%llu D=%llu A=%llu C=%llu E=%llu rejected=%llu commit_unemitted=%llu dispatch_attempts=%llu active_T=%d active_window=%d wait_commit=%d context_limit=%d cpu_jobs=%llu cpu_rows=%llu route_entries=%llu\n",
            (unsigned long long)attempted_windows,(unsigned long long)verified_windows,(unsigned long long)verified_rows,(unsigned long long)W,(unsigned long long)R,
            (unsigned long long)D,(unsigned long long)A,(unsigned long long)C,(unsigned long long)E,(unsigned long long)(D>=A?D-A:0),
            (unsigned long long)(C>=E?C-E:0),(unsigned long long)dispatch_attempts,current_t,window_active,wait_commit_ok,context_limit,
            (unsigned long long)cpu_jobs,(unsigned long long)cpu_rows,(unsigned long long)route_entries);
        for (int t=0;t<4;++t) if (by_t[t].attempted) {
            const auto& b=by_t[t];
            out.append(false,"window_hist T=%d attempted=%llu verified=%llu W=%llu R=%llu D=%llu A=%llu C=%llu E=%llu\n",t+1,
                (unsigned long long)b.attempted,(unsigned long long)b.verified,(unsigned long long)b.W,(unsigned long long)b.R,
                (unsigned long long)b.D,(unsigned long long)b.A,(unsigned long long)b.C,(unsigned long long)b.E);
        }
        for (int64_t l=0;l<layers && l<64;++l) {
            const auto& f=fmt[l];
            out.append(false,"format l=%lld gu=%d down=%d gu_act=%d down_act=%d gu_row=%zu down_row=%zu up=%zu down_off=%zu bytes=%zu\n",(long long)l,f.gu_type,f.d_type,f.gu_act,f.d_act,f.gu_row,f.d_row,f.up_off,f.down_off,f.bytes);
            for (int t=0;t<4;++t) {
                const auto& L=layer[l]; if (!L.attempted[t]) continue;
                out.append(false,"dispatch l=%lld T=%d attempted=%llu completed=%llu rows=%llu jobs=%llu cpu_rows=%llu entries=%llu zero_jobs=%llu\n",(long long)l,t+1,
                    (unsigned long long)L.attempted[t],(unsigned long long)L.completed[t],(unsigned long long)L.verified_rows[t],
                    (unsigned long long)L.cpu_jobs[t],(unsigned long long)L.cpu_rows[t],(unsigned long long)L.entries[t],(unsigned long long)L.zero_jobs[t]);
                for (int owner=0;owner<4;++owner) for (int nt=0;nt<4;++nt) {
                    const auto& b=rows[l][t][owner][nt]; if (!b.groups) continue;
                    out.append(false,"route l=%lld T=%d o=%d nt=%d g=%llu e=%llu gu=%llu down=%llu blob=%llu\n",(long long)l,t+1,owner,nt+1,
                        (unsigned long long)b.groups,(unsigned long long)b.entries,(unsigned long long)b.gu,(unsigned long long)b.down,(unsigned long long)b.blob);
                }
            }
        }
        for (int slot=0;slot<=workers && slot<=128;++slot) {
            const auto& s=service.slots[slot];
            for (int role=0;role<6;++role) {
                const auto& c=s.call[role]; if (!c.calls) continue;
                out.append(false,"service thread=%d role=%d calls=%llu cpu_ns=%llu wall_ns=%llu reads=%llu invalid=%d overflow=%d\n",slot-1,role,
                    (unsigned long long)c.calls,(unsigned long long)c.cpu_ns,(unsigned long long)c.wall_ns,(unsigned long long)c.reads,c.invalid,c.overflow);
            }
            for (int ph=0;ph<2;++ph) if (s.tasks[ph])
                out.append(false,"tasks thread=%d phase=%d claimed=%llu completed=%llu nonempty=%llu flat_rows=%llu overflow=%d\n",slot-1,ph,
                    (unsigned long long)s.tasks[ph],(unsigned long long)s.finished[ph],(unsigned long long)s.nonempty[ph],(unsigned long long)s.rows[ph],s.overflow);
        }
        for (int ph=0;ph<2;++ph) if (service.phases[ph]) out.append(false,"planned phase=%d phases=%llu completed=%llu tasks=%llu flat_rows=%llu\n",ph,
            (unsigned long long)service.phases[ph],(unsigned long long)service.phases_done[ph],(unsigned long long)service.planned_tasks[ph],(unsigned long long)service.planned_rows[ph]);
        if (snapshot_values_safe()) {
            for (int i=0;i<DecodeLedgerSnapshot::kReal;++i)
                if (after.real[i]!=before.real[i]) out.append(false,"existing_ms index=%d delta=%.6f\n",i,after.real[i]-before.real[i]);
            for (int i=0;i<DecodeLedgerSnapshot::kCount;++i)
                if (after.count[i]!=before.count[i]) out.append(false,"existing_count index=%d delta=%lld\n",i,(long long)(after.count[i]-before.count[i]));
        }
        out.append(true,"strata decode joined close request=%llu prompt_fnv64=%016llx status=%.32s complete=%d coverage_complete=%d service_complete=%d invalid=%d overflow=%d output_bounded=%d omitted_rows=%llu snapshot_before=%d snapshot_after=%d done_counts_available=%d commit_gpu_qualified=%d terminal_seen=%d storage_cap=262144 output_cap=262144 logical_bytes_not_dram_or_h2d=1 wall_spans_overlap=1 clock_domain=thread_cpu\n",
            (unsigned long long)sequence,(unsigned long long)prompt_hash,status?status:"unknown",coverage&&out.bounded,coverage,service.complete(workers),invalid,overflow,out.bounded,
            (unsigned long long)out.omitted,before.valid,after.valid,emitted_done>=0 && offered_done>=0 && accepted_done>=0,wait_commit_ok,terminal_seen);
    }
};
static_assert(sizeof(DecodeRouteLedger)<=256*1024,"joined request-state budget");
static_assert(sizeof(DecodeRouteLedger)+DecodeLedgerOutput::kCap+
              129*sizeof(strata::kernels::cpu::ServiceClockScope)+4096<=512*1024,
              "fixed observer state, close buffer, clock objects and formatter envelope");
} // namespace strata::core
