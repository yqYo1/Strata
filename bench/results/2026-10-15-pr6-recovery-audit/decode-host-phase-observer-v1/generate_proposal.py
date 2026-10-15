#!/usr/bin/env python3
"""Generate proposal artifacts only. Does not apply, compile or run a fixture."""
from pathlib import Path
import difflib, hashlib, json
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
A=P/'bench/results/2026-10-15-pr6-recovery-audit/decode-host-phase-observer-v1'
assert Path(__file__).resolve().parent==A
files={}
def change(name, fn):
    old=(P/name).read_text(); new=fn(old); assert old!=new,name; files[name]=(old,new)
def replace(s,old,new):
    assert s.count(old)==1,(old[:100],s.count(old)); return s.replace(old,new,1)
header=r'''// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: LGPL-3.0-or-later
// Proposal: bounded issuer-only ordinary SYCL G1 host wall observer.
#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::core {
class DecodeHostPhaseObserver {
public:
    // Every leaf is exclusive. RunTotal is their parent, not another leaf.
    enum Phase { Other, Capture, CaptureCommit, HostStaging, InitialGraphSubmit,
        LayerEventWait, PoolCallback, CopyStreamSync, NextGraphSubmit,
        TailComputeSync, TailCopySync, FinalReadout, RunTotal, PhaseCount };
    static constexpr unsigned Bins=33, LayerPhases=4, MaxLayers=64;
    static constexpr size_t OutputCap=256*1024, FooterReserve=2048;
    using ClockFn=bool (*)(uint64_t&) noexcept;
    struct Moments { uint64_t count=0,sum=0,min=UINT64_MAX,max=0; };
    struct Distribution { Moments value; uint64_t hist[Bins]{}; };
    // Marginals, never a window x T x layer cross-product or per-call vector.
    Distribution global[PhaseCount]{},by_t[4][PhaseCount]{},window_band[4]{};
    Moments layer[MaxLayers][LayerPhases]{};
    uint64_t windows=0,succeeded=0,failed=0,clock_reads=0;
    bool invalid=false,overflow=false,attached=false,bound_once=false,in_window=false;
    const int layers;
    explicit DecodeHostPhaseObserver(int n,ClockFn clock=steady_now) noexcept : layers(n),clock_(clock) {
        if (n<1 || n>int(MaxLayers) || !clock) invalid=true;
    }
    static const char* name(Phase p) noexcept {
        static const char* const names[]={"other","capture","capture_commit","host_staging",
            "initial_graphsubmit","layer_eventwait","poolcallback","copy_stream_sync",
            "next_graphsubmit","tail_compute_sync","tail_copy_sync","finalreadout","run_total"};
        return unsigned(p)<PhaseCount?names[p]:"invalid";
    }
    static bool steady_now(uint64_t& n) noexcept {
        const auto v=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (v<0) { n=0; return false; } n=uint64_t(v); return true;
    }
    void add(uint64_t& dst,uint64_t n) noexcept {
        if (n>UINT64_MAX-dst) { dst=UINT64_MAX; overflow=true; } else dst+=n;
    }
    bool now(uint64_t& n) noexcept {
        add(clock_reads,1);
        if (!clock_ || !clock_(n)) { invalid=true; n=0; return false; }
        return true;
    }
    static unsigned bin(uint64_t n) noexcept {
        // 0 = {0}; i>=1 = [4^(i-1),4^i-1], last = [2^62,2^64-1].
        if (!n) return 0;
        unsigned b=1; while ((n>>=2)!=0) ++b; return b;
    }
    static uint64_t low(unsigned b) noexcept { return b?uint64_t{1}<<(2*(b-1)):0; }
    static uint64_t high(unsigned b) noexcept {
        return b==0?0:b==32?UINT64_MAX:(uint64_t{1}<<(2*b))-1;
    }
    void record(Moments& m,uint64_t n) noexcept {
        add(m.count,1); add(m.sum,n);
        if (n<m.min) m.min=n;
        if (n>m.max) m.max=n;
    }
    void record(Distribution& d,uint64_t n) noexcept {
        record(d.value,n); add(d.hist[bin(n)],1);
    }
    void sample(Phase p,int T,int l,uint64_t n) noexcept {
        if (unsigned(p)>=PhaseCount || T<1 || T>4) { invalid=true; return; }
        record(global[p],n); record(by_t[T-1][p],n);
        const int i=p==LayerEventWait?0:p==PoolCallback?1:p==CopyStreamSync?2:p==NextGraphSubmit?3:-1;
        if (i>=0) {
            if (l<0 || l>=layers || l>=int(MaxLayers)) invalid=true;
            else record(layer[l][i],n);
        }
    }
    static unsigned quantile_bin(const Distribution& d,unsigned percent) noexcept {
        if (!d.value.count || percent<1 || percent>100) return 0;
        const uint64_t rank=(d.value.count/100)*percent+((d.value.count%100)*percent+99)/100;
        uint64_t seen=0;
        for (unsigned b=0;b<Bins;++b) {
            if (d.hist[b]>=rank-seen) return b;
            seen+=d.hist[b]; // seen < rank, cannot overflow
        }
        return Bins; // detected by receipt, not silently an exact quantile
    }
    struct Output {
        char* bytes; size_t used=0; bool bounded=true;
        bool append(bool footer,const char* format,...) noexcept {
            char line[1024]; va_list a; va_start(a,format);
            const int n=std::vsnprintf(line,sizeof line,format,a); va_end(a);
            const size_t limit=footer?OutputCap:OutputCap-FooterReserve;
            if (!bytes || n<0 || size_t(n)>=sizeof line || size_t(n)>limit || used>limit-size_t(n)) {
                bounded=false; return false;
            }
            std::memcpy(bytes+used,line,size_t(n)); used+=size_t(n); return true;
        }
    };
    void print_moments(Output& out,const char* view,int a,int b,const Moments& m) const noexcept {
        if (!m.count) return;
        out.append(false,"host_moment view=%s a=%d b=%d n=%llu sum_ns=%llu min_ns=%llu max_ns=%llu\n",
            view,a,b,(unsigned long long)m.count,(unsigned long long)m.sum,
            (unsigned long long)m.min,(unsigned long long)m.max);
    }
    void print_distribution(Output& out,const char* view,int a,int b,const Distribution& d) const noexcept {
        if (!d.value.count) return;
        print_moments(out,view,a,b,d.value);
        const unsigned q50=quantile_bin(d,50),q95=quantile_bin(d,95);
        if (q50>=Bins || q95>=Bins) { out.bounded=false; return; }
        out.append(false,"host_quantile_interval view=%s a=%d b=%d p50_lo_ns=%llu p50_hi_ns=%llu p95_lo_ns=%llu p95_hi_ns=%llu\n",
            view,a,b,(unsigned long long)low(q50),(unsigned long long)high(q50),
            (unsigned long long)low(q95),(unsigned long long)high(q95));
        // One bounded line with 33 raw counts, enough to recompute interval ranks.
        out.append(false,"host_hist4 view=%s a=%d b=%d bins=33 counts=",view,a,b);
        for (unsigned i=0;i<Bins;++i) out.append(false,"%s%llu",i?",":"",(unsigned long long)d.hist[i]);
        out.append(false,"\n");
    }
    bool structurally_complete(uint64_t attempted,uint64_t verified) const noexcept {
        if (invalid || overflow || attached || !bound_once || in_window || failed || !windows ||
            windows!=attempted || succeeded!=verified || windows!=succeeded) return false;
        uint64_t leaf_sum=0;
        for (int p=0;p<RunTotal;++p) {
            if (global[p].value.sum>UINT64_MAX-leaf_sum) return false;
            leaf_sum+=global[p].value.sum;
        }
        if (leaf_sum!=global[RunTotal].value.sum || global[RunTotal].value.count!=windows) return false;
        for (int p=Capture;p<PhaseCount;++p) {
            if (p>=LayerEventWait && p<=NextGraphSubmit) continue;
            if (global[p].value.count!=windows) return false;
        }
        for (int p=0;p<PhaseCount;++p) {
            uint64_t count=0,sum=0,hist[Bins]{};
            for (int t=0;t<4;++t) {
                const auto& d=by_t[t][p];
                if (d.value.count>UINT64_MAX-count || d.value.sum>UINT64_MAX-sum) return false;
                count+=d.value.count; sum+=d.value.sum;
                for (unsigned b=0;b<Bins;++b) {
                    if (d.hist[b]>UINT64_MAX-hist[b]) return false;
                    hist[b]+=d.hist[b];
                }
            }
            if (count!=global[p].value.count || sum!=global[p].value.sum) return false;
            uint64_t total=0;
            for (unsigned b=0;b<Bins;++b) {
                if (hist[b]!=global[p].hist[b] || hist[b]>UINT64_MAX-total) return false;
                total+=hist[b];
            }
            if (total!=count) return false;
        }
        for (int l=0;l<layers;++l) for (unsigned p=0;p<LayerPhases;++p)
            if (layer[l][p].count!=windows) return false;
        return true;
    }
    size_t receipt(char* bytes,uint64_t request,const char* status,bool joined_complete,
                   uint64_t attempted,uint64_t verified) const noexcept {
        Output out{bytes};
        out.append(false,"strata decode_host_phases begin schema=decode-host-phases-v1 request=%llu status=%.32s clock=steady_ns issuer_only=1 G=1 cap_bytes=%zu\n",
            (unsigned long long)request,status?status:"unknown",OutputCap);
        out.append(false,"host_schema parent=run_total children=other,capture,capture_commit,host_staging,initial_graphsubmit,layer_eventwait,poolcallback,copy_stream_sync,next_graphsubmit,tail_compute_sync,tail_copy_sync,finalreadout exclusive_children=1 poolcallback_includes_shared_hit_graphsubmit=1 hist=base4_quantile_intervals window_bands=1,2..8,9..64,65+ layer_phase_ids=5,6,7,8\n");
        for (int p=0;p<PhaseCount;++p) {
            out.append(false,"host_phase id=%d name=%s\n",p,name(Phase(p)));
            print_distribution(out,"all",p,0,global[p]);
            for (int t=0;t<4;++t) print_distribution(out,"T",p,t+1,by_t[t][p]);
        }
        for (int l=0;l<layers && l<int(MaxLayers);++l)
            for (unsigned p=0;p<LayerPhases;++p) print_moments(out,"layer",l,int(p)+LayerEventWait,layer[l][p]);
        for (int b=0;b<4;++b) print_distribution(out,"window_band",b,RunTotal,window_band[b]);
        const bool complete=joined_complete && structurally_complete(attempted,verified) && out.bounded;
        out.append(true,"strata decode_host_phases close request=%llu complete=%d joined_complete=%d windows=%llu succeeded=%llu failed=%llu attempted=%llu verified=%llu clock_reads=%llu invalid=%d overflow=%d attached=%d in_window=%d output_bounded=%d reason=%s\n",
            (unsigned long long)request,complete,joined_complete,(unsigned long long)windows,
            (unsigned long long)succeeded,(unsigned long long)failed,(unsigned long long)attempted,
            (unsigned long long)verified,(unsigned long long)clock_reads,invalid,overflow,attached,in_window,out.bounded,
            complete?"ok":"incomplete_diagnostic");
        return out.used;
    }
private:
    ClockFn clock_;
};
static_assert(sizeof(DecodeHostPhaseObserver)<64*1024,"request storage budget");

// Only the ordinary serve issuer creates this borrow. It never restores a previous pointer.
// Declare after DecodeServiceBudgetScope so early-exit detaches before owner close/destruction.
class DecodeHostPhaseBorrow {
    DecodeHostPhaseObserver** slot_=nullptr;
    DecodeHostPhaseObserver* owner_=nullptr;
public:
    DecodeHostPhaseBorrow(DecodeHostPhaseObserver*& slot,DecodeHostPhaseObserver* owner) noexcept {
        if (!owner) return;
        if (slot || owner->attached) {
            owner->invalid=true;
            if (slot) { slot->invalid=true; slot=nullptr; } // refuse stale/colliding pointer, never restore it
            return;
        }
        slot_=&slot; owner_=owner; slot=owner; owner->attached=owner->bound_once=true;
    }
    void detach() noexcept {
        if (!owner_) return;
        if (*slot_==owner_) *slot_=nullptr;
        else owner_->invalid=true;
        owner_->attached=false; slot_=nullptr; owner_=nullptr;
    }
    ~DecodeHostPhaseBorrow() { detach(); }
    DecodeHostPhaseBorrow(const DecodeHostPhaseBorrow&)=delete;
    DecodeHostPhaseBorrow& operator=(const DecodeHostPhaseBorrow&)=delete;
};

// A single active exclusive leaf. Every transition closes the prior leaf at
// the SAME host timestamp that opens the next. Parent uses the first/last stamps.
// Counter-update/clock overhead is observed wall; no GPU/device clock join.
class DecodeHostWindow {
    DecodeHostPhaseObserver* owner_=nullptr;
    int t_=0,layer_=-1;
    DecodeHostPhaseObserver::Phase phase_=DecodeHostPhaseObserver::Other;
    uint64_t start_=0,last_=0,ordinal_=0;
    bool valid_=false;
    bool read(uint64_t& n) noexcept {
        if (!owner_->now(n) || n<last_) { owner_->invalid=true; valid_=false; return false; }
        return valid_;
    }
    void close_leaf(uint64_t n) noexcept { owner_->sample(phase_,t_,layer_,n-last_); last_=n; }
public:
    explicit DecodeHostWindow(DecodeHostPhaseObserver* owner,int T) noexcept : owner_(owner),t_(T) {
        if (!owner_) return; // OFF: zero clocks, allocation or output
        if (owner_->in_window || !owner_->attached || T<1 || T>4) { owner_->invalid=true; owner_=nullptr; return; }
        owner_->in_window=true; owner_->add(owner_->windows,1); ordinal_=owner_->windows;
        valid_=owner_->now(start_); last_=start_;
    }
    void phase(DecodeHostPhaseObserver::Phase p,int layer=-1) noexcept {
        if (!owner_) return;
        if (unsigned(p)>=DecodeHostPhaseObserver::RunTotal) { owner_->invalid=true; return; }
        uint64_t n=0;
        if (read(n)) close_leaf(n);
        phase_=p; layer_=layer;
    }
    void envelope(bool eligible) noexcept { if (owner_ && !eligible) owner_->invalid=true; }
    void finish(bool success) noexcept {
        if (!owner_) return;
        uint64_t n=0;
        if (read(n)) {
            close_leaf(n); owner_->sample(DecodeHostPhaseObserver::RunTotal,t_,-1,n-start_);
            const int b=ordinal_==1?0:ordinal_<=8?1:ordinal_<=64?2:3;
            owner_->record(owner_->window_band[b],n-start_);
        }
        owner_->add(success?owner_->succeeded:owner_->failed,1);
        owner_->in_window=false; owner_=nullptr;
    }
    ~DecodeHostWindow() { finish(false); } // original inference return/error is untouched
    DecodeHostWindow(const DecodeHostWindow&)=delete;
    DecodeHostWindow& operator=(const DecodeHostWindow&)=delete;
};
} // namespace strata::core
'''
files['include/strata/core/decode_host_phases.hpp']=('',header)
def service(s):
    s=replace(s,'#include "strata/core/cpu_gu_reuse_observer.hpp"','#include "strata/core/cpu_gu_reuse_observer.hpp"\n#include "strata/core/decode_host_phases.hpp"')
    s=replace(s,'    std::unique_ptr<CpuGuReuseObserver> gu_reuse_;','    std::unique_ptr<CpuGuReuseObserver> gu_reuse_;\n    std::unique_ptr<DecodeHostPhaseObserver> host_phases_;\n    bool host_requested_=false;\n    void finish_host(bool joined_complete,const char* status);')
    return replace(s,'    ~DecodeServiceBudgetScope();','    ~DecodeServiceBudgetScope();\n    DecodeHostPhaseObserver* host_phases() const noexcept { return host_phases_.get(); }')
change('include/strata/core/decode_service_budget.hpp',service)
def expert(s):
    s=replace(s,'uint64_t joined_request_sequence=0;','bool decode_host_phases_requested() {\n    static const bool on=[] {\n        const char* v=std::getenv("STRATA_DECODE_HOST_PHASES");\n        return v && !std::strcmp(v,"1"); // exec-time configuration, read only\n    }();\n    return on;\n}\nuint64_t joined_request_sequence=0;')
    s=replace(s,'    if (!requested_) return; // no clocks, allocation, prompt scan or output on the off path','    host_requested_=decode_host_phases_requested();\n    if (!requested_) return; // no clocks, allocation, prompt scan or output on the off path')
    s=replace(s,'    d.decode_route_ledger=state_.get(); // after prompt; one synchronous ordinary issuer','    d.decode_route_ledger=state_.get(); // after prompt; one synchronous ordinary issuer\n    if (host_requested_) host_phases_.reset(new(std::nothrow) DecodeHostPhaseObserver((int)layers));')
    s=replace(s,'DecodeServiceBudgetScope::~DecodeServiceBudgetScope() { finish("error_or_early_exit",-1,-1,-1,0); }',r'''void DecodeServiceBudgetScope::finish_host(bool joined_complete,const char* status) {
    if (!host_requested_) return;
    if (!host_phases_) {
        std::fprintf(stderr,"strata decode_host_phases close request=%llu status=%.32s eligible=%d complete=0 reason=%s\n",
            (unsigned long long)sequence_,status?status:"unknown",state_!=nullptr,
            !requested_?"joined_required":!state_?"joined_ineligible":"state_allocation");
        return;
    }
    std::unique_ptr<char[]> bytes(new(std::nothrow) char[DecodeHostPhaseObserver::OutputCap]);
    if (!bytes) {
        std::fprintf(stderr,"strata decode_host_phases close request=%llu complete=0 reason=output_allocation\n",
            (unsigned long long)sequence_); return;
    }
    // Identity/work comes from the joined receipt with the same sequence. No worker reads here.
    const size_t used=host_phases_->receipt(bytes.get(),sequence_,status,joined_complete,
        state_->attempted_windows,state_->verified_windows);
    std::fwrite(bytes.get(),1,used,stderr); // <=256 KiB per request, only after close
}
DecodeServiceBudgetScope::~DecodeServiceBudgetScope() { finish("error_or_early_exit",-1,-1,-1,0); }''')
    s=replace(s,'    if (!requested_) return;\n    if (!state_) {','    if (!requested_) { finish_host(false,status); return; }\n    if (!state_) {\n        finish_host(false,status);')
    s=replace(s,'    if (!bytes) {\n        std::fprintf(stderr,"strata decode joined close request=', '    if (!bytes) {\n        finish_host(false,status);\n        std::fprintf(stderr,"strata decode joined close request=')
    s=replace(s,'    state_->receipt(output,status,emitted,offered,accepted,decode_ms);\n    state_->active=false;', '''    state_->receipt(output,status,emitted,offered,accepted,decode_ms);
    const bool normal=status && (!std::strcmp(status,"length") || !std::strcmp(status,"stop"));
    if (host_requested_) finish_host(output.bounded && state_->complete(normal,emitted,offered,accepted),status);
    state_->active=false;''')
    return s
change('src/core/expert_source.cpp',expert)
def verifyhpp(s):
    s=replace(s,'#include "strata/core/watchdog_lifecycle.hpp"','#include "strata/core/watchdog_lifecycle.hpp"\n#include "strata/core/decode_host_phases.hpp"')
    s=replace(s,'    Verifier() = default;','    Verifier() = default;\n    DecodeHostPhaseBorrow observe_decode_host_phases(DecodeHostPhaseObserver* owner) noexcept {\n        return DecodeHostPhaseBorrow(host_phases_,owner);\n    }')
    return replace(s,'private:\n    void signal_gpu_waits();','private:\n    DecodeHostPhaseObserver* host_phases_=nullptr; // issuer-only RAII request borrow; never captured\n    void signal_gpu_waits();')
change('include/strata/core/verify.hpp',verifyhpp)
def verifycpp(s):
    start=s.index('bool Verifier::run('); end=s.index('\nvoid Verifier::set_plan_slot',start)
    a=s[start:end]
    a=replace(a,'    using namespace strata::kernels;','''    using namespace strata::kernels;
    using HostPhase=DecodeHostPhaseObserver;
    DecodeHostWindow host_window(host_phases_,T); // ordinary issuer, no worker/callback writer''')
    a=replace(a,'    if (!capture(T, err) || !capture_commit(err)) return false;','''    host_window.envelope(next_==nullptr && lb_==0 && le_==g.n_layers);
    host_window.phase(HostPhase::Capture);
    if (!capture(T, err)) return false;
    host_window.phase(HostPhase::CaptureCommit);
    if (!capture_commit(err)) return false;
    host_window.envelope(groups_[T]<=1); // fail diagnostic only; never change inference dispatch
    host_window.phase(HostPhase::HostStaging);''')
    a=replace(a,'#ifdef STRATA_ENABLE_SYCL\n    const cudaError_t le =', '    host_window.phase(HostPhase::InitialGraphSubmit);\n#ifdef STRATA_ENABLE_SYCL\n    const cudaError_t le =')
    a=replace(a,'#endif\n    if (le != cudaSuccess)', '#endif\n    host_window.phase(HostPhase::Other);\n    if (le != cudaSuccess)')
    a=replace(a,'        const cudaError_t es = cudaEventSynchronize(layer_done_);','''        host_window.phase(HostPhase::LayerEventWait,(int)l);
        const cudaError_t es = cudaEventSynchronize(layer_done_);
        host_window.phase(HostPhase::Other);''')
    a=replace(a,'        const Clock::time_point b = Clock::now();','''        const Clock::time_point b = Clock::now();
        // This broad caller scope includes plan publish/fallback shared-hit graphsubmit.
        // publish_plan/launch_work and pool workers do not touch host_window.
        host_window.phase(HostPhase::PoolCallback,(int)l);''')
    a=replace(a,'        ms_pool += ms_since(b);','        ms_pool += ms_since(b);\n        host_window.phase(HostPhase::Other);')
    a=replace(a,'        const cudaError_t copies = cudaStreamSynchronize(copy_);','''        host_window.phase(HostPhase::CopyStreamSync,(int)l);
        const cudaError_t copies = cudaStreamSynchronize(copy_);
        host_window.phase(HostPhase::Other);''')
    a=replace(a,'        const cudaError_t next = strata::sycl_backend::compat::graph_launch_with_completion(','        host_window.phase(HostPhase::NextGraphSubmit,(int)l);\n        const cudaError_t next = strata::sycl_backend::compat::graph_launch_with_completion(')
    a=replace(a,'            segments_[T][size_t(k + 1)], cs_, layer_done_);','            segments_[T][size_t(k + 1)], cs_, layer_done_);\n        host_window.phase(HostPhase::Other);')
    a=replace(a,'    const cudaError_t se = cudaStreamSynchronize(cs_);','    host_window.phase(HostPhase::TailComputeSync);\n    const cudaError_t se = cudaStreamSynchronize(cs_);\n    host_window.phase(HostPhase::Other);')
    a=replace(a,'    const cudaError_t copy_status = cudaStreamSynchronize(copy_); // Host callbacks cannot reach the next window.','''    host_window.phase(HostPhase::TailCopySync);
    const cudaError_t copy_status = cudaStreamSynchronize(copy_); // Host callbacks cannot reach the next window.
    host_window.phase(HostPhase::Other);''')
    a=replace(a,'    window_scope.drained(); // Both stage streams completed; sampling below has no mapped spin waits.','    window_scope.drained(); // Both stage streams completed; sampling below has no mapped spin waits.\n    host_window.phase(HostPhase::FinalReadout);')
    a=replace(a,'    progress_beat();\n    return true;','    progress_beat();\n    host_window.finish(true);\n    return true;')
    return s[:start]+a+s[end:]
change('src/core/verify.cpp',verifycpp)
def generate(s):
    s=replace(s,'                n,max_new,o.max_context,mtp.coupled(),decode_sources);','''                n,max_new,o.max_context,mtp.coupled(),decode_sources);
            // Destroy/detach before decode_service on every early return. Never restore an old borrow.
            auto host_phase_borrow=ver.observe_decode_host_phases(decode_service.host_phases());''')
    return replace(s,'            decode_service.finish(finish,produced_n,draft_offered,draft_accepted,decode_ms);','''            host_phase_borrow.detach();
            decode_service.finish(finish,produced_n,draft_offered,draft_accepted,decode_ms);''')
change('src/program/generate.cpp',generate)
rows=[]; diffs=[]
for name,(old,new) in sorted(files.items()):
    out=A/'files'/name; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(new)
    diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
    rows.append({'path':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'proposed_sha256':hashlib.sha256(new.encode()).hexdigest(),'proposed_bytes':len(new.encode())})
(A/'source.diff').write_text(''.join(diffs))
(A/'source-hashes.json').write_text(json.dumps({'schema':'decode-host-phase-observer-proposal-v1','applied':False,'compiled':False,'executed':False,'product_source_changes':False,'base_binary_sha256':'92acf71e6b4fd914f68c8cefcae7e03f170adf62b5e3b94cf31af747344183f7','files':rows},indent=2)+'\n')
