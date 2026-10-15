// SPDX-FileCopyrightText: 2026 yayoi
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
        const auto sum_into=[](uint64_t& dst,uint64_t n) noexcept {
            if (n>UINT64_MAX-dst) return false;
            dst+=n; return true;
        };
        // Fixed ordinal bands must partition exactly the same RunTotal samples.
        const uint64_t expected_band_counts[]={1,
            windows>1?(windows<8?windows:8)-1:0,
            windows>8?(windows<64?windows:64)-8:0,
            windows>64?windows-64:0};
        uint64_t band_count=0,band_sum=0,band_hist[Bins]{};
        for (unsigned b=0;b<4;++b) {
            const auto& d=window_band[b];
            if (d.value.count!=expected_band_counts[b] ||
                !sum_into(band_count,d.value.count) || !sum_into(band_sum,d.value.sum)) return false;
            uint64_t count=0;
            for (unsigned i=0;i<Bins;++i)
                if (!sum_into(count,d.hist[i]) || !sum_into(band_hist[i],d.hist[i])) return false;
            if (count!=d.value.count) return false;
        }
        if (band_count!=global[RunTotal].value.count || band_sum!=global[RunTotal].value.sum) return false;
        for (unsigned i=0;i<Bins;++i)
            if (band_hist[i]!=global[RunTotal].hist[i]) return false;
        // Each layer appears once per window, and all layer moments must also
        // conserve the corresponding global phase. Checked additions avoid wrap.
        for (unsigned p=0;p<LayerPhases;++p) {
            uint64_t count=0,sum=0;
            for (int l=0;l<layers;++l) {
                const auto& m=layer[l][p];
                if (m.count!=windows || !sum_into(count,m.count) || !sum_into(sum,m.sum)) return false;
            }
            const auto& g=global[int(LayerEventWait)+int(p)].value;
            if (count!=g.count || sum!=g.sum) return false;
        }
        return true;
    }
    size_t receipt(char* bytes,uint64_t request,const char* status,bool joined_complete,
                   uint64_t attempted,uint64_t verified) const noexcept {
        Output out{bytes};
        out.append(false,"strata decode_host_phases begin schema=decode-host-phases-v2 request=%llu status=%.32s clock=steady_ns issuer_only=1 G=1 cap_bytes=%zu\n",
            (unsigned long long)request,status?status:"unknown",OutputCap);
        out.append(false,"host_schema parent=run_total children=other,capture,capture_commit,host_staging,initial_graphsubmit,layer_eventwait,poolcallback,copy_stream_sync,next_graphsubmit,tail_compute_sync,tail_copy_sync,finalreadout exclusive_children=1 band_and_layer_conservation=1 poolcallback_includes_shared_hit_graphsubmit=1 hist=base4_quantile_intervals window_bands=1,2..8,9..64,65+ layer_phase_ids=5,6,7,8\n");
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
            // slot is unproven: inspect only its value, never its pointee.
            // A still-live prior borrow diagnoses its own known owner on detach.
            slot=nullptr; // refuse stale/colliding pointer, never restore it
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
