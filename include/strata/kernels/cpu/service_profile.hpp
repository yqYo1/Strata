// Decode-only CPU leaf service diagnostic. Repository LICENSE; independently authored.
#pragma once

#if defined(STRATA_CPU_SERVICE_PROFILE)
#if !defined(__linux__)
#error STRATA_CPU_SERVICE_PROFILE requires Linux thread CPU clocks
#endif

#include "strata/kernels/cpu/native_expert.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace strata::kernels::cpu::service {

enum class Op : unsigned { Gu, HiddenQuant, Down };
inline constexpr unsigned kFormats = 7, kWidths = 4, kOps = 3, kWorkers = 64;
inline constexpr unsigned kCells = kFormats * kWidths * kOps;
inline constexpr int kTypes[kFormats][2] = {
    {18,20}, {18,42}, {21,20}, {21,42}, {22,20}, {22,42}, {23,20}
};

struct Counts {
    uint64_t calls=0, rows=0, token_rows=0, wall_ns=0, cpu_ns=0;
};

// Validate every field before changing any of them, including in snapshot sums.
inline bool add(Counts& a, const Counts& b) noexcept {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    if (b.calls>max-a.calls || b.rows>max-a.rows || b.token_rows>max-a.token_rows ||
        b.wall_ns>max-a.wall_ns || b.cpu_ns>max-a.cpu_ns) return false;
    a.calls+=b.calls; a.rows+=b.rows; a.token_rows+=b.token_rows;
    a.wall_ns+=b.wall_ns; a.cpu_ns+=b.cpu_ns; return true;
}

inline bool subtract(const Counts& a, const Counts& b, Counts& out) noexcept {
    if (a.calls<b.calls || a.rows<b.rows || a.token_rows<b.token_rows ||
        a.wall_ns<b.wall_ns || a.cpu_ns<b.cpu_ns) return false;
    out={a.calls-b.calls,a.rows-b.rows,a.token_rows-b.token_rows,
         a.wall_ns-b.wall_ns,a.cpu_ns-b.cpu_ns}; return true;
}

inline bool clock_ns(clockid_t id, uint64_t& out) noexcept {
    timespec ts{};
    if (clock_gettime(id,&ts) || ts.tv_sec<0 || ts.tv_nsec<0 || ts.tv_nsec>=1000000000 ||
        static_cast<uint64_t>(ts.tv_sec)>(std::numeric_limits<uint64_t>::max()-ts.tv_nsec)/1000000000)
        return false;
    out=static_cast<uint64_t>(ts.tv_sec)*1000000000+static_cast<uint64_t>(ts.tv_nsec);
    return true;
}

struct Snapshot {
    bool enabled=false, valid=false;
    // 1 input,2 clock,4 counter,8 worker,16 quiescence,32 delta,64 slot owner.
    uint64_t error_mask=0;
    std::array<Counts,kCells> cells{};
    struct Thread { uint64_t tid=0, errors=0; Counts total{}; };
    std::array<Thread,kWorkers+1> threads{};
};

class Profile {
    // Exactly one pool worker writes each slot; the calling thread owns slot64.
    // Batch retirement acquires parked_; the mutex also publishes the dedicated
    // dispatch host's HQ/main-slot writes to the separate generation thread.
    struct alignas(64) Slot { uint64_t tid=0, errors=0; std::array<Counts,kCells> cells{}; };
    std::array<Slot,kWorkers+1> slots_{};
    const bool enabled_;
    std::atomic<bool> overflow_{false};
    mutable std::mutex mutex_;
public:
    explicit Profile(bool enabled=false) noexcept : enabled_(enabled) {}
    bool enabled() const noexcept { return enabled_; }
    class BatchScope {
        std::unique_lock<std::mutex> lock_;
    public:
        BatchScope(Profile& p,NativeGuPhase phase) : lock_(p.mutex_,std::defer_lock) {
            if (p.enabled_ && phase==NativeGuPhase::Decode) lock_.lock();
        }
    };

    Snapshot snapshot(bool quiescent) const {
        Snapshot out; out.enabled=enabled_;
        if (!enabled_) return out;
        if (!quiescent) {out.error_mask=16;return out;}
        std::lock_guard<std::mutex> lock(mutex_); // no slot reads during a native decode batch
        out.valid=!overflow_.load(std::memory_order_relaxed);
        if (!out.valid) out.error_mask=8;
        for (unsigned i=0;i<slots_.size();++i) {
            const auto& s=slots_[i]; auto& t=out.threads[i]; t.tid=s.tid; t.errors=s.errors;
            if (s.errors) {out.valid=false;out.error_mask|=s.errors;}
            for (unsigned j=0;j<kCells;++j)
                if (!add(out.cells[j],s.cells[j]) || !add(t.total,s.cells[j])) {out.valid=false;out.error_mask|=4;}
        }
        return out;
    }

    // The dispatch caller owns BatchScope until these worker scopes have joined.
    // Slot counters themselves have one writer and never take the snapshot mutex.
    class Scope {
        Slot* slot_=nullptr; Counts* cell_=nullptr;
        uint64_t wall_=0, cpu_=0; int rows_=0, nt_=0;
    public:
        Scope(Profile& p,int worker,const NativeFmt& f,Op op,int nt,int rows,NativeGuPhase phase) noexcept {
            if (!p.enabled_ || phase!=NativeGuPhase::Decode) return;
            if (worker < -1 || worker>=static_cast<int>(kWorkers)) {
                p.overflow_.store(true,std::memory_order_relaxed); return;
            }
            slot_=&p.slots_[worker<0?kWorkers:static_cast<unsigned>(worker)];
            static thread_local const long raw_tid=syscall(SYS_gettid);
            const uint64_t tid=static_cast<uint64_t>(raw_tid);
            if (raw_tid<=0 || (slot_->tid && slot_->tid!=tid)) {slot_->errors|=64;return;}
            slot_->tid=tid;
            unsigned fmt=kFormats;
            for (unsigned i=0;i<kFormats;++i)
                if (kTypes[i][0]==f.gu_type && kTypes[i][1]==f.d_type) {fmt=i;break;}
            const auto operation=static_cast<unsigned>(op);
            const int maximum=op==Op::Down?2560:640;
            if (fmt==kFormats || f.n_embd!=2560 || f.n_ff!=640 || nt<1 || nt>4 ||
                operation>=kOps || rows<1 || rows>maximum) {slot_->errors|=1;return;}
            rows_=rows; nt_=nt;
            cell_=&slot_->cells[(fmt*kWidths+static_cast<unsigned>(nt-1))*kOps+operation];
            // CPU interval is nested inside wall. Classification/counter updates
            // are outside this interval, but remain charged to model execution.
            if (!clock_ns(CLOCK_MONOTONIC,wall_) || !clock_ns(CLOCK_THREAD_CPUTIME_ID,cpu_)) {
                slot_->errors|=2;cell_=nullptr;
            }
        }
        Scope(const Scope&)=delete;
        Scope& operator=(const Scope&)=delete;
        ~Scope() noexcept {
            if (!cell_) return;
            uint64_t cpu=0,wall=0;
            if (!clock_ns(CLOCK_THREAD_CPUTIME_ID,cpu) || !clock_ns(CLOCK_MONOTONIC,wall) ||
                cpu<cpu_ || wall<wall_ || cpu-cpu_>wall-wall_) {slot_->errors|=2;return;}
            const Counts sample{1,static_cast<uint64_t>(rows_),static_cast<uint64_t>(rows_*nt_),wall-wall_,cpu-cpu_};
            if (!add(*cell_,sample)) slot_->errors|=4;
        }
    };
};

// Both snapshots must come from the same live pool, outside all run calls.
// Aggregate CPU/wall service sums overlap across workers; neither is a critical path.
inline bool write_delta(FILE* file,uint64_t request,const Snapshot& before,const Snapshot& after) noexcept {
    if (!before.enabled && !after.enabled) return true;
    std::array<Counts,kCells> delta{};
    std::array<Counts,kWorkers+1> thread_delta{};
    Counts bin_sum{},thread_sum{};
    uint64_t error_mask=before.error_mask|after.error_mask;
    bool valid=before.enabled && after.enabled && before.valid && after.valid;
    for (unsigned i=0;i<kCells;++i)
        if (!subtract(after.cells[i],before.cells[i],delta[i]) || !add(bin_sum,delta[i])) valid=false;
    for (unsigned i=0;i<before.threads.size();++i) {
        if ((before.threads[i].tid && before.threads[i].tid!=after.threads[i].tid) ||
            !subtract(after.threads[i].total,before.threads[i].total,thread_delta[i]) ||
            !add(thread_sum,thread_delta[i]) || (thread_delta[i].calls && !after.threads[i].tid)) valid=false;
    }
    if (bin_sum.calls!=thread_sum.calls || bin_sum.rows!=thread_sum.rows ||
        bin_sum.token_rows!=thread_sum.token_rows || bin_sum.wall_ns!=thread_sum.wall_ns ||
        bin_sum.cpu_ns!=thread_sum.cpu_ns) valid=false;
    if (!valid) error_mask|=32;
    std::fprintf(file,"strata CPU_SERVICE request=%llu valid=%d error_mask=%llu formats=7 nt_max=4 operations=GU,HQ,DOWN "
        "wall_clock=CLOCK_MONOTONIC cpu_clock=CLOCK_THREAD_CPUTIME_ID scope=decode_pool_leaves "
        "overhead=charged physical_RAM_bytes=unknown critical_path=unknown\n",
        static_cast<unsigned long long>(request),int(valid),static_cast<unsigned long long>(error_mask));
    if (!valid) return false; // never publish partial timing as valid attribution
    constexpr const char* names[]={"GU","HQ","DOWN"};
    for (unsigned f=0;f<kFormats;++f) for (unsigned nt=1;nt<=kWidths;++nt) for (unsigned op=0;op<kOps;++op) {
        const auto& d=delta[(f*kWidths+nt-1)*kOps+op]; if (!d.calls) continue;
        std::fprintf(file,"strata CPU_SERVICE_BIN request=%llu gu=%d down=%d nt=%u op=%s "
            "calls=%llu rows=%llu token_rows=%llu wall_ns=%llu cpu_ns=%llu\n",
            static_cast<unsigned long long>(request),kTypes[f][0],kTypes[f][1],nt,names[op],
            static_cast<unsigned long long>(d.calls),static_cast<unsigned long long>(d.rows),
            static_cast<unsigned long long>(d.token_rows),static_cast<unsigned long long>(d.wall_ns),
            static_cast<unsigned long long>(d.cpu_ns));
    }
    for (unsigned i=0;i<after.threads.size();++i) {
        const auto& d=thread_delta[i];
        if (!d.calls) continue;
        std::fprintf(file,"strata CPU_SERVICE_THREAD request=%llu slot=%u tid=%llu calls=%llu wall_ns=%llu cpu_ns=%llu\n",
            static_cast<unsigned long long>(request),i,static_cast<unsigned long long>(after.threads[i].tid),
            static_cast<unsigned long long>(d.calls),static_cast<unsigned long long>(d.wall_ns),
            static_cast<unsigned long long>(d.cpu_ns));
    }
    return true;
}

} // namespace strata::kernels::cpu::service
#endif // STRATA_CPU_SERVICE_PROFILE
