// Main-authored private diagnostic; repository LICENSE. No production policy change.
#pragma once
#include "strata/kernels/cpu/native_expert.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <unistd.h>
namespace pool_observer {
enum Kind { Control, Spin, CvWaitScope, GuTask, DownTask, WaitParked, WaitDone, Quant, OtherTask, Kinds };
enum Phase { Unknown, Prefill, Decode, Phases };
struct Stamp { uint64_t wall, cpu; };
inline Stamp stamp() {
    timespec w{},c{};
    if(clock_gettime(CLOCK_MONOTONIC,&w)||clock_gettime(CLOCK_THREAD_CPUTIME_ID,&c))std::abort();
    return {uint64_t(w.tv_sec)*1000000000ull+uint64_t(w.tv_nsec),uint64_t(c.tv_sec)*1000000000ull+uint64_t(c.tv_nsec)};
}
struct Counter { uint64_t wall=0,cpu=0,intervals=0; };
struct alignas(64) Slot {
    uint64_t tid=0,start_wall=0,start_cpu=0,end_wall=0,end_cpu=0,errors=0;
    int core=-1,kind=Control,phase=Unknown,depth=0;
    long voluntary_start=0,voluntary_end=0,involuntary_start=0,involuntary_end=0;
    bool closed=false;
    Stamp last{};
    std::array<std::array<Counter,Kinds>,Phases> counters{};
};
// Each slot has one writer. Main reads only after all workers are joined.
inline std::array<Slot,6> slots;
inline thread_local Slot* current=nullptr;
inline Phase phase_of(strata::kernels::cpu::NativeGuPhase p){return p==strata::kernels::cpu::NativeGuPhase::Prefill?Prefill:Decode;}
inline Phase phase(){return current?Phase(current->phase):Unknown;}
inline void change(Kind kind,Phase selected) {
    if(!current)std::abort();
    const Stamp now=stamp();auto& s=*current;
    if(now.wall<s.last.wall||now.cpu<s.last.cpu){++s.errors;std::abort();}
    auto& c=s.counters[s.phase][s.kind];c.wall+=now.wall-s.last.wall;c.cpu+=now.cpu-s.last.cpu;++c.intervals;
    s.last=now;s.kind=kind;s.phase=selected;
}
struct Root {
    Slot& s;
    explicit Root(unsigned index):s(slots.at(index)) {
        if(current||s.tid)std::abort();
        rusage usage{};if(getrusage(RUSAGE_THREAD,&usage))std::abort();s.voluntary_start=usage.ru_nvcsw;s.involuntary_start=usage.ru_nivcsw;
        s.tid=uint64_t(syscall(SYS_gettid));s.core=sched_getcpu();s.last=stamp();s.start_wall=s.last.wall;s.start_cpu=s.last.cpu;current=&s;
    }
    ~Root(){if(s.depth)++s.errors;change(Control,Unknown);s.end_wall=s.last.wall;s.end_cpu=s.last.cpu;rusage usage{};if(getrusage(RUSAGE_THREAD,&usage))std::abort();s.voluntary_end=usage.ru_nvcsw;s.involuntary_end=usage.ru_nivcsw;s.closed=true;current=nullptr;}
};
struct Scope {
    Kind previous_kind;Phase previous_phase;
    Scope(Kind k,Phase p):previous_kind(Kind(current->kind)),previous_phase(Phase(current->phase)) {++current->depth;change(k,p);}
    ~Scope(){change(previous_kind,previous_phase);--current->depth;}
};
} // namespace pool_observer
