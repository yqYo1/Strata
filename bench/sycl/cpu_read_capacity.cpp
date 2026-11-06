// Main-authored; repository LICENSE. Linux/Zen3 AVX2 payload-read calibration.
// Integer modular sums are exact; this does not execute inference or count bus traffic.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sys/mman.h>
#include <sys/resource.h>
#include <thread>
#include <time.h>
#include <vector>

constexpr size_t kBytes = size_t{1} << 30, kBlocks = kBytes / 256;
constexpr uint64_t kMul = 0x9e3779b97f4a7c15ULL, kAdd = 0x123456789abcdef0ULL;
constexpr unsigned kPasses = 7, kSamples = 5;
[[noreturn, gnu::cold]] void fail(const char* text) {
    std::fprintf(stderr, "CPU read capacity: %s\n", text);
    std::_Exit(1);
}
uint64_t tick() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value)) fail("clock");
    return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}
void barrier(pthread_barrier_t& value) {
    const int result = pthread_barrier_wait(&value);
    if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD) fail("barrier");
}
uint64_t expected(size_t first_block, size_t last_block, unsigned passes) {
    const __uint128_t first = first_block * 32, n = (last_block-first_block) * 32;
    return uint64_t((n * (2*first+n-1) / 2 * kMul + n*kAdd) * passes);
}
extern "C" __attribute__((noinline, noclone)) uint64_t strata_read_payload(
        const uint64_t* __restrict__ data, size_t first, size_t last, unsigned passes) {
    // CPUID: AVX2. Eight independent integer accumulators, complete256-byte blocks.
    __m256i a = _mm256_setzero_si256(), b=a, c=a, d=a, e=a, f=a, g=a, h=a;
    for (unsigned pass=0; pass<passes; ++pass) {
        // Compiler barrier: every pass must reload memory, even for an unchanged array.
        asm volatile("" ::: "memory");
        for (size_t block=first; block<last; ++block) {
            const auto* p = reinterpret_cast<const __m256i*>(data + block*32);
            a=_mm256_add_epi64(a,_mm256_load_si256(p+0)); b=_mm256_add_epi64(b,_mm256_load_si256(p+1));
            c=_mm256_add_epi64(c,_mm256_load_si256(p+2)); d=_mm256_add_epi64(d,_mm256_load_si256(p+3));
            e=_mm256_add_epi64(e,_mm256_load_si256(p+4)); f=_mm256_add_epi64(f,_mm256_load_si256(p+5));
            g=_mm256_add_epi64(g,_mm256_load_si256(p+6)); h=_mm256_add_epi64(h,_mm256_load_si256(p+7));
        }
    }
    const auto sum = _mm256_add_epi64(_mm256_add_epi64(_mm256_add_epi64(a,b),_mm256_add_epi64(c,d)),
                                    _mm256_add_epi64(_mm256_add_epi64(e,f),_mm256_add_epi64(g,h)));
    alignas(32) uint64_t lanes[4]; _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),sum);
    _mm256_zeroupper();
    return lanes[0]+lanes[1]+lanes[2]+lanes[3];
}
struct alignas(64) Slot { uint64_t sum=0; bool pinned=false; int first_cpu=-1, last_cpu=-1; };
static_assert(sizeof(Slot) % 64 == 0);
int main() {
    cpu_set_t available; CPU_ZERO(&available);
    if (sched_getaffinity(0,sizeof(available),&available)) fail("affinity query");
    std::vector<int> cpus;
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&available)) cpus.push_back(cpu);
    if (cpus.size()!=12 || !__builtin_cpu_supports("avx2")) fail("requires this12-thread AVX2 host");
    std::set<int> physical;
    for (int i=0;i<6;++i) {
        int core=-1; std::ifstream in("/sys/devices/system/cpu/cpu"+std::to_string(cpus[i])+"/topology/core_id");
        if (!(in>>core)) fail("physical core query");
        physical.insert(core);
    }
    if (physical.size()!=6) fail("first six logical IDs must be distinct physical cores");
    cpu_set_t first; CPU_ZERO(&first); CPU_SET(cpus[0],&first);
    if (sched_setaffinity(0,sizeof(first),&first)) fail("first-touch affinity");
    auto* data = static_cast<uint64_t*>(mmap(nullptr,kBytes,PROT_READ|PROT_WRITE,
                                            MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if (data==MAP_FAILED) fail("1GiB allocation");
    if (madvise(data,kBytes,MADV_NOHUGEPAGE)) fail("4KiB-page policy");
    for (size_t i=0;i<kBytes/8;++i) data[i]=uint64_t(i)*kMul+kAdd;
    if (strata_read_payload(data,0,2,3)!=expected(0,2,3)) fail("small scalar/formula control");
    std::printf("{\"working_set_bytes\":%zu,\"passes\":%u,\"samples\":%u,\"first_touch_cpu\":%d,\"conditions\":[",
                kBytes,kPasses,kSamples,cpus[0]);
    bool comma=false;
    for (const unsigned count : {1U,2U,4U,6U,12U}) {
        pthread_barrier_t start, finish;
        if (pthread_barrier_init(&start,nullptr,count+1) || pthread_barrier_init(&finish,nullptr,count+1)) fail("barrier init");
        std::array<Slot,12> slots{};
        std::vector<std::thread> workers; workers.reserve(count);
        try {
            for (unsigned i=0;i<count;++i) workers.emplace_back([&,i] {
                cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpus[i],&mask);
                slots[i].pinned=pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)==0;
                const size_t begin=kBlocks*i/count, end=kBlocks*(i+1)/count;
                for (unsigned round=0;round<=kSamples;++round) {
                    barrier(start); slots[i].first_cpu=sched_getcpu();
                    slots[i].sum=strata_read_payload(data,begin,end,round ? kPasses : 1);
                    slots[i].last_cpu=sched_getcpu(); barrier(finish);
                }
            });
        } catch (...) { fail("worker creation"); }
        std::array<uint64_t,kSamples> elapsed{};
        std::array<long,kSamples> minor{}, major{};
        for (unsigned round=0;round<=kSamples;++round) {
            rusage before{}, after{};
            if (getrusage(RUSAGE_SELF,&before)) fail("initial fault counters");
            const auto begin=tick(); barrier(start); barrier(finish); const auto end=tick();
            if (getrusage(RUSAGE_SELF,&after)) fail("final fault counters");
            for (unsigned i=0;i<count;++i) if (!slots[i].pinned || slots[i].first_cpu!=cpus[i] ||
                slots[i].last_cpu!=cpus[i] || slots[i].sum!=expected(kBlocks*i/count,kBlocks*(i+1)/count,round?kPasses:1))
                    fail("pin or complete payload checksum");
            if (round) { elapsed[round-1]=end-begin; minor[round-1]=after.ru_minflt-before.ru_minflt; major[round-1]=after.ru_majflt-before.ru_majflt; }
        }
        for (auto& worker:workers) worker.join();
        if (pthread_barrier_destroy(&start) || pthread_barrier_destroy(&finish)) fail("barrier destroy");
        std::printf("%s{\"threads\":%u,\"cpus\":[",comma?",":"",count);comma=true;
        for (unsigned i=0;i<count;++i) std::printf("%s%d",i?",":"",cpus[i]);
        std::printf("],\"checksum\":\"%016llx\",\"elapsed_ns\":[",(unsigned long long)expected(0,kBlocks,kPasses));
        for (unsigned i=0;i<kSamples;++i) std::printf("%s%llu",i?",":"",(unsigned long long)elapsed[i]);
        std::printf("],\"minor_faults\":[");
        for (unsigned i=0;i<kSamples;++i) std::printf("%s%ld",i?",":"",minor[i]);
        std::printf("],\"major_faults\":[");
        for (unsigned i=0;i<kSamples;++i) std::printf("%s%ld",i?",":"",major[i]);
        std::printf("]}");
    }
    if (munmap(data,kBytes)) fail("unmap");
    std::printf("],\"checksums_and_affinity_passed\":true,\"GPU_executed\":false}\n");
}
