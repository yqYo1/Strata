#include <immintrin.h>
#include <sched.h>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>

// Measured logical streaming traffic, not memory-controller transactions.
// Three 512 MiB arrays exceed the Ryzen5600X 32 MiB LLC. Unsigned arithmetic
// intentionally wraps; complete output validation occurs outside timed spans.
using Clock=std::chrono::steady_clock;
constexpr size_t bytes=512ull<<20, count=bytes/sizeof(uint64_t);
constexpr int passes=4, samples=7;
static uint64_t mix(uint64_t x) {
    x+=0x9e3779b97f4a7c15ull; x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
    x=(x^(x>>27))*0x94d049bb133111ebull; return x^(x>>31);
}
int main() {
    setvbuf(stdout,nullptr,_IOLBF,0);
    cpu_set_t allowed;
    if(sched_getaffinity(0,sizeof(allowed),&allowed)) return 2;
    std::vector<int> cpus;
    // Real topology and affinity are retained by the controller; use only its
    // allowed CPUs. The benchmark does not change clocks/governors/services.
    for(int i=0;i<CPU_SETSIZE;++i) if(CPU_ISSET(i,&allowed)) cpus.push_back(i);
    if(cpus.size()<12) { std::fprintf(stderr,"requires12 allowed CPUs\n"); return 3; }
    auto alloc=[](){void* p=nullptr;if(posix_memalign(&p,64,bytes))std::exit(4);return (uint64_t*)p;};
    uint64_t *a=alloc(),*b=alloc(),*c=alloc(), expected=0;
    for(size_t i=0;i<count;++i){a[i]=mix(i);b[i]=mix(i+count);c[i]=0;expected+=a[i];}
    std::printf("{\"kind\":\"configuration\",\"array_bytes\":%zu,\"arrays\":3,\"passes\":%d,\"samples\":%d,\"logical_cpu_order\":[",bytes,passes,samples);
    for(size_t i=0;i<cpus.size();++i)std::printf("%s%d",i?",":"",cpus[i]);
    std::printf("]}\n");
    for(int nthreads: {1,2,4,6,12}) for(int op=0;op<4;++op){
        const char* name=op==0?"read":op==1?"copy_cached":op==2?"copy_nontemporal":"triad_cached";
        for(int sample=-1;sample<samples;++sample){
            std::barrier ready(nthreads+1), start(nthreads+1), done(nthreads+1);
            std::vector<std::thread> workers;std::vector<uint64_t> sums(nthreads);
            for(int t=0;t<nthreads;++t)workers.emplace_back([&,t](){
                cpu_set_t one;CPU_ZERO(&one);CPU_SET(cpus[t],&one);
                if(sched_setaffinity(0,sizeof(one),&one)){std::fprintf(stderr,"pin failed\n");std::_Exit(5);}
                const size_t lo=(count/4*t/nthreads)*4,hi=(count/4*(t+1)/nthreads)*4;
                ready.arrive_and_wait();start.arrive_and_wait();
                __m256i acc=_mm256_setzero_si256();
                for(int p=0;p<passes;++p){
                    for(size_t i=lo;i<hi;i+=4){
                        const __m256i x=_mm256_load_si256((const __m256i*)(a+i));
                        if(op==0)acc=_mm256_add_epi64(acc,x);
                        else if(op==1)_mm256_store_si256((__m256i*)(c+i),x);
                        else if(op==2)_mm256_stream_si256((__m256i*)(c+i),x);
                        else {const __m256i y=_mm256_load_si256((const __m256i*)(b+i));
                            _mm256_store_si256((__m256i*)(c+i),_mm256_add_epi64(x,_mm256_add_epi64(y,_mm256_slli_epi64(y,1))));}
                    }
                    if(op==2)_mm_sfence();
                }
                alignas(32)uint64_t lanes[4];_mm256_store_si256((__m256i*)lanes,acc);
                sums[t]=lanes[0]+lanes[1]+lanes[2]+lanes[3];done.arrive_and_wait();
            });
            ready.arrive_and_wait();const auto t0=Clock::now();
            start.arrive_and_wait();done.arrive_and_wait();const auto t1=Clock::now();
            for(auto& w:workers)w.join();
            if(op==0){uint64_t got=0;for(auto s:sums)got+=s;if(got!=expected*(uint64_t)passes)return 6;}
            const double seconds=std::chrono::duration<double>(t1-t0).count();
            const uint64_t traffic=bytes*(uint64_t)passes*(op==0?1:op==3?3:2);
            std::printf("{\"kind\":\"sample\",\"case\":\"%s\",\"threads\":%d,\"sample\":%d,\"warmup\":%s,\"seconds\":%.9f,\"logical_bytes\":%llu,\"logical_GBps\":%.6f}\n",name,nthreads,sample,sample<0?"true":"false",seconds,(unsigned long long)traffic,traffic/seconds/1e9);
        }
        if(op){for(size_t i=0;i<count;++i)if(c[i]!=(op==3?a[i]+3*b[i]:a[i])){std::fprintf(stderr,"full mismatch%zu\n",i);return 7;}}
        std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"threads\":%d,\"passed\":true,\"validated_elements\":%zu}\n",name,nthreads,count);
    }
    free(a);free(b);free(c);return 0;
}
