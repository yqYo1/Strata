// Synthetic resident packed-RAM -> ordinary host ring control, NOT live Stager.
// C++20 / pthread only. No GPU, pinning, shard I/O, dequant or model operations.
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t blob=1971200, corpus=192, ring=16, guard=64, stride=blob+guard;
constexpr size_t bytes=(corpus+ring)*stride;
constexpr unsigned samples=7, warmups=1;
static_assert(blob%64==0 && bytes<512ull*1024*1024 && corpus%ring==0);
uint64_t ns(Clock::time_point a,Clock::time_point b) {
    const auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
    if(n<0) throw std::runtime_error("nonmonotonic steady clock");
    return static_cast<uint64_t>(n);
}
uint64_t splitmix(uint64_t v) {
    v+=0x9e3779b97f4a7c15ull; v=(v^(v>>30))*0xbf58476d1ce4e5b9ull;
    v=(v^(v>>27))*0x94d049bb133111ebull; return v^(v>>31);
}
uint64_t hash(const uint8_t* p,size_t n) {
    uint64_t h=14695981039346656037ull;
    for(size_t i=0;i<n;++i) h=(h^p[i])*1099511628211ull;
    return h;
}
uint64_t number(std::string_view s) {
    uint64_t v=0;
    if(s.empty() || s.size()>20 || (s.size()>1 && s.front()=='0'))
        throw std::runtime_error("invalid unsigned decimal");
    auto r=std::from_chars(s.data(),s.data()+s.size(),v);
    if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size())
        throw std::runtime_error("invalid unsigned decimal");
    return v;
}
struct Arena {
    uint8_t* p=nullptr;
    Arena() {
        p=static_cast<uint8_t*>(std::aligned_alloc(64,bytes));
        if(!p) throw std::runtime_error("arena allocation failed");
    }
    ~Arena(){std::free(p);}
    Arena(const Arena&)=delete; Arena& operator=(const Arena&)=delete;
    uint8_t* source(size_t i){return p+i*stride;}
    uint8_t* dest(size_t i){return p+(corpus+i)*stride;}
};
struct JobResult { uint64_t generation=0, job=0, copy_ns=0; };
struct Pool {
    Arena& arena; unsigned workers;
    Clock::time_point budget_end;
    std::vector<std::thread> threads;
    std::vector<uint64_t> assignments;
    std::array<JobResult,ring> results{};
    std::atomic<size_t> next{0};
    std::mutex mu; std::condition_variable wake, finished;
    uint64_t generation=0, acknowledged=0; size_t first=0;
    unsigned startup=0, departed=0;
    bool quit=false, failure=false;
    Pool(Arena& a,unsigned w,Clock::time_point end):arena(a),workers(w),budget_end(end),assignments(w,0) {
        threads.reserve(w);
        try {
            for(unsigned i=0;i<w;++i) threads.emplace_back([this,i]{work(i);});
            std::unique_lock lock(mu);
            if(!finished.wait_until(lock,budget_end,[&]{return failure || startup==workers;}))
                throw std::runtime_error("worker startup budget exceeded");
            if(failure) throw std::runtime_error("worker startup failed");
        } catch(...) { stop(); throw; }
    }
    ~Pool(){stop();}
    void stop() noexcept {
        {std::lock_guard lock(mu); quit=true;}
        wake.notify_all();
        for(auto& t:threads) if(t.joinable()) t.join();
    }
    void work(unsigned worker) noexcept {
        try {
        uint64_t seen=0;
        {std::lock_guard lock(mu); ++startup; finished.notify_one();}
        for(;;) {
            size_t base; uint64_t current;
            {
                std::unique_lock lock(mu);
                wake.wait(lock,[&]{return quit || generation!=seen;});
                if(quit) return;
                seen=generation; current=seen; base=first;
            }
            bool bad=false;
            try {
                for(size_t slot;(slot=next.fetch_add(1,std::memory_order_relaxed))<ring;) {
                    const auto begin=Clock::now();
                    std::memcpy(arena.dest(slot),arena.source(base+slot),blob);
                    const auto end=Clock::now();
                    results[slot]={current,base+slot,ns(begin,end)};
                    ++assignments[worker];
                }
            } catch(...) {bad=true;}
            {
                std::lock_guard lock(mu);
                failure=failure || bad; ++departed;
                finished.notify_one();
            }
        }
        } catch(...) {
            // Wake either startup or batch waiter; main joins surviving workers
            // before releasing their arena. Broken mutex state cannot recover.
            try {std::lock_guard lock(mu); failure=true; finished.notify_all();}
            catch(...) {std::abort();}
        }
    }
    uint64_t batch(size_t base) {
        if(base>=corpus || base%ring || base+ring>corpus)
            throw std::runtime_error("batch bounds");
        // Prior batch workers have ALL departed; caller verified/acknowledged
        // every slot before invoking batch again. No concurrent verification.
        const auto begin=Clock::now();
        std::unique_lock lock(mu);
        if(generation && departed!=workers) throw std::runtime_error("live previous batch");
        if(acknowledged!=generation) throw std::runtime_error("previous batch not acknowledged");
        if(failure || generation==UINT64_MAX) throw std::runtime_error("pool state failure");
        first=base; departed=0; next.store(0,std::memory_order_relaxed);
        ++generation; wake.notify_all();
        if(!finished.wait_until(lock,budget_end,[&]{return failure || departed==workers;}))
            throw std::runtime_error("batch completion budget exceeded");
        const auto end=Clock::now();
        if(failure) throw std::runtime_error("worker failure");
        return ns(begin,end);
    }
    void acknowledge() {
        std::lock_guard lock(mu);
        if(departed!=workers || failure || acknowledged==generation)
            throw std::runtime_error("invalid acknowledgment");
        acknowledged=generation;
    }
};
struct Sample {
    std::array<uint64_t,corpus/ring> batch_ns{};
    std::array<uint64_t,corpus> copy_ns{}, job_hash{};
    std::array<uint64_t,6> worker_jobs{};
    uint64_t verify_ns=0, wall_ns=0, copy_sum_ns=0, jobs=0, guards=0;
};
void check_guard(const uint8_t* p) {
    for(size_t i=0;i<guard;++i) if(p[blob+i]!=0xa5)
        throw std::runtime_error("tail canary mismatch");
}
void deadline(Clock::time_point start) {
    if(Clock::now()-start>=std::chrono::seconds(60))
        throw std::runtime_error("60 second process work budget exceeded");
}
void array_json(const auto& a) {
    std::cout<<'[';
    for(size_t i=0;i<a.size();++i) {if(i) std::cout<<','; std::cout<<a[i];}
    std::cout<<']';
}
void error_text(const char* text) {
    std::cerr<<'"';
    const char* hex="0123456789abcdef";
    for(const unsigned char* p=reinterpret_cast<const unsigned char*>(text);*p;++p) {
        if(*p=='"' || *p=='\\') std::cerr<<'\\'<<static_cast<char>(*p);
        else if(*p<32) std::cerr<<"\\u00"<<hex[*p>>4]<<hex[*p&15];
        else std::cerr<<static_cast<char>(*p);
    }
    std::cerr<<'"';
}
}
int main(int argc,char** argv) {
    const char* stage="parse";
    try {
        const auto process_start=Clock::now();
        if(argc<2 || argc>4) throw std::runtime_error("usage: probe WORKERS [SEED] [--qualify]");
        const auto parsed=number(argv[1]);
        if(parsed!=1 && parsed!=3 && parsed!=4 && parsed!=6)
            throw std::runtime_error("workers must be 1,3,4,6");
        const unsigned workers=static_cast<unsigned>(parsed);
        uint64_t seed=1; bool seed_seen=false,qualify=false;
        for(int i=2;i<argc;++i) {
            if(std::string_view(argv[i])=="--qualify" && !qualify) qualify=true;
            else if(!seed_seen) {seed=number(argv[i]); seed_seen=true;}
            else throw std::runtime_error("unknown or duplicate option");
        }
        stage="source_initialization";
        Arena arena; std::array<uint64_t,corpus> expected{};
        for(size_t j=0;j<corpus;++j) {
            auto* p=arena.source(j);
            for(size_t off=0;off<blob;off+=8) {
                const uint64_t v=splitmix(seed ^ (static_cast<uint64_t>(j)<<32) ^ (off/8));
                for(unsigned b=0;b<8;++b) p[off+b]=static_cast<uint8_t>(v>>(8*b));
            }
            std::memset(p+blob,0xa5,guard);
            expected[j]=hash(p,blob); deadline(process_start);
        }
        for(size_t i=0;i<ring;++i) {
            std::memset(arena.dest(i),0,blob);
            std::memset(arena.dest(i)+blob,0xa5,guard);
        }
        stage="pool_start";
        Pool pool(arena,workers,process_start+std::chrono::seconds(60));
        // Thread creation/startup barrier outside timings.
        const unsigned measured=qualify?1:samples;
        std::vector<Sample> records(warmups+measured); // preallocated before timers
        stage="copy_batches_and_verification";
        for(unsigned s=0;s<records.size();++s) {
            auto& record=records[s];
            std::array<uint64_t,6> before{};
            for(unsigned w=0;w<workers;++w) before[w]=pool.assignments[w];
            for(size_t base=0;base<corpus;base+=ring) {
                deadline(process_start);
                const uint64_t elapsed=pool.batch(base);
                record.batch_ns[base/ring]=elapsed; record.wall_ns+=elapsed;
                const auto verify_begin=Clock::now();
                for(size_t slot=0;slot<ring;++slot) {
                    const size_t job=base+slot;
                    const auto& result=pool.results[slot];
                    if(result.generation!=pool.generation || result.job!=job)
                        throw std::runtime_error("generation/job identity mismatch");
                    const auto* src=arena.source(job); const auto* dst=arena.dest(slot);
                    if(std::memcmp(src,dst,blob)!=0) throw std::runtime_error("full blob mismatch");
                    check_guard(src); check_guard(dst); record.guards+=2;
                    record.job_hash[job]=hash(dst,blob);
                    if(record.job_hash[job]!=expected[job]) throw std::runtime_error("job hash mismatch");
                    record.copy_ns[job]=result.copy_ns; record.copy_sum_ns+=result.copy_ns;
                    ++record.jobs;
                }
                // Full-byte verification and hash are the host acknowledgment.
                // Only now may the next batch overwrite any ring slot.
                pool.acknowledge();
                record.verify_ns+=ns(verify_begin,Clock::now());
            }
            uint64_t assigned=0;
            for(unsigned w=0;w<workers;++w) {
                record.worker_jobs[w]=pool.assignments[w]-before[w]; assigned+=record.worker_jobs[w];
            }
            if(record.jobs!=corpus || assigned!=corpus || record.guards!=corpus*2)
                throw std::runtime_error("coverage reconciliation failure");
        }
        // All threads are parked after acknowledged final batch. Join untimed.
        pool.stop();
        stage="final_source_validation";
        for(size_t j=0;j<corpus;++j) {
            check_guard(arena.source(j));
            if(hash(arena.source(j),blob)!=expected[j])
                throw std::runtime_error("immutable source changed");
            deadline(process_start);
        }
        deadline(process_start);
        for(const auto& record:records) if(!record.wall_ns)
            throw std::runtime_error("zero timed duration");
        const uint64_t process_work_ns=ns(process_start,Clock::now());
        stage="receipt";
        std::cout<<"{\"success\":true,\"kind\":\"synthetic packed host ring-copy control\","
          <<"\"timing_claim\":"<<(qualify?"false":"true")
          <<",\"scope\":\"12 independently timed 16-slot copy batches per sample; verification outside timers; immediate host acknowledgment surrogate\","
          <<"\"not_production_stager_or_dma\":true,\"not_pinned\":true,\"gpu\":false,"
          <<"\"source\":\"deterministic ordinary aligned resident RAM\","
          <<"\"affinity\":\"inherited; no discovery or pinning\","
          <<"\"workers\":"<<workers<<",\"seed\":"<<seed<<",\"ring_slots\":16,\"corpus_blobs\":192,"
          <<"\"blob_bytes\":"<<blob<<",\"arena_bytes\":"<<bytes<<",\"alignment_bytes\":64,"
          <<"\"logical_source_payload_bytes_per_sample\":"<<corpus*blob
          <<",\"logical_destination_payload_bytes_per_sample\":"<<corpus*blob
          <<",\"logical_read_plus_write_bytes_per_sample\":"<<2*corpus*blob
          <<",\"warmups\":1,\"samples\":"<<measured
          <<",\"total_fully_byte_verified_jobs_including_warmup\":"<<records.size()*corpus
          <<",\"final_immutable_source_full_hash_checks\":192"
          <<",\"process_work_ns_before_output\":"<<process_work_ns
          <<",\"worker_clock_calls_per_job\":2,\"batch_clock_calls\":2,"
          <<"\"timing_includes\":\"mutex issue, notification, wakeup, claims, memcpy, worker clocks, completion barrier and host wait\","
          <<"\"copy_ns_is_summed_worker_spans_not_wall_or_exclusive_cpu_service\":true,"
          <<"\"expected_source_fnv1a64\":";
        array_json(expected);
        std::cout<<",\"individual_samples\":[";
        for(size_t s=0;s<records.size();++s) {
            if(s) std::cout<<',';
            const auto& r=records[s]; const double seconds=static_cast<double>(r.wall_ns)/1e9;
            std::cout<<"{\"warmup\":"<<(s<warmups?"true":"false")
              <<",\"jobs_fully_byte_verified\":"<<r.jobs<<",\"tail_canary_checks\":"<<r.guards
              <<",\"copy_issue_wait_batch_wall_ns_sum\":"<<r.wall_ns
              <<",\"correctness_ack_ns\":"<<r.verify_ns<<",\"worker_copy_span_ns_sum\":"<<r.copy_sum_ns;
            if(!qualify) std::cout<<",\"timed_batch_payload_GBps\":"<<(corpus*blob/seconds/1e9)
                                 <<",\"timed_batch_logical_read_plus_write_GBps\":"<<(2*corpus*blob/seconds/1e9);
            std::cout<<",\"batch_wall_ns\":"; array_json(r.batch_ns);
            std::cout<<",\"job_copy_ns\":"; array_json(r.copy_ns);
            std::cout<<",\"job_fnv1a64\":"; array_json(r.job_hash);
            std::cout<<",\"worker_jobs\":";
            std::cout<<'['; for(unsigned w=0;w<workers;++w) {if(w) std::cout<<','; std::cout<<r.worker_jobs[w];}
            std::cout<<"]}";
        }
        std::cout<<"]}\n";
        if(!std::cout) return 1;
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"{\"success\":false,\"no_capacity_result_valid\":true,\"stage\":";
        error_text(stage); std::cerr<<",\"error\":"; error_text(e.what()); std::cerr<<"}\n";
        return 1;
    } catch(...) {
        std::cerr<<"{\"success\":false,\"error\":\"unknown failure; no capacity result valid\"}\n"; return 1;
    }
}
