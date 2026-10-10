// Root builds only after pinned actual-body extraction. No SYCL/GPU dependency.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace sycl { struct exception : std::exception {}; }
namespace detail { struct RouteCensus { static void fatal_receipt(const char*) { std::abort(); } }; }
struct State;
thread_local State* current = nullptr;
namespace dpct { void sync_barrier(int&,void*); }

struct State {
    struct Entry { int32_t e; };
    struct Model {
        int ring=4;
        std::array<int,4> used{}; // Immutable event handles; event mutation is instrumented.
        std::array<std::atomic<int>,4> used_of{};
        void* cs=nullptr;
    } m;
    struct Census { bool on=false; } route_census;
    std::vector<Entry> seq;
    size_t k=0,kend=0,consumed=0;
    int gg_nslots=0;
    bool threaded;
    std::atomic<size_t> a_issued{0},a_consumed{0};
    std::mutex mutex;
    std::condition_variable cv;
    size_t ready=static_cast<size_t>(-1),permit=static_cast<size_t>(-1);
    bool reading=false;
    size_t violations=0,events=0,waits=0,flushes=0,reads=0;
    std::thread producer;
    std::chrono::steady_clock::time_point deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    explicit State(size_t count,bool t,bool grouped):threaded(t) {
        for(size_t i=0;i<count;++i) seq.push_back({static_cast<int32_t>(i)});
        kend=count; gg_nslots=grouped?2:0;
    }
    template<class Predicate> void await(std::unique_lock<std::mutex>& lock,Predicate p) {
        if(!cv.wait_until(lock,deadline,p)) {
            std::cerr<<"finite CPU schedule deadline\n"<<std::flush;
            std::_Exit(2); // no joinable-thread unwind on a broken contract
        }
    }
    void start() {
        if(!threaded) { a_issued.store(std::min(kend,size_t(m.ring))); return; }
        producer=std::thread([this] {
            for(size_t idx=0;idx<kend;++idx) {
                std::unique_lock lock(mutex);
                await(lock,[&]{return idx<a_consumed.load(std::memory_order_acquire)+size_t(m.ring);});
                ready=idx; reading=true; cv.notify_all();
                // Hold producer's read interval open deterministically. Actual shared
                // objects replaced by atomics; no real conflicting non-atomic access.
                (void)m.used_of[idx%size_t(m.ring)].load();
                await(lock,[&]{return permit==idx;});
                (void)m.used_of[idx%size_t(m.ring)].load();
                ++reads; reading=false;
                a_issued.store(idx+1,std::memory_order_release); cv.notify_all();
            }
        });
    }
    void publish_until(size_t index,bool event_boundary) {
        if(!threaded) return;
        std::unique_lock lock(mutex);
        if(a_issued.load(std::memory_order_acquire)>index) return;
        await(lock,[&]{return ready==index && reading;});
        if(event_boundary) ++violations;
        permit=index; cv.notify_all();
        await(lock,[&]{return a_issued.load(std::memory_order_acquire)>index;});
    }
    void wait_issued(size_t index) {
        ++waits;
        if(!threaded) return; // actual production helper's inline no-op
        publish_until(index,false);
        if(a_issued.load(std::memory_order_acquire)<=index) std::abort();
    }
    void flush() {
        ++flushes; gg_nslots=0;
        // Production flush returns previously gathered slots before wait_issued.
        give_back(consumed);
    }
    void give_back(size_t upto) {
        std::lock_guard lock(mutex); // Pair CV predicate changes with its mutex.
        a_consumed.store(upto,std::memory_order_release);
        if(!threaded) a_issued.store(std::min(kend,upto+size_t(m.ring)));
        cv.notify_all();
    }
    void event() {
        ++events;
        // Observe ACTUAL event publication-call boundary while issuer still
        // reading; then finish read BEFORE simulated metadata mutation. This
        // negative control observes ordering without exercising race/UB.
        publish_until(k,true);
        if(a_issued.load(std::memory_order_acquire)<=k) std::abort();
    }
    void old_release(int32_t stop) {
        #include "old_release.inc"
        release_to(stop);
    }
    void new_release(int32_t stop) {
        #include "new_release.inc"
        release_to(stop);
    }
    bool run(bool newer,bool split) {
        current=this; start();
        if(split) {
            for(size_t end=1;end<=kend;end+=3) {
                if(newer) new_release(static_cast<int32_t>(end));
                else old_release(static_cast<int32_t>(end));
            }
        }
        if(newer) new_release(static_cast<int32_t>(kend));
        else old_release(static_cast<int32_t>(kend));
        if(producer.joinable()) producer.join();
        current=nullptr;
        return k==kend && consumed==kend && events==kend && a_consumed.load()==kend
            && a_issued.load()==kend && (!threaded || reads==kend)
            && violations==(threaded && !newer ? kend : 0)
            && waits==(newer ? kend : 0);
    }
};
namespace dpct { void sync_barrier(int&,void*) { current->event(); } }

int main() {
    size_t cases=0;
    for(size_t count:{size_t(0),size_t(1),size_t(3),size_t(4),size_t(5),size_t(17),size_t(65)})
        for(bool threaded:{false,true}) for(bool grouped:{false,true})
            for(bool split:{false,true}) for(bool newer:{false,true}) {
                State state(count,threaded,grouped);
                if(!state.run(newer,split)) {
                    std::cerr<<"{\"passed\":false,\"count\":"<<count
                        <<",\"threaded\":"<<threaded<<",\"new\":"<<newer<<"}\n";
                    return 1;
                }
                ++cases;
            }
    std::cout<<"{\"passed\":true,\"cases\":"<<cases
        <<",\"old_threaded_expected_negative\":true,\"new_publication_order_passed\":true,\"gpu_executed\":false}\n";
}
