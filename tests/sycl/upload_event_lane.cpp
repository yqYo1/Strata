// Main-authored safety fixture; repository LICENSE. No inference or speed claim.
#include "strata/prefill/mapped_host_source.hpp"
#include "prefill_host_pages.hpp"
#ifdef STRATA_PREFILL_EVENT_RUNTIME_CONTEXT
#include "strata/sycl/runtime.hpp"
#endif
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {
using Lane = strata::prefill::detail::PrefillUploadEventLane;
using Mapped = strata::prefill::detail::PrefillMappedHostSource;
constexpr auto imported_contract = Mapped::PropertyContract::PinnedNeoHost;
using Pages = strata::prefill::test_detail::PrefillHostPages;
using Clock = std::chrono::steady_clock;
void need(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void ze(ze_result_t code) { need(code == ZE_RESULT_SUCCESS, "fixture LevelZero call"); }
void gate(const char* name) {
    std::cout << "{\"kind\":\"gate\",\"name\":\"" << name << "\"}" << std::endl;
    std::string line;
    need(bool(std::getline(std::cin, line)) && line == "continue", "Main gate refused");
}
void progress(const char* name) { std::cerr << "upload lane progress: " << name << std::endl; }
struct Memory {
    sycl::queue queue;
    std::unique_ptr<Pages> ordinary_a, ordinary_b;
    std::uint32_t *a = nullptr, *b = nullptr, *seen = nullptr, *storage = nullptr, *output = nullptr;
    explicit Memory(sycl::queue q, bool mapped) : queue(std::move(q)) {
      try {
        if (mapped) {
            ordinary_a.reset(new Pages(66 * 4)); ordinary_b.reset(new Pages(66 * 4));
            a = static_cast<std::uint32_t*>(ordinary_a->data());
            b = static_cast<std::uint32_t*>(ordinary_b->data());
        } else {
            a = sycl::malloc_host<std::uint32_t>(66, queue);
            b = sycl::malloc_host<std::uint32_t>(66, queue);
        }
        seen = sycl::malloc_host<std::uint32_t>(66, queue);
        storage = sycl::malloc_device<std::uint32_t>(66, queue);
        output = sycl::malloc_device<std::uint32_t>(66, queue);
        need(a && b && seen && storage && output, "fixture allocation");
        for (int i = 0; i < 66; ++i) a[i] = b[i] = seen[i] = 0x7162abcd;
        for (int i = 1; i <= 64; ++i) { a[i] = std::uint32_t(i * 7919); b[i] = a[i] ^ 0x5ac31e07; }
        if (mapped) { ordinary_a->read_only(); ordinary_b->read_only(); }
        queue.memcpy(storage, seen, 66 * 4).wait_and_throw();
        queue.memcpy(output, seen, 66 * 4).wait_and_throw();
        // Prewarm the EXACT reader lambda before a two-second host gate exists.
        reader(1); queue.wait_and_throw();
      } catch (...) { retire(); throw; }
    }
    ~Memory() noexcept { retire(); }
    void retire() noexcept {
        strata::sycl_backend::checked_cleanup([&] { queue.wait_and_throw(); }, [&] {
            if (!ordinary_a && a) sycl::free(a, queue);
            if (!ordinary_b && b) sycl::free(b, queue);
            for (auto* p : {seen,storage,output}) if (p) sycl::free(p, queue);
            a=b=seen=storage=output=nullptr;
        }, "upload fixture memory retirement");
    }
    sycl::event reader(std::size_t out) {
        auto* src = storage + 1; auto* dst = output + out;
        return queue.single_task([=] { *dst = src[0] + 3u * src[63]; });
    }
    void verify(std::initializer_list<std::pair<int,std::uint32_t>> values,
                const std::uint32_t* final_source) {
        queue.memcpy(seen, storage, 66 * 4).wait_and_throw();
        need(seen[0] == 0x7162abcd && seen[65] == 0x7162abcd, "device edge guard");
        for (int i = 1; i <= 64; ++i) need(seen[i] == final_source[i], "full copied payload");
        queue.memcpy(seen, output, 66 * 4).wait_and_throw();
        need(seen[0] == 0x7162abcd && seen[65] == 0x7162abcd, "output edge guard");
        for (auto [at,value] : values) need(seen[at] == value, "reader saw overwritten/stale blob");
        for (int i = 0; i < 66; ++i) {
            const auto expected = i == 0 || i == 65 ? 0x7162abcd : std::uint32_t(i * 7919);
            need(a[i] == expected && b[i] == (i == 0 || i == 65 ? expected : expected ^ 0x5ac31e07),
                 "source owner changed before retirement");
        }
    }
};

// Independent host release prevents a failed import/submit from becoming a
// permanent external wait. This owner releases BEFORE draining either queue,
// and destroys its borrowed event only AFTER that drain. No host task or spin kernel.
class Held {
    Lane& lane_;
    sycl::queue queue_;
    ze_event_pool_handle_t pool_ = nullptr;
    ze_event_handle_t event_ = nullptr;
    std::optional<sycl::event> imported_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool release_ = false, expired_ = false;
    ze_result_t signal_ = ZE_RESULT_NOT_READY;
    std::thread thread_;
public:
    Held(Lane& lane, sycl::queue q) : lane_(lane), queue_(std::move(q)) {
      try {
        const auto context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue_.get_context());
        auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue_.get_device());
        ze_event_pool_desc_t pool{}; pool.stype=ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pool.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE; pool.count=1;
        ze(zeEventPoolCreate(context,&pool,1,&device,&pool_));
        ze_event_desc_t desc{}; desc.stype=ZE_STRUCTURE_TYPE_EVENT_DESC;
        desc.signal=desc.wait=ZE_EVENT_SCOPE_FLAG_HOST;
        ze(zeEventCreate(pool_,&desc,&event_));
        thread_=std::thread([this] {
            std::unique_lock lock(mutex_);
            expired_=!condition_.wait_for(lock,std::chrono::seconds(2),[&] {return release_;});
            lock.unlock();
            const auto result=zeEventHostSignal(event_);
            lock.lock();signal_=result;
        });
        const auto start=Clock::now();
        imported_=sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
            {event_,sycl::ext::oneapi::level_zero::ownership::keep},queue_.get_context());
        need(Clock::now()-start < std::chrono::milliseconds(100) && held(), "host-gate import blocked");
        queue_.ext_oneapi_submit_barrier({*imported_});
        need(held(), "host gate expired during queue submission");
      } catch (...) { retire(); throw; }
    }
    bool held() {
        { std::lock_guard lock(mutex_); if (release_ || expired_) return false; }
        const auto result=zeEventQueryStatus(event_);
        if(result!=ZE_RESULT_NOT_READY && result!=ZE_RESULT_SUCCESS) ze(result);
        return result == ZE_RESULT_NOT_READY;
    }
    void release() {
        {std::lock_guard lock(mutex_);release_=true;condition_.notify_all();}
        if(thread_.joinable()) thread_.join();
        ze(signal_); need(!expired_, "host-gate deadline released dependency");
    }
    ~Held() noexcept { retire(); }
private:
    void retire() noexcept {
        strata::sycl_backend::checked_cleanup([&] {
            const bool signalling=thread_.joinable();
            {std::lock_guard lock(mutex_);release_=true;condition_.notify_all();}
            if(thread_.joinable()) thread_.join();
            if(signalling) ze(signal_);
            lane_.drain();
        },[&] {
            imported_.reset();
            if(event_) ze(zeEventDestroy(event_));
            if(pool_) ze(zeEventPoolDestroy(pool_));
            event_=nullptr;pool_=nullptr;
        },"upload fixture host gate retirement");
    }
};

template<class F> void refuses(F&& fn) {
    bool rejected=false;
    try { fn(); } catch(const std::invalid_argument&) {rejected=true;}
    need(rejected,"invalid owner/range/token accepted");
}
template<class F> auto bounded(const char* name,F&& fn) {
    progress(name);const auto start=Clock::now();auto value=fn();
    need(Clock::now()-start<std::chrono::milliseconds(100),"dependency submission/import exceeded100ms");
    return value;
}

int run(const std::string& mode, bool mapped) {
    need(mode=="metadata" || mode=="forward" || mode=="reverse" || mode=="permanent" ||
         mode=="cancel" || mode=="cancel-throw" || mode=="refuse" ||
         (mapped && (mode=="import" || mode=="retire")), "known fixture case required");
    gate("before-runtime");
    {
#ifdef STRATA_PREFILL_EVENT_RUNTIME_CONTEXT
        auto runtime = strata::sycl_backend::runtime_for();
        sycl::queue queue = runtime->compute();
        const auto check = [runtime] { runtime->check(); };
#else
        sycl::queue queue(sycl::gpu_selector_v,
            [](sycl::exception_list) {strata::sycl_backend::cleanup_fail_stop("upload fixture asynchronous SYCL error");},
            sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
        const auto check = [] {};
#endif
        need(queue.get_device().get_info<sycl::info::device::name>().find("B570")!=std::string::npos,"B570 required");
        // Declaration order retains buffers until lane and host-gate retirement.
        std::unique_ptr<Memory> memory_owner;
        Lane lane(queue,check,16);
        gate("runtime-ready");
        // Imported owners retire BEFORE the lane; heap pages retire AFTER it.
        std::unique_ptr<Mapped> imported_a, imported_b;
        if(mode=="import") {
            Pages pages(4096);
            auto* data=static_cast<std::uint32_t*>(pages.data());data[0]=0x73621eab;
            pages.read_only();
            // The pinned provider reports HOST rather than HOST_IMPORTED.
            // Generic standard admission must still reject it and unimport.
            refuses([&] {Mapped standard(lane,pages.data(),pages.bytes());});
            need(sycl::get_pointer_type(pages.data(),queue.get_context())==sycl::usm::alloc::unknown,
                 "rejected standard import left a registered heap pointer");
            {
                Mapped imported(lane,pages.data(),pages.bytes(),imported_contract);
                const auto proof=imported.property_proof();
                need(proof.type==ZE_MEMORY_TYPE_HOST && proof.allocation_id>0,"explicit NEO import proof");
                std::cout << "{\"kind\":\"mapping-proof\",\"contract\":\"mapped-neo\",\"property_type\":"
                          << unsigned(proof.type) << ",\"allocation_id\":" << proof.allocation_id
                          << ",\"strict_standard_rejected\":true}" << std::endl;
                need(imported.view(4,64*4).bytes==64*4,"mapped interior view");
                refuses([&] {(void)imported.view(pages.bytes(),1);});
                refuses([&] {Mapped overlap(lane,pages.data(),pages.bytes());});
                refuses([&] {Mapped unaligned(lane,static_cast<std::byte*>(pages.data())+1,pages.bytes());});
                refuses([&] {Mapped empty(lane,pages.data(),0);});
                gate("buffers-ready");
            }
            need(data[0]==0x73621eab,"unimport modified/freed original heap owner");
            need(sycl::get_pointer_type(pages.data(),queue.get_context())==sycl::usm::alloc::unknown,
                 "unimport left a registered heap pointer");
        } else if(mode!="metadata") {
            memory_owner.reset(new Memory(queue,mapped));auto& memory=*memory_owner;
            if(mapped) {
                imported_a.reset(new Mapped(lane,memory.ordinary_a->data(),memory.ordinary_a->bytes(),imported_contract));
                imported_b.reset(new Mapped(lane,memory.ordinary_b->data(),memory.ordinary_b->bytes(),imported_contract));
            }
            gate("buffers-ready");
            const Lane::Region dst{memory.storage+1,64*4};
            const Lane::Source a=mapped ? imported_a->view(4,64*4) : Lane::Source{memory.a+1,64*4};
            const Lane::Source b=mapped ? imported_b->view(4,64*4) : Lane::Source{memory.b+1,64*4};
            const auto expected_a=memory.a[1]+3u*memory.a[64], expected_b=memory.b[1]+3u*memory.b[64];
            need(expected_a!=expected_b,"fixture patterns must distinguish overwrites");
            progress(mode.c_str());
            if(mode=="forward") {
                Held hold(lane,queue);
                const auto prior=bounded("reader-before-copy",[&] {return lane.readers_done();});
                const auto ready=bounded("copy-after-reader",[&] {return lane.copy_after(dst,a,64*4,prior);});
                progress("wait-copy-ready");lane.wait_ready(ready);memory.reader(1);
                const auto done=bounded("reader-after-copy",[&] {return lane.readers_done();});
                const auto start=Clock::now();
                do {need(hold.held() && !lane.ready(prior) && !lane.ready(ready) && !lane.ready(done),"forward dependency escaped hold");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } while(Clock::now()-start<std::chrono::milliseconds(50));
                hold.release();lane.drain();memory.verify({{1,expected_a}},memory.a);
            } else if(mode=="reverse") {
                const auto ready=lane.copy_after(dst,a,64*4);lane.wait_ready(ready);
                memory.reader(1);queue.wait_and_throw();
                Held hold(lane,queue);
                memory.reader(2); // A second dequant-like reader is deliberately held.
                const auto both=bounded("last-held-reader",[&] {return lane.readers_done();});
                const auto overwritten=bounded("overwrite-after-last-reader",[&] {return lane.copy_after(dst,b,64*4,both);});
                const auto start=Clock::now();
                do {need(hold.held() && lane.ready(ready) && !lane.ready(both) && !lane.ready(overwritten),"scratch overwrite escaped last reader");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } while(Clock::now()-start<std::chrono::milliseconds(50));
                hold.release();lane.wait_ready(overwritten);memory.reader(3);lane.drain();
                memory.verify({{1,expected_a},{2,expected_a},{3,expected_b}},memory.b);
            } else if(mode=="permanent") {
                const auto ready=lane.copy_after(dst,a,64*4);
                for(int i=1;i<=4;++i) {lane.wait_ready(ready);memory.reader(i);}
                const auto last=lane.readers_done();
                const auto next=lane.copy_after(dst,b,64*4,last);lane.wait_ready(next);memory.reader(5);
                lane.drain();memory.verify({{1,expected_a},{2,expected_a},{3,expected_a},{4,expected_a},{5,expected_b}},memory.b);
            } else if(mode=="cancel" || mode=="cancel-throw") {
              bool unwound=false;
              try {
                { Held hold(lane,queue);
                  const auto prior=lane.readers_done();(void)lane.copy_after(dst,a,64*4,prior);
                  // Simulated stop after accepted enqueue: scope exit releases
                  // the host dependency, then drains before either source frees.
                  if(mode=="cancel-throw") throw std::runtime_error("injected accepted-upload cancellation");
                }
              } catch(const std::runtime_error& error) {
                  need(mode=="cancel-throw" && std::string(error.what())=="injected accepted-upload cancellation",
                       "unexpected cancellation failure");
                  unwound=true;
              }
                need(unwound==(mode=="cancel-throw"),"cancellation unwind was not exercised");
                lane.drain();memory.verify({},memory.a);
            } else if(mode=="retire") {
                Held hold(lane,queue);
                const auto prior=lane.readers_done();
                const auto copied=lane.copy_after(dst,a,64*4,prior);
                std::mutex release_mutex; std::condition_variable release_condition;
                bool armed=false;
                struct Release {
                    std::thread thread;
                    ~Release() { if(thread.joinable()) thread.join(); }
                } release{std::thread([&] {
                    {std::unique_lock lock(release_mutex);release_condition.wait(lock,[&] {return armed;});}
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));hold.release();
                })};
                const auto before=Clock::now();
                {std::lock_guard lock(release_mutex);armed=true;release_condition.notify_all();}
                imported_a.reset(); // Must drain the accepted held transfer before unimport.
                need(Clock::now()-before>=std::chrono::milliseconds(40),"mapped source retired before held copy");
                release.thread.join();
                refuses([&] {(void)lane.copy_after(dst,a,64*4);}); // Stale mapped view after unimport.
                imported_a.reset(new Mapped(lane,memory.ordinary_a->data(),memory.ordinary_a->bytes(),imported_contract));
                refuses([&] {(void)lane.copy_after(dst,a,64*4);}); // Also stale after the same address is reimported.
                lane.wait_ready(copied);memory.reader(1);lane.drain();
                memory.verify({{1,expected_a}},memory.a);
            } else {
                refuses([&] {(void)lane.copy_after(dst,a,65*4);});
                refuses([&] {(void)lane.copy_after({nullptr,64*4},a,64*4);});
                refuses([&] {lane.wait_ready({0,0});});
                refuses([&] {(void)lane.copy_after(dst,a,64*4,Lane::Readers{0,0});});
                Lane other(queue,check,4);
                Pages unknown(4096);unknown.read_only();
                refuses([&] {(void)lane.copy_after(dst,{unknown.data(),64*4},64*4);});
                if(mapped) {
                    refuses([&] {(void)other.copy_after(dst,a,64*4);});
                    refuses([&] {(void)lane.copy_after(dst,{memory.a+1,64*4},64*4);});
                }
                const auto r=lane.copy_after(dst,a,64*4);lane.wait_ready(r);lane.drain();
                refuses([&] {other.wait_ready(r);});
                memory.verify({},memory.a);
            }
            lane.drain();
        }
        lane.drain();
    }
    gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true,\"case\":\"" << mode
              << "\",\"source\":\"" << (mapped ? "mapped-neo" : "usm") << "\"}" << std::endl;
    return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {
        need(argc==2 || argc==3,"case and optional source required");
        const std::string source=argc==3 ? argv[2] : "usm";
        need(source=="usm" || source=="mapped-neo","source must be usm or mapped-neo");
        return run(argv[1],source=="mapped-neo");
    }
    catch(const std::exception& error) {std::cerr << "upload lane failure: " << error.what() << std::endl;return 1;}
}
