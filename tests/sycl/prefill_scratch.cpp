// Original Strata fixture; repository LICENSE applies. No model inference.
#include "strata/prefill/prefill.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_slot_sizes.hpp"
#include "strata/core/residency_publication.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace core = strata::core;
using strata::prefill::Prefill;
std::shared_ptr<sb::Runtime> runtime;
constexpr size_t guard = 256;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(cudaError_t status) { need(status == cudaSuccess, cudaGetErrorString(status)); }
void passed(const char* name) {
    std::cout << "prefill_scratch_device case=" << name << " status=PASS\n" << std::flush;
}
struct Session {
    core::ModelGeometry geometry;
    core::SessionState state;
    sb::Allocation allocation;
    void init() {
        const auto bytes = core::session_bytes(geometry, 1024, 10);
        need(bytes > 0 && bytes <= (512ull << 20), "fixed session budget");
        allocation = sb::Allocation(runtime, size_t(bytes), sb::MemoryKind::Device);
        need(core::session_init(geometry, 1024, 10, allocation.data(), state) != 0, "session init");
    }
    ~Session() noexcept {
        sb::checked_cleanup([&] { runtime->wait(); }, [&] {
            core::session_release(state);
            for (int64_t i = 0; state.qsa_states && i < geometry.n_qsa_layers(); ++i) {
                auto& q = state.qsa_states[i];
                if (q.host_step) check(cudaFreeHost(q.host_step));
                if (q.host_pos) check(cudaFreeHost(q.host_pos));
            }
            delete[] state.qsa_states;
            state.qsa_states = nullptr;
        }, "prefill fixture session retirement");
    }
};
struct Parent {
    sb::Allocation allocation;
    size_t bytes;
    explicit Parent(size_t n) : allocation(runtime, n + 2 * guard, sb::MemoryKind::Device), bytes(n) {
        need(n <= (512ull << 20) && uintptr_t(data()) % guard == 0, "parent size/alignment");
        runtime->compute().memset(allocation.data(), 0xa5, allocation.size());
        runtime->wait();
    }
    unsigned char* data() const { return allocation.as<unsigned char>() + guard; }
    void unchanged() const {
        // Every byte, including outer guards, must remain untouched by layout
        // construction. No tensor/session payload is retained outside this job.
        std::vector<unsigned char> observed(std::min<size_t>(allocation.size(), 8u << 20));
        for (size_t at = 0; at < allocation.size(); at += observed.size()) {
            const auto n = std::min(observed.size(), allocation.size() - at);
            try {
                runtime->compute().memcpy(observed.data(), allocation.as<unsigned char>() + at, n);
                runtime->wait();
            } catch (...) { sb::cleanup_fail_stop("prefill fixture readback did not retire"); }
            need(std::all_of(observed.begin(), observed.begin() + n,
                [](auto byte) { return byte == 0xa5; }), "parent or guard was written by layout");
        }
    }
};
size_t exact_bytes(const Session& s, int64_t rows) {
    // Alias1/MMQ0 uses the same production take sequence in bytes_needed/init.
    // Subtract ONLY the explicit sizing slack, retaining every take's guard.
    const auto need_bytes = Prefill::bytes_needed(s.geometry, s.state, rows);
    need(need_bytes > (8u << 20) && need_bytes <= (512ull << 20), "fixed prefill budget");
    return size_t(need_bytes - (8u << 20));
}
bool init(Prefill& p, Session& s, core::WeightTable& weights, sycl::queue& queue,
          void* base, size_t bytes, std::string& error) {
    return p.init(weights, s.geometry, s.state, nullptr, nullptr, nullptr, 256,
                  &queue, error, base, bytes);
}
void poisoned(Prefill& p) {
    std::string error;
    need(p.chunk() == 0, "poison reports a runnable chunk");
    need(!p.run(nullptr, 1, 0, error) && error.find("layout unavailable") != std::string::npos,
         "poison reached run");
    core::MtpDrafter mtp;
    error.clear();
    need(!p.draft_kv(mtp, nullptr, nullptr, 1, 0, error) &&
         error.find("layout unavailable") != std::string::npos, "poison reached draft KV");
    error.clear();
    need(!p.relayout(128, nullptr, 1, error) && error.find("layout unavailable") != std::string::npos,
         "poison reached relayout");
}
template<class F> void rejects(F&& call, const char* why) {
    bool refused = false;
    try { call(); } catch (const std::exception&) { refused = true; }
    need(refused, why);
}
void loan_run(const std::string& mode) {
    Session session; session.init();
    core::ExpertCache cache;
    core::WeightTable weights;
    auto queue = runtime->make_queue(true);
    Prefill prompt;
    core::CacheScratchLease lease;
    std::string error;
    try {
        const auto maximum = std::numeric_limits<int64_t>::max();
        core::ExpertCache invalid;
        need(!invalid.open(2,1,1,maximum,error) && !invalid.valid(), "uniform cache product overflow");
        need(!invalid.open(1,maximum,2,1,error) && !invalid.valid(), "cache residency overflow");
        need(!invalid.open(int64_t(std::numeric_limits<int32_t>::max())+1,1,1,1,error), "cache slot index overflow");
        for (const auto& sizes : std::vector<std::vector<int64_t>>{{0},{-1},{maximum},{maximum/2,maximum/2}})
            need(!invalid.open_sized(sizes,1,1,error) && !invalid.valid(), "sized cache geometry admitted");
        if (mode == "loan") passed("cache_geometry_refusal");
        const auto full = exact_bytes(session,256), small = exact_bytes(session,128);
        constexpr size_t slab = 1u << 20;
        const auto tail = (full+slab-1)/slab;
        std::vector<int64_t> sizes(128+tail,slab);
        std::fill_n(sizes.begin(),128,int64_t(256));
        need(tail>1 && tail*slab+32768 <= (513u<<20) &&
             (tail-1)*slab >= small && sizes.size() <= size_t(session.geometry.n_layers*session.geometry.n_expert),
             "loan fixture budget");
        need(cache.open_sized(sizes,session.geometry.n_layers,session.geometry.n_expert,error), "loan cache allocation");
        std::vector<std::vector<uint8_t>> payload(sizes.size());
        std::vector<int32_t> resident(size_t(session.geometry.n_layers*session.geometry.n_expert),-1);
        for (size_t i=0; i<sizes.size(); ++i) {
            payload[i].assign(size_t(sizes[i]),uint8_t(1+(i*11)%251));
            need(cache.admit(int64_t(i)/session.geometry.n_expert,int64_t(i)%session.geometry.n_expert)==int32_t(i),
                 "cache slot identity");
            resident[i]=int32_t(i);
            need(cache.fill_slot_queued(int32_t(i),payload[i].data(),error), "initial sized payload fill");
        }
        need(cache.sync_queued(error), "initial cache completion");
        auto baseline = resident;
        sb::Allocation device_map(runtime,resident.size()*sizeof(int32_t),sb::MemoryKind::Device);
        const auto publish = [&](const char* phase) {
            return core::publish_residency_tables(1,phase,[&](size_t) {
                return cudaMemcpy(device_map.data(),resident.data(),resident.size()*sizeof(int32_t),cudaMemcpyHostToDevice);
            },error);
        };
        const auto map_matches = [&] {
            std::vector<int32_t> observed(resident.size());
            check(cudaMemcpy(observed.data(),device_map.data(),observed.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));
            need(observed==resident,"device residency snapshot differs");
        };
        const auto payload_matches = [&](size_t end) {
            for (size_t i=0;i<end;++i)
                if (baseline[i]>=0)
                    need(cache.verify_slot(int32_t(i),payload[i].data(),error),"cache payload/slot identity differs");
        };
        need(publish("initial"),"initial residency publication"); map_matches(); payload_matches(sizes.size());
        cache.enable_scratch_loan_tracking();
        if (mode=="loan") passed("sized_payload_initialization");
        { auto reader=cache.scratch_reader();
          rejects([&]{(void)cache.prepare_scratch_loan(128);},"reader did not prevent prepare"); }
        if (mode=="loan") passed("reader_prepare_exclusion");
        need(prompt.init(weights,session.geometry,session.state,nullptr,&cache,resident.data(),256,
                         &queue,error,cache.device_slot(128),tail*slab),"tracked Prefill init");
        need(!prompt.run(nullptr,1,0,error) && error.find("explicit parent lease")!=std::string::npos,
             "tracked unbound Prefill reached inference");
        if (mode=="loan") passed("tracked_prefill_entry_refusal");
        uint64_t first_generation=0;
        for (int cycle=0;cycle<3;++cycle) {
            if (cycle==2) {
                // Reopen the same cache object, retaining the old ticket as a
                // witness. This epoch has only prefix residents: an empty
                // residency tail still owns a real nonempty scratch interval.
                const auto old=lease;
                prompt.reset(); cache.close();
                need(cache.open_sized(sizes,session.geometry.n_layers,session.geometry.n_expert,error),"reopened cache");
                resident.assign(resident.size(),-1);
                for (size_t i=0;i<128;++i) {
                    need(cache.admit(0,int64_t(i))==int32_t(i),"reopened prefix slot"); resident[i]=int32_t(i);
                    need(cache.fill_slot_queued(int32_t(i),payload[i].data(),error),"reopened prefix payload");
                }
                need(cache.sync_queued(error),"reopened prefix completion"); baseline=resident;
                cache.enable_scratch_loan_tracking();
                need(!old.pending() && !old.contains(&cache,cache.device_slot(128),1,false),"old allocation epoch admitted");
                need(prompt.init(weights,session.geometry,session.state,nullptr,&cache,resident.data(),256,
                                 &queue,error,cache.device_slot(128),tail*slab),"reopened Prefill init");
                need(!prompt.bind_cache_scratch_lease(old,error),"old owner epoch bound after reopen");
                need(publish("reopened"),"reopened map"); map_matches();
                need(core::occupied_expert_tail_first(128,int32_t(sizes.size()-1),cache.slots(),
                     resident.data(),resident.size())==-1 && cache.decode_ready(),"empty-tail preflight changed parent");
            }
            const int32_t first=cycle==1 ? 129 : 128;
            const int64_t rows=cycle ? 128 : 256;
            const auto bytes=uint64_t(sizes.size()-size_t(first))*slab;
            auto* base=cache.device_slot(first);
            lease=cache.prepare_scratch_loan(first);
            if (!cycle) first_generation=lease.generation();
            else {
                need(lease.generation()==(cycle==1 ? first_generation+1 : 1),"loan generation/epoch mismatch");
                need(prompt.relayout(rows,base,bytes,error),"ready prior ticket prevented moved layout");
            }
            need(lease.pending() && lease.contains(&cache,base,bytes,false) &&
                 !lease.contains(&cache,base,bytes) && !lease.contains(&invalid,base,1,false) &&
                 !lease.contains(&cache,base+bytes-1,2,false) &&
                 !lease.contains(&cache,reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-255),512,false),
                 "parent capability owner/extent/admission");
            if (!cycle && mode=="loan") passed("owner_and_interval_capability");
            if (!cycle) {
                core::ExpertCache foreign;
                need(foreign.open(129,1,129,256,error),"foreign parent allocation");
                foreign.enable_scratch_loan_tracking();
                auto other=foreign.prepare_scratch_loan(128);
                need(!prompt.bind_cache_scratch_lease(other,error),"foreign capability was bound");
                need(foreign.mask_scratch_loan(other,[]{return true;}) &&
                     foreign.activate_scratch_loan(other,[]{return true;}) &&
                     foreign.retire_scratch_loan(other,[]{return true;}),"foreign unused lease retirement");
                foreign.begin_scratch_refill(other);
                need(foreign.publish_scratch_refill(other,[]{return true;}),"foreign unused lease completion");
                foreign.close();
                need(!other.pending() && !other.contains(&foreign,base,1,false),"closed foreign epoch alive");
                if (mode=="loan") passed("foreign_ticket_refusal");
            }
            need(prompt.bind_cache_scratch_lease(lease,error),"current cache capability bind");
            const auto layout_retained = [&] {
                need(lease.pending() && !prompt.relayout(128,base,bytes,error) && prompt.chunk()==rows &&
                     error.find("until cache loan refill/publication")!=std::string::npos,"pending loan changed layout");
            };
            layout_retained();
            if (!cycle && mode=="loan") passed("prepared_layout_retention");
            if (!cycle && (mode=="loan-release-prepared" || mode=="loan-parent-close")) {
                std::cout << "prefill_cache_loan retirement=" << mode << " phase=prepared owners_live=1\n" << std::flush;
                if (mode=="loan-parent-close") cache.close(); else prompt.reset();
                std::cout << "prefill_cache_loan UNEXPECTED_RETURN\n" << std::flush; std::_Exit(99);
            }
            for (size_t i=0;i<sizes.size();++i) if (resident[i]>=first) resident[i]=-1;
            need(cache.mask_scratch_loan(lease,[&] {
                layout_retained();
                need(!lease.contains(&cache,base,bytes,false),"transition granted access");
                rejects([&]{(void)cache.scratch_reader();},"mask admitted decode");
                need(!cache.fill_slot_queued(first,payload[size_t(first)].data(),error),"mask admitted writer");
                return publish("lend");
            }),"mask publication"); map_matches();
            if (!cycle && mode=="loan") passed("mask_publication_liveness");
            need(prompt.activate_cache_scratch_lease([&] {
                check(cudaDeviceSynchronize()); layout_retained(); return true;
            },error) && lease.contains(&cache,base,bytes),"checked queue retirement/activation");
            if (!cycle && mode=="loan") passed("checked_queue_retirement");
            need(!cache.fill_slot(first,payload[size_t(first)].data(),&queue,error) &&
                 !cache.fill_slot_blocking(first,payload[size_t(first)].data(),error) &&
                 !cache.fill_slot_queued(first,payload[size_t(first)].data(),error),"borrower did not exclude writers");
            if (!cycle && mode=="loan") passed("exclusive_write_protection");
            queue.memset(base,0x7e,size_t(bytes)); check(cudaStreamSynchronize(&queue));
            payload_matches(size_t(first));
            if (!cycle && mode=="loan") passed("borrowed_overwrite_prefix_guard");
            need(prompt.retire_cache_scratch_lease(error),"checked borrower retirement"); layout_retained();
            rejects([&]{(void)cache.scratch_reader();},"used loan admitted decode");
            need(!prompt.run(nullptr,1,0,error) && error.find("stale/inactive")!=std::string::npos,
                 "inactive loan reached inference");
            if (!cycle && mode=="loan") passed("used_decode_refusal");
            if (!cycle && mode=="loan-release-used") {
                std::cout << "prefill_cache_loan retirement=" << mode << " phase=used owners_live=1\n" << std::flush;
                prompt.reset(); std::cout << "prefill_cache_loan UNEXPECTED_RETURN\n" << std::flush; std::_Exit(99);
            }
            cache.begin_scratch_refill(lease); layout_retained();
            if (!cycle && mode=="loan-release-refilling") {
                std::cout << "prefill_cache_loan retirement=" << mode << " phase=refilling owners_live=1\n" << std::flush;
                prompt.reset(); std::cout << "prefill_cache_loan UNEXPECTED_RETURN\n" << std::flush; std::_Exit(99);
            }
            need(!cache.fill_slot(0,payload[0].data(),&queue,error) &&
                 !cache.fill_slot_blocking(0,payload[0].data(),error) &&
                 !cache.fill_slot_queued(0,payload[0].data(),error) &&
                 !cache.fill_slot_queued(first,payload[size_t(first)].data(),error,int64_t(slab)+1),
                 "refill escaped loan/slot interval");
            if (!cycle && mode=="loan") passed("refill_interval_guard");
            size_t refilled=0;
            for (size_t i=size_t(first);i<sizes.size();++i) if (baseline[i]>=first) {
                const auto* source=payload[i].data();
                const bool filled=i%3==0 ? cache.fill_slot(int32_t(i),source,&queue,error) :
                    i%3==1 ? cache.fill_slot_blocking(int32_t(i),source,error) : cache.fill_slot_queued(int32_t(i),source,error);
                need(filled,"same-slot refill submission");
                ++refilled;
            }
            need(cycle==2 ? refilled==0 : refilled>0,"empty/populated refill case mismatch");
            check(cudaStreamSynchronize(&queue)); need(cache.sync_queued(error),"same-slot refill completion");
            payload_matches(sizes.size());
            if (!cycle && mode=="loan") passed("refill_exact_slots_and_payload");
            std::copy(baseline.begin(),baseline.end(),resident.begin());
            if (!cycle && mode=="loan-release-poisoned") {
                need(!cache.publish_scratch_refill(lease,[]{return false;}),"injected publication refusal ignored");
                layout_retained();
                std::cout << "prefill_cache_loan retirement=" << mode << " phase=poisoned owners_live=1\n" << std::flush;
                prompt.reset(); std::cout << "prefill_cache_loan UNEXPECTED_RETURN\n" << std::flush; std::_Exit(99);
            }
            need(cache.publish_scratch_refill(lease,[&] {
                layout_retained();
                rejects([&]{(void)cache.scratch_reader();},"publication admitted decode");
                need(!cache.fill_slot_queued(first,payload[size_t(first)].data(),error),"publication admitted writer");
                return publish("refill");
            }),"refill publication"); map_matches();
            if (!cycle && mode=="loan") passed("publication_reentry_refusal");
            need(!lease.pending() && !lease.contains(&cache,base,bytes,false) && cache.decode_ready(),"ready admission");
            { auto reader=cache.scratch_reader(); }
            if (!cycle && mode=="loan") passed("ready_reader_reopened");
            if (cycle==1) passed("moved_smaller_next_generation");
            if (cycle==2) { passed("empty_resident_tail_cycle"); passed("reopened_owner_epoch_refusal"); }
        }
        need(mode=="loan","unknown loan retirement mode");
        prompt.reset(); cache.close();
        need(!lease.pending(),"closed parent retained a pending ticket"); passed("closed_epoch_refusal");
        std::cout << "prefill_cache_loan_fixture cases=20 cycles=3 model_PP=UNRUN status=PASS\n" << std::flush;
    } catch (const std::exception& failure) {
        std::cerr << "prefill cache loan fixture: " << failure.what() << " detail=" << error << '\n' << std::flush;
        if (lease.pending()) sb::cleanup_fail_stop("cache loan fixture failed with pending owners");
        throw;
    }
}
void run(const std::string& mode) {
    if (mode.rfind("loan",0)==0) { loan_run(mode); return; }
    Session session; session.init();
    core::WeightTable weights;
    auto queue = runtime->make_queue(true);
    const size_t full = exact_bytes(session, 256), small = exact_bytes(session, 128);
    Parent a(full), b(full);
    std::string error;
    if (mode == "smoke") {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),(64u << 20)-1,error) &&
             error.find("GEMM scratch does not fit") != std::string::npos, "first take refusal");
        poisoned(p); passed("first_take_refusal");
        a.unchanged(); b.unchanged();
        std::cout << "prefill_scratch_device_fixture mode=smoke cases=1 status=PASS\n";
        return;
    }
    if (mode == "retirement") {
        Prefill p; need(init(p,session,weights,queue,a.data(),full,error), "retirement fixture init");
        need(queue.get_context() == runtime->context(), "retirement origin context");
        check(cudaStreamSynchronize(&queue));
        // Controlled host-domain rejection, not a stalled/invalid GPU command.
        // The external queue object stays alive; no work remains on its old
        // value. Replacing it violates Prefill's queue-ownership contract.
        sycl::context other(runtime->device());
        sycl::queue foreign(other, runtime->device(), sycl::property::queue::in_order{});
        need(foreign.get_context() != runtime->context() &&
             foreign.has_property<sycl::property::queue::in_order>(), "retirement foreign queue setup");
        queue = std::move(foreign);
        std::cout << "prefill_scratch_device retirement=reset queue_domain_changed=1 owners_live=1 old_queue_drained=1 foreign_in_order=1\n" << std::flush;
        p.reset(); // must _Exit before releasing resources after failed sync
        std::cout << "prefill_scratch_device retirement=UNEXPECTED_RETURN\n" << std::flush;
        std::_Exit(99);
    }
    need(mode == "full", "unknown case");
    {
        Prefill p;
        need(init(p,session,weights,queue,a.data(),full,error) && p.chunk() == 256, "exact init");
        passed("exact_init");
        need(p.relayout(256,b.data(),full,error) && p.chunk() == 256, "moved layout");
        passed("moved_relayout");
        need(p.relayout(128,a.data(),small,error) && p.chunk() == 128, "smaller layout");
        passed("smaller_relayout");
        need(!p.relayout(257,b.data(),full,error) && p.chunk() == 128, "invalid growth changed valid layout");
        passed("growth_refusal");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),full-1,error) &&
            error.find("do not fit") != std::string::npos, "one byte short was admitted");
        passed("one_byte_short_poison"); poisoned(p); passed("poison_entries");
        need(!init(p,session,weights,queue,a.data(),full,error), "init bypassed reset");
        p.reset();
        need(init(p,session,weights,queue,a.data(),full,error), "reset failed after checked retirement");
        passed("reset_recovery");
        need(!p.relayout(128,b.data(),small-1,error), "short relayout was admitted");
        poisoned(p); passed("relayout_short_poison");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data()+2,full,error) &&
            error.find("GEMM") != std::string::npos, "GEMM init alignment refusal");
        poisoned(p); passed("gemm_init_refusal");
    }
    {
        Prefill p; need(init(p,session,weights,queue,a.data(),full,error), "rebind fixture init");
        bool threw = false;
        try { (void)p.relayout(128,b.data()+2,small,error); }
        catch (const std::invalid_argument&) { threw = true; }
        need(threw, "GEMM rebind did not refuse alignment"); poisoned(p); passed("gemm_rebind_refusal");
    }
    {
        auto bad = session.state; bad.qsa_states = nullptr;
        bool threw = false;
        try { (void)Prefill::bytes_needed(session.geometry,bad,256); }
        catch (const std::invalid_argument&) { threw = true; }
        need(threw, "planner accepted missing QSA state");
        Prefill p;
        need(!p.init(weights,session.geometry,bad,nullptr,nullptr,nullptr,256,&queue,error,a.data(),full),
            "init accepted missing QSA state");
        poisoned(p); passed("malformed_qsa_refusal");
    }
    {
        Prefill p;
        need(!init(p,session,weights,queue,a.data(),a.allocation.size(),error), "oversize parent span admitted");
        poisoned(p); passed("parent_bounds_refusal");
        sb::Allocation host(runtime,512,sb::MemoryKind::Host);
        p.reset();
        need(!init(p,session,weights,queue,host.data(),host.size(),error), "host USM admitted as scratch");
        poisoned(p); passed("host_usm_refusal");
        p.reset();
        need(queue.get_context() == runtime->context(), "full origin context");
        check(cudaStreamSynchronize(&queue));
        sycl::context other(runtime->device());
        sycl::queue foreign(other,runtime->device(),sycl::property::queue::in_order{});
        need(foreign.get_context() != runtime->context() &&
             foreign.has_property<sycl::property::queue::in_order>(), "full foreign queue setup");
        need(!init(p,session,weights,foreign,a.data(),full,error), "foreign queue admitted");
        need(error.find("runtime domain") != std::string::npos, "foreign queue refused for wrong reason");
        poisoned(p); passed("foreign_queue_refusal");
    }
    a.unchanged(); b.unchanged(); passed("canary_and_retirement");
    std::cout << "prefill_scratch_device_fixture mode=full cases=15 status=PASS\n";
}
}
int main(int argc, char** argv) try {
    need(argc == 3 && std::string(argv[1]) == "--case", "use --case smoke/full/retirement/loan/loan-release-<phase>/loan-parent-close");
    runtime = sb::runtime_for();
    run(argv[2]);
    runtime->wait();
    std::cout << "prefill_scratch_device_complete=1 model_PP=UNRUN\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "prefill scratch fixture: " << error.what() << '\n'; return 1;
}
