// Authored for Strata; repository LICENSE applies. Exercises the production
// upload helper and ExpertCache, not model arithmetic or session rollback.
#include "strata/prefill/expert_bank_upload.hpp"
#include "strata/prefill/draft_uploads.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/command_profile.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sys/mman.h>
#include <time.h>
#include <iostream>
#include <vector>

namespace pd = strata::prefill::detail;
namespace sb = strata::sycl_backend;
constexpr uint32_t Guard = 0xa5a5a5a5;
constexpr size_t G = 64;
void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t value(int expert, size_t word, unsigned generation) {
    uint32_t x = uint32_t(word) * 747796405u ^ uint32_t(expert + 1) * 2891336453u ^ generation * 277803737u;
    x ^= x >> 16; x *= 2246822519u;
    return x ^ (x >> 13);
}
struct Stop {};

void run(int capacity, size_t stride, int experts, unsigned layers, bool unwind, size_t source_stride = 0) {
    if (!source_stride) source_stride = stride;
    auto runtime = sb::runtime_for();
    auto& q = runtime->compute();
    std::vector<uint32_t> host(size_t(512) * source_stride / 4 + 2 * G, Guard);
    auto* slab = reinterpret_cast<uint8_t*>(host.data() + G);
    const size_t slab_bytes = size_t(512) * source_stride, words = source_stride / 4 - 1;
    strata::core::ExpertCache cache;
    std::string error;
    if (!cache.open_uninitialized(capacity, 1, 512, int64_t(stride), error))
        throw std::runtime_error(error);
    sb::Allocation status_owner(runtime, 4, sb::MemoryKind::Device);
    auto* status = status_owner.as<uint32_t>();
    pd::ExpertBankSlots slots(capacity);
    uint32_t observed = ~uint32_t(0);
    unsigned drains = 0, cancellations = 0;
    uint64_t uploads = 0, hits = 0, consumers = 0;
    for (unsigned generation = 1; generation <= layers; ++generation) {
        // Previous generation's queue was checked before changing the slab/map.
        slots.reset();
        for (int e = 0; e < 512; ++e)
            for (size_t i = 0; i < source_stride / 4; ++i)
                host[G + size_t(e) * source_stride / 4 + i] = value(e, i, generation);
        bool unwound = false;
        try {
            pd::DraftUploadGuard lease([&] { runtime->wait(); ++drains; return true; },
                [] { sb::cleanup_fail_stop("expert-bank fixture retirement"); });
            lease.arm();
            q.memset(cache.device_slot(0), 0xa5, size_t(cache.bytes()));
            q.memset(status, 0, 4);
            for (int round = 0; round < 2; ++round) for (int e = 0; e < experts; ++e) {
                const auto ticket = pd::stage_expert_bank(slots, cache, slab, slab_bytes,
                    stride, words * 4, e, &q, source_stride);
                uploads += ticket.upload; hits += !ticket.upload;
                need(cache.slot_of(0, e) < 0, "private scratch/permanent map leaked to general cache");
                const auto* blob = reinterpret_cast<const uint32_t*>(cache.device_slot(ticket.slot));
                // Two queued readers before the next scratch overwrite model
                // multiple consumers without adding a wait after each expert.
                for (int job = 0; job < 2; ++job) {
                    const size_t global = (words + 255) / 256 * 256;
                    q.parallel_for(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_linear_id();
                        const bool bad = i < words && blob[i] != value(e, i, generation);
                        const bool any = sycl::any_of_group(item.get_group(), bad);
                        if (item.get_local_linear_id() == 0 && any)
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>(*status).fetch_or(1);
                    });
                    ++consumers;
                }
                if (unwind && e == 4) {
                    q.memcpy(&observed, status, 4);
                    throw Stop{}; // slab/cache/status destinations outlive checked lease
                }
            }
            const auto* base = reinterpret_cast<const uint32_t*>(cache.device_slot(0));
            q.parallel_for(sycl::range<1>(size_t(capacity)), [=](sycl::id<1> item) {
                if (base[(size_t(item[0]) + 1) * (stride / 4) - 1] != Guard)
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device, sycl::access::address_space::global_space>(*status).fetch_or(2);
            });
            q.memcpy(&observed, status, 4);
            runtime->wait(); ++drains; lease.retire();
        } catch (const Stop&) { unwound = true; ++cancellations; }
        need(unwound == unwind && observed == 0, "queued reader, guard or unwind result failed");
        for (size_t i = 0; i < G; ++i)
            need(host[i] == Guard && host[host.size() - 1 - i] == Guard, "host guard changed");
        for (int e = 0; e < 512; ++e)
            for (size_t i = 0; i < source_stride / 4; ++i)
                need(host[G + size_t(e) * source_stride / 4 + i] == value(e, i, generation), "immutable slab changed");
    }
    if (!unwind) {
        const uint64_t permanent = uint64_t(std::min(capacity - 1, experts));
        need(uploads == layers * (2 * uint64_t(experts) - permanent) && hits == layers * permanent,
            "permanent/scratch admission totals");
    }
    need(drains == layers && cancellations == (unwind ? layers : 0), "checked retirement count");
    unsigned rejects = 0;
    auto reject = [&](auto action) {
        bool caught = false;
        try { action(); } catch (const std::invalid_argument&) { caught = true; }
        need(caught, "invalid admission accepted"); ++rejects;
    };
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes - 1, stride, words * 4, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, stride + 1, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 512, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, nullptr, slab_bytes, stride, words * 4, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, nullptr, source_stride); });
    pd::ExpertBankSlots wrong(capacity == 1 ? 2 : 1);
    reject([&] { pd::stage_expert_bank(wrong, cache, slab, slab_bytes, stride, words * 4, 0, &q, source_stride); });
    sycl::queue foreign{sycl::context{runtime->device()}, runtime->device(), sycl::property::queue::in_order{}};
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &foreign, source_stride); });
    if (source_stride != stride) {
        reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &q, source_stride - 8); });
        reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &q, stride); });
    }
    runtime->wait();
    std::cout << "expert_bank_case slots=" << capacity << " stride=" << stride << " experts=" << experts
        << " source_stride=" << source_stride
        << " layers=" << layers << " uploads=" << uploads << " hits=" << hits << " consumers=" << consumers
        << " unwind=" << cancellations << " admission_rejections=" << rejects
        << " status=" << observed << " immutable_slab=PASS guards=PASS private_map=PASS checked_drain=PASS\n";
}

class BankTimingMarker;
sycl::event bank_marker(sycl::queue& q);
double thread_seconds();

// Disjoint host intervals. Wrapper buckets can contain internal runtime waits;
// neither wrapper CPU time nor final-drain CPU time means CPU model arithmetic.
struct BankHostAccounting {
    using Clock = std::chrono::steady_clock;
    enum Phase {other, upload, readers, drain};
    bool enabled, active = false;
    Phase phase = other;
    Clock::time_point previous;
    double previous_cpu = 0;
    std::array<double,4> wall{}, cpu{};
    std::array<uint64_t,4> entries{};
    explicit BankHostAccounting(bool on) : enabled(on) {}
    void begin(Clock::time_point w, double c) {
        if (!enabled) return;
        need(!active,"nested host accounting");active=true;previous=w;previous_cpu=c;
    }
    void record(Clock::time_point w,double c) {
        need(active && w>=previous && c>=previous_cpu,"host accounting clock/state");
        wall[phase]+=std::chrono::duration<double,std::milli>(w-previous).count();
        cpu[phase]+=(c-previous_cpu)*1000;previous=w;previous_cpu=c;
    }
    void enter(Phase next) {
        if (!enabled) return;
        const auto w=Clock::now();const auto c=thread_seconds();record(w,c);
        phase=next;++entries[next];
    }
    void finish(Clock::time_point w,double c) {
        if (!enabled) return;
        record(w,c);active=false;
    }
};

// Exercise the production private view through checked transitions in ONE
// allocation. The source stride and destination stride deliberately differ.
void run_private(int backing_slots, uint64_t backing_stride, int experts,
                 const std::vector<uint64_t>& strides, uint64_t source_stride, bool unwind, bool full_copy=false,
                 bool staged=false, bool custom_queue=false, bool copy_runs=false, bool timing=false,
                 int run_limit=512, bool host_profile=false) {
    need(run_limit>=1 && run_limit<=512 && (!host_profile || timing),"copy-size/profile contract");
    if (copy_runs) need(full_copy && !staged &&
        std::all_of(strides.begin(),strides.end(),[&](uint64_t s) {return s==source_stride;}),
        "copy runs require exact packed payloads without staging");
    auto runtime = sb::runtime_for();
    auto q = custom_queue ? runtime->make_queue(true) : runtime->compute();
    auto drain = [&] { q.wait_and_throw(); runtime->check(); runtime->wait(); };
    const uint64_t slab_bytes = 512 * source_stride;
    std::vector<uint32_t> host(size_t(slab_bytes/4)+2*G,Guard);
    auto* slab = reinterpret_cast<uint8_t*>(host.data()+G);
    strata::core::ExpertCache cache;
    std::string error;
    if (!cache.open_uninitialized(backing_slots,1,512,int64_t(backing_stride),error))
        throw std::runtime_error(error);
    sb::Allocation status_owner(runtime,4,sb::MemoryKind::Device);
    auto* status = status_owner.as<uint32_t>();
    uint32_t observed = ~uint32_t(0);
    std::unique_ptr<pd::ExpertUploadStaging> staging;
    if (staged) staging.reset(new pd::ExpertUploadStaging(cache.device_slot(0),
        size_t(cache.bytes()), size_t(source_stride), &q));
    uint64_t uploads=0,hits=0,consumers=0,expected_uploads=0,expected_hits=0;
    uint64_t copy_commands=0,runs=0,run_experts=0;
    double wall_ms=0,queue_ms=0,cpu_ms=0,empty_ms=0;
    double clock_change_ns=0;
    BankHostAccounting accounting(host_profile);
    unsigned drains=0,cancellations=0;
    for (size_t layer=0;layer<strides.size();++layer) {
        // The previous iteration's lease drained before rebinding these views
        // or changing host source bytes. No layer allocation/reallocation.
        pd::PrivateExpertBank bank(cache,strides[layer]);
        const auto geometry=bank.layout();
        pd::ExpertBankSlots slots(geometry.slots,geometry.full_fit());
        const auto blob_bytes=std::min(source_stride,geometry.stride)-(full_copy?0:4);
        const auto words=size_t(blob_bytes/4);
        const auto generation=unsigned(layer+1);
        for (int e=0;e<512;++e)
            for (size_t i=0;i<size_t(source_stride/4);++i)
                host[G+size_t(e)*size_t(source_stride/4)+i]=value(e,i,generation);
        bool unwound=false;
        try {
            pd::DraftUploadGuard lease([&] {drain();++drains;return true;},
                [] {sb::cleanup_fail_stop("private expert-bank retirement");});
            lease.arm();
            q.memset(cache.device_slot(0),0xa5,size_t(cache.bytes()));
            q.memset(status,0,4);
            using Wall=std::chrono::steady_clock;
            Wall::time_point timed_start;
            double cpu_start=0;
            std::unique_ptr<sycl::event> begin_marker;
            if (timing) {
                need(custom_queue && !staged && !unwind && strides.size()==1,"timing owner/shape contract");
                drain(); // allocation/source preparation and reset are not timed
                auto start=Wall::now();bank_marker(q);bank_marker(q);drain();
                empty_ms=std::chrono::duration<double,std::milli>(Wall::now()-start).count();
                if (host_profile) {
                    BankHostAccounting calibration(true);
                    const auto c=thread_seconds();const auto w=Wall::now();calibration.begin(w,c);
                    for (int i=0;i<4096;++i) calibration.enter(BankHostAccounting::Phase(i%3));
                    const auto end=Wall::now();calibration.finish(end,thread_seconds());
                    clock_change_ns=std::chrono::duration<double,std::nano>(end-w).count()/4096;
                }
                begin_marker=std::make_unique<sycl::event>(bank_marker(q));
                cpu_start=thread_seconds();timed_start=Wall::now();
                accounting.begin(timed_start,cpu_start);
            }
            for (int round=0;round<2;++round) {
              pd::ExpertBankSlots::UploadRun accepted;
              for (int e=0;e<experts;++e) {
                accounting.enter(BankHostAccounting::upload);
                pd::ExpertBankSlots::Ticket ticket;
                const int count=copy_runs && !accepted.contains(e) ?
                    slots.permanent_run_length(e,std::min(run_limit,experts-e)) : 0;
                if (copy_runs && accepted.contains(e)) ticket=accepted.ticket(e);
                else if (count>1) {
                    accepted=pd::stage_private_expert_bank_run(slots,bank,slab,slab_bytes,
                        blob_bytes,e,count,&q,source_stride);
                    ticket=accepted.ticket(e);++copy_commands;++runs;run_experts+=uint64_t(count);
                } else {
                    ticket=pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,
                        blob_bytes,e,&q,source_stride,staging.get());
                    copy_commands+=ticket.upload;
                }
                uploads+=ticket.upload;hits+=!ticket.upload;
                need(cache.slot_of(0,e)<0,"private map leaked into general cache");
                const auto* blob=reinterpret_cast<const uint32_t*>(bank.device_slot(ticket.slot));
                accounting.enter(BankHostAccounting::readers);
                for (int job=0;job<2;++job) {
                    const size_t global=(words+255)/256*256;
                    q.parallel_for(sycl::nd_range<1>(global,256),[=](sycl::nd_item<1> item) {
                        const size_t i=item.get_global_linear_id();
                        const bool bad=i<words && blob[i]!=value(e,i,generation);
                        const bool any=sycl::any_of_group(item.get_group(),bad);
                        if (item.get_local_linear_id()==0 && any)
                            sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                                sycl::memory_scope::device,sycl::access::address_space::global_space>(*status).fetch_or(1);
                    });
                    ++consumers;
                }
                // Keep loop/control bookkeeping in the inclusive reader bucket.
                // Two clock pairs per expert avoid charging a third pair for
                // only these few instructions; the initial prefix is 'other'.
                if (unwind && e==4) {
                    q.memcpy(&observed,status,4);
                    throw Stop{};
                }
              }
            }
            if (timing) {
                accounting.enter(BankHostAccounting::drain);
                auto end_marker=bank_marker(q);drain();
                const auto wall_end=Wall::now();const auto cpu_end=thread_seconds();
                wall_ms=std::chrono::duration<double,std::milli>(wall_end-timed_start).count();
                cpu_ms=(cpu_end-cpu_start)*1000;
                accounting.finish(wall_end,cpu_end);
                const auto start=begin_marker->get_profiling_info<sycl::info::event_profiling::command_end>();
                const auto end=end_marker.get_profiling_info<sycl::info::event_profiling::command_start>();
                need(end>start && wall_ms>0 && cpu_ms>=0,"copy-run timing clock/order invalid");
                queue_ms=double(end-start)/1e6;
            }
            const auto* base=reinterpret_cast<const uint32_t*>(cache.device_slot(0));
            // Check every padding word, including the unused backing tail.
            q.parallel_for(sycl::range<1>(size_t(geometry.backing_bytes/4)),[=](sycl::id<1> item) {
                const uint64_t at=uint64_t(item[0])*4;
                if ((at>=geometry.used_bytes || at%geometry.stride>=blob_bytes) &&
                    base[item[0]]!=Guard)
                    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                        sycl::memory_scope::device,sycl::access::address_space::global_space>(*status).fetch_or(2);
            });
            q.memcpy(&observed,status,4);
            drain();++drains;lease.retire();
        } catch (const Stop&) {unwound=true;++cancellations;}
        need(unwound==unwind && observed==0,"private bank reader/guard/unwind failed");
        for (size_t i=0;i<G;++i)
            need(host[i]==Guard && host[host.size()-1-i]==Guard,"private source guard changed");
        for (int e=0;e<512;++e)
            for (size_t i=0;i<size_t(source_stride/4);++i)
                need(host[G+size_t(e)*size_t(source_stride/4)+i]==value(e,i,generation),"private immutable source changed");
        const auto permanent=uint64_t(std::min(experts,geometry.slots-(geometry.full_fit()?0:1)));
        expected_uploads+=2*uint64_t(experts)-permanent;expected_hits+=permanent;
    }
    if (staging) {
        staging->check_guards();
        need(staging->copies()==uploads,"staging upload admission count");
    }
    if (!unwind) {
        need(uploads==expected_uploads && hits==expected_hits,"private upload/reuse totals");
        need(run_experts<=uploads && copy_commands==uploads-run_experts+runs,"private copy-run conservation");
    }
    need(drains==strides.size() && cancellations==(unwind?strides.size():0),"private checked retirement count");
    unsigned rejects=0;
    auto reject=[&](auto action) {
        bool caught=false;
        try {action();} catch (const std::invalid_argument&) {caught=true;}
        need(caught,"private invalid admission accepted");++rejects;
    };
    pd::PrivateExpertBank bank(cache,strides[0]);
    pd::ExpertBankSlots slots(bank.layout().slots,bank.layout().full_fit());
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes-1,4,0,&q,source_stride);});
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,bank.layout().stride+1,0,&q,source_stride);});
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,4,512,&q,source_stride);});
    reject([&] {pd::stage_private_expert_bank(slots,bank,nullptr,slab_bytes,4,0,&q,source_stride);});
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,4,0,nullptr,source_stride);});
    pd::ExpertBankSlots wrong(bank.layout().slots==1?2:1);
    reject([&] {pd::stage_private_expert_bank(wrong,bank,slab,slab_bytes,4,0,&q,source_stride);});
    sycl::queue foreign{sycl::context{runtime->device()},runtime->device(),sycl::property::queue::in_order{}};
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,4,0,&foreign,source_stride);});
    reject([&] {pd::stage_private_expert_bank(slots,bank,slab,slab_bytes,4,0,&q,source_stride-4);});
    reject([&] {(void)bank.device_slot(bank.layout().slots);});
    cache.enable_scratch_loan_tracking();
    reject([&] {(void)bank.device_slot(0);});
    reject([&] {pd::PrivateExpertBank invalid(cache,strides[0]);});
    drain();
    std::cout<<"expert_bank_private_case backing_slots="<<backing_slots<<" backing_stride="<<backing_stride
        <<" source_stride="<<source_stride<<" experts="<<experts<<" layers="<<strides.size()
        <<" uploads="<<uploads<<" hits="<<hits<<" consumers="<<consumers<<" unwind="<<cancellations
        <<" admission_rejections="<<rejects<<" status="<<observed<<" full_copy="<<int(full_copy)
        <<" staged="<<staged<<" custom_queue="<<custom_queue<<" copy_runs="<<copy_runs
        <<" copy_commands="<<copy_commands<<" runs="<<runs<<" run_experts="<<run_experts
        <<" immutable_slab=PASS guards=PASS private_map=PASS checked_drain=PASS\n";
    if (timing) {
        const bool eligible=wall_ms>=100 && queue_ms>=100 && empty_ms<=0.05*wall_ms;
        std::cout<<std::setprecision(17)<<"expert_bank_copy_timing candidate="<<copy_runs
            <<" payload="<<source_stride<<" logical_uploads="<<uploads<<" requested_H2D_bytes="<<uploads*source_stride
            <<" commands="<<copy_commands<<" wall_ms="<<wall_ms<<" queue_ms="<<queue_ms
            <<" thread_ms="<<cpu_ms<<" empty_ms="<<empty_ms<<" eligible="<<eligible
            <<" preparation_timed=0 guards_timed=0 actual_copy_read_drain_timed=1 model_qualified=0\n";
        if (host_profile) {
            double sum_wall=0,sum_cpu=0;uint64_t switches=0;
            for (size_t i=0;i<4;++i) {sum_wall+=accounting.wall[i];sum_cpu+=accounting.cpu[i];switches+=accounting.entries[i];}
            need(!accounting.active && std::abs(sum_wall-wall_ms)<1e-5 && std::abs(sum_cpu-cpu_ms)<1e-5 &&
                accounting.entries[1]==uint64_t(2*experts) && accounting.entries[2]==uint64_t(2*experts) &&
                accounting.entries[0]==0 && accounting.entries[3]==1,
                "disjoint host accounting incomplete");
            std::cout<<"expert_bank_copy_host_profile cap="<<run_limit<<" payload="<<source_stride;
            const std::array<const char*,4> names{"other_submit","upload_wrapper","reader_submit_control","final_drain"};
            for (size_t i=0;i<4;++i) std::cout<<" "<<names[i]<<"_wall_ms="<<accounting.wall[i]
                <<" "<<names[i]<<"_thread_ms="<<accounting.cpu[i];
            std::cout<<" clock_change_ns="<<clock_change_ns<<" clock_switches="<<switches
                <<" timer_estimate_ms="<<clock_change_ns*double(switches)/1e6
                <<" intervals_disjoint=1 coverage_conserved=1 exceptions=0 profiling_only=1 reader_loop_control_included=1\n";
        }
    }
}

void run_private_run_enqueue_failure() {
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    for (int mode=0;mode<3;++mode) {
        strata::core::ExpertCache cache;std::string error;
        if (!cache.open_uninitialized(4,1,512,256,error)) throw std::runtime_error(error);
        pd::PrivateExpertBank bank(cache,256);pd::ExpertBankSlots slots(4);
        std::vector<uint8_t> slab(512*256,0x7b),observed(4*256,0);
        unsigned attempts=0,drains=0;bool rejected=false;
        try {
            pd::DraftUploadGuard lease([&] {runtime->wait();++drains;return true;},
                [] {sb::cleanup_fail_stop("private run enqueue rejection retirement");});lease.arm();
            q.memset(cache.device_slot(0),0xa5,4*256);
            try {
                pd::stage_private_expert_bank_run_enqueue(slots,bank,slab.data(),slab.size(),256,5,3,&q,256,
                    [&](void* dst,const void* src,size_t bytes,void* stream) {
                        need(dst==bank.device_slot(0) && src==slab.data()+5*256 &&
                            bytes==3*256 && stream==&q,"wrong rejected run arguments");++attempts;
                        if (mode==1) throw std::runtime_error("injected enqueue throw");
                        if (mode==2) q.memcpy(dst,src,bytes); // accepted work,ambiguous caller result
                        return false;
                    });
            } catch (const std::runtime_error&) {rejected=true;}
            need(rejected && slots.permanent_used()==0,"failed run published mappings");
            auto refuse=[&](auto action) {
                bool caught=false;try {action();} catch(const std::invalid_argument&) {caught=true;}
                need(caught,"ambiguous run permits retry/reset");
            };
            refuse([&] {(void)slots.begin(5);});refuse([&] {slots.reset();});
            refuse([&] {(void)slots.begin_run(5,3);});
            q.memcpy(observed.data(),cache.device_slot(0),observed.size());
            throw Stop{};
        } catch (const Stop&) {}
        need(attempts==1 && drains==1,"run rejection lost checked retirement");
        for (size_t i=0;i<observed.size();++i)
            need(observed[i]==uint8_t(mode==2 && i<3*256 ? 0x7b : 0xa5),"ambiguous run bytes not drained");
    }
    std::cout<<"expert_bank_run_enqueue_failure modes=3 published=0 no_retry=PASS checked_drain=PASS\n";
}

void run_private_run_admissions() {
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    strata::core::ExpertCache cache;std::string error;
    if (!cache.open_uninitialized(4,1,512,256,error)) throw std::runtime_error(error);
    pd::PrivateExpertBank bank(cache,256);pd::ExpertBankSlots slots(4),wrong(3);
    std::vector<uint8_t> slab(512*256,0x7b);unsigned rejected=0,attempts=0;
    auto enqueue=[&](void*,const void*,size_t,void*) {++attempts;return true;};
    auto reject=[&](auto action) {
        bool caught=false;try {action();} catch(const std::invalid_argument&) {caught=true;}
        need(caught,"invalid private copy run accepted");++rejected;
    };
    auto issue=[&](auto& s,const uint8_t* source,uint64_t bytes,uint64_t payload,
                  int first,int count,void* stream,uint64_t stride) {
        return pd::stage_private_expert_bank_run_enqueue(s,bank,source,bytes,payload,
            first,count,stream,stride,enqueue);
    };
    reject([&] {issue(slots,slab.data(),slab.size()-1,256,0,2,&q,256);});
    reject([&] {issue(slots,slab.data(),slab.size(),252,0,2,&q,256);});
    reject([&] {issue(slots,slab.data(),slab.size(),256,0,2,&q,512);});
    reject([&] {issue(slots,nullptr,slab.size(),256,0,2,&q,256);});
    reject([&] {issue(slots,slab.data(),slab.size(),256,0,2,nullptr,256);});
    reject([&] {issue(slots,slab.data(),slab.size(),256,511,2,&q,256);});
    reject([&] {issue(slots,slab.data(),slab.size(),256,0,4,&q,256);});
    reject([&] {issue(wrong,slab.data(),slab.size(),256,0,2,&q,256);});
    sycl::queue foreign{sycl::context{runtime->device()},runtime->device(),sycl::property::queue::in_order{}};
    reject([&] {issue(slots,slab.data(),slab.size(),256,0,2,&foreign,256);});
    cache.enable_scratch_loan_tracking();
    reject([&] {issue(slots,slab.data(),slab.size(),256,0,2,&q,256);});
    need(attempts==0 && slots.permanent_used()==0,"run refusal mutated state or enqueued");
    runtime->wait();
    std::cout<<"expert_bank_run_admissions rejections="<<rejected<<" enqueues=0 mapping_unchanged=PASS\n";
}

void run_private_enqueue_failure() {
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    strata::core::ExpertCache cache;std::string error;
    if (!cache.open_uninitialized(4,1,512,256,error)) throw std::runtime_error(error);
    pd::PrivateExpertBank bank(cache,256);pd::ExpertBankSlots slots(4);
    std::vector<uint8_t> slab(512*256,0x7b),observed(4*256,0);
    unsigned attempts=0,drains=0;bool rejected=false,pending_rejected=false;
    try {
        pd::DraftUploadGuard lease([&] {runtime->wait();++drains;return true;},
            [] {sb::cleanup_fail_stop("private enqueue rejection retirement");});
        lease.arm();
        q.memset(cache.device_slot(0),0xa5,4*256);
        q.memcpy(observed.data(),cache.device_slot(0),observed.size());
        try {
            pd::stage_private_expert_bank_enqueue(slots,bank,slab.data(),slab.size(),256,0,&q,256,
                [&](void* dst,const void* src,size_t bytes,void* stream) {
                    need(dst==bank.device_slot(0) && src==slab.data() && bytes==256 && stream==&q,
                        "wrong rejected upload arguments");
                    ++attempts;return false; // no provider fault or device reset
                });
        } catch (const std::runtime_error& e) {
            rejected=std::string(e.what())=="private expert bank upload enqueue failed";
        }
        need(rejected && slots.permanent_used()==0,"failed enqueue published a private mapping");
        try {(void)slots.begin(0);} catch (const std::invalid_argument&) {pending_rejected=true;}
        need(pending_rejected,"ambiguous upload was retryable");
        throw Stop{}; // normal error unwind must retire already-queued work
    } catch (const Stop&) {}
    need(attempts==1 && drains==1 && std::all_of(observed.begin(),observed.end(),
        [](uint8_t byte) {return byte==0xa5;}),"enqueue rejection failed checked retirement");
    std::cout<<"expert_bank_private_enqueue_failure attempts=1 published=0 pending_rejected=PASS checked_drain=PASS\n";
}



void staging_admissions() {
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    strata::core::ExpertCache cache;std::string error;
    if(!cache.open_uninitialized(4,1,512,512,error))throw std::runtime_error(error);
    std::vector<uint8_t> source(512,0x3b);
    pd::ExpertUploadStaging staging(cache.device_slot(0),size_t(cache.bytes()),256,&q);
    unsigned rejected=0;
    auto reject=[&](auto action){bool caught=false;try{action();}catch(const std::invalid_argument&){caught=true;}
        need(caught,"staging invalid admission accepted");++rejected;};
    reject([&]{staging.enqueue(cache.device_slot(0),nullptr,256,&q);});
    reject([&]{staging.enqueue(cache.device_slot(0),source.data(),0,&q);});
    reject([&]{staging.enqueue(cache.device_slot(0),source.data(),257,&q);});
    reject([&]{staging.enqueue(cache.device_slot(0),source.data(),256,nullptr);});
    reject([&]{staging.enqueue(source.data(),source.data()+256,256,&q);});
    reject([&]{staging.enqueue(cache.device_slot(0),cache.device_slot(1),256,&q);});
    auto other=runtime->make_queue(true);
    reject([&]{staging.enqueue(cache.device_slot(0),source.data(),256,&other);});
    reject([&]{pd::ExpertUploadStaging bad(cache.device_slot(0),size_t(cache.bytes()),0,&q);});
    reject([&]{pd::ExpertUploadStaging bad(cache.device_slot(0),size_t(cache.bytes()),pd::ExpertUploadStaging::maximum_blob+1,&q);});
    sycl::queue foreign{sycl::context{runtime->device()},runtime->device(),sycl::property::queue::in_order{}};
    reject([&]{pd::ExpertUploadStaging bad(cache.device_slot(0),size_t(cache.bytes()),256,&foreign);});
    need(staging.copies()==0,"rejected staging published a copy");
    {
        pd::ExpertUploadStaging retiring(cache.device_slot(0),size_t(cache.bytes()),256,&q);
        for(int i=0;i<17;++i)retiring.enqueue(cache.device_slot(i%4),source.data(),256,&q);
        // Destruction itself retires the final four host-copy events.
    }
    std::vector<uint8_t> observed(256);
    for(int slot=0;slot<4;++slot){q.memcpy(observed.data(),cache.device_slot(slot),observed.size());runtime->wait();
        need(std::all_of(observed.begin(),observed.end(),[](uint8_t v){return v==0x3b;}),"staging destructor copy lifetime");}
    auto custom=runtime->make_queue(true);
    sycl::event tail;
    {
        pd::ExpertUploadStaging retiring(cache.device_slot(0),size_t(cache.bytes()),256,&custom);
        for(int i=0;i<17;++i)retiring.enqueue(cache.device_slot(i%4),source.data(),256,&custom);
        auto* destination=reinterpret_cast<uint32_t*>(cache.device_slot(3));
        tail=custom.single_task([=]{ uint32_t x=7;
            for(int i=0;i<65536;++i)x=(x^(x>>13))*1664525U+1013904223U;
            destination[64]=x; });
        // The destructor must retire the actual custom queue, including tail.
    }
    need(tail.get_info<sycl::info::event::command_execution_status>()==
        sycl::info::event_command_status::complete,"custom queue was not drained by destructor");
    custom.wait_and_throw();runtime->check();
    staging.check_guards();
    std::cout<<"expert_staging_custom_queue destructor_tail_complete=PASS host_copy_owners_live=PASS\n";
    std::cout<<"expert_staging_admissions rejections="<<rejected<<" destructor_inflight=PASS host_guards=PASS\n";
}

// Isolate the actual private-bank upload entrypoint and its queued readers.
// This measures no dequantization/GEMM or model throughput; those need their
// own production-path comparison before model advancement.
class BankTimingMarker;
sycl::event bank_marker(sycl::queue& q) { return q.single_task<BankTimingMarker>([]{}); }
double thread_seconds() {
    timespec t{};need(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0,"thread clock");
    return double(t.tv_sec)+double(t.tv_nsec)/1e9;
}
struct ReadonlySlab {
    void* pointer = nullptr;size_t bytes;
    explicit ReadonlySlab(size_t n):bytes(n) {
        pointer=mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        need(pointer!=MAP_FAILED,"benchmark source mapping");
    }
    ~ReadonlySlab() { if(pointer && pointer!=MAP_FAILED) munmap(pointer,bytes); }
    ReadonlySlab(const ReadonlySlab&)=delete;
};
void bank_padding_guard(sycl::queue& q,const uint32_t* base,size_t stride,size_t payload,uint32_t* status) {
    need(stride==payload+256 && payload%4==0 && stride<=(16u<<20),"padding guard geometry");
    q.parallel_for(sycl::range<1>(384*64),[=](sycl::id<1> item) {
        const size_t slot=item[0]/64,word=item[0]%64;
        if(base[(slot*stride+payload)/4+word]!=Guard)
            sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(*status).fetch_or(2);
    });
}
void bank_padding_guard_smoke() {
    auto runtime=sb::runtime_for();auto q=runtime->make_queue(true);
    sb::Allocation data(runtime,384*512,sb::MemoryKind::Device),flag(runtime,4,sb::MemoryKind::Device);
    auto drain=[&]{q.wait_and_throw();runtime->check();};
    pd::DraftUploadGuard lease([&]{drain();return true;},
        []{sb::cleanup_fail_stop("padding guard fixture retirement");});lease.arm();
    const std::array<int64_t,8> offsets{{-1,252,256,508,191*512+256,191*512+508,383*512+256,383*512+508}};
    for(size_t test=0;test<offsets.size();++test) {
        uint32_t zero=0,observed=~uint32_t(0);
        q.memset(data.data(),0xa5,data.size());q.memset(flag.data(),0,4);
        if(offsets[test]>=0)q.memcpy(data.as<uint8_t>()+offsets[test],&zero,4);
        bank_padding_guard(q,data.as<uint32_t>(),512,256,flag.as<uint32_t>());
        q.memcpy(&observed,flag.data(),4);drain();
        need(observed==(test<2?0u:2u),"padding guard missed boundary or rejected payload bytes");
    }
    for(int mode=0;mode<2;++mode) {
        std::vector<uint8_t> observed_bytes(data.size());
        uint32_t observed=~uint32_t(0);unsigned drains=0;bool unwound=false;
        try {
            pd::DraftUploadGuard check_lease([&]{drain();++drains;return true;},
                []{sb::cleanup_fail_stop("padding exception fixture retirement");});check_lease.arm();
            q.memset(data.data(),0xa5,data.size());q.memset(flag.data(),0,4);
            if(mode==0)q.memcpy(observed_bytes.data(),data.data(),data.size());
            else {
                bank_padding_guard(q,data.as<uint32_t>(),512,256,flag.as<uint32_t>());
                q.memcpy(&observed,flag.data(),4);
            }
            throw Stop{}; // Only a host exception;never a provider/device fault.
        } catch(const Stop&) {unwound=true;}
        need(unwound && drains==1,"padding validation unwind missed checked retirement");
        if(mode==0)need(std::all_of(observed_bytes.begin(),observed_bytes.end(),
            [](uint8_t v){return v==0xa5;}),"D2H validation destination released before drain");
        else need(observed==0,"GPU validation status destination released before drain");
    }
    lease.retire();std::cout<<"padding_guard_cases=8 first_middle_last_boundaries=PASS payload_exclusion=PASS\n"
        <<"padding_guard_unwind_cases=2 checked_drain_before_owner_release=PASS\n";
}
void staging_benchmark(int process, bool custom_queue=false, bool attribution=false,
                       bool device_guard=false, int rounds=9) {
    using Clock=std::chrono::steady_clock;
    auto runtime=sb::runtime_for();
    auto q=custom_queue?runtime->make_queue(true):runtime->compute();
    auto drain=[&]{sb::cpu_wait_profile::blocking(sb::cpu_wait_profile::direct_queue,
        [&]{q.wait_and_throw();});runtime->check();runtime->wait();};
    std::vector<double> clock_ns;
    for (int i=0;i<1000;++i) {const auto a=Clock::now(),b=Clock::now();clock_ns.push_back(std::chrono::duration<double,std::nano>(b-a).count());}
    std::sort(clock_ns.begin(),clock_ns.end());
    need(clock_ns[989]<=10000,"timer p99 above10us");
    for (size_t payload:{size_t(1971200),size_t(2662400)}) {
        const size_t stride=payload+256,slab_bytes=512*stride;
        ReadonlySlab slab(slab_bytes);
        auto* source=static_cast<uint32_t*>(slab.pointer);
        for (int e=0;e<512;++e) for (size_t i=0;i<stride/4;++i)
            source[size_t(e)*(stride/4)+i]=value(e,i,1);
        need(mprotect(slab.pointer,slab_bytes,PROT_READ)==0,"readonly slab protection");
        strata::core::ExpertCache cache;std::string error;
        if(!cache.open_uninitialized(384,1,512,int64_t(stride),error))throw std::runtime_error(error);
        pd::PrivateExpertBank bank(cache,stride);
        pd::ExpertUploadStaging staging(cache.device_slot(0),size_t(cache.bytes()),payload,&q);
        sb::Allocation status_owner(runtime,4,sb::MemoryKind::Device);
        auto* status=status_owner.as<uint32_t>();
        // Keep timing-instrument initialization separate from bank first-use.
        // All five subsequent empty controls and the cold cost are reported.
        const auto setup_begin=Clock::now();bank_marker(q);bank_marker(q);drain();
        const double marker_setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup_begin).count();
        std::array<double,5> empty_controls{};double empty_wall_ms=0;
        for (int i=0;i<5;++i) {
            const auto t=Clock::now();bank_marker(q);bank_marker(q);
            drain();empty_controls[size_t(i)]=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
            empty_wall_ms=std::max(empty_wall_ms,empty_controls[size_t(i)]);
        }
        for(int round=-1;round<rounds;++round) for(int order=0;order<2;++order) {
            const int arm=(process+round+2+order)%2;
            drain();pd::ExpertBankSlots slots(384);
            q.memset(cache.device_slot(0),0xa5,size_t(cache.bytes()));q.memset(status,0,4);drain();
            uint64_t uploads=0,hits=0,consumers=0;uint32_t observed=~uint32_t(0);
            const auto copies_before=staging.copies(),bytes_before=staging.copied_bytes();
            double wall=0,cpu=0,queue_ms=0;
            const auto observer_whole_begin=Clock::now();
            { // Observer setup,planning,queries,formatting and destruction are charged below.
                std::unique_ptr<sb::command_profile::Trace> trace;
                if(attribution) {
                    std::cerr<<"staging attribution sample: process="<<process<<" round="<<round
                        <<" arm="<<arm<<" payload="<<payload<<'\n';
                    trace.reset(new sb::command_profile::Trace(&q,1));
                }
                auto frame=trace?trace->begin(0,0):sb::command_profile::Frame{};
                for(int pass=0;pass<2;++pass)for(int e=0;e<512;++e)frame.consider(e,pass*512+e,1);
                const auto marker=bank_marker(q);
                auto cpu_scope=std::make_unique<sb::cpu_wait_profile::Scope>(attribution);
                const auto begin=Clock::now();const double begin_cpu=thread_seconds();
                pd::DraftUploadGuard lease([&]{drain();return true;},
                    []{sb::cleanup_fail_stop("staging benchmark retirement");});lease.arm();
                for(int pass=0;pass<2;++pass)for(int e=0;e<512;++e) {
                    auto upload_tag=frame.tag(sb::command_profile::Kind::expert_upload,e,pass*512+e,1);
                    const auto ticket=pd::stage_private_expert_bank(slots,bank,
                        static_cast<const uint8_t*>(slab.pointer),slab_bytes,payload,e,&q,stride,arm?&staging:nullptr);
                    uploads+=ticket.upload;hits+=!ticket.upload;
                    const auto* blob=reinterpret_cast<const uint32_t*>(bank.device_slot(ticket.slot));
                    const size_t words=payload/4;
                    for(int job=0;job<2;++job) {
                        const size_t global=(words+255)/256*256;
                        auto reader_tag=frame.tag(sb::command_profile::Kind::gather,e,pass*512+e,1);
                        auto event=q.parallel_for(sycl::nd_range<1>(global,256),[=](sycl::nd_item<1> item){
                            const size_t i=item.get_global_linear_id();
                            const bool bad=i<words && blob[i]!=value(e,i,1);
                            const bool any=sycl::any_of_group(item.get_group(),bad);
                            if(item.get_local_linear_id()==0 && any)
                                sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(*status).fetch_or(1);
                        });sb::command_profile::observe(&q,event);++consumers;
                    }
                }
                {
                    sb::cpu_wait_profile::Host status_enqueue(sb::cpu_wait_profile::moe_status);
                    auto status_tag=frame.tag(sb::command_profile::Kind::status,-1,0,1);
                    auto event=q.memcpy(&observed,status,4);sb::command_profile::observe(&q,event);
                }
                const auto end=bank_marker(q);drain();lease.retire();
                wall=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
                cpu=(thread_seconds()-begin_cpu)*1e3;
                cpu_scope->complete(true);cpu_scope.reset();frame.after_checked_drain();
                const auto a=marker.get_profiling_info<sycl::info::event_profiling::command_end>();
                const auto b=end.get_profiling_info<sycl::info::event_profiling::command_start>();
                need(b>a && uploads==641 && hits==383 && consumers==2048 && observed==0,"timed production bank contract");
                need(staging.copies()-copies_before==(arm?uploads:0) &&
                    staging.copied_bytes()-bytes_before==(arm?uploads*payload:0),"timed staging dispatch counters");
                queue_ms=double(b-a)/1e6;
            } // Frame dies before Trace;its after_checked_drain already retired events.
            const double observer_whole_ms=std::chrono::duration<double,std::milli>(Clock::now()-observer_whole_begin).count();
            staging.check_guards();
            if(device_guard) {
                pd::DraftUploadGuard check_lease([&]{drain();return true;},
                    []{sb::cleanup_fail_stop("GPU padding validation retirement");});check_lease.arm();
                // Same complete padding coverage as the host check,without a
                // full-bank D2H transfer/allocation between measured batches.
                // Payload bytes are independently checked by every reader.
                bank_padding_guard(q,reinterpret_cast<const uint32_t*>(cache.device_slot(0)),
                                   stride,payload,status);
                q.memcpy(&observed,status,4);drain();
                check_lease.retire();
                need(observed==0,"GPU bank padding guard");
            } else {
                std::vector<uint8_t> check(size_t(cache.bytes()));
                // The destination is constructed BEFORE its checked lease so
                // every exceptional exit drains before vector destruction.
                pd::DraftUploadGuard check_lease([&]{drain();return true;},
                    []{sb::cleanup_fail_stop("D2H padding validation retirement");});check_lease.arm();
                q.memcpy(check.data(),cache.device_slot(0),check.size());drain();
                check_lease.retire();
                for(size_t slot=0;slot<384;++slot)for(size_t i=payload;i<stride;++i)
                    need(check[slot*stride+i]==0xa5,"timed bank padding guard");
            }
            for(int e=0;e<512;++e)need(cache.slot_of(0,e)<0,"timed private map leaked");
            const bool eligible=wall>=10 && queue_ms>=10 && empty_wall_ms<=.05*wall;
            std::cout<<std::setprecision(17)<<"{\"kind\":\"staging_bank\",\"process\":"<<process
                <<",\"custom_queue\":"<<(custom_queue?"true":"false")<<",\"round\":"<<round<<",\"order\":"<<order<<",\"arm\":"<<arm
                <<",\"attribution\":"<<(attribution?"true":"false")
                <<",\"device_guard\":"<<(device_guard?"true":"false")<<",\"observer_whole_wall_ms\":"<<observer_whole_ms
                <<",\"payload_bytes\":"<<payload<<",\"source_stride\":"<<stride<<",\"bank_slots\":384"
                <<",\"uploads\":"<<uploads<<",\"hits\":"<<hits<<",\"consumers\":"<<consumers
                <<",\"wall_ms\":"<<wall<<",\"thread_CPU_ms\":"<<cpu<<",\"queue_interval_ms\":"<<queue_ms
                <<",\"marker_setup_wall_ms\":"<<marker_setup_ms<<",\"empty_control_wall_ms\":[";
            for(size_t i=0;i<empty_controls.size();++i)std::cout<<(i?",":"")<<empty_controls[i];
            std::cout<<"],\"empty_control_max_wall_ms\":"<<empty_wall_ms<<",\"timer_p99_ns\":"<<clock_ns[989]
                <<",\"eligible\":"<<(eligible?"true":"false")
                <<",\"guards_and_every_consumer\":true,\"private_map_unchanged\":true,\"model_qualified\":false}\n";
        }
        for(int e=0;e<512;++e)for(size_t i=0;i<stride/4;++i)
            need(source[size_t(e)*(stride/4)+i]==value(e,i,1),"benchmark immutable source");
    }
}

int main(int argc, char** argv) try {
    if (argc==2 && std::string(argv[1])=="--copy-runs-size-smoke") {
        for (int cap : {1,4,16,64,512}) {
            std::cout<<"expert_bank_copy_size cap="<<cap<<" profiling=0\n";
            run_private(4,256,10,{256},256,false,true,false,true,cap>1,false,cap);
            run_private(384,256,512,{256},256,false,true,false,true,cap>1,false,cap);
            run_private(512,256,512,{256},256,false,true,false,true,cap>1,false,cap);
        }
        run_private(1,256,10,{256},256,false,true,false,true,true,false,4);
        run_private(8,256,10,{256,256},256,true,true,false,true,true,false,4);
        run_private_run_enqueue_failure();run_private_run_admissions();
        std::cout<<"expert_bank_copy_size_smoke PASS model=UNRUN\n";return 0;
    }
    if (argc==3 && (std::string(argv[1])=="--copy-runs-size" ||
                   std::string(argv[1])=="--copy-runs-size-profile")) {
        const std::string arg=argv[2];
        need(arg=="1" || arg=="4" || arg=="16" || arg=="64" || arg=="512","copy-size cap must be1,4,16,64or512");
        const int cap=std::stoi(arg);const bool profile=std::string(argv[1])=="--copy-runs-size-profile";
        std::cout<<"expert_bank_copy_size cap="<<cap<<" profiling="<<profile<<"\n";
        run_private(4,256,10,{256},256,false,true,false,true,cap>1,false,cap);
        auto runtime=sb::runtime_for();bank_marker(runtime->compute());runtime->wait();
        run_private(384,2662400,512,{2662400},2662400,false,true,false,true,cap>1,true,cap,profile);
        run_private(384,2662400,512,{1510400},1510400,false,true,false,true,cap>1,true,cap,profile);
        std::cout<<"expert_bank_copy_size PASS model=UNRUN\n";return 0;
    }
#if defined(STRATA_SYCL_EXPERT_UPLOAD_ATTRIBUTION)
    if(argc==2 && std::string(argv[1])=="--padding-guard-smoke") {
        bank_padding_guard_smoke();std::cout<<"expert_bank_fixture PASS model=UNRUN\n";return 0;
    }
    if(argc==3 && (std::string(argv[1])=="--staging-guard-copy-off" ||
                  std::string(argv[1])=="--staging-guard-device-off" ||
                  std::string(argv[1])=="--staging-guard-device-on")) {
        const int process=std::stoi(argv[2]);need(process>=0 && process<3,"guard process ordinal");
        const std::string mode=argv[1];
        staging_benchmark(process,true,mode=="--staging-guard-device-on",
                          mode!="--staging-guard-copy-off",10);
        std::cout<<"expert_bank_staging_benchmark PASS model=UNRUN\n";return 0;
    }
    if(argc==3 && (std::string(argv[1])=="--staging-attribution-off" ||
                  std::string(argv[1])=="--staging-attribution-on")) {
        const int process=std::stoi(argv[2]);need(process>=0 && process<3,"attribution process ordinal");
        staging_benchmark(process,true,std::string(argv[1])=="--staging-attribution-on");
        std::cout<<"expert_bank_staging_benchmark PASS model=UNRUN\n";return 0;
    }
#endif
    if (argc==3 && (std::string(argv[1])=="--staging-bench" || std::string(argv[1])=="--staging-custom-bench")) {
        const int process=std::stoi(argv[2]);need(process>=0 && process<3,"benchmark process ordinal");
        staging_benchmark(process,std::string(argv[1])=="--staging-custom-bench");
        std::cout<<"expert_bank_staging_benchmark PASS model=UNRUN\n";return 0;
    }
    need(argc == 2 && (std::string(argv[1]) == "--smoke" || std::string(argv[1]) == "--validate" ||
        std::string(argv[1]) == "--packed-smoke" || std::string(argv[1]) == "--packed-validate" ||
        std::string(argv[1]) == "--staging-smoke" || std::string(argv[1]) == "--staging-validate" ||
        std::string(argv[1]) == "--packed-custom-smoke" || std::string(argv[1]) == "--staging-custom-smoke" ||
        std::string(argv[1]) == "--staging-custom-validate"),
        "use --smoke|--validate|--packed-smoke|--packed-validate");
    if (std::string(argv[1]) == "--packed-custom-smoke") {
        run_private(4,512,10,{256,512,256},256,false,true,false,true);
        run_private(4,512,10,{256,512},256,true,false,false,true);
    } else if (std::string(argv[1]) == "--staging-custom-smoke") {
        run_private(4,512,10,{256,512,256},256,false,true,true,true);
        run_private(4,512,10,{256,512},256,true,false,true,true);
        staging_admissions();
    } else if (std::string(argv[1]) == "--staging-custom-validate") {
        run_private(384,512,512,{256,512,768,256},256,false,false,true,true);
        run_private(384,2662400,512,{1510400,2662400,2176000},2662400,false,true,true,true);
        run_private(384,512,512,{256,512},256,true,false,true,true);
    } else if (std::string(argv[1]) == "--staging-smoke") {
        run_private(4,512,10,{256,512,256},256,false,true,true);
        run_private(4,512,10,{256,512},256,true,false,true);
        run_private_enqueue_failure();
        staging_admissions();
    } else if (std::string(argv[1]) == "--staging-validate") {
        run_private(384,512,512,{256,512,768,256},256,false,false,true);
        run_private(384,2662400,512,{1510400,2662400,2176000},2662400,false,true,true);
        run_private(384,512,512,{256,512},256,true,false,true);
        run_private_enqueue_failure();
    } else if (std::string(argv[1]) == "--packed-smoke") {
        run_private(4,512,10,{256,512,256},256,false,true);
        run_private(4,512,10,{256,512},256,true);
        run_private_enqueue_failure();
    } else if (std::string(argv[1]) == "--packed-validate") {
        run_private(384,512,512,{256,512,768,256},256,false);
        run_private(384,2662400,512,{1510400,2662400,2176000},2662400,false,true);
        run_private(384,512,512,{256,512},256,true);
        run_private_enqueue_failure();
    } else
    if (std::string(argv[1]) == "--smoke") {
        run(4, 256, 6, 2, false);
        run(1, 256, 6, 1, false);
        run(4, 256, 6, 1, true);
        run(4, 512, 6, 2, false, 256);
        run(1, 512, 6, 1, false, 256);
        run(4, 512, 6, 1, true, 256);
    } else {
        run(383, 256, 512, 1, false);
        run(384, 256, 512, 2, false);
        run(512, 256, 512, 1, false);
        run(384, 2662400, 512, 1, false);
        run(384, 512, 512, 2, false, 256);
        run(384, 2662400, 512, 1, false, 1382400);
    }
    std::cout << "expert_bank_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "expert_bank_fixture failed: " << error.what() << '\n';
    return 1;
}
