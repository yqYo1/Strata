// Isolated, opt-in IQ4NL helper component service fixture. No model/adoption gate.
// Calls only the existing helpers; exact events come from their private receipt hook.
#if !defined(STRATA_SYCL_PREFILL_IQ4NL_DEQUANT) || !defined(STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT)
#error "IQ4NL service requires private dequant and event receipt qualification build"
#endif
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include "strata/kernels/iq_kernels.hpp"
#include "../prefill/iq4nl_dequant.hpp"
#include "../prefill/iq4nl_event_receipt.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using Arm = strata::kernels::detail::Iq4nlArm;
using Receipt = strata::kernels::detail::Iq4nlReceipt;
using Sink = strata::kernels::detail::Iq4nlReceiptSink;
using Scope = strata::kernels::detail::Iq4nlReceiptScope;
using Clock = std::chrono::steady_clock;
constexpr size_t n = 2560 * 640, inputs = 64, outputs = 2, batch_calls = 64;
constexpr size_t input_pad = 64, output_pad = 32, alignment = 64;
constexpr size_t packed_bytes = n / 32 * 18, input_bytes = packed_bytes + 2 * input_pad;
constexpr size_t output_words = n + 2 * output_pad, output_bytes = output_words * sizeof(uint16_t);
constexpr size_t usm_bytes = inputs * input_bytes + outputs * output_bytes;
// One canonical input, a full-cohort readback, oracle, initial output, two readbacks.
constexpr size_t host_payload_bytes = (1 + inputs) * input_bytes + 4 * output_bytes;
constexpr unsigned warm_passes = 2, service_blocks = 8;
constexpr std::array<std::array<unsigned,4>,4> orders{{{{0,1,2,3}},{{1,2,3,0}},{{2,3,0,1}},{{3,0,1,2}}}};
constexpr std::array<int,16> codes{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// Finite varied scales, including signed zero and binary16 subnormal boundaries.
// Products in this workload remain finite, unlike the separate overflow diagnostic.
constexpr std::array<uint16_t,14> scales{0,1,0x3ff,0x400,0x401,0x2000,0x3c00,
    0x8000,0x8001,0x83ff,0x8400,0x8401,0xa000,0xbc00};
static_assert(n == 1638400 && n / 256 == 6400 && packed_bytes == 921600);
static_assert(input_bytes == 921728 && output_words == 1638464 && usm_bytes < 96 * 1024 * 1024);
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void flush() { require(std::fflush(stdout) == 0, "stdout_flush"); }
uint64_t elapsed(Clock::time_point a, Clock::time_point b) {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
    require(value >= 0, "host_clock_order"); return uint64_t(value);
}
const char* arm_name(Arm arm) { return arm == Arm::generic ? "generic" : "private"; }
const char* regime_name(unsigned cell) { return cell < 2 ? "same_address" : "rotation64"; }
Arm cell_arm(unsigned cell) { return cell % 2 ? Arm::private_down : Arm::generic; }
// Copied independent scalar RNE oracle from prefill_iq4nl_parity.cpp. No SYCL half,
// ggml decode, lane/group arithmetic, or device kernel body participates in it.
float decode(uint16_t h) {
    const unsigned e=(h>>10)&31, m=h&1023;
    require(e!=31,"nonfinite_scale_forbidden");
    float v=e?std::ldexp(float(1024+m),int(e)-25):std::ldexp(float(m),-24);
    return (h&0x8000)?-v:v;
}
uint32_t rne_shift(uint32_t v,unsigned shift) {
    if(shift>31) return 0;
    const uint32_t base=v>>shift, rem=v&((uint32_t(1)<<shift)-1), half=uint32_t(1)<<(shift-1);
    return base+(rem>half || (rem==half && (base&1)));
}
uint16_t encode(float f) {
    const uint32_t u=std::bit_cast<uint32_t>(f), sign=(u>>16)&0x8000;
    const unsigned e=(u>>23)&255;const uint32_t m=u&0x7fffff;
    require(e!=255,"oracle_input_not_finite");
    if(e==0) return uint16_t(sign);
    const int he=int(e)-112;
    if(he>=31) return uint16_t(sign|0x7c00);
    if(he<=0) {
        if(he< -10) return uint16_t(sign);
        return uint16_t(sign|rne_shift(m|0x800000,unsigned(14-he)));
    }
    const uint32_t rounded=rne_shift(m,13);
    return uint16_t(sign|((unsigned(he)<<10)+rounded));
}
uint64_t hash_bytes(const void* data, size_t size) {
    auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ull;
    for (size_t i=0;i<size;++i) { h ^= bytes[i]; h *= 1099511628211ull; }
    return h;
}
struct Host {
    std::vector<uint8_t> input, restored;
    std::vector<uint16_t> oracle, initial;
    std::array<std::vector<uint16_t>,outputs> result;
    Host():input(input_bytes,0x6d),restored(inputs*input_bytes),
        oracle(output_words,0xa55a),initial(output_words,0xa55a) {
        for (auto& r:result) r.resize(output_words);
        for (size_t block=0;block<n/32;++block) {
            const size_t row=block/20, inrow=block%20;
            const uint16_t scale=scales[(block+3*row)%scales.size()];
            uint8_t* p=input.data()+input_pad+block*18;
            p[0]=uint8_t(scale);p[1]=uint8_t(scale>>8);
            for(size_t j=0;j<16;++j) {
                const unsigned low=unsigned((j+row+3*inrow)&15);
                const unsigned high=unsigned((15-j+5*row+7*inrow)&15);
                p[2+j]=uint8_t(low|(high<<4));
            }
            for(size_t j=0;j<32;++j) {
                const uint8_t packed=p[2+(j%16)];
                const unsigned code=j<16?(packed&15):(packed>>4);
                oracle[output_pad+block*32+j]=encode(decode(scale)*float(codes[code]));
            }
        }
    }
};
struct Views {
    sycl::queue* queue{};
    std::array<const uint8_t*,inputs> src{};
    std::array<uint16_t*,outputs> dst{};
};
struct Call {
    Arm arm{};
    unsigned input_index{}, output_slot{};
    sycl::queue* queue{};
    const void* src{};
    uint16_t* dst{};
    uint64_t scope_id{}, serial{};
};
struct Plan {
    std::array<Call,batch_calls> calls{};
    size_t count{};
    uint64_t scope_id{};
    bool replay{};
    unsigned cell{}, block{}, input_index{};
};
struct Sample { uint64_t submit{}, start{}, end{}, host_submit{}; };
constexpr size_t host_fixed_bytes = sizeof(Host)+sizeof(Views)+sizeof(Plan)+
    sizeof(std::array<Receipt,batch_calls>)+sizeof(std::array<Sample,batch_calls>);
static_assert(host_payload_bytes + host_fixed_bytes < 256 * 1024 * 1024);
void make_plan(Plan& p, const Views& v, uint64_t scope, unsigned cell, unsigned block,
               bool replay=false, unsigned index=0) {
    require(scope && cell < 4 && index < inputs, "plan_parameters");
    p.scope_id=scope;p.cell=cell;p.block=block;p.replay=replay;p.input_index=index;
    p.count=replay?2:batch_calls;
    for(size_t i=0;i<p.count;++i) {
        const unsigned source=replay?index:(cell<2?0:unsigned((i+block*9)%inputs));
        const unsigned slot=replay?unsigned((index+i)%outputs):unsigned(i%outputs);
        p.calls[i]={replay?(i?Arm::private_down:Arm::generic):cell_arm(cell),source,slot,
            v.queue,v.src[source],v.dst[slot],scope,uint64_t(i+1)};
    }
}
void validate_plan(const Plan& p, const Views& v, size_t capacity) {
    require(p.scope_id && p.cell<4 && p.input_index<inputs && p.count==(p.replay?2:batch_calls)
        && p.count<=capacity && capacity<=batch_calls && v.queue, "plan_preflight");
    std::array<bool,inputs> seen{};
    for(size_t i=0;i<p.count;++i) {
        const auto& c=p.calls[i];
        const unsigned source=p.replay?p.input_index:(p.cell<2?0:unsigned((i+p.block*9)%inputs));
        const unsigned slot=p.replay?unsigned((p.input_index+i)%outputs):unsigned(i%outputs);
        const Arm arm=p.replay?(i?Arm::private_down:Arm::generic):cell_arm(p.cell);
        require(c.arm==arm && c.input_index==source && c.output_slot==slot && c.queue==v.queue
            && c.src==v.src[source] && c.dst==v.dst[slot] && c.src && c.dst
            && c.scope_id==p.scope_id && c.serial==i+1, "plan_tuple");
        if(!p.replay && p.cell>=2) { require(!seen[source],"plan_rotation_duplicate");seen[source]=true; }
    }
}
void validate_metadata(const Plan& p,const std::array<Receipt,batch_calls>& receipts,
                       size_t count,uint64_t serial,bool invalid,bool need_events) {
    require(!invalid && count==p.count && serial==p.count,"receipt_accounting");
    for(size_t i=0;i<p.count;++i) {
        const auto& r=receipts[i];const auto& c=p.calls[i];
        require(r.arm==c.arm && r.type==20 && r.n==int64_t(n) && r.queue==c.queue
            && r.src==c.src && r.dst==c.dst && r.scope_id==c.scope_id && r.serial==c.serial,
            "receipt_tuple");
        if(need_events) require(r.event.has_value(),"receipt_event_missing");
    }
    for(size_t i=p.count;i<receipts.size();++i) require(!receipts[i].event,"receipt_excess_event");
}
template<class F> void expect_reject(F f,const char* reason) {
    bool rejected=false;try {f();} catch(const std::runtime_error&) {rejected=true;}
    require(rejected,reason);
}
void host_checks(Host& h) {
    require(std::endian::native==std::endian::little,"little_endian_required");
    const std::array<uint16_t,14> halves{0,1,0x3ff,0x400,0x401,0x3c00,0x7bff,
        0x8000,0x8001,0x83ff,0x8400,0x8401,0xbc00,0xfbff};
    for(auto x:halves) require(encode(decode(x))==x,"half_roundtrip");
    require(encode(1.00048828125f)==0x3c00,"even_normal_tie");
    require(encode(1.00146484375f)==0x3c02,"odd_normal_tie");
    require(encode(std::ldexp(1.f,-25))==0,"zero_subnormal_tie");
    require(encode(std::ldexp(3.f,-25))==2,"odd_subnormal_tie");
    require(encode(65520.f)==0x7c00 && encode(-0.f)==0x8000,"overflow_and_negative_zero");
    std::array<bool,16> code_seen{};std::array<bool,scales.size()> scale_seen{};
    for(size_t block=0;block<n/32;++block) {
        const uint8_t* p=h.input.data()+input_pad+block*18;
        const uint16_t scale=uint16_t(p[0])|(uint16_t(p[1])<<8);
        const auto it=std::find(scales.begin(),scales.end(),scale);
        require(it!=scales.end(),"packed_scale");scale_seen[size_t(it-scales.begin())]=true;
        for(size_t j=0;j<32;++j) {
            const unsigned code=j<16?(p[2+j]&15):(p[2+j-16]>>4);
            code_seen[code]=true;
            const uint16_t expected=encode(decode(scale)*float(codes[code]));
            require((expected&0x7c00)!=0x7c00 && h.oracle[output_pad+block*32+j]==expected,"packed_oracle");
        }
    }
    for(bool hit:code_seen) require(hit,"packed_code_coverage");
    for(bool hit:scale_seen) require(hit,"packed_scale_coverage");
    for(size_t i=0;i<input_pad;++i)
        require(h.input[i]==0x6d && h.input[input_bytes-1-i]==0x6d,"host_input_guards");
    for(size_t i=0;i<output_pad;++i)
        require(h.oracle[i]==0xa55a && h.oracle[output_words-1-i]==0xa55a,"host_output_guards");
    // Synthetic pointers are only compared. No queue object or event is constructed.
    Views v;v.queue=reinterpret_cast<sycl::queue*>(uintptr_t(0x30000000));
    for(size_t i=0;i<inputs;++i) v.src[i]=reinterpret_cast<const uint8_t*>(uintptr_t(0x10000000)+i*0x100000);
    for(size_t i=0;i<outputs;++i) v.dst[i]=reinterpret_cast<uint16_t*>(uintptr_t(0x20000000)+i*0x400000);
    Plan p;std::array<Receipt,batch_calls> r{};
    for(unsigned b=0;b<service_blocks;++b) for(unsigned pos=0;pos<4;++pos) {
        make_plan(p,v,1+b*4+pos,orders[b%4][pos],b);validate_plan(p,v,batch_calls);
    }
    for(unsigned i=0;i<inputs;++i) {make_plan(p,v,1+i,0,0,true,i);validate_plan(p,v,batch_calls);}
    make_plan(p,v,99,2,3);validate_plan(p,v,batch_calls);
    expect_reject([&]{validate_plan(p,v,1);},"capacity_rejection_missing");
    for(size_t i=0;i<p.count;++i) {
        const auto& c=p.calls[i];auto& x=r[i];
        x.arm=c.arm;x.type=20;x.n=n;x.queue=c.queue;x.src=c.src;x.dst=c.dst;x.scope_id=c.scope_id;x.serial=c.serial;
    }
    validate_metadata(p,r,64,64,false,false);
    auto corrupt=[&](auto mutate) { const Receipt saved=r[1];mutate(r[1]);
        expect_reject([&]{validate_metadata(p,r,64,64,false,false);},"metadata_corruption_accepted");r[1]=saved; };
    corrupt([](Receipt& x){x.serial=1;});corrupt([](Receipt& x){x.serial=3;});
    corrupt([](Receipt& x){x.arm=Arm::private_down;});corrupt([](Receipt& x){x.type=21;});
    corrupt([](Receipt& x){x.n-=256;});corrupt([](Receipt& x){x.queue=nullptr;});
    corrupt([](Receipt& x){x.src=nullptr;});corrupt([](Receipt& x){x.dst=nullptr;});
    corrupt([](Receipt& x){++x.scope_id;});
    expect_reject([&]{validate_metadata(p,r,0,0,false,false);},"zero_count_accepted");
    expect_reject([&]{validate_metadata(p,r,65,65,false,false);},"excess_count_accepted");
    expect_reject([&]{validate_metadata(p,r,63,64,false,false);},"dropped_count_accepted");
    expect_reject([&]{validate_metadata(p,r,64,64,true,false);},"sticky_invalid_accepted");
    // Empty optional slots remain empty: these are unsubmitted scope/argument checks.
    r={};Sink outer{r.data(),r.size(),0,100};
    require(!strata::kernels::detail::iq4nl_receipt_sink(),"host_TLS_not_empty");
    {
        Scope scope(outer);Sink inner{r.data(),r.size(),0,101};bool nested=false;
        try { Scope denied(inner); } catch(const std::logic_error&) {nested=true;}
        require(nested && strata::kernels::detail::iq4nl_receipt_sink()==&outer && !outer.count
            && !outer.serial && !outer.invalid,"nested_scope_state");
        int marker=0;auto* dst=reinterpret_cast<uint16_t*>(&marker);
        struct Bad {int type;int64_t count;bool nullsrc,nulldst;};
        const std::array<Bad,7> bad{{{42,256,false,false},{20,0,false,false},{20,-256,false,false},
            {20,257,false,false},{20,256,true,false},{20,256,false,true},
            {20,(int64_t(std::numeric_limits<unsigned>::max())+1)*256,false,false}}};
        for(const auto& b:bad) {
            bool rejected=false;
            try { strata::kernels::iq_dequant_f16_prefill_iq4nl(b.type,b.nullsrc?nullptr:&marker,
                b.count,b.nulldst?nullptr:dst,nullptr); }
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected && !outer.count && !outer.serial && !outer.invalid,"invalid_private_receipt");
        }
        require(marker==0,"host_marker_modified");
    }
    require(!strata::kernels::detail::iq4nl_receipt_sink(),"host_TLS_not_restored");
    for(const auto& x:r) require(!x.event,"host_event_constructed");
    std::printf("HOST_PASS,packing_blocks=%zu,oracle_active_words=%zu,roundtrips=14,boundaries=6,private_rejections=7,plan_batches=32,replay_plans=64,metadata_adversaries=13,nested_rejection=1,no_queue=1,no_device=1,generic_invalid=source_only\n",n/32,n);
    flush();
}
struct Async {
    std::atomic<unsigned> errors{0};
    // Written only by the submitting thread; the handler touches errors only.
    size_t completed_drains=0, successful_wrappers=0;
};
[[noreturn]] void fail_stop(const char* stage,size_t attempted,size_t successful) noexcept {
    std::fprintf(stderr,"FAIL_STOP,stage=%s,attempted=%zu,successful_returns=%zu,completion=unknown,no_USM_or_event_unwind=1\n",stage,attempted,successful);
    std::fflush(stderr);std::_Exit(2);
}
void progress(const char* phase,uint64_t id) {
    // Announce the submitted block and its planned drain before the timed window.
    std::printf("PROGRESS,phase=%s,id=%llu,planned_completion_boundary=wait_and_throw\n",phase,(unsigned long long)id);flush();
}
// Only exceptions after a normal wait return may unwind. The async handler never
// throws; its sticky state invalidates a completed stage outside this boundary.
template<class F> void transfer_stage(sycl::queue& q,Async& a,const char* name,uint64_t id,F f) {
    require(!a.errors.load(),"pre_submit_sticky_async");progress(name,id);
    try { f();q.wait_and_throw();++a.completed_drains; } catch(...) { fail_stop(name,0,0); }
    require(!a.errors.load(),"post_completion_sticky_async");
}
void identity(sycl::queue& q) {
    require(q.get_backend()==sycl::backend::ext_oneapi_level_zero && q.is_in_order()
        && q.has_property<sycl::property::queue::enable_profiling>(),"queue_admission");
    const auto selected=q.get_device();
    require(selected.has(sycl::aspect::fp16) && selected.has(sycl::aspect::queue_profiling)
        && selected.has(sycl::aspect::usm_device_allocations),"device_capability_admission");
    const auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(selected);
    ze_device_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    require(zeDeviceGetProperties(dev,&p)==ZE_RESULT_SUCCESS,"device_identity_query");
    require(p.vendorId==0x8086 && p.deviceId==0xe20c && !(p.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),"B570_root_required");
    ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    require(zeDevicePciGetPropertiesExt(dev,&pci)==ZE_RESULT_SUCCESS,"PCI_query");
    require(pci.address.domain==0 && pci.address.bus==5 && pci.address.device==0 && pci.address.function==0,"PCI_0000_05_00_0_required");
    std::printf("QUEUE,queue=%p,backend=level_zero,in_order=1,enable_profiling=1,queue_profiling=1,fp16=1,usm_device=1,vendor=8086,device=e20c,pci=0000:05:00.0,root=1,resolution_ns=%zu\n",
        static_cast<void*>(&q),size_t(selected.get_info<sycl::info::device::profiling_timer_resolution>()));
    auto flags=selected.get_info<sycl::info::device::half_fp_config>();
    std::sort(flags.begin(),flags.end(),[](auto a,auto b){return unsigned(a)<unsigned(b);});
    std::printf("FP16_CONFIG,advertised_capabilities_only=1");
    for(auto flag:flags) std::printf(",flag=%u",unsigned(flag));
    std::puts("");flush();
}
struct Buffers {
    sycl::queue& queue;
    std::array<uint8_t*,inputs> input{};
    std::array<uint16_t*,outputs> output{};
    explicit Buffers(sycl::queue& q):queue(q) {}
    void release() noexcept {
        // A free exception is terminal too: do not retry an ambiguous free.
        try {
            for(auto& p:input) if(p) {auto* allocation=p;p=nullptr;sycl::free(allocation,queue);}
            for(auto& p:output) if(p) {auto* allocation=p;p=nullptr;sycl::free(allocation,queue);}
        } catch(...) {
            std::fprintf(stderr,"FAIL_CLEANUP,USM_free_failed=1,no_retry=1\n");std::fflush(stderr);std::_Exit(2);
        }
    }
    ~Buffers() noexcept {release();}
    Buffers(const Buffers&)=delete;Buffers& operator=(const Buffers&)=delete;
};
Views allocate(Buffers& b,const Host& h) {
    Views v;v.queue=&b.queue;
    std::array<uintptr_t,inputs+outputs> bases{};
    std::array<size_t,inputs+outputs> sizes{};
    for(size_t i=0;i<inputs;++i) {
        b.input[i]=static_cast<uint8_t*>(sycl::aligned_alloc_device(alignment,input_bytes,b.queue));
        require(b.input[i],"input_USM_allocation");v.src[i]=b.input[i]+input_pad;
        bases[i]=reinterpret_cast<uintptr_t>(b.input[i]);sizes[i]=input_bytes;
    }
    for(size_t i=0;i<outputs;++i) {
        b.output[i]=static_cast<uint16_t*>(sycl::aligned_alloc_device(alignment,output_bytes,b.queue));
        require(b.output[i],"output_USM_allocation");v.dst[i]=b.output[i]+output_pad;
        bases[inputs+i]=reinterpret_cast<uintptr_t>(b.output[i]);sizes[inputs+i]=output_bytes;
    }
    for(size_t i=0;i<bases.size();++i) {
        require(bases[i]%alignment==0 && bases[i]<=std::numeric_limits<uintptr_t>::max()-sizes[i],"USM_alignment_or_range");
        for(size_t j=0;j<i;++j)
            require(bases[i]+sizes[i]<=bases[j] || bases[j]+sizes[j]<=bases[i],"USM_overlap");
    }
    const uint64_t input_hash=hash_bytes(h.input.data(),h.input.size());
    for(size_t i=0;i<inputs;++i)
        std::printf("INPUT,index=%zu,allocation=device,queue=%p,base=%p,view=%p,total_bytes=%zu,packed_bytes=%zu,guard_each_bytes=%zu,alignment=%zu,fnv1a64_full=%016llx\n",
            i,static_cast<void*>(&b.queue),static_cast<void*>(b.input[i]),static_cast<const void*>(v.src[i]),input_bytes,packed_bytes,input_pad,alignment,
            (unsigned long long)input_hash);
    for(size_t i=0;i<outputs;++i)
        std::printf("OUTPUT,slot=%zu,allocation=device,queue=%p,base=%p,view=%p,total_bytes=%zu,active_words=%zu,guard_each_words=%zu,alignment=%zu\n",
            i,static_cast<void*>(&b.queue),static_cast<void*>(b.output[i]),static_cast<void*>(v.dst[i]),output_bytes,n,output_pad,alignment);
    std::puts("ALLOCATION_RESULT,input_allocations=64,output_allocations=2,distinct=1,nonoverlap=1,view_alignment=64");flush();
    return v;
}
void invoke(const Call& c) {
    if(c.arm==Arm::generic) strata::kernels::iq_dequant_f16(20,c.src,int64_t(n),c.dst,c.queue);
    else strata::kernels::iq_dequant_f16_prefill_iq4nl(20,c.src,int64_t(n),c.dst,c.queue);
}
void readback(sycl::queue& q,Async& a,Buffers& b,Host& h,uint64_t id,bool all,unsigned index) {
    transfer_stage(q,a,"readback",id,[&] {
        for(size_t slot=0;slot<outputs;++slot) q.memcpy(h.result[slot].data(),b.output[slot],output_bytes);
        if(all) for(size_t i=0;i<inputs;++i) q.memcpy(h.restored.data()+i*input_bytes,b.input[i],input_bytes);
        else q.memcpy(h.restored.data()+index*input_bytes,b.input[index],input_bytes);
    });
    for(size_t slot=0;slot<outputs;++slot) require(h.result[slot]==h.oracle,"last_occupant_oracle_or_output_guards");
    const size_t begin=all?0:index, end=all?inputs:index+1;
    for(size_t i=begin;i<end;++i)
        require(std::memcmp(h.restored.data()+i*input_bytes,h.input.data(),input_bytes)==0,"input_immutability_or_guards");
}
void batch(sycl::queue& q,Async& a,Buffers& b,Host& h,const Views& v,Plan& p,
           std::array<Receipt,batch_calls>& receipts,std::array<Sample,batch_calls>& samples,
           const char* phase,bool timed,unsigned position,int predecessor) {
    validate_plan(p,v,receipts.size());require(!a.errors.load(),"pre_batch_sticky_async");
    // Scope setup and all plan checks happen before the possible-accepted-work boundary.
    Sink sink{receipts.data(),receipts.size(),0,p.scope_id};Scope scope(sink);
    progress(phase,p.scope_id);
    size_t attempted=0,successful=0,queried=0;bool raw_calls_logged=false;
    Clock::time_point begin,submitted,completed;
    try {
        begin=Clock::now();
        for(size_t i=0;i<p.count;++i) {
            ++attempted;const auto before=Clock::now();invoke(p.calls[i]);const auto after=Clock::now();
            ++successful;++a.successful_wrappers;
            // No throwing validator, allocation, logging, copy, profile query or wait here.
            samples[i].host_submit=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count());
        }
        submitted=Clock::now();q.wait_and_throw();completed=Clock::now();++a.completed_drains;
    } catch(...) { fail_stop(phase,attempted,successful); }
    // Completion is known even if retain() lost an event or the handler saw an error.
    try {
        require(!a.errors.load(),"post_batch_sticky_async");
        require(attempted==p.count && successful==p.count,"wrapper_return_accounting");
        validate_metadata(p,receipts,sink.count,sink.serial,sink.invalid,true);
        for(size_t i=0;i<p.count;++i) {
            for(size_t j=0;j<i;++j) require(*receipts[i].event!=*receipts[j].event,"duplicate_event_identity");
            auto& s=samples[i];const auto& e=*receipts[i].event;
            s.submit=e.get_profiling_info<sycl::info::event_profiling::command_submit>();
            s.start=e.get_profiling_info<sycl::info::event_profiling::command_start>();
            s.end=e.get_profiling_info<sycl::info::event_profiling::command_end>();++queried;
            require(s.submit<=s.start && s.start<=s.end,"event_timestamp_order");
            if(i) require(samples[i-1].end<=s.start,"in_order_event_order");
        }
        for(size_t i=0;i<p.count;++i) {
            const auto& c=p.calls[i];const auto& s=samples[i];
            std::printf("CALL,phase=%s,timed=%u,batch=%llu,block=%u,position=%u,cell=%u,predecessor=%d,scope=%llu,serial=%llu,arm=%s,regime=%s,input_index=%u,output_slot=%u,queue=%p,src=%p,dst=%p,type=20,n=%zu,submit=%llu,start=%llu,end=%llu,service_ns=%llu,queue_delay_ns=%llu,host_submit_ns=%llu,metadata_match=1,event_distinct=1\n",
                phase,unsigned(timed),(unsigned long long)p.scope_id,p.block,position,p.cell,predecessor,
                (unsigned long long)c.scope_id,(unsigned long long)c.serial,arm_name(c.arm),p.replay?"oracle_replay":regime_name(p.cell),
                c.input_index,c.output_slot,static_cast<void*>(c.queue),c.src,static_cast<void*>(c.dst),n,
                (unsigned long long)s.submit,(unsigned long long)s.start,(unsigned long long)s.end,
                (unsigned long long)(s.end-s.start),(unsigned long long)(s.start-s.submit),(unsigned long long)s.host_submit);
        }
        // Keep queried samples even if an untimed numerical check rejects this batch.
        raw_calls_logged=true;
        flush();
        // Both slots survive only as their last occupants. Full pre/post replay is separate.
        readback(q,a,b,h,p.scope_id,!p.replay,p.input_index);
        std::array<unsigned,outputs> last_input{};std::array<Arm,outputs> last_arm{};
        for(size_t i=0;i<p.count;++i) { const auto& c=p.calls[i];last_input[c.output_slot]=c.input_index;last_arm[c.output_slot]=c.arm; }
        std::printf("BATCH,phase=%s,timed=%u,batch=%llu,block=%u,position=%u,cell=%u,predecessor=%d,planned=%zu,attempted=%zu,successful_returns=%zu,receipts=%zu,queried=%zu,dq_waits=1,readback_waits=1,host_submit_window_ns=%llu,host_wait_ns=%llu,host_wall_ns=%llu,completion=known,async_errors=0,receipt_invalid=0,oracle_last_occupants_only=1,last_slot0_input=%u,last_slot0_arm=%s,last_slot1_input=%u,last_slot1_arm=%s,output_words_checked=%zu,input_bytes_checked=%zu,all_guards_checked=1,input_immutable=1,result=PASS\n",
            phase,unsigned(timed),(unsigned long long)p.scope_id,p.block,position,p.cell,predecessor,p.count,attempted,successful,sink.count,queried,
            (unsigned long long)elapsed(begin,submitted),(unsigned long long)elapsed(submitted,completed),(unsigned long long)elapsed(begin,completed),
            last_input[0],arm_name(last_arm[0]),last_input[1],arm_name(last_arm[1]),outputs*output_words,
            (p.replay?1:inputs)*input_bytes);flush();
        // All queries/readbacks are complete. No optional event survives into the next batch.
        for(auto& r:receipts) r.event.reset();
    } catch(...) {
        // Preserve complete raw triples read before a later query/validation error.
        // No service subtraction or validity claim is made for these rejected rows.
        if(!raw_calls_logged) for(size_t i=0;i<queried;++i)
            std::printf("QUERY_PARTIAL,phase=%s,batch=%llu,serial=%zu,submit=%llu,start=%llu,end=%llu,timing_valid=0\n",
                phase,(unsigned long long)p.scope_id,i+1,(unsigned long long)samples[i].submit,
                (unsigned long long)samples[i].start,(unsigned long long)samples[i].end);
        std::printf("BATCH_REJECT,phase=%s,batch=%llu,planned=%zu,attempted=%zu,successful_returns=%zu,receipts=%zu,queried=%zu,completion=known,receipt_invalid=%u,async_errors=%u,reason=post_completion_check_or_query_failure\n",
            phase,(unsigned long long)p.scope_id,p.count,attempted,successful,sink.count,queried,unsigned(sink.invalid),a.errors.load());
        std::fflush(stdout);throw;
    }
}
void replay(sycl::queue& q,Async& a,Buffers& b,Host& h,const Views& v,Plan& p,
            std::array<Receipt,batch_calls>& receipts,std::array<Sample,batch_calls>& samples,
            uint64_t& scope,const char* phase) {
    std::printf("REPLAY_BEGIN,phase=%s,inputs=64,arms=2,calls=128,active_words_per_arm=%zu,timing_samples=0\n",phase,inputs*n);flush();
    for(unsigned i=0;i<inputs;++i) {
        make_plan(p,v,++scope,0,i,true,i);batch(q,a,b,h,v,p,receipts,samples,phase,false,0,-1);
    }
    // Close replay with a simultaneous full-cohort scan, including earlier views.
    readback(q,a,b,h,scope,true,0);
    std::printf("REPLAY_END,phase=%s,calls=128,oracle=full_independent_RNE,input_immutable=1,full_guards=1,both_slots_exercised=1,closing_full_cohort_scan=1,closing_readback_waits=1\n",phase);flush();
}
void warm(sycl::queue& q,Async& a,const Views& v) {
    require(!strata::kernels::detail::iq4nl_receipt_sink() && !a.errors.load(),"warm_preflight");
    progress("warm_two_passes",0);size_t attempted=0,successful=0;
    try {
        for(unsigned pass=0;pass<warm_passes;++pass) for(unsigned index=0;index<inputs;++index)
            for(unsigned arm=0;arm<2;++arm) {
                const unsigned slot=unsigned(attempted%outputs);++attempted;
                invoke({arm?Arm::private_down:Arm::generic,index,slot,v.queue,v.src[index],v.dst[slot],0,0});
                ++successful;++a.successful_wrappers;
            }
        q.wait_and_throw();++a.completed_drains;
    } catch(...) {fail_stop("warm",attempted,successful);}
    require(!a.errors.load() && attempted==warm_passes*inputs*2 && successful==attempted,"warm_completed_check");
    std::puts("WARM,passes=2,inputs_per_arm_per_pass=64,arms=2,calls=256,dq_waits=1,receipt_scope=inactive,timing_samples=0,completion=known,async_errors=0");flush();
}
void run(sycl::queue& q,Async& a,Host& h,bool service) {
    // All host metadata and optional event storage exist before the first H2D submit.
    Plan plan;std::array<Receipt,batch_calls> receipts{};std::array<Sample,batch_calls> samples{};
    Buffers b(q);const Views v=allocate(b,h);
    transfer_stage(q,a,"initial_upload",0,[&] {
        for(auto* p:b.input) q.memcpy(p,h.input.data(),input_bytes);
        for(auto* p:b.output) q.memcpy(p,h.initial.data(),output_bytes);
    });
    uint64_t scope=0;replay(q,a,b,h,v,plan,receipts,samples,scope,"prephase");
    warm(q,a,v);
    const unsigned blocks=service?service_blocks:1;int predecessor=-1;
    for(unsigned block=0;block<blocks;++block) for(unsigned position=0;position<4;++position) {
        const unsigned cell=orders[block%4][position];
        make_plan(plan,v,++scope,cell,block);
        batch(q,a,b,h,v,plan,receipts,samples,service?"service":"qualification_cell",service,position,predecessor);
        predecessor=int(cell);
    }
    replay(q,a,b,h,v,plan,receipts,samples,scope,"final_replay");
    require(!a.errors.load() && !strata::kernels::detail::iq4nl_receipt_sink(),"final_completed_state");
    for(const auto& r:receipts) require(!r.event,"final_event_not_released");
    const size_t expected_drains=260+blocks*8, expected_wrappers=512+blocks*4*batch_calls;
    require(a.completed_drains==expected_drains && a.successful_wrappers==expected_wrappers,"final_fixed_counts");
    std::printf("FINAL_COUNTS,completed_queue_waits=%zu,expected_queue_waits=%zu,successful_wrappers=%zu,expected_wrappers=%zu,upload_waits=1,warm_waits=1,replay_DQ_waits=128,replay_readback_waits=128,replay_closing_waits=2,cell_DQ_waits=%u,cell_readback_waits=%u,result=PASS\n",
        a.completed_drains,expected_drains,a.successful_wrappers,expected_wrappers,blocks*4,blocks*4);flush();
    // Last readback drain established completion; explicit free happens before queue teardown.
    progress("USM_release",scope);b.release();
    std::printf("USM_RELEASED,allocations=66,bytes=%zu,completion=known,events_released=1\n",usm_bytes);flush();
}
}
int main(int argc,char** argv) {
    try {
        // No default GPU action. Reject malformed CLI before host checks or queue creation.
        require(argc==2,"usage_prefill_iq4nl_service_--host-only_or_--qualify_or_--service_no_default");
        const bool host_only=!std::strcmp(argv[1],"--host-only");
        const bool qualify=!std::strcmp(argv[1],"--qualify");
        const bool service=!std::strcmp(argv[1],"--service");
        require(host_only||qualify||service,"invalid_mode_no_queue_created");
        std::puts("SCHEMA,version=1,format=CSV_key_value,device_clock=event_backend_ns,host_clock=steady_clock_ns,event_source=existing_wrapper_exact_submit,zero_service=valid,synthetic_component=1,model=0,adoption=0,async_errors=sticky_handler_flag");
        std::puts("SOURCE,base=9e54ed18c0e374289293838b026e3e69b1c1fe70,receipt_header_bytes=2756,receipt_header_sha256=5a194e5ad480ae36c63e164c1ae0de51b828919a7aade112ddd29e303c3e0ef0,iq_kernel_bytes=269005,iq_kernel_sha256=cc5080af0c9e341a69edc423c272726922c95985aa7d3125cd1b344a06ad2b12,pins=reference_requires_owner_verification");
        std::printf("CONFIG,mode=%s,type=20,rows=2560,cols=640,n=%zu,groups=6400,local_size=32,input_count=64,output_slots=2,calls_per_cell=64,cells=4,blocks=%u,warm_passes=2,warm_calls=256,prephase_calls=128,final_replay_calls=128,planned_cell_calls=%u,receipt_capacity=64,rotation=(i+block*9)%%64,USM_bytes=%zu,USM_budget_bytes=%zu,host_payload_bytes=%zu,host_fixed_accounted_bytes=%zu,host_budget_bytes=%zu,host_allocator_overhead=implementation_defined,performance_inference=external,default_mode=reject\n",
            argv[1]+2,n,host_only?0u:(service?8u:1u),host_only?0u:(service?2048u:256u),usm_bytes,size_t(96*1024*1024),
            host_payload_bytes,host_fixed_bytes,size_t(256*1024*1024));
        for(unsigned block=0;block<(host_only?8u:(service?8u:1u));++block)
            std::printf("ORDER,block=%u,cell0=%u,cell1=%u,cell2=%u,cell3=%u\n",block,orders[block%4][0],orders[block%4][1],orders[block%4][2],orders[block%4][3]);
        flush();Host h;host_checks(h);
        std::printf("INPUT_PATTERN,packed_bytes=%zu,full_bytes=%zu,fnv1a64_full=%016llx,oracle_fnv1a64_full=%016llx,scale_count=14,code_count=16,all_products_finite=1,identical_cohort_bytes=1\n",
            packed_bytes,input_bytes,(unsigned long long)hash_bytes(h.input.data(),h.input.size()),
            (unsigned long long)hash_bytes(h.oracle.data(),output_bytes));flush();
        if(host_only) {std::puts("PASS,mode=host-only,no_queue=1,no_device=1,synthetic_component=1,model=0,adoption=0");flush();return 0;}
        Async async;
        {
            sycl::queue q(sycl::gpu_selector_v,[&](sycl::exception_list errors) noexcept {
                // Bounded sticky flag, not an exception count; no iteration, CAS loop,
                // allocation, string building, logging or throw from the handler.
                (void)errors;async.errors.store(1,std::memory_order_relaxed);
            },sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
            identity(q);require(!async.errors.load(),"admission_sticky_async");run(q,async,h,service);
        } // All USM and retained events freed first; queue teardown precedes terminal PASS.
        require(!async.errors.load(),"queue_teardown_sticky_async");
        std::printf("PASS,mode=%s,cell_batches=%u,cell_calls=%u,replay_calls=256,warm_calls=256,completed_queue_waits=%zu,successful_wrappers=%zu,USM_freed=1,queue_teardown=1,synthetic_component=1,model=0,adoption=0\n",
            service?"service":"qualify",service?32u:4u,service?2048u:256u,async.completed_drains,async.successful_wrappers);flush();return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL,%s\n",e.what());std::fflush(stderr);return 1;
    } catch(...) {std::fprintf(stderr,"FAIL,unknown_host_exception\n");std::fflush(stderr);return 1;}
}
