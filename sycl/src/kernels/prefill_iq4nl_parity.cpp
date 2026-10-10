// Synthetic private-prefill IQ4_NL exact-bit discriminator; no timing/model claims.
// Scalar mapping/table provenance: R163, ggml-common.h at reviewed 609a270d.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include "strata/kernels/iq_kernels.hpp"
#include "../prefill/iq4nl_dequant.hpp"
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
#include "../prefill/iq4nl_event_receipt.hpp"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
constexpr std::array<int,16> codes{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// Explicit IEEE binary16 decode: all finite inputs are exact binary32 values.
float decode(uint16_t h) {
    const unsigned e=(h>>10)&31, m=h&1023;
    require(e!=31,"nonfinite scale forbidden");
    float v=e?std::ldexp(float(1024+m),int(e)-25):std::ldexp(float(m),-24);
    return (h&0x8000)?-v:v;
}
uint32_t rne_shift(uint32_t v,unsigned shift) {
    if(shift>31) return 0;
    const uint32_t base=v>>shift, rem=v&((uint32_t(1)<<shift)-1), half=uint32_t(1)<<(shift-1);
    return base+(rem>half || (rem==half && (base&1)));
}
// Independent integer FP32 -> FP16 round-to-nearest-even; no SYCL half/ggml decoder.
uint16_t encode(float f) {
    const uint32_t u=std::bit_cast<uint32_t>(f), sign=(u>>16)&0x8000;
    const unsigned e=(u>>23)&255;const uint32_t m=u&0x7fffff;
    require(e!=255,"oracle input not finite");
    if(e==0) return uint16_t(sign); // every FP32 subnormal is below half halfway
    const int he=int(e)-112;
    if(he>=31) return uint16_t(sign|0x7c00);
    if(he<=0) {
        if(he< -10) return uint16_t(sign);
        return uint16_t(sign|rne_shift(m|0x800000,unsigned(14-he)));
    }
    const uint32_t rounded=rne_shift(m,13);
    const uint32_t bits=(unsigned(he)<<10)+rounded;
    return uint16_t(sign|bits); // mantissa carry includes exponent/overflow
}
void host_checks() {
    require(std::endian::native==std::endian::little,"little-endian packed input required");
    const std::array<uint16_t,14> halves{0,1,0x3ff,0x400,0x401,0x3c00,0x7bff,0x8000,0x8001,0x83ff,0x8400,0x8401,0xbc00,0xfbff};
    for(auto h:halves) require(encode(decode(h))==h,"half roundtrip");
    require(encode(1.00048828125f)==0x3c00,"even normal tie");
    require(encode(1.00146484375f)==0x3c02,"odd normal tie");
    require(encode(std::ldexp(1.f,-25))==0,"zero subnormal tie");
    require(encode(std::ldexp(3.f,-25))==2,"odd subnormal tie");
    require(encode(65520.f)==0x7c00,"overflow tie");
    require(encode(-0.f)==0x8000,"negative zero");
    int marker=0;auto* src=&marker;auto* dst=reinterpret_cast<uint16_t*>(&marker);
    struct Bad {int type;int64_t n;bool nullsrc,nulldst;};
    const std::array<Bad,7> bad{{{42,256,false,false},{20,0,false,false},{20,-256,false,false},{20,257,false,false},
        {20,256,true,false},{20,256,false,true},{20,(int64_t(std::numeric_limits<unsigned>::max())+1)*256,false,false}}};
    for(const auto& b:bad) {
        bool rejected=false;try {strata::kernels::iq_dequant_f16_prefill_iq4nl(b.type,b.nullsrc?nullptr:src,b.n,b.nulldst?nullptr:dst,nullptr);}
        catch(const std::invalid_argument&) {rejected=true;} require(rejected,"private invalid admission accepted");
    }
    require(marker==0,"host markers modified");
    std::puts("HOST_PASS,14_roundtrips,6_boundaries,7_private_rejections,no_queue_no_GPU");
}
void identity(sycl::queue& q) {
    require(q.get_backend()==sycl::backend::ext_oneapi_level_zero && q.is_in_order(),"LevelZero in-order required");
    auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_device_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    require(zeDeviceGetProperties(dev,&p)==ZE_RESULT_SUCCESS,"device identity query");
    require(p.vendorId==0x8086&&p.deviceId==0xe20c&&!(p.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),"B570 root identity");
    ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    require(zeDevicePciGetPropertiesExt(dev,&pci)==ZE_RESULT_SUCCESS,"PCI query");
    require(pci.address.domain==0&&pci.address.bus==5&&pci.address.device==0&&pci.address.function==0,"PCI0000:05:00.0 required");
    std::puts("IDENTITY,8086,e20c,0000:05:00.0,root");require(std::fflush(stdout)==0,"stdout flush");
    const auto selected=q.get_device();require(selected.has(sycl::aspect::fp16),"fp16 required");
    auto flags=selected.get_info<sycl::info::device::half_fp_config>();
    std::sort(flags.begin(),flags.end(),[](auto a,auto b){return unsigned(a)<unsigned(b);});
    std::printf("FP16_CONFIG,advertised_capabilities_only,fp16,1,flags");
    for(auto flag:flags)std::printf(",%u",unsigned(flag));
    std::puts("");require(std::fflush(stdout)==0,"capability flush");
}
struct Async {std::atomic<unsigned> errors{0};};
// Every potentially submitted operation must stay inside this fail-stop boundary.
// Queue timeout is owner-supervised; no unsafe free/unwind on unknown completion.
template<class F> void stage(sycl::queue& q,Async& a,const char* name,F f) {
    std::printf("STAGE,%s\n",name);if(std::fflush(stdout)!=0) std::_Exit(2);
    try {f();q.wait_and_throw();require(a.errors.load()==0,"sticky async error");}
    catch(...) {std::fprintf(stderr,"FAIL_STOP,%s,completion_unknown,no_USM_unwind\n",name);std::fflush(stderr);std::_Exit(2);}
}
struct Buffers {
    sycl::queue& q;uint8_t* input=nullptr;uint16_t *a=nullptr,*b=nullptr;
    Buffers(sycl::queue& queue):q(queue) {}
    ~Buffers(){if(input)sycl::free(input,q);if(a)sycl::free(a,q);if(b)sycl::free(b,q);}
};
void run(sycl::queue& q,Async& async) {
    constexpr size_t maxn=1638400, pad=32, inputpad=64;
    constexpr size_t inputbytes=maxn/32*18+2*inputpad, outputwords=maxn+2*pad;
    static_assert(inputbytes+2*outputwords*2 <16*1024*1024);
    Buffers dev(q);dev.input=sycl::malloc_device<uint8_t>(inputbytes,q);
    dev.a=sycl::malloc_device<uint16_t>(outputwords,q);dev.b=sycl::malloc_device<uint16_t>(outputwords,q);
    require(dev.input&&dev.a&&dev.b,"USM allocation");
    std::vector<uint8_t> input(inputbytes), restored(inputbytes);
    std::vector<uint16_t> expected(outputwords), a(outputwords),b(outputwords),initial(outputwords,0xa55a);
    // Test the same device constructor used by store_run separately from IQ inputs.
    // Independent RNE expected bits are not a portable constructor/FP-mode claim.
    const std::array<float,20> constructor_inputs{0.f,-0.f,1.f,-1.f,
        1.00048828125f,1.00146484375f,-1.00048828125f,-1.00146484375f,
        65504.f,-65504.f,65520.f,-65520.f,65536.f,-65536.f,
        std::ldexp(1.f,-24),-std::ldexp(1.f,-24),std::ldexp(1.f,-25),
        std::ldexp(3.f,-25),std::ldexp(1023.f,-24),std::ldexp(2047.f,-25)};
    std::fill(input.begin(),input.end(),0x6d);expected=initial;
    std::memcpy(input.data()+inputpad,constructor_inputs.data(),sizeof(constructor_inputs));
    for(size_t i=0;i<constructor_inputs.size();++i)expected[pad+i]=encode(constructor_inputs[i]);
    stage(q,async,"constructor_upload",[&]{q.memcpy(dev.input,input.data(),inputbytes);q.memcpy(dev.a,initial.data(),outputwords*2);});
    stage(q,async,"constructor_device",[&]{
        const float* source=reinterpret_cast<const float*>(dev.input+inputpad);
        sycl::half* destination=reinterpret_cast<sycl::half*>(dev.a+pad);
        const size_t count=constructor_inputs.size();
        q.parallel_for<class HalfConstructorDiagnostic>(sycl::nd_range<1>{32,32},[=](sycl::nd_item<1> item){
            const size_t i=item.get_global_linear_id();if(i<count)destination[i]=sycl::half(source[i]);
        });
    });
    stage(q,async,"constructor_readback",[&]{q.memcpy(restored.data(),dev.input,inputbytes);q.memcpy(a.data(),dev.a,outputwords*2);});
    require(restored==input,"constructor input immutability/guards");
    for(size_t i=0;i<constructor_inputs.size();++i)
        std::printf("CONSTRUCTOR,index=%zu,float_bits=%08x,IEEE_RNE_reference=%04x,observed=%04x\n",i,
            std::bit_cast<uint32_t>(constructor_inputs[i]),unsigned(expected[pad+i]),unsigned(a[pad+i]));
    require(std::fflush(stdout)==0,"constructor flush");require(a==expected,"independent RNE constructor bits/guards mismatch");
    std::puts("CONSTRUCTOR_RESULT,20_words,independent_RNE_bits_equal,target_specific_observation,portable_contract_unqualified");
    require(std::fflush(stdout)==0,"constructor result flush");
    const std::array<uint16_t,14> scales{0,1,0x3ff,0x400,0x401,0x3c00,0x7bff,0x8000,0x8001,0x83ff,0x8400,0x8401,0xbc00,0xfbff};
    size_t cases=0, compared=0;
    for(size_t n: {size_t(256),size_t(512),maxn}) for(auto scale:scales) {
        std::fill(input.begin(),input.end(),0x6d);expected=initial;
        for(size_t block=0;block<n/32;++block) {
            uint8_t* p=input.data()+inputpad+block*18;p[0]=uint8_t(scale);p[1]=uint8_t(scale>>8);
            // Actual Down:2560 rows x640 columns,20 packed blocks/row.
            // Odd rows split flat groups; each two rows is five groups.
            const size_t row=block/20, inrow=block%20;
            for(size_t j=0;j<16;++j) {
                unsigned low=unsigned((j+row+3*inrow)&15), high=unsigned((15-j+5*row+7*inrow)&15);
                p[2+j]=uint8_t(low|(high<<4));
            }
            // Scalar byte mapping is block-local, independent of GPU lane/group arithmetic.
            for(size_t j=0;j<32;++j) {
                const uint8_t packed=p[2+(j%16)];const unsigned code=j<16?(packed&15):(packed>>4);
                const float product=decode(scale)*float(codes[code]);
                expected[pad+block*32+j]=encode(product);
            }
        }
        for(unsigned repeat=0;repeat<2;++repeat) {
            stage(q,async,"upload",[&]{q.memcpy(dev.input,input.data(),inputbytes);q.memcpy(dev.a,initial.data(),outputwords*2);q.memcpy(dev.b,initial.data(),outputwords*2);});
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
            // Slots/scopes are host-owned before submit and survive both existing drains.
            std::array<strata::kernels::detail::Iq4nlReceipt,2> receipts{};
            strata::kernels::detail::Iq4nlReceiptSink sink{receipts.data(),receipts.size(),0,cases+1};
            strata::kernels::detail::Iq4nlReceiptScope receipt_scope(sink);
#endif
            stage(q,async,"generic",[&]{strata::kernels::iq_dequant_f16(20,dev.input+inputpad,int64_t(n),dev.a+pad,&q);});
            stage(q,async,"private",[&]{strata::kernels::iq_dequant_f16_prefill_iq4nl(20,dev.input+inputpad,int64_t(n),dev.b+pad,&q);});
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
            // Both stages have already completed successfully; profiling queries add no waits.
            require(!sink.invalid&&sink.count==2&&sink.serial==2,"receipt missing/overflow/retention failure");
            for(size_t r=0;r<receipts.size();++r) {
                const auto& receipt=receipts[r];
                require(receipt.event.has_value()&&receipt.scope_id==cases+1&&receipt.serial==r+1&&
                    receipt.type==20&&receipt.n==int64_t(n)&&receipt.queue==&q&&
                    receipt.src==dev.input+inputpad&&receipt.dst==(r==0?dev.a:dev.b)+pad&&
                    receipt.arm==(r==0?strata::kernels::detail::Iq4nlArm::generic:
                                      strata::kernels::detail::Iq4nlArm::private_down),"receipt metadata");
                const auto submit=receipt.event->get_profiling_info<sycl::info::event_profiling::command_submit>();
                const auto start=receipt.event->get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto end=receipt.event->get_profiling_info<sycl::info::event_profiling::command_end>();
                require(submit<=start&&start<=end,"receipt timestamp ordering");
                std::printf("RECEIPT,scope=%llu,serial=%llu,arm=%s,type=20,n=%zu,metadata_match=1,submit=%llu,start=%llu,end=%llu\n",
                    (unsigned long long)receipt.scope_id,(unsigned long long)receipt.serial,
                    r==0?"generic":"private",n,(unsigned long long)submit,
                    (unsigned long long)start,(unsigned long long)end);
            }
            require(std::fflush(stdout)==0,"receipt flush");
#endif
            stage(q,async,"readback",[&]{q.memcpy(restored.data(),dev.input,inputbytes);q.memcpy(a.data(),dev.a,outputwords*2);q.memcpy(b.data(),dev.b,outputwords*2);});
            require(restored==input,"input immutability/guards");
            size_t ab=0,oracle_a=0,oracle_b=0;bool first=true;
            for(size_t i=0;i<outputwords;++i) {
                ab+=a[i]!=b[i];oracle_a+=a[i]!=expected[i];oracle_b+=b[i]!=expected[i];
                if(first&&(a[i]!=b[i]||a[i]!=expected[i]||b[i]!=expected[i])) {
                    std::printf("FIRST_DIFF,n=%zu,scale=%04x,repeat=%u,word=%zu,expected=%04x,generic=%04x,private=%04x\n",n,unsigned(scale),repeat,i,unsigned(expected[i]),unsigned(a[i]),unsigned(b[i]));first=false;
                }
            }
            std::printf("CASE,n=%zu,rows=%zu,cols=%zu,scale=%04x,class=%s,repeat=%u,active=%zu,ab=%zu,oracle_generic=%zu,oracle_private=%zu,all_guards_checked=1\n",n,n==maxn?size_t(2560):size_t(0),n==maxn?size_t(640):size_t(0),unsigned(scale),(scale&0x8000)?"synthetic_negative_robustness":"synthetic_finite",repeat,n,ab,oracle_a,oracle_b);
            require(std::fflush(stdout)==0,"case flush");require(ab==0&&oracle_a==0&&oracle_b==0,"exact bits/guards mismatch");
            ++cases;compared+=n;
        }
    }
    stage(q,async,"final_drain",[]{});
    std::printf("PASS,cases=%zu,active_words=%zu,synthetic_only=1,model=0,performance=0,fullKV=0,adoption=0\n",cases,compared);
    require(std::fflush(stdout)==0,"terminal flush");
} // buffers freed only after known successful drain; endpoints live throughout every stage
}
int main(int argc,char** argv) {
    try {
        require(argc==1||(argc==2&&!std::strcmp(argv[1],"--host-only")),"usage: prefill_iq4nl_parity [--host-only]");
        host_checks();require(std::fflush(stdout)==0,"host flush");if(argc==2)return 0;
        Async async;
        {
        sycl::queue q(sycl::gpu_selector_v,[&](sycl::exception_list errors) noexcept {
            for(const auto& ignored:errors){(void)ignored;unsigned v=async.errors.load();while(v<1024&&!async.errors.compare_exchange_weak(v,v+1)){} }
        },sycl::property_list{sycl::property::queue::in_order{}
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
            ,sycl::property::queue::enable_profiling{}
#endif
        });
        identity(q);
#ifdef STRATA_SYCL_PREFILL_IQ4NL_EVENT_RECEIPT
        require(q.has_property<sycl::property::queue::enable_profiling>(),"receipt requires profiling queue");
        require(q.get_device().has(sycl::aspect::queue_profiling),"receipt requires device profiling aspect");
        const auto resolution=q.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
        std::printf("RECEIPT_QUEUE,profiling=1,in_order=1,backend=level_zero,resolution_ns=%zu\n",size_t(resolution));
        require(std::fflush(stdout)==0,"receipt queue flush");
#endif
        run(q,async);
        stage(q,async,"post_USM_teardown_drain",[]{});
        }
        require(async.errors.load()==0,"queue teardown async error");
        std::puts("TERMINAL,pass,synthetic_only,model_false,performance_false,fullKV_false,adoption_false");
        require(std::fflush(stdout)==0&&std::fflush(stderr)==0,"terminal flush");return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL,%s\n",e.what());std::fflush(stderr);return 1;}
}
