// Root-owned qualification fixture for the real prefill GDN producer/conv/recurrence.
// Synthetic component evidence only: no model, decode or physical KV lifecycle claim.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_gate_factor.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t Heads=48, Keys=16, Width=128, Channels=10240;
constexpr size_t State=Width*Heads*Width, History=Channels*3, Guard=32;
constexpr uint32_t FloatPoison=0x7fc12345;
constexpr uint16_t HalfPoison=0x7e55;
using Encoding=strata::prefill::GdnGateEncoding;
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
uint32_t bits(float value) { return std::bit_cast<uint32_t>(value); }
struct Errors {
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::vector<std::string> messages;
    void collect(sycl::exception_list exceptions) noexcept {
        failed.store(true);
        try { std::lock_guard<std::mutex> lock(mutex);
            for(auto error:exceptions) try { std::rethrow_exception(error); }
            catch(const std::exception& error) {
                size_t length=0;while(length<1024&&error.what()[length]) ++length;
                if(messages.size()<8) messages.emplace_back(error.what(),length);
            } catch(...) {}
        } catch(...) {}
    }
};
template<class F> void stage(sycl::queue& q,Errors& errors,const char* name,F action) {
    std::printf("STAGE,%s\n",name); std::fflush(stdout);
    try { action(); q.wait_and_throw(); require(!errors.failed.load(),"asynchronous error"); }
    catch(const std::exception& error) {
        std::fprintf(stderr,"FAIL_STOP,%s,%s,completion_unknown,unwind_skipped\n",name,error.what());
        { std::lock_guard<std::mutex> lock(errors.mutex);
          for(const auto& message:errors.messages) std::fprintf(stderr,"ASYNC,%s\n",message.c_str()); }
        std::fflush(stderr); std::_Exit(2);
    } catch(...) { std::fprintf(stderr,"FAIL_STOP,%s,unknown\n",name); std::fflush(stderr); std::_Exit(2); }
}
uint64_t hash(const void* data,size_t bytes) {
    uint64_t value=14695981039346656037ULL;
    for(size_t i=0;i<bytes;++i) { value^=static_cast<const uint8_t*>(data)[i]; value*=1099511628211ULL; }
    return value;
}
template<class T> T poison();
template<> float poison<float>() { return std::bit_cast<float>(FloatPoison); }
template<> uint16_t poison<uint16_t>() { return HalfPoison; }
template<class T> struct Buffer {
    sycl::queue& q;
    T* base;
    size_t capacity;
    std::vector<T> host;
    Buffer(sycl::queue& queue,size_t count):q(queue),base(nullptr),capacity(count),host(count+2*Guard,poison<T>()) {
        base=sycl::malloc_device<T>(host.size(),q); require(base!=nullptr,"allocation failed");
    }
    ~Buffer() noexcept { try { sycl::free(base,q); } catch(...) { std::fputs("FAIL USM release\n",stderr); std::fflush(stderr); std::_Exit(2); } }
    Buffer(const Buffer&)=delete;
    T* data() { return base+Guard; }
    T* values() { return host.data()+Guard; }
    void reset() { std::fill(host.begin(),host.end(),poison<T>()); }
    void upload() { q.memcpy(base,host.data(),host.size()*sizeof(T)); }
    void download() { q.memcpy(host.data(),base,host.size()*sizeof(T)); }
    void immutable(Errors& errors) {
        std::vector<T> observed(host.size());
        stage(q,errors,"immutable_input_readback",[&]{q.memcpy(observed.data(),base,observed.size()*sizeof(T));});
        require(std::memcmp(observed.data(),host.data(),host.size()*sizeof(T))==0,"immutable input changed");
    }
    void guards(size_t live) const {
        require(live<=capacity,"live bound");
        const T expected=poison<T>();
        for(size_t i=0;i<host.size();++i) if(i<Guard||i>=Guard+live)
            require(std::memcmp(&host[i],&expected,sizeof(T))==0,"guard or inactive tail changed");
    }
};
template<class T> void equal(const Buffer<T>& a,const Buffer<T>& b,size_t live,const char* label) {
    a.guards(live); b.guards(live);
    for(size_t i=0;i<live;++i) {
        const T x=a.host[Guard+i],y=b.host[Guard+i];
        if(std::memcmp(&x,&y,sizeof(T))) {
            std::fprintf(stderr,"FIRST_DIFFERENCE,%s,index,%zu\n",label,i); require(false,"bitwise difference");
        }
        if constexpr(sizeof(T)==sizeof(float)) require(std::isfinite(x),"nonfinite FP32 result");
        else require((x&0x7c00)!=0x7c00,"nonfinite FP16 result");
    }
}
class MaterializedLogGateExp;
struct Fixture {
    sycl::queue& q; Errors& errors; size_t cap;
    Buffer<float> stateA,stateB,historyA,historyB,qkv,convW,hA,hB,ab,dt,a,gA,gB,bA,bB,z,gamma,yA,yB,expectedFactor;
    Buffer<uint16_t> halfA,halfB;
    Fixture(sycl::queue& queue,Errors& e,size_t capacity):q(queue),errors(e),cap(capacity),
        stateA(q,State),stateB(q,State),historyA(q,History),historyB(q,History),qkv(q,cap*Channels),convW(q,Channels*4),
        hA(q,cap*Channels),hB(q,cap*Channels),ab(q,cap*2*Heads),dt(q,Heads),a(q,Heads),gA(q,cap*Heads),gB(q,cap*Heads),
        bA(q,cap*Heads),bB(q,cap*Heads),z(q,cap*Heads*Width),gamma(q,Width),yA(q,cap*Heads*Width),yB(q,cap*Heads*Width),
        expectedFactor(q,cap*Heads),halfA(q,cap*Heads*Width),halfB(q,cap*Heads*Width) {}
    ~Fixture() noexcept { stage(q,errors,"final_drain",[]{}); }
    void initialize(bool zero) {
        for(size_t i=0;i<State;++i) stateA.values()[i]=stateB.values()[i]=zero?0.f:float(int((i*7+i/128*11)%31)-15)*.001f;
        for(size_t i=0;i<History;++i) historyA.values()[i]=historyB.values()[i]=zero?0.f:float(int((i*13)%23)-11)*.002f;
        for(size_t i=0;i<Channels*4;++i) convW.values()[i]=.08f+float(i%7)*.01f;
        for(size_t h=0;h<Heads;++h) { dt.values()[h]=float(int(h%7)-3)*.02f; a.values()[h]=-.04f-float(h%13)*.007f; }
        for(size_t i=0;i<Width;++i) gamma.values()[i]=.9f+float(i%17)*.01f;
        stage(q,errors,"initialize",[&]{ for(auto* buffer:{&stateA,&stateB,&historyA,&historyB,&convW,&dt,&a,&gamma}) buffer->upload(); });
    }
    void prepare(size_t live,size_t offset) {
        require(live>0&&live<=cap,"chunk bound");
        for(auto* buffer:{&qkv,&ab,&z,&hA,&hB,&gA,&gB,&bA,&bB,&yA,&yB}) buffer->reset();
        halfA.reset(); halfB.reset(); expectedFactor.reset();
        for(size_t t=0;t<live;++t) {
            const size_t token=offset+t;
            for(size_t c=0;c<Channels;++c) qkv.values()[t*Channels+c]=float(int((token*13+c*7+c/128*11)%43)-21)*.005f;
            for(size_t h=0;h<Heads;++h) {
                ab.values()[t*2*Heads+h]=float(int((token*3+h*7)%61)-30)*.1f;
                ab.values()[t*2*Heads+Heads+h]=float(int((token*11+h*13)%31)-15)*.04f;
                for(size_t c=0;c<Width;++c) z.values()[t*Heads*Width+h*Width+c]=float(int((token*7+h*11+c*3)%37)-18)*.02f;
            }
        }
        stage(q,errors,"prepare",[&]{
            for(auto* buffer:{&qkv,&ab,&z,&hA,&hB,&gA,&gB,&bA,&bB,&yA,&yB}) buffer->upload();
            halfA.upload();halfB.upload();expectedFactor.upload();
        });
    }
    void run(size_t live,size_t offset,int variant,bool factorFirst) {
        prepare(live,offset);
        auto arm=[&](bool factor) {
            const Encoding encoding=factor?Encoding::DecayFactor:Encoding::LogGate;
            auto& gate=factor?gB:gA; auto& beta=factor?bB:bA; auto& state=factor?stateB:stateA;
            auto& history=factor?historyB:historyA;auto& h=factor?hB:hA;auto& y=factor?yB:yA;auto& half=factor?halfB:halfA;
            stage(q,errors,factor?"factor_gates_conv_recurrence":"log_gates_conv_recurrence",[&]{
                strata::prefill::gdn_gates_encoded(encoding,ab.data(),dt.data(),a.data(),gate.data(),beta.data(),int64_t(live),&q);
                strata::prefill::gdn_conv(history.data(),qkv.data(),convW.data(),h.data(),int64_t(live),1e-6f,&q);
                strata::prefill::gdn_recurrence_encoded(encoding,variant,state.data(),h.data(),gate.data(),beta.data(),z.data(),gamma.data(),1e-6f,y.data(),half.data(),int64_t(live),&q);
            });
        };
        arm(factorFirst);arm(!factorFirst);
        stage(q,errors,"materialized_loggate_exp",[&]{
            const size_t liveGates=live*Heads,global=(liveGates+255)/256*256;
            const float* log=gA.data();float* expected=expectedFactor.data();
            q.parallel_for<MaterializedLogGateExp>(sycl::nd_range<1>{global,256},[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const size_t i=item.get_global_linear_id();if(i<liveGates) expected[i]=sycl::native::exp(log[i]);
            });
        });
        stage(q,errors,"readback",[&]{
            for(auto* buffer:{&stateA,&stateB,&historyA,&historyB,&hA,&hB,&gA,&gB,&bA,&bB,&yA,&yB}) buffer->download();
            halfA.download();halfB.download();expectedFactor.download();
        });
        equal(stateA,stateB,State,"state");equal(historyA,historyB,History,"conv_history");
        equal(hA,hB,live*Channels,"conv_output");equal(bA,bB,live*Heads,"beta");
        equal(yA,yB,live*Heads*Width,"FP32_output");equal(halfA,halfB,live*Heads*Width,"FP16_output");
        equal(gB,expectedFactor,live*Heads,"materialized_loggate_exp");
        gA.guards(live*Heads);gB.guards(live*Heads);
        for(size_t i=0;i<live*Heads;++i) require(std::isfinite(gA.values()[i])&&std::isfinite(gB.values()[i])&&gB.values()[i]>=0.f&&gB.values()[i]<=1.f,"gate classification");
        for(auto* buffer:{&ab,&dt,&a,&qkv,&convW,&z,&gamma}) buffer->immutable(errors);
        // These are exact state/output comparisons, not tolerance-based native-exp claims.
        std::printf("CHUNK_PASS,offset,%zu,live,%zu,variant,%d,first,%s,state,%llu,conv_history,%llu,y,%llu,half,%llu\n",
            offset,live,variant,factorFirst?"factor":"log",(unsigned long long)hash(stateA.host.data(),stateA.host.size()*4),
            (unsigned long long)hash(historyA.host.data(),historyA.host.size()*4),(unsigned long long)hash(yA.host.data(),yA.host.size()*4),
            (unsigned long long)hash(halfA.host.data(),halfA.host.size()*2)); std::fflush(stdout);
    }
};
size_t decimal(const char* text,size_t limit) {
    require(text&&*text,"empty decimal");size_t value=0;
    for(const char* p=text;*p;++p) { require(*p>='0'&&*p<='9',"nondecimal input"); require(value<=limit/10,"decimal bound"); value=value*10+size_t(*p-'0');require(value<=limit,"decimal bound"); }
    return value;
}
void host_contract() {
    (void)strata::prefill::gdn_prefill_gate_factor_enabled();
    std::array<float,8> state{1.f,-2.f,3.f,4.f,-5.f,6.f,7.f,-8.f},output=state;
    std::array<uint16_t,8> half{1,2,3,4,5,6,7,8};
    const auto initial=state,initialOutput=output;const auto initialHalf=half;
    for(Encoding encoding:{Encoding::LogGate,Encoding::DecayFactor}) {
        // Each API must execute independently: a producer refusal must not skip
        // the recurrence refusal assertion in a feature-OFF qualification build.
        for(int api:{0,1}) {
            bool refused=false;
            try {
                if(api==0) strata::prefill::gdn_gates_encoded(encoding,nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr);
                else strata::prefill::gdn_recurrence_encoded(encoding,0,state.data(),nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,output.data(),half.data(),0,nullptr);
            } catch(const std::invalid_argument&) { refused=true; }
#if defined(STRATA_GDN_FACTOR_PARITY_ENABLED)
            require(!refused,"enabled empty API refused");
#else
            require(refused==(encoding==Encoding::DecayFactor),"build-OFF factor refusal");
#endif
            require(state==initial&&output==initialOutput&&half==initialHalf,"empty API changed markers");
        }
    }
    for(int64_t length:{int64_t(-1),int64_t(262145)}) {
        bool refused=false;
        try {strata::prefill::gdn_gates_encoded(Encoding::LogGate,nullptr,nullptr,nullptr,nullptr,nullptr,length,nullptr);}
        catch(const std::invalid_argument&) {refused=true;}require(refused,"invalid length accepted");
    }
    bool refused=false;
    try {strata::prefill::gdn_recurrence_encoded(Encoding::LogGate,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,nullptr,nullptr,0,nullptr,6145);}
    catch(const std::invalid_argument&) {refused=true;}require(refused,"invalid output stride accepted");
    refused=false;
    try {strata::prefill::gdn_recurrence_encoded(Encoding::LogGate,3,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,nullptr,nullptr,0,nullptr);}
    catch(const std::invalid_argument&) {refused=true;}require(refused,"unsupported variant accepted");
    refused=false;
    try {strata::prefill::gdn_gates_encoded(static_cast<Encoding>(2),nullptr,nullptr,nullptr,nullptr,nullptr,0,nullptr);}
    catch(const std::invalid_argument&) {refused=true;}require(refused,"unknown encoding accepted");
    std::puts("HOST_PASS,empty_markers,independent_empty_API_calls,4,invalid_length_stride_variant_encoding,no_queue_construction,no_GPU_submission");
}
void selected_identity(sycl::queue& q) {
    require(q.get_backend()==sycl::backend::ext_oneapi_level_zero&&q.is_in_order(),"LevelZero in-order required");
    auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    ze_device_properties_t properties{};properties.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    require(zeDeviceGetProperties(device,&properties)==ZE_RESULT_SUCCESS,"device identity query");
    require(properties.vendorId==0x8086&&properties.deviceId==0xe20c&&!(properties.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),"B570 root identity required");
    ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    require(zeDevicePciGetPropertiesExt(device,&pci)==ZE_RESULT_SUCCESS,"public loader PCI query");
    require(pci.address.domain==0&&pci.address.bus==5&&pci.address.device==0&&pci.address.function==0,"exact PCI identity required");
    std::puts("IDENTITY,LevelZero,8086,e20c,0000:05:00.0,root,passed");std::fflush(stdout);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc==2&&!std::strcmp(argv[1],"--host-only")) {host_contract();require(std::fflush(stdout)==0,"output failure");return 0;}
        size_t total=0,cap=257,endpoint=0;int variant=0;
        if(argc>1) {
            require(argc==7&&!std::strcmp(argv[1],"--prefix")&&!std::strcmp(argv[3],"--chunk")&&!std::strcmp(argv[5],"--tail"),"usage: --prefix 32768|262144 --chunk 1..8192 --tail 0|1|4");
            total=decimal(argv[2],262144);cap=decimal(argv[4],8192);endpoint=decimal(argv[6],4);
            require((total==32768||total==262144)&&cap>0&&(endpoint==0||endpoint==1||endpoint==4)&&endpoint<=cap,"prefix admission");
        }
        const char* serial=std::getenv("STRATA_GDN_REC_HEADS"); if(serial) variant=2;
        Errors errors;
        sycl::queue q(sycl::gpu_selector_v,[&](sycl::exception_list exceptions){errors.collect(exceptions);},sycl::property_list{sycl::property::queue::in_order{}});
        selected_identity(q);
        std::printf("SCOPE,synthetic_real_gdn_component,model,false,performance,false,physical_KV_lifecycle,false,capacity,%zu\n",cap);
        {
            Fixture fixture(q,errors,cap);
            if(total) {
                fixture.initialize(false);size_t offset=0,calls=0;
                const size_t body=total-endpoint;
                while(offset<body) { size_t live=std::min(cap,body-offset);fixture.run(live,offset,variant,(calls++%2)!=0);offset+=live; }
                if(endpoint) { fixture.run(endpoint,offset,variant,(calls++%2)!=0);offset+=endpoint; }
                require(offset==total,"complete carried prefix");
                std::printf("PREFIX_PASS,total,%zu,calls,%zu,tail,%zu,carried_state_and_conv,true\n",offset,calls,endpoint);
            } else {
                for(bool zero:{true,false}) {
                    fixture.initialize(zero);size_t offset=0,index=0;
                    for(size_t live:{size_t(1),size_t(3),size_t(5),size_t(31),size_t(32),size_t(33),size_t(255),size_t(256),size_t(257),size_t(3),size_t(257)}) {
                        fixture.run(live,offset,variant,(index++%2)!=0);offset+=live;
                    }
                }
            }
        }
        std::puts("TERMINAL,pass,real_producer_conv_recurrence_bitwise,synthetic_only,model_unqualified,full_KV_lifecycle_unqualified,adopted_false");
        require(std::fflush(stdout)==0&&std::fflush(stderr)==0,"output failure");return 0;
    } catch(const std::exception& error) { std::fprintf(stderr,"FAIL,%s\n",error.what());return 1; }
}
