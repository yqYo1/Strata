// Main-authored NativeHead ownership/graph fixture; repository LICENSE.
// Real mapped GGUF head, no model forward or Verifier ownership qualification.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify_window_policy.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/runtime.hpp"
#include "../../src/kernels/sycl/q6_dpas_component.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace k = strata::kernels;
using strata::core::NativeHead;
using strata::core::VerifyInput;
constexpr int K=2560,N=248320;
constexpr size_t WeightBytes=size_t(N)*(K/256)*210;
void need(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
void check(cudaError_t value) {need(value==cudaSuccess,cudaGetErrorString(value));}
int64_t now(clockid_t id) {
    timespec t{};need(clock_gettime(id,&t)==0,"clock");
    return int64_t(t.tv_sec)*1000000000+t.tv_nsec;
}
struct Stamp {int64_t wall=now(CLOCK_MONOTONIC),cpu=now(CLOCK_THREAD_CPUTIME_ID);};
template<class Fn> void timed(const char* phase,int cycle,Fn&& fn) {
    const Stamp a;fn();const Stamp b;
    std::cout<<"TRANSITION phase="<<phase<<" cycle="<<cycle<<" wall_ns="<<b.wall-a.wall
        <<" thread_cpu_ns="<<b.cpu-a.cpu<<" scope=whole_wrapper\n";
}
uint32_t mix(uint32_t x) {x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
uint16_t half_at(const uint8_t* b,size_t p) {return uint16_t(b[p])|uint16_t(uint16_t(b[p+1])<<8);}
int signed_byte(uint8_t b) {return int(std::bit_cast<int8_t>(b));}
void gate(const char* name) {
    need(std::getenv("STRATA_Q6_HEAD_OWNER_GATE")!=nullptr,"supervised gate required");
    std::cout<<name<<" pid="<<getpid()<<'\n';std::string line;
    need(bool(std::getline(std::cin,line))&&line=="continue","supervisor gate");
}
template<class T> struct Guard {
    static constexpr size_t Edge=128/sizeof(T);
    static constexpr T Mark=T(-73);
    std::shared_ptr<sb::Runtime> rt;sb::Allocation storage;size_t count;
    Guard(std::shared_ptr<sb::Runtime> r,size_t n):rt(r),
        storage(r,(n+2*Edge)*sizeof(T),sb::MemoryKind::Device),count(n) {reset();}
    T* data() {return storage.template as<T>()+Edge;}
    void reset() {std::vector<T> b(count+2*Edge,Mark);rt->wait(rt->compute().memcpy(storage.data(),b.data(),b.size()*sizeof(T)));}
    void put(const std::vector<T>& b) {need(b.size()==count,"input span");rt->wait(rt->compute().memcpy(data(),b.data(),b.size()*sizeof(T)));}
    std::vector<T> get() {
        std::vector<T> b(count+2*Edge);rt->wait(rt->compute().memcpy(b.data(),storage.data(),b.size()*sizeof(T)));
        for(size_t i=0;i<Edge;++i)need(b[i]==Mark&&b[count+Edge+i]==Mark,"allocation canary");
        return {b.begin()+Edge,b.begin()+Edge+count};
    }
};
struct Fixture {
    std::shared_ptr<sb::Runtime> rt;
    strata::GgufModel source;
    NativeHead head; // Retires after the drained graphs and temporary buffers.
    const uint8_t* w=nullptr;
    std::unique_ptr<sb::Allocation> reference_weights;
    Guard<float> mixed,reference,candidate,single;
    Guard<uint8_t> q8;
    cudaStream_t stream=nullptr;
    std::array<cudaGraphExec_t,3> graphs{};
    std::array<uint64_t,3> launches{};
    bool dpas=false,packed=false,host_backed=false;
    size_t numerical_cases=0,blocked_cases=0,partial_cases=0;
    Fixture(std::shared_ptr<sb::Runtime> r,const std::vector<std::string>& shards):rt(r),source(shards),
        mixed(r,K*4),reference(r,N*4),candidate(r,N*4),single(r,N*4),q8(r,k::native_q8_1_bytes(K,4)) {
        const char* flag=std::getenv("STRATA_SYCL_DECODE_Q6_DPAS_HEAD");
        need(flag&&(!std::strcmp(flag,"0")||!std::strcmp(flag,"1")||!std::strcmp(flag,"2")),"explicit fixture arm");
        dpas=!std::strcmp(flag,"1");
        packed=!std::strcmp(flag,"2");
        const char* backing=std::getenv("STRATA_PREFILL_HEAD_HOST_BACKING");
        need(!backing||!std::strcmp(backing,"0")||!std::strcmp(backing,"1"),"explicit host backing policy");
        host_backed=backing&&!std::strcmp(backing,"1");
        size_t at=0;const auto* tensor=source.find("output.weight",&at);
        need(tensor&&tensor->type==14&&tensor->shape==std::vector<uint64_t>{K,N}&&
            source.in_bounds(*tensor,at),"actual Q6 head extent");
        w=static_cast<const uint8_t*>(source.shard(at).tensor_data(*tensor));
        check(cudaStreamCreate(&stream));
        std::string error;timed("load",-1,[&]{if(!head.load(shards,K,N,error))throw std::runtime_error(error);});
        if(packed) {
            // The independent reference is fixture-only,not a production owner.
            reference_weights=std::make_unique<sb::Allocation>(rt,WeightBytes,sb::MemoryKind::Device);
            check(cudaMemcpy(reference_weights->data(),w,WeightBytes,cudaMemcpyHostToDevice));
        }
        metadata();need(head.loaded()&&head.decode_dpas_ready()==dpas,"loaded owner");
        std::cout<<"HEAD admitted=1 dpas="<<int(dpas)<<" layout="<<(packed?"sole-packed":"canonical")<<" canonical_bytes="<<head.weight_bytes()
            <<" extra_device_weight_bytes=0 extra_host_weight_bytes="<<head.host_backing_bytes()
            <<" max_alloc_bytes="<<rt->device().get_info<sycl::info::device::max_mem_alloc_size>()<<'\n';
        if(packed)std::cout<<"FIXTURE_REFERENCE device_weight_bytes="<<WeightBytes<<" production_owner=0\n";
        capture();
    }
    ~Fixture() {
        // A partial/failed retirement may not unwind borrowed GPU operands.
        try {retire();check(cudaStreamDestroy(stream));stream=nullptr;}
        catch(...) {std::cerr<<"head fixture teardown completion unknown\n"<<std::flush;std::_Exit(4);}
    }
    void metadata() const {
        need(head.type()==14&&head.weight_bytes()==WeightBytes&&head.row_bytes()==2100&&
            head.decode_dpas_enabled()==dpas&&head.primary_packed()==packed,"head metadata/immutable selector");
        need(head.has_host_backing()==host_backed&&head.host_backing_bytes()==(host_backed?WeightBytes:0),
            "immutable initial host backing/actual extra RAM bytes");
        if(packed && head.loaded()) {
            bool refused=false;try{(void)head.weights();}catch(const std::logic_error&){refused=true;}
            need(refused,"packed owner exposed as row-major weights");
        }
    }
    const void* storage() const {return packed?head.packed_weights():head.weights();}
    void drain() {check(cudaStreamSynchronize(stream));rt->wait();}
    void retire() {
        drain();for(auto& g:graphs)if(g){check(cudaGraphExecDestroy(g));g=nullptr;}
    }
    void entry(int arm) {
        if(arm==2){std::string e;if(!head.run(mixed.data(),single.data(),stream,e))throw std::runtime_error(e);return;}
        k::native_quantize_q8_1(mixed.data(),q8.data(),K,4,stream);
        const auto phase=arm==0?VerifyInput::prompt:VerifyInput::decode;
        const bool selected=strata::core::verify_q6_dpas_head(head.decode_dpas_ready(),phase,
            k::native_mmvq_multi_exact(),head.type(),K,N,4);
        need(selected==(arm==1&&dpas),"actual fixture phase/owner selection");
        float* out=arm==0?reference.data():candidate.data();
        if(packed && arm==1)head.run_packed_columns(q8.data(),out,4,stream);
        else if(selected)head.run_decode_c4(q8.data(),out,stream);
        else k::native_mmvq(head.type(),packed?reference_weights->data():head.weights(),q8.data(),out,K,N,4,stream);
    }
    void capture() {
        for(int arm=0;arm<3;++arm) {
            need(!graphs[size_t(arm)],"old reader survived retirement");
            cudaGraph_t g=nullptr;
            try {
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));entry(arm);
                check(cudaStreamEndCapture(stream,&g));size_t nodes=0;check(cudaGraphGetNodes(g,nullptr,&nodes));
                need(nodes==2,"quantize+projection two-node graph");
                check(cudaGraphInstantiate(&graphs[size_t(arm)],g,0ull));check(cudaGraphDestroy(g));
            } catch(...) {std::cerr<<"head fixture capture incomplete\n"<<std::flush;std::_Exit(3);}
        }
        std::cout<<"CAPTURE graphs=3 nodes_each=2 dpas_decode_calls="<<int(dpas||packed)<<" dpas_prompt_calls=0 dpas_C1_calls="<<int(packed)<<"\n";
    }
    std::pair<double,double> oracle(int row,int column,const std::vector<uint8_t>& x) const {
        double sum=0,l1=0;
        for(int j=0;j<K;++j) {
            const size_t b=(size_t(row)*(K/256)+size_t(j/256))*210;
            const int u=j%256,part=(u%128)/32,l=u%32;
            const unsigned low=(w[b+size_t((u/128)*64+(part%2)*32+l)]>>(4*(part/2)))&15;
            const unsigned high=(w[b+128+size_t((u/128)*32+l)]>>(2*part))&3;
            const size_t a=(size_t(column)*(K/32)+size_t(j/32))*36;
            const double value=double(int(low|(high<<4))-32)*signed_byte(w[b+192+size_t(u/16)])*
                k::f32_from_f16(half_at(w,b+208))*signed_byte(x[a+4+size_t(j%32)])*
                k::f32_from_f16(half_at(x.data(),a));
            sum+=value;l1+=std::abs(value);
        }
        return {sum,l1};
    }
    void numeric(int cycle,uint32_t seed) {
        drain();std::vector<float> values(K*4);
        for(size_t i=0;i<values.size();++i)values[i]=float(int(mix(uint32_t(i)^seed)&65535)-32768)/65536.f;
        mixed.put(values);reference.reset();candidate.reset();single.reset();
        for(size_t arm=0;arm<graphs.size();++arm){check(cudaGraphLaunch(graphs[arm],stream));++launches[arm];drain();}
        const auto a=reference.get(),b=candidate.get(),c=single.get();const auto x=q8.get();
        std::vector<double> forward_bounds(size_t(N)*4);
        need(mixed.get()==values,"mixed input changed");
        std::array<std::array<double,K/16>,4> xmax{};
        for(int col=0;col<4;++col)for(int unit=0;unit<K/16;++unit) {
            const size_t base=(size_t(col)*(K/32)+size_t(unit/2))*36;
            for(int j=0;j<16;++j)xmax[size_t(col)][size_t(unit)]=std::max(xmax[size_t(col)][size_t(unit)],
                std::abs(double(k::f32_from_f16(half_at(x.data(),base)))*signed_byte(x[base+4+size_t(unit%2)*16+size_t(j)])));
        }
        const double u=std::numeric_limits<float>::epsilon()/2,count=8.*(K/32)+64,gamma=count*u/(1-count*u);
        double worst=0;size_t rows=0;
        for(int row=0;row<N;++row) {
            const bool selected=row<16||row>=N-16||row%(N/512)==0;
            rows+=selected;
            for(int col=0;col<4;++col) {
                double envelope=0;
                for(int block=0;block<K/256;++block) {
                    const size_t base=(size_t(row)*(K/256)+size_t(block))*210;
                    const double scale=std::abs(double(k::f32_from_f16(half_at(w,base+208))));
                    need(std::isfinite(scale),"finite source block scale");
                    for(int unit=0;unit<16;++unit)envelope+=scale*std::abs(signed_byte(w[base+192+size_t(unit)]))*
                        16*32*xmax[size_t(col)][size_t(block*16+unit)];
                }
                const size_t index=size_t(col)*N+size_t(row);
                forward_bounds[index]=2*gamma*envelope+1e-12;
                need(std::isfinite(a[index])&&std::isfinite(b[index]),"finite full output");
                const double delta=std::abs(double(a[index])-b[index]);worst=std::max(worst,delta);
                need(delta<=2*gamma*envelope+1e-12,"full-output forward envelope");
                if(col==0)need(std::isfinite(c[size_t(row)])&&
                    std::abs(double(c[size_t(row)])-a[index])<=2*gamma*envelope+1e-12,"unchanged C1");
                else need(c[index]==Guard<float>::Mark,"C1 wrote another column");
                if(selected) {
                    const auto expected=oracle(row,col,x);const double bound=gamma*expected.second+1e-12;
                    need(std::abs(double(a[index])-expected.first)<=bound&&
                        std::abs(double(b[index])-expected.first)<=bound,"independent actual-weight oracle");
                    if(col==0)need(std::abs(double(c[size_t(row)])-expected.first)<=bound,"C1 scalar oracle");
                }
            }
        }
        ++numerical_cases;
        std::cout<<"NUMERIC cycle="<<cycle<<" seed="<<seed<<" outputs="<<N*4<<" oracle_rows="<<rows
            <<" max_abs_pair_error="<<std::setprecision(12)<<worst<<" C1=PASS prompt=reference guards=PASS\n";
        if(packed)for(int columns:{2,3}) {
            candidate.reset();head.run_packed_columns(q8.data(),candidate.data(),columns,stream);drain();
            const auto partial=candidate.get();
            for(size_t i=0;i<partial.size();++i) {
                if(i>=size_t(columns)*N)need(partial[i]==Guard<float>::Mark,"partial head wrote inactive column");
                else {
                    need(std::isfinite(partial[i]) && std::abs(double(partial[i])-a[i])<=forward_bounds[i],
                         "partial head exceeds the full-output forward envelope");
                    const int row=int(i%N),col=int(i/N);
                    if(row<16 || row>=N-16 || row%(N/512)==0) {
                        const auto expected=oracle(row,col,x);
                        need(std::abs(double(partial[i])-expected.first)<=gamma*expected.second+1e-12,
                             "partial-column independent scalar oracle");
                    }
                }
            }
            need(q8.get()==x,"partial head modified Q8 input");++partial_cases;
            std::cout<<"PARTIAL columns="<<columns<<" outputs="<<columns*N<<" guards=PASS input_immutable=PASS\n";
        }
    }
    void blocked(const char* phase) {
        std::string e;need(head.head_detached()&&!head.decode_dpas_ready()&&
            !head.run(mixed.data(),single.data(),stream,e),"detached C1 accepted");
        bool refused=false;try{head.run_decode_c4(q8.data(),candidate.data(),stream);}
        catch(const std::invalid_argument&){refused=true;}
        need(refused,"detached DPAS projection accepted");++blocked_cases;
        refused=false;try{head.run_packed_columns(q8.data(),candidate.data(),4,stream);}
        catch(const std::invalid_argument&){refused=true;}
        need(refused,"detached packed projection accepted");
        std::cout<<"BLOCKED phase="<<phase<<" canonical=PASS dpas=PASS\n";
    }
    void immutable() {
        drain();
        {std::vector<uint8_t> bytes(WeightBytes);
            check(cudaMemcpy(bytes.data(),storage(),bytes.size(),cudaMemcpyDeviceToHost));
            if(packed) {
                std::vector<uint16_t> expected(WeightBytes/2);
                k::component::q6_dpas_pack({w,WeightBytes},expected,K,N);
                need(std::memcmp(bytes.data(),expected.data(),bytes.size())==0,"packed head byte mutation");
            } else need(std::memcmp(bytes.data(),w,bytes.size())==0,"canonical head byte mutation");}
        metadata();
    }
    void retired_span(const void* pointer) {
        bool retired=false;
        try{sb::validate_device_span(pointer,WeightBytes,*stream);}
        catch(const std::invalid_argument& e){const std::string why=e.what();
            retired=why=="unregistered SYCL device span"||why=="SYCL device span exceeds allocation or queue domain";}
        need(retired,"retired head remains a live registered allocation");
    }
    void run() {
        numeric(-1,7);std::string error;
        if(!head.prepare_prefill_weights(error))throw std::runtime_error(error);
        head.cancel_prefill_weights();
        numeric(-1,19);immutable();
        for(int cycle=0;cycle<2;++cycle) {
            timed("prepare",cycle,[&]{if(!head.prepare_prefill_weights(error))throw std::runtime_error(error);});
            timed("retire_readers",cycle,[&]{retire();});
            const void* old_canonical=storage();
            sb::validate_device_span(old_canonical,WeightBytes,*stream);
            timed("detach",cycle,[&]{head.detach_prefill_weights();});
            need(!head.loaded()&&storage()==nullptr,"head owner not detached");metadata();blocked("detached");
            retired_span(old_canonical);
            std::cout<<"RETIREMENT cycle="<<cycle<<" canonical_span=retired extra_weight_span=absent\n";
            {
                const size_t bytes=WeightBytes;
                sb::Allocation overwrite(rt,bytes,sb::MemoryKind::Device);
                check(cudaMemset(overwrite.data(),0x3c,bytes));rt->wait();
                std::array<uint8_t,128> tail{};
                check(cudaMemcpy(tail.data(),static_cast<uint8_t*>(overwrite.data())+bytes-tail.size(),tail.size(),cudaMemcpyDeviceToHost));
                need(std::all_of(tail.begin(),tail.end(),[](uint8_t v){return v==0x3c;}),"temporary reclaimed allocation tail");
                std::cout<<"RECLAIM cycle="<<cycle<<" overwritten_device_bytes="<<bytes<<" tail=PASS\n";
            }
            timed("restore",cycle,[&]{head.restore_prefill_weights();});blocked("restored_not_published");
            need(head.loaded(),"canonical allocation not restored");
            timed("resume",cycle,[&]{head.resume_prefill_weights();});
            need(head.decode_dpas_ready()==dpas,"DPAS owner not restored");
            capture();numeric(cycle,uint32_t(31+cycle));immutable();
        }
        need(numerical_cases==4&&blocked_cases==4&&launches==std::array<uint64_t,3>{4,4,4},"complete owner extent");
        need(partial_cases==(packed?8u:0u),"complete partial-column checks");
        std::cout<<"OWNER_CHECK dpas="<<int(dpas)<<" numeric_cases=4 blocked_cases=4 restore_cycles=2 cancel_cases=1 graph_replays=12 canonical_immutable=PASS extra_weights="
            <<(host_backed?"host-backing":"absent")<<" model=0 Verifier_owner=0\n";
        if(host_backed)std::cout<<"HOST_BACKING_CHECK passed=1 bytes="<<head.host_backing_bytes()
            <<" detach_D2H_bytes=0 restore_H2D_bytes="<<WeightBytes<<" restore_cycles=2 extra_device_weight_bytes=0\n";
        if(packed)std::cout<<"PACKED_OWNER partial_cases=8 widths=1,2,3,4 row_major_access=refused sole_weight_bytes="<<WeightBytes<<" model=0\n";
    }
    void timing(bool candidate_first) {
        need(packed,"only the sole-packed candidate has a new timing plan");
        constexpr int Replays=256,Samples=5;
        timespec resolution{};need(clock_getres(CLOCK_MONOTONIC,&resolution)==0,"timer resolution");
        std::vector<int64_t> overhead(1001);
        for(auto& value:overhead){const auto a=now(CLOCK_MONOTONIC);value=now(CLOCK_MONOTONIC)-a;}
        std::sort(overhead.begin(),overhead.end());
        const int64_t unit=std::max<int64_t>({1,overhead[overhead.size()/2],
            int64_t(resolution.tv_sec)*1000000000+resolution.tv_nsec});
        const int64_t floor=std::max<int64_t>(10000000,300*unit);
        std::cout<<"TIMER median_overhead_ns="<<overhead[overhead.size()/2]<<" unit_ns="<<unit<<" floor_ns="<<floor<<'\n';
        for(int arm:{0,1})for(int i=0;i<3;++i){check(cudaGraphLaunch(graphs[size_t(arm)],stream));drain();}
        for(int sample=0;sample<Samples;++sample)for(int index=0;index<2;++index) {
            const int arm=candidate_first?1-index:index;
            drain();const int64_t start=now(CLOCK_MONOTONIC);
            for(int i=0;i<Replays;++i)check(cudaGraphLaunch(graphs[size_t(arm)],stream));
            check(cudaStreamSynchronize(stream));const int64_t elapsed=now(CLOCK_MONOTONIC)-start;
            need(elapsed>=floor,"ineligible packed-head timing batch");
            std::cout<<"TIMING sample="<<sample<<" arm="<<arm<<" replays="<<Replays<<" elapsed_ns="<<elapsed
                <<" floor_ns="<<floor<<" scope=quantize-projection-graph-submit-complete\n";
        }
        immutable();std::cout<<"TIMING_DONE samples_per_arm=5 model_speed=unqualified\n";
    }
};
void host() {
    NativeHead h;std::string e;
    need(!h.has_host_backing()&&h.host_backing_bytes()==0,"empty owner has host backing");
    need(!h.prepare_prefill_weights(e),"empty prepare accepted");
    need(!h.run(nullptr,nullptr,nullptr,e),"empty projection accepted");
    bool refused=false;try{h.run_decode_c4(nullptr,nullptr,nullptr);}catch(const std::invalid_argument&){refused=true;}
    need(refused,"empty DPAS owner reached runtime");
    refused=false;try{h.run_packed_columns(nullptr,nullptr,1,nullptr);}catch(const std::invalid_argument&){refused=true;}
    need(refused,"empty packed owner reached runtime");
    std::cout<<"HOST owner_negative_cases=3 GPU=0\n";
    std::cout<<"PACKED_HOST owner_negative_cases=1 GPU=0\n";
}
} // namespace
int main(int argc,char** argv) {
    std::cout<<std::unitbuf;std::shared_ptr<sb::Runtime> keep;
    try {
        if(argc==2&&std::string(argv[1])=="--host"){host();std::_Exit(0);}
        if(argc==3&&std::string(argv[1])=="--reject-switch") {
            NativeHead h;std::string e;need(!h.load({"/nonexistent-q6-head-fixture"},K,N,e)&&
                e.find(argv[2])!=std::string::npos,"switch rejection before file/device access");
            std::cout<<"REJECT_SWITCH passed=1 GPU=0\n";std::_Exit(0);
        }
        const bool timing=argc>=4&&std::string(argv[1])=="--timing";
        need((argc>=3&&std::string(argv[1])=="--check") ||
            (timing&&(std::string(argv[2])=="rc"||std::string(argv[2])=="cr")),
            "use --host|--reject-switch substring|--check shards...|--timing rc|cr shards...");
        gate("STRATA_Q6_HEAD_OWNER_GATE");keep=sb::runtime_for();
        need(sb::Runtime::devices().size()==1&&keep->device().is_gpu()&&
            keep->device().get_info<sycl::info::device::name>().find("B570")!=std::string::npos,"actual B570");
        gate("STRATA_Q6_HEAD_OWNER_DEVICE_GATE");
        const std::vector<std::string> shards(argv+(timing?3:2),argv+argc);
        {Fixture fixture(keep,shards);fixture.run();if(timing)fixture.timing(std::string(argv[2])=="cr");}
        keep->wait();std::cout<<"HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified\n";
        std::cout.flush();std::cerr.flush();std::_Exit(0);
    }catch(const std::exception& e){std::cerr<<"head owner fixture failed: "<<e.what()<<'\n'<<std::flush;std::_Exit(2);}
}
