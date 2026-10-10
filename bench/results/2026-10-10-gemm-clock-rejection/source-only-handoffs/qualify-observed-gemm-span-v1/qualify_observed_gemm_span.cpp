// ROOT-RUN ONLY. Link the unchanged production Gemm object and DPCT/oneMKL.
// Output/completion qualification; PTI full-operation coverage is external.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/gemm.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

class ObservedGemmSpanBegin;
class ObservedGemmSpanEnd;
namespace {
constexpr size_t guard=256, memory_cap=128u*1024u*1024u;
constexpr uint64_t begin_tag=0x425547454d4d0001ull, end_tag=0x454e47454d4d0001ull;
constexpr uint16_t half_bits[4]={0x0000,0x3800,0x3c00,0x3400};
constexpr float values[4]={0.0f,0.5f,1.0f,0.25f};
struct Stamp { uint64_t submit,start,end; };
struct HostSubmit { uint64_t before,after; };
struct HostClockPair { uint64_t raw_before,monotonic,raw_after; };
uint64_t clock_ns(clockid_t id) {
    timespec t{}; if(clock_gettime(id,&t)!=0 || t.tv_sec<0 || (uint64_t)t.tv_sec>UINT64_MAX/1000000000ull || t.tv_nsec<0 || t.tv_nsec>=1000000000)
        throw std::runtime_error("host clock query failed");
    return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
}
uint64_t raw_ns(){return clock_ns(CLOCK_MONOTONIC_RAW);}
HostClockPair clock_pair(){const auto a=raw_ns();const auto m=clock_ns(CLOCK_MONOTONIC);const auto b=raw_ns();return {a,m,b};}
int parse_M(const char* s) {
    if(!s||!*s)throw std::runtime_error("M required");
    unsigned v=0;
    for(const char*p=s;*p;++p){if(p-s>=4||*p<'0'||*p>'9'||v>6495u/10u)throw std::runtime_error("M must be decimal1..6495, at most4digits");v=v*10u+(unsigned)(*p-'0');if(v>6495)throw std::runtime_error("M out of range");}
    if(!v)throw std::runtime_error("M out of range");
    return (int)v;
}
Stamp stamp(const sycl::event&e) {
    if(e.get_info<sycl::info::event::command_execution_status>()!=sycl::info::event_command_status::complete)
        throw std::runtime_error("event incomplete");
    Stamp s{e.get_profiling_info<sycl::info::event_profiling::command_submit>(),
            e.get_profiling_info<sycl::info::event_profiling::command_start>(),
            e.get_profiling_info<sycl::info::event_profiling::command_end>()};
    if(s.submit>s.start||s.start>s.end)throw std::runtime_error("backward event fields");
    return s;
}
uint16_t input_bits(size_t i,int cols){return half_bits[((i/(size_t)cols)+(i%(size_t)cols))%4];}
float expected(int t,int n,int K){float s=0;for(int k=0;k<4;++k)s+=values[(t+k)%4]*values[(n+k)%4];return s*(float)(K/4);}
// One reusable 4MiB host slab; no full host mirror of the largest output.
// Slab/guard/payload sizes are multiples of4, so a typed element never straddles.
void upload(sycl::queue&q,uint8_t*d,std::vector<uint8_t>&h,size_t bytes,int cols,bool output){
    const size_t unit=output?4:2;
    for(size_t off=0;off<bytes+2*guard;off+=h.size()){
        const size_t len=std::min(h.size(),bytes+2*guard-off);
        std::fill(h.begin(),h.begin()+len,0xA5);
        for(size_t j=0;j<len;j+=unit){const size_t at=off+j;if(at<guard||at>=guard+bytes)continue;
            if(output){const float v=-12345.0f;std::memcpy(h.data()+j,&v,4);}
            else{const uint16_t v=input_bits((at-guard)/2,cols);std::memcpy(h.data()+j,&v,2);}}
        q.memcpy(d+off,h.data(),len).wait_and_throw();
    }
}
uint64_t verify(sycl::queue&q,const uint8_t*d,std::vector<uint8_t>&h,size_t bytes,int cols,int K,bool output,const char*label){
    uint64_t hash=14695981039346656037ull;const size_t unit=output?4:2;
    for(size_t off=0;off<bytes+2*guard;off+=h.size()){
        const size_t len=std::min(h.size(),bytes+2*guard-off);q.memcpy(h.data(),d+off,len).wait_and_throw();
        for(size_t j=0;j<len;++j){const size_t at=off+j;hash=(hash^h[j])*1099511628211ull;
            if((at<guard||at>=guard+bytes)&&h[j]!=0xA5){std::fprintf(stderr,"guard mismatch %s byte=%zu\n",label,at);throw std::runtime_error("canary mismatch");}}
        for(size_t j=0;j<len;j+=unit){const size_t at=off+j;if(at<guard||at>=guard+bytes)continue;
            const size_t i=(at-guard)/unit;
            if(output){float got;std::memcpy(&got,h.data()+j,4);const float want=expected((int)(i/(size_t)cols),(int)(i%(size_t)cols),K);
                if(got!=want){std::fprintf(stderr,"first output mismatch row=%zu col=%zu got=%.9g expected=%.9g\n",i/(size_t)cols,i%(size_t)cols,(double)got,(double)want);throw std::runtime_error("full exact output mismatch");}}
            else{uint16_t got;std::memcpy(&got,h.data()+j,2);if(got!=input_bits(i,cols)){std::fprintf(stderr,"source changed %s element=%zu got=%04x\n",label,i,(unsigned)got);throw std::runtime_error("source whole-buffer mismatch");}}
        }
    }
    return hash;
}
void print_stamp(const char*label,const Stamp&s,const HostSubmit&h){
    std::printf("\"%s\":{\"submit_ns\":%llu,\"start_ns\":%llu,\"end_ns\":%llu,\"host_RAW_submit_before_ns\":%llu,\"host_RAW_submit_after_ns\":%llu}",label,
        (unsigned long long)s.submit,(unsigned long long)s.start,(unsigned long long)s.end,(unsigned long long)h.before,(unsigned long long)h.after);
}
}
int main(int argc,char**argv) try {
    // All argument/geometry/budget rejection precedes ANY GPU/runtime allocation.
    if(argc!=4 || (std::strcmp(argv[1],"GU")&&std::strcmp(argv[1],"Down")) || std::strcmp(argv[3],"--observed-shape"))
        throw std::runtime_error("usage: qualify_observed_gemm_span GU|Down M --observed-shape; root must verify observed M");
    const int M=parse_M(argv[2]);const bool gu=std::strcmp(argv[1],"GU")==0;
    const int N=gu?1280:2560,K=gu?2560:640,ldy=N;
    const size_t xb=(size_t)M*K*2,wb=(size_t)N*K*2,yb=(size_t)M*ldy*4;
    constexpr size_t hb=4u*1024u*1024u;
    const size_t owned_bytes=xb+wb+yb+6*guard+hb+2*sizeof(uint64_t)+2;
    if(owned_bytes>=memory_cap)throw std::runtime_error("owned buffer budget exceeded");
    // Existing full/transfer flags have no purpose here; prevent misleading mode.
    if(std::getenv("STRATA_PREFILL_TRANSFER_TIMING")||std::getenv("STRATA_PREFILL_SERVICE_TIMING"))
        throw std::runtime_error("full/transfer service flags must be absent");
    std::vector<uint8_t> h(hb);
    auto*q=dpct::get_current_device().create_in_order_queue(true); // Production DPCT factory/property path.
    if(!q->has_property<sycl::property::queue::in_order>()||!q->has_property<sycl::property::queue::enable_profiling>()||
       !q->get_device().has(sycl::aspect::queue_profiling))throw std::runtime_error("queue admission failed");
    if(q->get_device().get_info<sycl::info::device::name>()!="Intel(R) Arc(TM) B570 Graphics")throw std::runtime_error("B570 required");
    // Raw owners intentionally survive error paths until fatal _Exit. No
    // exception may unwind a Gemm or USM owner with completion unknown.
    auto*dx=static_cast<uint8_t*>(sycl::aligned_alloc_device(256,xb+2*guard,*q));
    auto*dw=static_cast<uint8_t*>(sycl::aligned_alloc_device(256,wb+2*guard,*q));
    auto*dy=static_cast<uint8_t*>(sycl::aligned_alloc_device(256,yb+2*guard,*q));
    auto*tags=sycl::malloc_device<uint64_t>(2,*q);
    auto*gm=new strata::prefill::Gemm;std::string err;
    if(!dx||!dw||!dy||!tags)throw std::runtime_error("allocation failed");
    if(!gm->init(q,1,err))throw std::runtime_error(err);
    upload(*q,dx,h,xb,K,false);upload(*q,dw,h,wb,K,false);upload(*q,dy,h,yb,ldy,true);
    q->memset(tags,0,2*sizeof(uint64_t));
    auto*X=reinterpret_cast<uint16_t*>(dx+guard);auto*W=reinterpret_cast<uint16_t*>(dw+guard);auto*Y=reinterpret_cast<float*>(dy+guard);
    sycl::event warm;
    if(!gm->f16_event(X,W,Y,M,N,K,ldy,warm))throw std::runtime_error("unsupported warm event path");
    q->wait_and_throw();q->throw_asynchronous(); // Warm/JIT fully outside target.
    upload(*q,dy,h,yb,ldy,true);
    const auto clock_before=clock_pair();
    HostSubmit bh{raw_ns(),0};
    const auto begin=q->parallel_for<ObservedGemmSpanBegin>(sycl::nd_range<1>{32,32},
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {if(it.get_local_linear_id()==0)tags[0]=begin_tag;});bh.after=raw_ns();
    HostSubmit gh{raw_ns(),0};sycl::event returned;
    if(!gm->f16_event(X,W,Y,M,N,K,ldy,returned))throw std::runtime_error("unsupported target event path");
    gh.after=raw_ns();
    HostSubmit eh{raw_ns(),0};
    const auto end=q->parallel_for<ObservedGemmSpanEnd>(sycl::nd_range<1>{32,32},
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {if(it.get_local_linear_id()==0)tags[1]=end_tag;});eh.after=raw_ns();
    const auto wait_before=raw_ns();end.wait_and_throw();q->throw_asynchronous();const auto wait_after=raw_ns();
    const auto clock_after=clock_pair();
    if(bh.before>bh.after||gh.before>gh.after||eh.before>eh.after||wait_before>wait_after||
       clock_before.raw_before>clock_before.raw_after||clock_after.raw_before>clock_after.raw_after||
       clock_before.raw_after>clock_after.raw_before||clock_before.monotonic>clock_after.monotonic)
        throw std::runtime_error("backward host clock fields");
    const Stamp bs=stamp(begin),gs=stamp(returned),es=stamp(end);
    if(bs.end>gs.end||gs.end>es.end)throw std::runtime_error("same-queue completion order invalid");
    uint64_t tag_h[2]{};q->memcpy(tag_h,tags,sizeof(tag_h)).wait_and_throw();
    if(tag_h[0]!=begin_tag||tag_h[1]!=end_tag)throw std::runtime_error("marker proof missing");
    const auto yh=verify(*q,dy,h,yb,ldy,K,true,"Y");
    const auto xh=verify(*q,dx,h,xb,K,K,false,"X");
    const auto wh=verify(*q,dw,h,wb,K,K,false,"W");
    q->wait_and_throw();q->throw_asynchronous();
    // All commands complete before release. Exceptions during teardown still
    // take fatal no-unwind exit; no attempted GPU recovery or replay.
    delete gm;sycl::free(tags,*q);sycl::free(dx,*q);sycl::free(dw,*q);sycl::free(dy,*q);
    dpct::get_current_device().destroy_queue(q);
    std::printf("{\"kind\":\"observed_gemm_span_fixture\",\"output_completion_qualified\":true,\"full_operation_span_qualified\":false,\"performance_eligible\":false,\"role\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ldy\":%d,\"observed_membership_controller_required\":true,\"input_type\":\"FP16\",\"output_type\":\"FP32\",\"target_calls\":1,\"warm_calls\":1,\"markers\":2,\"row_padding_values\":0,\"guard_bytes_each_end\":%zu,\"full_output_values_checked\":%zu,\"source_bytes_checked\":%zu,\"explicit_owned_buffer_bytes\":%zu,\"X_fnv1a64\":\"%016llx\",\"W_fnv1a64\":\"%016llx\",\"Y_fnv1a64\":\"%016llx\",",argv[1],M,N,K,ldy,guard,(size_t)M*N,xb+wb,owned_bytes,(unsigned long long)xh,(unsigned long long)wh,(unsigned long long)yh);
    print_stamp("begin_marker",bs,bh);std::printf(",");print_stamp("returned_event",gs,gh);std::printf(",");print_stamp("end_marker",es,eh);
    std::printf(",\"host_RAW_wait_before_ns\":%llu,\"host_RAW_wait_after_ns\":%llu,\"clock_pairs\":[{\"RAW_before_ns\":%llu,\"MONOTONIC_ns\":%llu,\"RAW_after_ns\":%llu},{\"RAW_before_ns\":%llu,\"MONOTONIC_ns\":%llu,\"RAW_after_ns\":%llu}],\"clock_mapping_proven\":false,\"device\":\"Intel Arc B570\",\"queue\":\"production_DPCT_in_order_profiling\"}\n",
        (unsigned long long)wait_before,(unsigned long long)wait_after,(unsigned long long)clock_before.raw_before,(unsigned long long)clock_before.monotonic,(unsigned long long)clock_before.raw_after,
        (unsigned long long)clock_after.raw_before,(unsigned long long)clock_after.monotonic,(unsigned long long)clock_after.raw_after);
    if(std::fflush(stdout)!=0)throw std::runtime_error("receipt write failed");
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"observed GEMM fixture FAIL: %s; no span qualification\n",e.what());std::fflush(stderr);std::_Exit(1);}
catch(...){std::fprintf(stderr,"observed GEMM fixture FAIL: unknown exception; no span qualification\n");std::fflush(stderr);std::_Exit(1);}
