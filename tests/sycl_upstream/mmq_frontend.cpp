#include "strata/core/graph.hpp"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/sycl_upstream/mmq_adapter.hpp"
#include "upstream_cpu_dequant.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <vector>
namespace api = strata::prefill::mmq;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T* p = nullptr; size_t n; bool host;
    Buffer(size_t n, bool host = false): n(n), host(host) {
        ck(host ? cudaMallocHost(&p,n*sizeof(T)) : cudaMalloc(&p,n*sizeof(T)));
    }
    ~Buffer() { if (host) cudaFreeHost(p); else cudaFree(p); }
    size_t bytes() const { return n*sizeof(T); }
};
struct Stream {
    cudaStream_t s{};
    Stream() { ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); }
    ~Stream() {
        cudaStreamCaptureStatus status{};
        if (cudaStreamIsCapturing(s,&status)==cudaSuccess && status!=cudaStreamCaptureStatusNone) {
            cudaGraph_t g{}; cudaStreamEndCapture(s,&g); if (g) cudaGraphDestroy(g);
        }
        cudaStreamDestroy(s);
    }
};
struct Event {
    cudaEvent_t e{};
    Event() { ck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming)); }
    ~Event() { cudaEventDestroy(e); }
};
struct Kind { ggml_type type; const char* name; int qk, bytes; void (*decode)(const void*,float*,int64_t); };
#define KIND(T,N,Q) Kind{T,#N,Q,sizeof(block_##N), [](const void* p,float* f,int64_t n){ upstream_cpu::dequantize_row_##N(static_cast<const block_##N*>(p),f,n); }}
const Kind kinds[] = {
    KIND(GGML_TYPE_Q2_0,q2_0,QK2_0), KIND(GGML_TYPE_IQ2_XXS,iq2_xxs,QK_K),
    KIND(GGML_TYPE_IQ2_XS,iq2_xs,QK_K), KIND(GGML_TYPE_IQ2_S,iq2_s,QK_K),
    KIND(GGML_TYPE_IQ3_XXS,iq3_xxs,QK_K), KIND(GGML_TYPE_IQ3_S,iq3_s,QK_K),
    KIND(GGML_TYPE_IQ4_NL,iq4_nl,QK4_NL), KIND(GGML_TYPE_IQ4_XS,iq4_xs,QK_K),
    KIND(GGML_TYPE_Q8_0,q8_0,QK8_0)};
#undef KIND
int main(int argc, char** argv) try {
    if (argc>1) {
        if (!std::strcmp(argv[1],"--invalid-stream")) api::iota(nullptr,1,reinterpret_cast<void*>(0x1234));
        if (!std::strcmp(argv[1],"--sticky-error")) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(0x1234));
            Buffer<int32_t> out(1); api::iota(out.p,1,nullptr);
        }
        if (!std::strcmp(argv[1],"--cold-capture")) {
            Buffer<uint8_t> w(65536), xq(api::q8_bytes(5,1024));
            Buffer<int32_t> bounds(3),ids(5); Buffer<float> out(5*23);
            Stream stream; api::Context context;
            ck(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
            context.run({w.p,GGML_TYPE_Q8_0,19,1024,20672,2,xq.p,bounds.p,ids.p,5,3,out.p,23},stream.s);
        }
        throw std::runtime_error("expected fatal MMQ launch error did not terminate");
    }
    size_t products=0, guards=0, bytes_checked=0; double max_l1=0;
    uint32_t rng=0x17382981;
    auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
    for (const auto& kind:kinds) {
        constexpr int cols=1024, wr=19, rows=5, ld=23, experts=2;
        constexpr float sentinel=-1234567.f;
        const size_t row_bytes=cols/kind.qk*kind.bytes, eb=api::matrix_bytes(kind.type,wr,cols)+4096;
        Buffer<uint8_t> hw(experts*eb+4096,true), dw(hw.n), dq(api::q8_bytes(rows,cols));
        Buffer<float> hx(rows*cols,true), dx(hx.n), ho(rows*ld,true), out(ho.n);
        Buffer<int32_t> hb(3,true), db(3), hi(rows,true), di(rows), hg(rows,true), dg(rows);
        std::memset(hw.p,0,hw.bytes());
        for(int e=0;e<experts;++e) for(int r=0;r<wr;++r) for(int b=0;b<cols/kind.qk;++b) {
            auto* block=hw.p+e*eb+r*row_bytes+b*kind.bytes;
            for(int j=2;j<kind.bytes;++j)block[j]=uint8_t(next());
            const uint16_t scale=0x2800;std::memcpy(block,&scale,2);
        }
        auto input=[&](int run){
            for(int t=0;t<rows;++t) {
                hi.p[t]=(t+run)%rows;hg.p[t]=(rows-t+run)%rows;
                for(int k=0;k<cols;++k)hx.p[t*cols+k]=float(k%32==31?127:k%61-30)*std::ldexp(1.f,t+run%3);
            }
            hb.p[0]=0;hb.p[1]=2+run%2;hb.p[2]=rows;
            for(int e=0;e<experts;++e)for(int r=0;r<wr;++r)for(int b=0;b<cols/kind.qk;++b){
                const uint16_t scale=uint16_t(0x2800+(run%3)*0x400);std::memcpy(hw.p+e*eb+r*row_bytes+b*kind.bytes,&scale,2);
            }
        };
        auto copy_inputs=[&](cudaStream_t s){
            ck(cudaMemcpyAsync(dw.p,hw.p,hw.bytes(),cudaMemcpyHostToDevice,s));
            ck(cudaMemcpyAsync(dx.p,hx.p,hx.bytes(),cudaMemcpyHostToDevice,s));
            ck(cudaMemcpyAsync(db.p,hb.p,hb.bytes(),cudaMemcpyHostToDevice,s));
            ck(cudaMemcpyAsync(di.p,hi.p,hi.bytes(),cudaMemcpyHostToDevice,s));
            ck(cudaMemcpyAsync(dg.p,hg.p,hg.bytes(),cudaMemcpyHostToDevice,s));
        };
        api::Context context;
        api::Product p{dw.p,kind.type,wr,cols,eb,experts,dq.p,db.p,di.p,rows,3,out.p,ld};
        auto compute=[&](cudaStream_t s){api::quantize(dx.p,dg.p,dq.p,kind.type,cols,cols,rows,s);context.run(p,s);};
        auto result=[&](cudaStream_t s){ck(cudaMemcpyAsync(ho.p,out.p,ho.bytes(),cudaMemcpyDeviceToHost,s));};
        auto verify=[&](){
            std::vector<float> weight(cols);
            for(int e=0;e<experts;++e)for(int r=0;r<wr;++r){
                kind.decode(hw.p+e*eb+r*row_bytes,weight.data(),cols);
                for(int t=hb.p[e];t<hb.p[e+1];++t){
                    double ref=0,l1=0;for(int k=0;k<cols;++k){const double v=double(weight[k])*hx.p[hg.p[t]*cols+k];ref+=v;l1+=std::abs(v);}
                    const float got=ho.p[hi.p[t]*ld+r];const double error=std::abs(got-ref);
                    if(!std::isfinite(got)||error>3e-6*l1+1e-6){std::cerr<<kind.name<<" got="<<got<<" ref="<<ref<<'\n';throw std::runtime_error("frontend routed product");}
                    max_l1=std::max(max_l1,error/std::max(l1,1e-30));++products;
                }
            }
            for(int t=0;t<rows;++t)for(int r=wr;r<ld;++r){check(ho.p[t*ld+r]==sentinel,"output stride guard");++guards;}
        };
        input(0);std::fill(ho.p,ho.p+ho.n,sentinel);ck(cudaMemcpy(out.p,ho.p,ho.bytes(),cudaMemcpyHostToDevice));
        // Real default stream, followed by named streams; no raw SYCL queue
        // pointer enters any original public API in this test.
        copy_inputs(nullptr);compute(nullptr);result(nullptr);ck(cudaStreamSynchronize(nullptr));verify();
        Stream compute_stream,copy_stream;Event copied,used;
        copy_inputs(copy_stream.s);ck(cudaEventRecord(copied.e,copy_stream.s));
        ck(cudaStreamWaitEvent(compute_stream.s,copied.e));compute(compute_stream.s);
        ck(cudaStreamSynchronize(compute_stream.s)); // Warm this stream's scratch before capture.
        strata::core::CapturedGraph graph;std::string error;
        check(graph.begin(compute_stream.s,error),error.c_str());
        ck(cudaEventRecord(used.e,compute_stream.s));ck(cudaStreamWaitEvent(copy_stream.s,used.e));
        copy_inputs(copy_stream.s);ck(cudaEventRecord(copied.e,copy_stream.s));
        ck(cudaStreamWaitEvent(compute_stream.s,copied.e));compute(compute_stream.s);result(compute_stream.s);
        check(graph.end(compute_stream.s,error),error.c_str());
        check(graph.nodes()==13,"copy/quantize/main/fixup graph node count");
        for(int run=1;run<=8;++run){input(run);check(graph.launch(compute_stream.s,error),error.c_str());check(graph.wait_ms(10000),"MMQ graph timeout");verify();}
        if (kind.type==GGML_TYPE_Q8_0) {
            constexpr int runs=16;
            Buffer<float> twice(hx.n), concurrent(runs*out.n), received(concurrent.n,true);
            Buffer<uint8_t> qtwice(dq.n);
            for(size_t i=0;i<hx.n;++i)hx.p[i]*=2;
            ck(cudaMemcpy(twice.p,hx.p,hx.bytes(),cudaMemcpyHostToDevice));
            api::quantize(twice.p,dg.p,qtwice.p,kind.type,cols,cols,rows,compute_stream.s);
            ck(cudaStreamSynchronize(compute_stream.s));
            for(size_t i=0;i<hx.n;++i)hx.p[i]/=2;
            std::fill(received.p,received.p+received.n,sentinel);
            ck(cudaMemcpy(concurrent.p,received.p,received.bytes(),cudaMemcpyHostToDevice));
            std::promise<void> release;
            auto gate=release.get_future().share();
            ck(cudaLaunchHostFunc(compute_stream.s,[](void* p){static_cast<std::shared_future<void>*>(p)->wait();},&gate));
            auto submit=[&](int parity) {
                for(int run=parity;run<runs;run+=2) {
                    auto task=p;task.xq=parity?qtwice.p:dq.p;task.dst=concurrent.p+run*out.n;
                    context.run(task,compute_stream.s);
                }
            };
            auto first=std::async(std::launch::async,submit,0);
            auto second=std::async(std::launch::async,submit,1);
            const bool ready=first.wait_for(std::chrono::seconds(2))==std::future_status::ready &&
                second.wait_for(std::chrono::seconds(2))==std::future_status::ready;
            release.set_value();first.get();second.get();
            ck(cudaMemcpyAsync(received.p,concurrent.p,received.bytes(),cudaMemcpyDeviceToHost,compute_stream.s));
            ck(cudaStreamSynchronize(compute_stream.s));check(ready,"MMQ submission waited for preceding device work");
            for(int run=0;run<runs;++run)for(int t=0;t<rows;++t)for(int r=0;r<ld;++r) {
                const float expected=r<wr?ho.p[t*ld+r]*(run%2?2:1):sentinel;
                check(received.p[run*out.n+t*ld+r]==expected,"concurrent frontend scratch cross-use");
            }
            std::cout<<"concurrent_host_submissions=16 blocked_stream_nonblocking=1\n";
        }
        std::cout<<"format="<<kind.name<<" nodes="<<graph.nodes()<<" changed_replays=8\n";
    }
    {
        // Surrounding public wrappers inside the original graph manager.
        constexpr size_t half=288, down=160, upoff=320, downoff=640, gs=640, ds=192;
        Buffer<uint8_t> hb(4*1024,true), blob(hb.n), gu(4*gs), dn(4*ds), hgu(gu.n,true), hdn(dn.n,true);
        Buffer<float> hinput(4*32,true), input(hinput.n), sw(4*16), hsw(sw.n,true);
        Buffer<int32_t> ids(17), hids(ids.n,true);
        constexpr size_t ngu=1280*40, nd=2560*10, dcode=1280*640,
            gscale=dcode+2560*160, dscale=gscale+ngu*2, blobsize=dscale+nd*2;
        Buffer<uint8_t> hpacked(blobsize,true), packed(blobsize), qgu(ngu*18+32), qdn(nd*18+32),
            hqgu(qgu.n,true), hqdn(qdn.n,true);
        Stream stream;api::GatherGroup group;group.first=1;group.n=4;
        for(int e=0;e<4;++e)group.blob[e]=blob.p+e*1024;
        strata::core::CapturedGraph graph;std::string error;
        auto body=[&](){
            ck(cudaMemcpyAsync(blob.p,hb.p,hb.bytes(),cudaMemcpyHostToDevice,stream.s));
            ck(cudaMemsetAsync(gu.p,0xce,gu.bytes(),stream.s));ck(cudaMemsetAsync(dn.p,0xce,dn.bytes(),stream.s));
            api::gather_native(blob.p,blob.p+upoff,half,blob.p+downoff,down,gu.p,dn.p,stream.s);
            check(api::gather_native_group(group,upoff,half,downoff,down,gu.p,gs,dn.p,ds,stream.s),"group admission");
            ck(cudaMemcpyAsync(input.p,hinput.p,hinput.bytes(),cudaMemcpyHostToDevice,stream.s));
            api::swiglu(input.p,sw.p,4,16,false,stream.s);api::iota(ids.p,17,stream.s);
            ck(cudaMemcpyAsync(packed.p,hpacked.p,packed.bytes(),cudaMemcpyHostToDevice,stream.s));
            ck(cudaMemsetAsync(qgu.p,0xce,qgu.bytes(),stream.s));ck(cudaMemsetAsync(qdn.p,0xce,qdn.bytes(),stream.s));
            api::gather_strata_q2(packed.p,qgu.p,qdn.p,stream.s);
            ck(cudaMemcpyAsync(hgu.p,gu.p,gu.bytes(),cudaMemcpyDeviceToHost,stream.s));
            ck(cudaMemcpyAsync(hdn.p,dn.p,dn.bytes(),cudaMemcpyDeviceToHost,stream.s));
            ck(cudaMemcpyAsync(hsw.p,sw.p,sw.bytes(),cudaMemcpyDeviceToHost,stream.s));
            ck(cudaMemcpyAsync(hids.p,ids.p,ids.bytes(),cudaMemcpyDeviceToHost,stream.s));
            ck(cudaMemcpyAsync(hqgu.p,qgu.p,qgu.bytes(),cudaMemcpyDeviceToHost,stream.s));
            ck(cudaMemcpyAsync(hqdn.p,qdn.p,qdn.bytes(),cudaMemcpyDeviceToHost,stream.s));
        };
        check(graph.begin(stream.s,error),error.c_str());body();check(graph.end(stream.s,error),error.c_str());
        check(graph.nodes()==18,"stage wrapper added a completion kernel");
        for(int run=0;run<4;++run){
            for(size_t i=0;i<hb.n;++i)hb.p[i]=uint8_t(i*31+i/1024+run*17);
            for(size_t i=0;i<hinput.n;++i)hinput.p[i]=float(int(i%17)-8+run)/4;
            for(size_t i=0;i<hpacked.n;++i)hpacked.p[i]=uint8_t(i*31+i/1024+run*17);
            check(graph.launch(stream.s,error),error.c_str());check(graph.wait_ms(10000),"stage graph timeout");
            for(int e=0;e<4;++e){
                for(size_t i=0;i<gs;++i){const auto expected=i>=2*half?uint8_t(0xce):hb.p[e*1024+(i<half?i:upoff+i-half)];check(hgu.p[e*gs+i]==expected,"gather gate/up bytes");++bytes_checked;}
                for(size_t i=0;i<ds;++i){const auto expected=i>=down?uint8_t(0xce):hb.p[e*1024+downoff+i];check(hdn.p[e*ds+i]==expected,"gather down bytes");++bytes_checked;}
            }
            for(int r=0;r<4;++r)for(int k=0;k<16;++k){const double g=hinput.p[r*32+k],u=hinput.p[r*32+16+k];const double ref=g/(1+std::exp(-g))*u;check(std::abs(hsw.p[r*16+k]-ref)<2e-6,"frontend swiglu");}
            for(int i=0;i<17;++i)check(hids.p[i]==i,"frontend iota");
            for(int which=0;which<2;++which) {
                const auto& b=which?hqdn:hqgu;const size_t count=which?nd:ngu;
                for(size_t i=0;i<b.n;++i) {
                    const size_t block=i/18, offset=i%18;
                    const uint8_t expected=i>=count*18?uint8_t(0xce):
                        hpacked.p[offset<2?(which?dscale:gscale)+block*2+offset:
                            (which?dcode:0)+block*16+offset-2];
                    check(b.p[i]==expected,"Strata Q2 gather/guard");++bytes_checked;
                }
            }
        }
        api::Product empty;api::Context context;context.run(empty,reinterpret_cast<void*>(0x1234));
        api::quantize(nullptr,nullptr,nullptr,0,0,0,0,reinterpret_cast<void*>(0x1234));
        api::swiglu(nullptr,nullptr,0,0,false,reinterpret_cast<void*>(0x1234));api::iota(nullptr,0,reinterpret_cast<void*>(0x1234));
        api::GatherGroup invalid;
        check(!api::gather_native_group(invalid,0,16,32,16,nullptr,32,nullptr,16,reinterpret_cast<void*>(0x1234)),"empty group used stream");
    }
    std::cout<<"PASS frontend_formats=9 product_comparisons="<<products<<" stride_guards="<<guards
        <<" changed_product_replays=72 max_error_over_l1="<<max_l1<<" stage_bytes="<<bytes_checked<<'\n';
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
