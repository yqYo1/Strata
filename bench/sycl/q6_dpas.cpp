// Main-authored production-entry/graph comparison; repository LICENSE.
#include "q6_dpas_component.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <time.h>
#include <vector>
#include <unistd.h>

namespace {
namespace sb = strata::sycl_backend;
namespace cp = strata::kernels::component;
namespace k = strata::kernels;
void need(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
void check(cudaError_t e) { need(e == cudaSuccess, cudaGetErrorString(e)); }
int64_t clock_ns(clockid_t id) {
    timespec t{}; need(clock_gettime(id, &t) == 0, "clock");
    return int64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct Stamp { int64_t wall, cpu; };
Stamp stamp() { return {clock_ns(CLOCK_MONOTONIC), clock_ns(CLOCK_THREAD_CPUTIME_ID)}; }
uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
uint16_t half_at(const std::vector<uint8_t>& b, size_t p) { return uint16_t(b[p]) | uint16_t(uint16_t(b[p+1]) << 8); }
void half_put(std::vector<uint8_t>& b, size_t p, uint16_t h) { b[p]=uint8_t(h); b[p+1]=uint8_t(h >> 8); }
int signed_byte(uint8_t v) { return int(std::bit_cast<int8_t>(v)); }

void host() {
    int cases=0;
    for (int ni : {256, 2560}) for (int no : {1, 17}) {
        std::vector<uint8_t> w(size_t(ni/256)*no*210);
        for (size_t i=0;i<w.size();++i) w[i]=uint8_t(mix(uint32_t(i)));
        std::vector<uint16_t> packed(cp::q6_dpas_packed_bytes(ni,no)/2,0xdead);
        cp::q6_dpas_pack(w,packed,ni,no);
        const size_t blocks=size_t(ni/256);
        for (size_t r=0;r<((size_t(no)+15)/16)*16;++r)
            for (size_t b=0;b<blocks;++b) for (size_t j=0;j<105;++j) {
                uint16_t expected=0;
                if (r<size_t(no)) expected=half_at(w,(r*blocks+b)*210+2*j);
                need(packed[(r/16*blocks+b)*16*105+j*16+r%16]==expected,"pack byte/tail roundtrip");
            }
        ++cases;
    }
    for (const auto s : {std::array<int,3>{0,16,4},{255,16,4},{256,0,4},{256,16,1},{256,16,2},{256,16,8},{2147483392,2147483647,4}}) {
        need(!cp::q6_dpas_c4_shape(s[0],s[1],s[2]),"negative shape control"); ++cases;
    }
    bool rejected=false;
    try { cp::q6_dpas_c4(nullptr,nullptr,nullptr,256,16,1,nullptr); }
    catch (const std::invalid_argument&) { rejected=true; }
    need(rejected,"C1 rejected before runtime access"); ++cases;
    std::cout << "HOST passed=1 cases=" << cases << " GPU=0\n";
}

template<class T> T sentinel();
template<> uint8_t sentinel<uint8_t>() { return 0xa5; }
template<> uint16_t sentinel<uint16_t>() { return 0xa5a5; }
template<> float sentinel<float>() { return -731.25f; }
template<class T> struct Buffer {
    static constexpr size_t Guard=128/sizeof(T);
    std::shared_ptr<sb::Runtime> rt;
    sb::Allocation a; size_t count;
    Buffer(std::shared_ptr<sb::Runtime> r,size_t n):rt(r),a(r,(n+2*Guard)*sizeof(T),sb::MemoryKind::Device),count(n) {
        std::vector<T> fill(n+2*Guard,sentinel<T>());rt->wait(rt->compute().memcpy(a.data(),fill.data(),fill.size()*sizeof(T)));
    }
    T* data() { return a.template as<T>()+Guard; }
    void put(const std::vector<T>& x) { need(x.size()==count,"buffer input size");rt->wait(rt->compute().memcpy(data(),x.data(),count*sizeof(T))); }
    std::vector<T> get() {
        std::vector<T> x(count+2*Guard);rt->wait(rt->compute().memcpy(x.data(),a.data(),x.size()*sizeof(T)));
        for(size_t i=0;i<Guard;++i) need(x[i]==sentinel<T>()&&x[Guard+count+i]==sentinel<T>(),"allocation canary");
        return {x.begin()+Guard,x.begin()+Guard+count};
    }
};

struct Fixture {
    std::shared_ptr<sb::Runtime> rt; int ni,no;
    std::vector<uint8_t> w,x;std::vector<uint16_t> packed;
    Buffer<uint8_t> dw,dx;Buffer<uint16_t> dp;Buffer<float> yr,yc;
    cudaStream_t stream=nullptr;cudaEvent_t completion=nullptr;
    std::array<cudaGraphExec_t,2> graphs{};uint64_t calls=0;
    Fixture(std::shared_ptr<sb::Runtime> r,int input,int output):rt(r),ni(input),no(output),
        w(size_t(ni/256)*no*210),x(size_t(ni/32)*36*4),packed(cp::q6_dpas_packed_bytes(ni,no)/2),
        dw(r,w.size()),dx(r,x.size()),dp(r,packed.size()),yr(r,size_t(no)*4),yc(r,size_t(no)*4) {
        check(cudaStreamCreate(&stream));check(cudaEventCreateWithFlags(&completion,cudaEventDisableTiming));
        fill(7);
        for(int arm=0;arm<2;++arm) {
            cudaGraph_t graph=nullptr;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            try {
                entry(arm,4);check(cudaStreamEndCapture(stream,&graph));
                size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));need(nodes==1,"exact one-node graph");
                check(cudaGraphInstantiate(&graphs[size_t(arm)],graph,0ull));check(cudaGraphDestroy(graph));
            } catch (...) {
                // A partial capture cannot cross source retirement. Terminate
                // the owned fixture instead of unwinding uncertain graph state.
                std::cerr << "Q6 capture construction failed\n" << std::flush;
                std::_Exit(3);
            }
        }
        std::cout << "ADMISSION k=" << ni << " n=" << no << " canonical_bytes=" << w.size()
            << " packed_bytes=" << packed.size()*2 << " device_payload_bytes=" << w.size()+x.size()+packed.size()*2+size_t(no)*32
            << " max_alloc_bytes=" << rt->device().get_info<sycl::info::device::max_mem_alloc_size>()
            << " graphs=2 nodes_each=1\n" << std::flush;
    }
    ~Fixture() {
        // Never free graph storage or allocations while either queue owns work.
        if(stream && cudaStreamSynchronize(stream)!=cudaSuccess) std::_Exit(4);
        try { rt->wait(); } catch (...) { std::_Exit(4); }
        for(auto g:graphs) if(g&&cudaGraphExecDestroy(g)!=cudaSuccess)std::_Exit(4);
        if(completion&&cudaEventDestroy(completion)!=cudaSuccess)std::_Exit(4);
        if(stream&&cudaStreamDestroy(stream)!=cudaSuccess)std::_Exit(4);
    }
    void drain() { check(cudaStreamSynchronize(stream));rt->wait(); }
    void entry(int arm,int columns) {
        if(arm==0) k::native_q6_k_columns_mmvq(dw.data(),dx.data(),yr.data(),ni,no,columns,stream);
        else cp::q6_dpas_c4(dp.data(),dx.data(),yc.data(),ni,no,columns,stream);
    }
    void fill(uint32_t seed) {
        if(stream)drain();
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(mix(uint32_t(i)^seed));
        constexpr uint16_t ds[]{0x0c00,0x1000,0x1400,0x1800,0x0001,0x8000,0x8c00};
        for(size_t b=0;b<w.size()/210;++b)half_put(w,b*210+208,ds[(b+seed)%7]);
        for(size_t i=0;i<x.size();++i)x[i]=uint8_t(mix(uint32_t(i)+seed*17));
        for(size_t b=0;b<x.size()/36;++b) {half_put(x,b*36,uint16_t(0x1800+(b%3)*0x400));half_put(x,b*36+2,0x7bff);}
        const auto a=stamp();cp::q6_dpas_pack(w,packed,ni,no);const auto b=stamp();
        dw.put(w);dx.put(x);dp.put(packed);const auto c=stamp();
        std::cout << "SETUP k=" << ni << " n=" << no << " seed=" << seed << " pack_wall_ns=" << b.wall-a.wall
            << " pack_cpu_ns=" << b.cpu-a.cpu << " upload_wall_ns=" << c.wall-b.wall << " upload_cpu_ns=" << c.cpu-b.cpu << '\n' << std::flush;
    }
    void launch(int arm) {
        check(sb::compat::graph_launch_with_completion(graphs[size_t(arm)],stream,completion));++calls;
    }
    void wait() { check(cudaEventSynchronize(completion)); }
    // Independent byte-wise scalar dot and actual L1 envelope for selected rows.
    std::pair<double,double> oracle(int row,int column) const {
        double sum=0,l1=0;
        for(int j=0;j<ni;++j) {
            const size_t b=(size_t(row)*(ni/256)+size_t(j/256))*210;
            const int u=j%256,part=(u%128)/32,l=u%32;
            const unsigned q=(w[b+size_t((u/128)*64+(part%2)*32+l)]>>(4*(part/2)))&15;
            const unsigned h=(w[b+128+size_t((u/128)*32+l)]>>(2*part))&3;
            const auto a=(size_t(column)*(ni/32)+size_t(j/32))*36;
            const double value=double(int(q|(h<<4))-32)*signed_byte(w[b+192+size_t(u/16)])*
                k::f32_from_f16(half_at(w,b+208))*signed_byte(x[a+4+size_t(j%32)])*k::f32_from_f16(half_at(x,a));
            sum+=value;l1+=std::abs(value);
        }
        return {sum,l1};
    }
    double envelope(int row) const {
        double bound=0;
        for(int b=0;b<ni/256;++b) {
            const size_t base=(size_t(row)*(ni/256)+size_t(b))*210;
            int scales=0;for(int s=0;s<16;++s)scales+=std::abs(signed_byte(w[base+192+size_t(s)]));
            bound+=std::abs(double(k::f32_from_f16(half_at(w,base+208))))*scales*16*32*128/128;
        }
        return bound; // |dx|<=1/128, |qx|<=128, |qw|<=32,16 elements/scale
    }
    void verify(bool immutable=false) {
        drain();const auto ref=yr.get(),cand=yc.get();
        const double u=std::numeric_limits<float>::epsilon()/2;
        const double count=8.0*(ni/32)+64,gamma=count*u/(1-count*u);
        double worst=0;size_t oracle_rows=0;
        for(int row=0;row<no;++row) {
            const double bound=2*gamma*envelope(row)+1e-12;
            const bool selected=no<=1024||row<16||row>=no-16||row%std::max(1,no/512)==0;
            if(selected)++oracle_rows;
            for(int c=0;c<4;++c) {
                const size_t at=size_t(c)*no+row;
                need(std::isfinite(ref[at])&&std::isfinite(cand[at]),"finite Q6 results");
                const double err=std::abs(double(cand[at])-ref[at]);worst=std::max(worst,err);
                need(err<=bound,"DPAS/DP4A forward envelope");
                if(selected) {
                    const auto expected=oracle(row,c);const double limit=gamma*expected.second+1e-12;
                    need(std::abs(double(cand[at])-expected.first)<=limit&&std::abs(double(ref[at])-expected.first)<=limit,"independent Q6/Q8 scalar oracle");
                }
            }
        }
        if(immutable)need(dw.get()==w&&dx.get()==x&&dp.get()==packed,"input changed");
        std::cout << "NUMERIC passed=1 k=" << ni << " n=" << no << " columns=4 outputs=" << size_t(no)*4
            << " oracle_rows=" << oracle_rows << " max_abs_error=" << std::setprecision(12) << worst << " guards=1\n" << std::flush;
    }
    void c1() {
        drain();entry(0,1);check(cudaStreamSynchronize(stream));const auto ref=yr.get();
        for(int r=0;r<std::min(no,64);++r) {
            auto expected=oracle(r,0);const double u=std::numeric_limits<float>::epsilon()/2,ops=8.0*(ni/32)+64;
            need(std::abs(double(ref[size_t(r)])-expected.first)<=ops*u/(1-ops*u)*expected.second+1e-12,"C1 unchanged reference control");
        }
        bool refused=false;try{entry(1,1);}catch(const std::invalid_argument&){refused=true;}
        need(refused,"C1 must not launch DPAS");std::cout << "C1_CONTROL passed=1 dpas_launched=0\n";
    }
};

std::shared_ptr<sb::Runtime> runtime() {
    auto r=sb::runtime_for();need(sb::Runtime::devices().size()==1&&r->device().is_gpu()&&r->device().get_info<sycl::info::device::name>().find("B570")!=std::string::npos,"actual B570");
    std::cout << "DEVICE name=" << r->device().get_info<sycl::info::device::name>() << '\n';return r;
}
void checks(bool small,std::shared_ptr<sb::Runtime>& keep) {
    auto r=runtime();keep=r;int cases=0;
    const std::vector<std::array<int,2>> shapes=small?std::vector<std::array<int,2>>{{256,1},{256,17},{512,33}}:
        std::vector<std::array<int,2>>{{2560,640},{2560,248320}};
    for(auto shape:shapes) {Fixture f(r,shape[0],shape[1]);for(uint32_t seed:{7u,19u}){
        f.fill(seed);for(int arm=0;arm<2;++arm){f.launch(arm);f.wait();}f.verify(true);++cases;
    }f.c1();}
    std::cout << "CHECK passed=1 small=" << int(small) << " cases=" << cases << " route=compat_graph columns=4\n";
}
void timing(int process,std::shared_ptr<sb::Runtime>& keep) {
    auto r=runtime();keep=r;
    std::vector<int64_t> overhead;for(int i=0;i<1000;++i){const auto a=stamp(),b=stamp();overhead.push_back(b.wall-a.wall);}std::sort(overhead.begin(),overhead.end());
    const auto timer=overhead[500];std::cout << "CLOCK median_stamp_ns=" << timer << " floor_ns=10000000 process=" << process << '\n';
    int samples=0;
    for(int n:{640,248320}) {
        Fixture f(r,2560,n);for(int arm=0;arm<2;++arm){f.launch(arm);f.wait();}f.verify();
        const int batch=n==640?4096:256;
        for(int at=0;at<2;++at) {const int arm=process==1?1-at:at;
            for(int sample=-2;sample<5;++sample) {
                const auto before=f.calls;const auto a=stamp();for(int i=0;i<batch;++i)f.launch(arm);const auto b=stamp();f.wait();const auto c=stamp();
                need(f.calls-before==uint64_t(batch)&&c.wall-a.wall>=std::max<int64_t>(10000000,300*timer),"timing eligibility");
                if(sample>=0){++samples;std::cout << "SAMPLE process=" << process << " arm=" << arm << " k=2560 n=" << n << " columns=4 sample=" << sample
                    << " launches=" << batch << " graph_launches=" << f.calls-before << " nodes_per_graph=1 route=compat_graph wall_ns=" << c.wall-a.wall
                    << " cpu_ns=" << c.cpu-a.cpu << " submit_wall_ns=" << b.wall-a.wall << " submit_cpu_ns=" << b.cpu-a.cpu
                    << " wait_wall_ns=" << c.wall-b.wall << " wait_cpu_ns=" << c.cpu-b.cpu << '\n' << std::flush;}
            }
        }f.verify(true);
    }
    need(samples==20,"complete timing extent");std::cout << "TIMING passed=1 samples=20 model_speed=0 physical_traffic=0\n";
}
} // namespace

int main(int argc,char** argv) {
    std::cout << std::unitbuf;
    try {
        need(argc==2||argc==3,"use --host|--check-small|--check|--time 0..2");
        const std::string mode=argv[1];if(mode=="--host"){host();std::cout.flush();std::_Exit(0);}
        if(std::getenv("STRATA_Q6_DPAS_GATE")) {
            std::cout << "STRATA_Q6_DPAS_GATE pid=" << getpid() << '\n';std::string reply;need(bool(std::getline(std::cin,reply))&&reply=="continue","native gate");
        }
        std::shared_ptr<sb::Runtime> keep;
        if(mode=="--check-small"||mode=="--check")checks(mode=="--check-small",keep);
        else {need(mode=="--time"&&argc==3&&std::strlen(argv[2])==1&&argv[2][0]>='0'&&argv[2][0]<='2',"time process0..2");timing(argv[2][0]-'0',keep);}
        std::cout.flush();std::cerr.flush();std::_Exit(0); // qualified retained-loader exit;ordinary provider teardown remains unqualified
    } catch(const std::exception& e) {std::cerr << "Q6 DPAS component failed: " << e.what() << '\n' << std::flush;std::_Exit(2);}
}
