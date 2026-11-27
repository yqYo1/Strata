// Main-authored canonical-layout C4 and C5 comparisons. Repository LICENSE.
// Reuse the numerical/oracle/graph harness from our f97893576704e6679c89ea6c4613d912665a0824.
// Canonical controls share one allocation. The packed C5 comparison uses
// separate buffers for the two layouts of the same Q6 weights.
#include "q6_dpas_component.hpp"
#include "strata/core/verify_window_policy.hpp"
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
#include <span>
#include <time.h>
#include <vector>
#include <unistd.h>

namespace {
namespace sb = strata::sycl_backend;
namespace cp = strata::kernels::component;
namespace k = strata::kernels;
namespace core = strata::core;
// Actual request5/C4 shapes in 4ac5c8af8:q6_packed_head_generation_speed_v1,
// processes[0].next_component_basis. Counts there are static capture calls,
// not elapsed time or physical traffic. This fixture keeps every shape separate.
struct Shape { int input, output; bool columns; };
constexpr std::array<Shape, 6> production_shapes{{
    {2560, 512, false}, {6144, 2560, true}, {2560, 6144, true},
    {2560, 10240, true}, {2560, 12288, true}, {2560, 248320, true}
}};
bool use_columns(int input, int output, int columns) {
    return core::verify_q6_columns(true, core::VerifyInput::decode,
        k::native_mmvq_multi_exact(), 14, input, output, columns);
}
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
void gate(const char* tag) {
    if (!std::getenv("STRATA_Q6_DPAS_GATE")) return;
    std::cout << tag << " pid=" << getpid() << '\n';
    std::string reply;
    need(bool(std::getline(std::cin,reply))&&reply=="continue","native gate");
}

void host() {
    int cases=0;
    need(k::native_mmvq_multi_exact(),"exact native activation layout required"); ++cases;
    for (const auto& s : production_shapes) {
        need(cp::q6_dpas_c4_shape(s.input,s.output,4),"eligible actual production shape"); ++cases;
        need(use_columns(s.input,s.output,4)==s.columns,"actual production reference policy"); ++cases;
        need(!core::verify_q6_columns(true,core::VerifyInput::prompt,true,14,s.input,s.output,4),
             "decode policy must not select a prompt route"); ++cases;
    }
    for (const auto s : {std::array<int,2>{256,1},{512,33},{2560,248320}}) {
        need(cp::q6_dpas_c4_shape(s[0],s[1],4),"eligible canonical shape"); ++cases;
    }
    alignas(2) uint8_t bad[4]{};
    bool unaligned=false;
    try { cp::q6_dpas_c4_canonical(bad+1,nullptr,nullptr,256,1,4,
                                  reinterpret_cast<void*>(uintptr_t(1))); }
    catch (const std::invalid_argument&) { unaligned=true; }
    need(unaligned,"unaligned canonical input rejected before runtime"); ++cases;
    for (const auto s : {std::array<int,3>{0,16,4},{255,16,4},{256,0,4},{256,16,1},{256,16,2},{256,16,8},{2147483392,2147483647,4}}) {
        need(!cp::q6_dpas_c4_shape(s[0],s[1],s[2]),"negative shape control"); ++cases;
    }
    bool rejected=false;
    try { cp::q6_dpas_c4_canonical(nullptr,nullptr,nullptr,256,16,1,nullptr); }
    catch (const std::invalid_argument&) { rejected=true; }
    need(rejected,"C1 rejected before runtime access"); ++cases;
    std::cout << "HOST passed=1 cases=" << cases << " GPU=0\n";
}

void host_c5() {
    int cases = 0;
    for (const auto& shape : production_shapes) {
        need(!use_columns(shape.input, shape.output, 5), "C5 compares with the production generic reference");
        ++cases;
    }
    alignas(2) uint8_t weights[4]{};
    auto queue = reinterpret_cast<void*>(uintptr_t(1));
    for (int width : {1, 4, 6}) {
        bool refused = false;
        try { cp::q6_dpas_c5_canonical(weights, nullptr, nullptr, 256, 1, width, queue); }
        catch (const std::invalid_argument&) { refused = true; }
        need(refused, "C5 rejects the wrong column count before runtime access");
        ++cases;
    }
    bool refused = false;
    try { cp::q6_dpas_c5_canonical(weights + 1, nullptr, nullptr, 256, 1, 5, queue); }
    catch (const std::invalid_argument&) { refused = true; }
    need(refused, "C5 rejects unaligned weights before runtime access");
    ++cases;
    std::cout << "HOST_C5 passed=1 cases=" << cases << " GPU=0\n";
}

void host_packed_c5() {
    need(cp::q6_dpas_packed_bytes(2560, 248320) == size_t(248320) * 10 * 210,
         "packed model head must have the same weight bytes as the canonical head");
    need(cp::q6_dpas_packed_bytes(256, 17) == 32 * 210, "partial output tile must have packed padding");
    alignas(2) uint16_t weights[2]{};
    const auto* unaligned = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const uint8_t*>(weights) + 1);
    auto queue = reinterpret_cast<void*>(uintptr_t(1));
    struct Invalid {
        const uint16_t* weights;
        int n, columns;
        void* stream;
    };
    const Invalid cases[]{
        {weights, 17, 4, queue}, {weights, 17, 6, queue}, {weights, 0, 5, queue},
        {weights, 17, 5, nullptr}, {unaligned, 17, 5, queue},
    };
    for (const auto& test : cases) {
        bool refused = false;
        try { cp::q6_dpas_c5_packed(test.weights, nullptr, nullptr, 256, test.n, test.columns, test.stream); }
        catch (const std::invalid_argument&) { refused = true; }
        need(refused, "packed C5 must reject an invalid launch before runtime access");
    }
    std::cout << "HOST_PACKED_C5 passed=1 cases=" << std::size(cases) + 2 << " GPU=0\n";
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
    std::shared_ptr<sb::Runtime> rt; int ni,no,columns;
    std::vector<uint8_t> w,x;
    Buffer<uint8_t> dw,dx;Buffer<float> yr,yc;
    std::vector<uint16_t> packed_weights;
    std::unique_ptr<Buffer<uint16_t>> packed_device;
    cudaStream_t stream=nullptr;cudaEvent_t completion=nullptr;
    std::array<cudaGraphExec_t,2> graphs{};uint64_t calls=0;
    Fixture(std::shared_ptr<sb::Runtime> r,int input,int output,int width,bool packed_c5=false):rt(r),ni(input),no(output),columns(width),
        w(size_t(ni/256)*no*210),x(size_t(ni/32)*36*columns),
        dw(r,w.size()),dx(r,x.size()),yr(r,size_t(no)*columns),yc(r,size_t(no)*columns) {
        if (packed_c5) {
            need(width == 5, "packed comparison requires five columns");
            packed_weights.resize(cp::q6_dpas_packed_bytes(ni, no) / sizeof(uint16_t));
            packed_device = std::make_unique<Buffer<uint16_t>>(r, packed_weights.size());
        }
        check(cudaStreamCreate(&stream));check(cudaEventCreateWithFlags(&completion,cudaEventDisableTiming));
        need(k::native_mmvq_multi_exact(),"production reference requires exact layout");
        fill(7);
        for(int arm=0;arm<2;++arm) {
            cudaGraph_t graph=nullptr;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            try {
                entry(arm,columns);check(cudaStreamEndCapture(stream,&graph));
                size_t nodes=0;check(cudaGraphGetNodes(graph,nullptr,&nodes));need(nodes==1,"exact one-node graph");
                check(cudaGraphInstantiate(&graphs[size_t(arm)],graph,0ull));check(cudaGraphDestroy(graph));
            } catch (...) {
                // A partial capture cannot cross source retirement. Terminate
                // the owned fixture instead of unwinding uncertain graph state.
                std::cerr << "Q6 capture construction failed\n" << std::flush;
                std::_Exit(3);
            }
        }
        const size_t packed_bytes = packed_weights.size() * sizeof(uint16_t);
        std::cout << "ADMISSION k=" << ni << " n=" << no << " canonical_bytes=" << w.size()
            << " packed_bytes=" << packed_bytes << " additional_candidate_weight_bytes=" << packed_bytes
            << " device_payload_bytes="
            << w.size()+packed_bytes+x.size()+size_t(no)*columns*2*sizeof(float)
            << " max_alloc_bytes=" << rt->device().get_info<sycl::info::device::max_mem_alloc_size>()
            << " graphs=2 nodes_each=1 reference=" << reference_route()
            << " scope=prequantized-" << (packed_device ? "packed-C" : "canonical-C") << columns << '\n' << std::flush;
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
    const char* reference_route() const {
        if (packed_device) return "canonical_c5";
        return use_columns(ni,no,columns) ? "columns" : "generic";
    }
    void entry(int arm,int width) {
        if (packed_device) {
            if (arm == 0) cp::q6_dpas_c5_canonical(dw.data(), dx.data(), yr.data(), ni, no, width, stream);
            else cp::q6_dpas_c5_packed(packed_device->data(), dx.data(), yc.data(), ni, no, width, stream);
        } else if(arm==0) {
            if(use_columns(ni,no,width))
                k::native_q6_k_columns_mmvq(dw.data(),dx.data(),yr.data(),ni,no,width,stream);
            else k::native_mmvq(14,dw.data(),dx.data(),yr.data(),ni,no,width,stream);
        } else if (columns == 5) {
            cp::q6_dpas_c5_canonical(dw.data(),dx.data(),yc.data(),ni,no,width,stream);
        } else {
            cp::q6_dpas_c4_canonical(dw.data(),dx.data(),yc.data(),ni,no,width,stream);
        }
    }
    void fill(uint32_t seed) {
        if(stream)drain();
        for(size_t i=0;i<w.size();++i)w[i]=uint8_t(mix(uint32_t(i)^seed));
        constexpr uint16_t ds[]{0x0c00,0x1000,0x1400,0x1800,0x0001,0x8000,0x8c00};
        for(size_t b=0;b<w.size()/210;++b)half_put(w,b*210+208,ds[(b+seed)%7]);
        for(size_t i=0;i<x.size();++i)x[i]=uint8_t(mix(uint32_t(i)+seed*17));
        for(size_t b=0;b<x.size()/36;++b) {half_put(x,b*36,uint16_t(0x1800+(b%3)*0x400));half_put(x,b*36+2,0x7bff);}
        const auto pack_start = packed_device ? stamp() : Stamp{};
        if (packed_device) cp::q6_dpas_pack(w, packed_weights, ni, no);
        const auto a=stamp();
        dw.put(w);dx.put(x);
        if (packed_device) packed_device->put(packed_weights);
        const auto b=stamp();
        std::cout << "SETUP k=" << ni << " n=" << no << " seed=" << seed
            << " pack_wall_ns=" << (packed_device ? a.wall - pack_start.wall : 0)
            << " pack_cpu_ns=" << (packed_device ? a.cpu - pack_start.cpu : 0)
            << " upload_wall_ns=" << b.wall-a.wall
            << " upload_cpu_ns=" << b.cpu-a.cpu << '\n' << std::flush;
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
            for(int c=0;c<columns;++c) {
                const size_t at=size_t(c)*no+row;
                need(std::isfinite(ref[at])&&std::isfinite(cand[at]),"finite Q6 results");
                const double err=std::abs(double(cand[at])-ref[at]);worst=std::max(worst,err);
                need(err<=bound,"Q6 forward envelope");
                if(selected) {
                    const auto expected=oracle(row,c);const double limit=gamma*expected.second+1e-12;
                    need(std::abs(double(cand[at])-expected.first)<=limit&&std::abs(double(ref[at])-expected.first)<=limit,"independent Q6/Q8 scalar oracle");
                }
            }
        }
        if(immutable)need(dw.get()==w&&dx.get()==x,"input changed");
        if(immutable&&packed_device)need(packed_device->get()==packed_weights,"packed input changed");
        std::cout << "NUMERIC passed=1 k=" << ni << " n=" << no << " columns=" << columns
            << " outputs=" << size_t(no)*columns
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
    std::cout << "DEVICE name=" << r->device().get_info<sycl::info::device::name>() << '\n';
    // Device discovery loads the forced adapter/driver. Stop before Fixture
    // allocations,copies or kernels so Main can attest those actual mappings.
    gate("STRATA_Q6_DPAS_DEVICE_GATE");return r;
}
void checks(bool small,int columns,std::shared_ptr<sb::Runtime>& keep,bool packed_c5=false) {
    auto r=runtime();keep=r;int cases=0;
    std::vector<std::array<int,2>> shapes;
    if(small)shapes={{256,1},{256,17},{512,33},{6144,33}};
    else if(packed_c5)shapes={{2560,248320}};
    else for(const auto& shape:production_shapes)shapes.push_back({shape.input,shape.output});
    for(auto shape:shapes) {Fixture f(r,shape[0],shape[1],columns,packed_c5);for(uint32_t seed:{7u,19u}){
        f.fill(seed);for(int arm=0;arm<2;++arm){f.launch(arm);f.wait();}f.verify(true);++cases;
    }if(!packed_c5)f.c1();}
    std::cout << "CHECK passed=1 small=" << int(small) << " cases=" << cases
        << " route=compat_graph columns=" << columns << '\n';
}
void timing(int process,int columns,std::shared_ptr<sb::Runtime>& keep,bool packed_c5=false) {
    auto r=runtime();keep=r;
    std::vector<int64_t> overhead;for(int i=0;i<1000;++i){const auto a=stamp(),b=stamp();overhead.push_back(b.wall-a.wall);}std::sort(overhead.begin(),overhead.end());
    const auto timer=overhead[500];std::cout << "CLOCK median_stamp_ns=" << timer << " floor_ns=10000000 process=" << process << '\n';
    int samples=0;
    std::span<const Shape> shapes(production_shapes);
    if(packed_c5)shapes=shapes.last(1);
    for(const auto& shape:shapes) {
        Fixture f(r,shape.input,shape.output,columns,packed_c5);for(int arm=0;arm<2;++arm){f.launch(arm);f.wait();}f.verify();
        const int batch=shape.output<=1024?4096:(shape.output==248320?256:1024);
        for(int at=0;at<2;++at) {const int arm=process==1?1-at:at;
            for(int sample=-2;sample<5;++sample) {
                const auto before=f.calls;const auto a=stamp();for(int i=0;i<batch;++i)f.launch(arm);const auto b=stamp();f.wait();const auto c=stamp();
                need(f.calls-before==uint64_t(batch)&&c.wall-a.wall>=std::max<int64_t>(10000000,300*timer),"timing eligibility");
                if(sample>=0){++samples;std::cout << "SAMPLE process=" << process << " arm=" << arm << " k=" << f.ni << " n=" << f.no
                    << " columns=" << columns << " sample=" << sample
                    << " launches=" << batch << " graph_launches=" << f.calls-before << " nodes_per_graph=1 route=compat_graph wall_ns=" << c.wall-a.wall
                    << " cpu_ns=" << c.cpu-a.cpu << " submit_wall_ns=" << b.wall-a.wall << " submit_cpu_ns=" << b.cpu-a.cpu
                    << " wait_wall_ns=" << c.wall-b.wall << " wait_cpu_ns=" << c.cpu-b.cpu
                    << " reference=" << f.reference_route() << '\n' << std::flush;}
            }
        }f.verify(true);
    }
    need(size_t(samples)==10*shapes.size(),"complete timing extent");
    std::cout << "TIMING passed=1 samples=" << samples << " shapes=" << shapes.size()
        << " model_speed=0 physical_traffic=0 cache=one-repeated-matrix\n";
}
} // namespace

int main(int argc,char** argv) {
    std::cout << std::unitbuf;
    try {
        need(argc==2||argc==3,"use --host, --check[-small], or --time 0..2, with optional -c5 or -packed-c5");
        const std::string mode=argv[1];
        if (mode=="--host" || mode=="--host-c5" || mode=="--host-packed-c5") {
            if (mode=="--host") host();
            else if (mode=="--host-c5") host_c5();
            else host_packed_c5();
            std::cout.flush();
            std::_Exit(0);
        }
        gate("STRATA_Q6_DPAS_GATE");
        std::shared_ptr<sb::Runtime> keep;
        if (mode=="--check-small" || mode=="--check") checks(mode=="--check-small",4,keep);
        else if (mode=="--check-c5-small" || mode=="--check-c5") checks(mode=="--check-c5-small",5,keep);
        else if (mode=="--check-packed-c5-small" || mode=="--check-packed-c5")
            checks(mode=="--check-packed-c5-small",5,keep,true);
        else {
            need((mode=="--time" || mode=="--time-c5" || mode=="--time-packed-c5") && argc==3 && std::strlen(argv[2])==1 &&
                 argv[2][0]>='0' && argv[2][0]<='2',"time process 0..2");
            timing(argv[2][0]-'0',mode=="--time"?4:5,keep,mode=="--time-packed-c5");
        }
        std::cout.flush();std::cerr.flush();std::_Exit(0); // qualified retained-loader exit;ordinary provider teardown remains unqualified
    } catch(const std::exception& e) {std::cerr << "Q6 DPAS component failed: " << e.what() << '\n' << std::flush;std::_Exit(2);}
}
