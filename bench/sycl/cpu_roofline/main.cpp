// Main-authored CPU calibration. Production math is linked, never copied.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int group = 6, reps = 5;
constexpr size_t guard = 64;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
double cpu_seconds() {
    rusage r{}; require(getrusage(RUSAGE_SELF, &r)==0, "getrusage");
    return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;
}
struct Buffer {
    uint8_t* base = nullptr; size_t bytes;
    explicit Buffer(size_t n) : bytes(n) {
        require(posix_memalign(reinterpret_cast<void**>(&base), 64, n+2*guard)==0, "allocation");
        std::memset(base, 0xA7, n+2*guard);
    }
    ~Buffer() { std::free(base); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    uint8_t* data() const { return base+guard; }
    void check() const {
        for (size_t i=0;i<guard;++i) require(base[i]==0xA7 && base[guard+bytes+i]==0xA7, "buffer canary");
    }
};
void pin(int core) {
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(core,&mask);
    require(sched_setaffinity(0,sizeof mask,&mask)==0, "CPU affinity");
}
void pread_exact(int fd, uint8_t* dest, size_t n, uint64_t off) {
    struct stat st{}; require(fstat(fd,&st)==0, "model stat");
    require(off<=static_cast<uint64_t>(st.st_size) && n<=static_cast<uint64_t>(st.st_size)-off, "model slice bounds");
    while (n) { const auto k=pread(fd,dest,n,static_cast<off_t>(off));
        require(k>0,"model slice read"); dest+=k; off+=k; n-=k; }
}
struct Inputs {
    cpu::NativeFmt fmt;
    std::vector<std::unique_ptr<Buffer>> blobs, outputs;
    std::array<std::unique_ptr<Buffer>,4> acts;
    std::array<std::array<float,cpu::H>,4> x;
    std::vector<cpu::ExpertJobMulti> jobs;
    int count;
    Inputs(const char* path, int gu, int dn, uint64_t go, uint64_t uo, uint64_t dno, size_t blob_bytes) {
        std::string err; require(cpu::native_fmt(gu,dn,cpu::H,cpu::FF,fmt,err),err.c_str());
        require(fmt.bytes==blob_bytes && fmt.gu_type!=42, "unsupported fixture geometry");
        count=group*static_cast<int>((((96ull<<20)+fmt.bytes-1)/fmt.bytes+group-1)/group);
        require(count<=96 && count*fmt.bytes>=96ull<<20, "stream extent");
        const int fd=open(path,O_RDONLY|O_CLOEXEC); require(fd>=0,"open model shard");
        for (int e=0;e<count;++e) {
            auto b=std::make_unique<Buffer>(fmt.bytes);
            pread_exact(fd,b->data(),fmt.up_off,go+e*fmt.up_off);
            pread_exact(fd,b->data()+fmt.up_off,fmt.up_off,uo+e*fmt.up_off);
            const auto down_bytes=fmt.bytes-fmt.down_off;
            pread_exact(fd,b->data()+fmt.down_off,down_bytes,dno+e*down_bytes);
            blobs.push_back(std::move(b)); outputs.push_back(std::make_unique<Buffer>(4*cpu::H*sizeof(float)));
        }
        close(fd);
        // A reproducible finite activation; token t retains its input at every NT.
        uint32_t state=20261020;
        for (int t=0;t<4;++t) {
            for (auto& value:x[t]) { state=1664525u*state+1013904223u; value=(int(state>>8)-8388608)*0x1p-23f; }
            acts[t]=std::make_unique<Buffer>(fmt.act_bytes);
        }
        jobs.resize(count);
        for (int e=0;e<count;++e) {
            jobs[e].blob=blobs[e]->data();
            for (int t=0;t<4;++t) {
                jobs[e].nact[t]=acts[t]->data();
                jobs[e].out[t]=reinterpret_cast<float*>(outputs[e]->data())+t*cpu::H;
            }
        }
    }
    void quant(int nt) { for (int t=0;t<nt;++t) cpu::native_quant_act(fmt,x[t].data(),acts[t]->data()); }
    void check(int n, int nt) {
        for (auto& b:blobs) b->check();
        for (auto& b:acts) b->check();
        for (int e=0;e<n;++e) {
            outputs[e]->check();
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::H;++r)
                require(std::isfinite(jobs[e].out[t][r]),"nonfinite result");
        }
    }
    // Outside timing: check GU and Down separately against ggml's same-input
    // vec_dot, then the complete production pool against the serial same route.
    std::array<double,4> validate(cpu::ExpertPool& pool, int nt) {
        quant(nt); for (auto& j:jobs) j.nt=nt;
        pool.run_split_multi_native(fmt,jobs.data(),group);
        double gu_error=0,down_error=0,pool_error=0,q2_ggml_gap=0;
        for (int e=0;e<group;++e) {
            Buffer ff(4*cpu::FF*sizeof(float)), out(4*cpu::H*sizeof(float));
            std::array<std::unique_ptr<Buffer>,4> hq;
            std::array<cpu::ActQ,4> a2{};
            const void* ap[4]; const void* hp[4]; const cpu::ActQ* a2p[4]; float* fp[4]; float* op[4];
            for (int t=0;t<nt;++t) {
                ap[t]=acts[t]->data(); fp[t]=reinterpret_cast<float*>(ff.data())+t*cpu::FF;
                op[t]=reinterpret_cast<float*>(out.data())+t*cpu::H;
                hq[t]=std::make_unique<Buffer>(fmt.h_bytes); hp[t]=hq[t]->data(); a2p[t]=&a2[t];
            }
            cpu::native_gu_rows(fmt,blobs[e]->data(),ap,nt,fp,0,cpu::FF);
            double gn=0,gd=0;
            const auto dotg=ggml_get_type_traits_cpu(static_cast<ggml_type>(fmt.gu_type))->vec_dot;
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::FF;++r) {
                float g=0,u=0;
                dotg(cpu::H,&g,0,blobs[e]->data()+r*fmt.gu_row,0,ap[t],0,1);
                dotg(cpu::H,&u,0,blobs[e]->data()+fmt.up_off+r*fmt.gu_row,0,ap[t],0,1);
                float want=(g/(1.f+std::exp(-g)))*u;
                require(std::isfinite(want),"nonfinite reference");
                gn+=std::abs(double(fp[t][r])-want); gd+=std::abs(double(want));
            }
            gu_error=std::max(gu_error,gn/(gd+1e-30));
            for (int t=0;t<nt;++t) {
                cpu::native_quant_h(fmt,fp[t],hq[t]->data());
                if (fmt.d_type==42) cpu::act_quant_any(fp[t],cpu::FF,a2[t]);
            }
            if (fmt.d_type==42) cpu::q2_rows_any(blobs[e]->data()+fmt.down_off,fmt.d_row,cpu::FF/64,a2p,nt,op,0,cpu::H);
            else cpu::native_down_rows(fmt,blobs[e]->data(),hp,nt,op,0,cpu::H);
            const auto dotd=ggml_get_type_traits_cpu(static_cast<ggml_type>(fmt.d_type))->vec_dot;
            double dn=0,dd=0,pn=0,pd=0,ggn=0,ggd=0;
            for (int t=0;t<nt;++t) for (int r=0;r<cpu::H;++r) {
                float want=0; dotd(cpu::FF,&want,0,blobs[e]->data()+fmt.down_off+r*fmt.d_row,0,hp[t],0,1);
                require(std::isfinite(want) && std::isfinite(op[t][r]),"nonfinite Down reference");
                double same_contract=want;
                if(fmt.d_type==42) {
                    // Q2_0's production ActQ retains FLOAT scales. GGML Q8_0
                    // stores HALF scales, so it is a different quantization
                    // contract. Independent scalar dot, same ActQ codes/scales;
                    // no vector unpack/reduction or production dot is reused.
                    same_contract=0;
                    const uint8_t* row=blobs[e]->data()+fmt.down_off+r*fmt.d_row;
                    for(int b=0;b<cpu::FF/64;++b) {
                        ggml_fp16_t wh;std::memcpy(&wh,row+b*18,2);
                        const double scale=ggml_fp16_to_fp32(wh);
                        for(int half=0;half<2;++half) {
                            int sum=0;
                            for(int j=0;j<32;++j) {
                                const int i=half*32+j;
                                const int code=((row[b*18+2+i/4]>>(2*(i%4)))&3)-1;
                                sum+=code*int(a2[t].q[b*64+i]);
                            }
                            same_contract+=scale*double(a2[t].scale[2*b+half])*sum;
                        }
                    }
                    ggn+=std::abs(double(op[t][r])-want);ggd+=std::abs(double(want));
                }
                dn+=std::abs(double(op[t][r])-same_contract); dd+=std::abs(same_contract);
                pn+=std::abs(double(jobs[e].out[t][r])-op[t][r]); pd+=std::abs(double(op[t][r]));
            }
            down_error=std::max(down_error,dn/(dd+1e-30)); pool_error=std::max(pool_error,pn/(pd+1e-30));
            q2_ggml_gap=std::max(q2_ggml_gap,ggn/(ggd+1e-30));
            ff.check(); out.check(); for(int t=0;t<nt;++t) hq[t]->check();
        }
        check(group,nt);
        if(!(gu_error<=1e-5 && down_error<=1e-5 && pool_error==0))
            std::cerr<<"GU="<<gu_error<<" Down="<<down_error<<" pool="<<pool_error<<" Q2/GGML-scale-gap="<<q2_ggml_gap<<std::endl;
        require(gu_error<=1e-5 && down_error<=1e-5 && pool_error==0, "numerical calibration gate");
        return {gu_error,down_error,pool_error,q2_ggml_gap};
    }
};
template<class F> void measure(F pass, uint64_t bytes, uint64_t visits, const std::string& fields, cpu::ExpertPool* pool=nullptr) {
    pass(); const auto pilot_start=Clock::now(); for(int i=0;i<2;++i) pass();
    uint64_t iterations=std::max<uint64_t>(1,static_cast<uint64_t>(std::ceil(1.3/(seconds(pilot_start)/2))));
    require(iterations<=1000000,"timing iteration budget");
    // Under-floor attempts stay in the record. Extend the entire cell once,
    // prospectively; failure after that is terminal, never silently discarded.
    for(int attempt=0;attempt<2;++attempt) {
        bool eligible=true;
        for(int r=0;r<reps;++r) {
            const double gu=pool?pool->ms_multi_gu:0, q=pool?pool->ms_multi_q:0, down=pool?pool->ms_multi_down:0;
            const auto c0=cpu_seconds(); const auto t0=Clock::now();
            for(uint64_t i=0;i<iterations;++i) pass();
            const double wall=seconds(t0), process_cpu=cpu_seconds()-c0; eligible &= wall>=1.;
            std::cout<<"{\"kind\":\"sample\","<<fields<<",\"attempt\":"<<attempt<<",\"rep\":"<<r
                <<",\"iterations\":"<<iterations<<",\"visits\":"<<visits*iterations
                <<",\"requested_bytes\":"<<bytes*iterations<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<process_cpu;
            if(pool) std::cout<<",\"gu_s\":"<<(pool->ms_multi_gu-gu)*.001<<",\"q_s\":"<<(pool->ms_multi_q-q)*.001<<",\"down_s\":"<<(pool->ms_multi_down-down)*.001;
            std::cout<<"}"<<std::endl;
        }
        if(eligible) return;
        iterations*=2;
    }
    throw std::runtime_error("timing floor failed after one whole-cell extension");
}
// Every loaded word contributes to a checksum. No stores to the measured
// buffer; workers persist across timed passes. This is a sequential-read
// reference, not physical DRAM traffic or a low-bit arithmetic upper bound.
__attribute__((noinline)) uint64_t scan(const uint64_t* p, size_t n) {
    __m256i v=_mm256_setzero_si256();
    for(size_t i=0;i<n;i+=4) v=_mm256_add_epi64(v,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)));
    alignas(32) uint64_t a[4]; _mm256_store_si256(reinterpret_cast<__m256i*>(a),v);
    return a[0]+a[1]+a[2]+a[3];
}
void ram_scan(const std::vector<int>& cores) {
    constexpr size_t words=(256ull<<20)/8;
    std::vector<uint64_t> data(words); uint64_t expected=0;
    for(size_t i=0;i<words;++i) { data[i]=i*6364136223846793005ull+1442695040888963407ull; expected+=data[i]; }
    std::barrier phase(static_cast<ptrdiff_t>(cores.size()));
    std::atomic<bool> stop{false}; std::vector<uint64_t> sums(cores.size()); std::vector<std::thread> workers;
    auto work=[&](size_t id) {
        pin(cores[id]);
        const size_t b=(words/4*id/cores.size())*4,e=(words/4*(id+1)/cores.size())*4;
        for(;;) { phase.arrive_and_wait(); if(stop.load()) break; sums[id]=scan(data.data()+b,e-b); phase.arrive_and_wait(); }
    };
    for(size_t i=1;i<cores.size();++i) workers.emplace_back(work,i);
    pin(cores[0]);
    auto pass=[&] {
        phase.arrive_and_wait(); const size_t end=(words/4/cores.size())*4;
        sums[0]=scan(data.data(),end); phase.arrive_and_wait();
        require(std::accumulate(sums.begin(),sums.end(),uint64_t(0))==expected,"RAM checksum");
    };
    try { measure(pass,data.size()*8,1,"\"mode\":\"scan\",\"threads\":"+std::to_string(cores.size())); }
    catch(...) { stop=true; phase.arrive_and_wait(); for(auto& t:workers)t.join(); throw; }
    stop=true; phase.arrive_and_wait(); for(auto& t:workers)t.join();
}
int main(int argc,char** argv) try {
    std::cout<<std::setprecision(12);
    const auto topo=cpu::detect_cpu_topology(true);
    require(topo.host_core>=0 && topo.worker_cores.size()>=5,"six physical CPUs required");
    if(argc==3 && std::string(argv[1])=="scan") {
        std::vector<int> ids; std::string s=argv[2]; size_t at=0;
        while(at<s.size()) { size_t end=s.find(',',at); ids.push_back(std::stoi(s.substr(at,end-at))); if(end==std::string::npos)break; at=end+1; }
        require(ids.size()==1 || ids.size()==6 || ids.size()==12,"scan thread count"); ram_scan(ids);
    } else {
        require(argc==11,"expected shard gu down gate_offset up_offset down_offset bytes workers check|measure");
        Inputs input(argv[1],std::stoi(argv[2]),std::stoi(argv[3]),std::stoull(argv[4]),std::stoull(argv[5]),std::stoull(argv[6]),std::stoull(argv[7]));
        const int workers=std::stoi(argv[8]); const bool host=workers==5;
        require(workers==1 || workers==5,"worker arm");
        const bool timing=std::string(argv[9])=="measure"; require(timing || std::string(argv[9])=="check","mode");
        require(std::string(argv[10])=="20261020","schedule version");
        cpu::ExpertPool pool(workers,true,host); pin(topo.host_core);
        std::cout<<"{\"kind\":\"placement\",\"workers\":"<<pool.workers()<<",\"host_works\":"<<(host?"true":"false")<<",\"host_cpu\":"<<topo.host_core<<",\"worker_cpus\":[";
        for(int i=0;i<workers;++i) std::cout<<(i?",":"")<<topo.worker_cores[i]; std::cout<<"]}"<<std::endl;
        for(int nt=1;nt<=4;++nt) {
            const auto v=input.validate(pool,nt);
            std::cout<<"{\"kind\":\"validation\",\"nt\":"<<nt<<",\"gu_rel_l1\":"<<v[0]<<",\"down_rel_l1\":"<<v[1]<<",\"pool_rel_l1\":"<<v[2]<<",\"q2_ggml_scale_gap\":"<<v[3]<<",\"finite_canaries\":true}"<<std::endl;
        }
        std::vector<std::pair<int,bool>> cells;
        for(int nt=1;nt<=4;++nt) for(bool stream:{false,true}) cells.emplace_back(nt,stream);
        std::mt19937 rng(20261020); std::shuffle(cells.begin(),cells.end(),rng);
        if(timing) for(auto [nt,stream]:cells) {
            const int n=stream?input.count:group;
            std::vector<int> order(n/group); std::iota(order.begin(),order.end(),0); std::shuffle(order.begin(),order.end(),rng);
            for(auto& j:input.jobs) j.nt=nt;
            auto pass=[&] {
                for(int i:order) { input.quant(nt); pool.run_split_multi_native(input.fmt,input.jobs.data()+i*group,group); }
            };
            measure(pass,input.fmt.bytes*n,n,"\"mode\":\""+std::string(stream?"stream":"hot")+"\",\"nt\":"+std::to_string(nt)+",\"working_set_bytes\":"+std::to_string(input.fmt.bytes*n),&pool);
            input.check(n,nt); input.validate(pool,nt);
        }
    }
    std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl; return 0;
} catch(const std::exception& e) { std::cerr<<"CPU calibration failed: "<<e.what()<<std::endl; return 1; }
