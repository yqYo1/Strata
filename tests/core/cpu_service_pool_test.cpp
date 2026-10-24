// Real pool publication/row partitioning with opt-in leaf attribution.
// Independently authored; repository LICENSE. No GPU, model-speed or quality claim.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_CPU_SERVICE_PROFILE
#error This fixture must link the real pool compiled with its diagnostic enabled
#endif

namespace cpu = strata::kernels::cpu;
namespace svc = cpu::service;
namespace {
constexpr int kH=2560, kF=640, kT=4, kBlobs=3;
static_assert(kH==cpu::H && kF==cpu::FF && kT<=cpu::MAXT);
constexpr float kGuard=-1234567.0f;
constexpr int kPairs[][2]={{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}};
using Output=std::array<std::vector<float>,kT>;
using References=std::array<std::array<Output,kT>,kBlobs>;

void require(bool ok,const char* why) {
    if (!ok) throw std::runtime_error(why);
}

// These formats have total integer codebooks and an initial finite FP16 scale
// in every block. Use linked GGML's geometry, without a quantizer/large capture.
std::vector<uint8_t> weights(const cpu::NativeFmt& f,uint32_t state) {
    std::vector<uint8_t> blob(f.bytes);
    auto next=[&] {state=state*747796405u+2891336453u;return state;};
    for (auto& b:blob) b=static_cast<uint8_t>(next()>>24);
    auto scales=[&](int type,int rows,int cols,size_t row,size_t offset) {
        const int block=static_cast<int>(ggml_blck_size(ggml_type(type)));
        const size_t bytes=ggml_type_size(ggml_type(type));
        require(cols%block==0 && row==size_t(cols/block)*bytes,"block geometry");
        for (int r=0;r<rows;++r) for (int b=0;b<cols/block;++b) {
            const uint16_t half=static_cast<uint16_t>(0x1800u|(next()&0x1ffu));
            std::memcpy(blob.data()+offset+size_t(r)*row+size_t(b)*bytes,&half,2);
        }
    };
    scales(f.gu_type,kF,kH,f.gu_row,0);
    scales(f.gu_type,kF,kH,f.gu_row,f.up_off);
    scales(f.d_type,kH,kF,f.d_row,f.down_off);
    return blob;
}

struct Inputs {
    std::array<std::array<uint8_t,cpu::kNativeActBytes>,kT> quant{};
    explicit Inputs(const cpu::NativeFmt& f) {
        std::array<float,kH> x{};
        for (int t=0;t<kT;++t) {
            for (int i=0;i<kH;++i) x[i]=t==3?0.f:float((i*29+t*17)%263-131)/131.f;
            cpu::native_quant_act(f,x.data(),quant[t].data());
        }
    }
};

Output reference(const cpu::NativeFmt& f,const std::vector<uint8_t>& blob,
                 const Inputs& input,int nt,cpu::NativeGuPhase phase) {
    std::array<std::array<float,kF>,kT> hidden{};
    std::array<std::array<uint8_t,cpu::kNativeHBytes>,kT> hq{};
    std::array<cpu::ActQ,kT> a2{};
    const void* a[kT]{};const void* h[kT]{};const cpu::ActQ* q[kT]{};
    float* ff[kT]{};float* out[kT]{};Output result;
    for (int t=0;t<nt;++t) {a[t]=input.quant[t].data();ff[t]=hidden[t].data();}
    // Same selected leaves, independent of pool partitioning and its scratch.
    // This is observer/pool parity, not an independent arithmetic oracle.
    cpu::native_gu_rows(f,blob.data(),a,nt,ff,0,kF,phase);
    for (int t=0;t<nt;++t) {
        result[t].resize(kH);out[t]=result[t].data();
        if (f.d_type==42) {cpu::act_quant_any(ff[t],kF,a2[t]);q[t]=&a2[t];}
        else {cpu::native_quant_h(f,ff[t],hq[t].data());h[t]=hq[t].data();}
    }
    if (f.d_type==42)
        cpu::q2_rows_any(blob.data()+f.down_off,f.d_row,kF/64,q,nt,out,0,kH);
    else cpu::native_down_rows(f,blob.data(),h,nt,out,0,kH);
    return result;
}

void counters(const svc::Snapshot& before,const svc::Snapshot& after,
              int format,int jobs,bool observed) {
    require(before.enabled==after.enabled && before.error_mask==0 && after.error_mask==0,
            "collector state/error");
    if (before.enabled) require(before.valid && after.valid,"invalid pool snapshot");
    svc::Counts bin{},thread{};
    for (unsigned cell=0;cell<svc::kCells;++cell) {
        svc::Counts d;require(svc::subtract(after.cells[cell],before.cells[cell],d),"cell underflow");
        const int fmt=cell/(kT*svc::kOps),nt=(cell/svc::kOps)%kT+1,op=cell%svc::kOps;
        const uint64_t count=observed && fmt==format ? (jobs+kT-nt)/kT : 0;
        const uint64_t rows=count*(op==2?kH:kF);
        require(d.rows==rows && d.token_rows==rows*nt,"format/width/operation row census");
        if (count) {
            require(d.calls>0 && d.cpu_ns>0 && d.wall_ns>=d.cpu_ns,"leaf timing/calls");
            if (op==1) require(d.calls==count,"one hidden quantization per expert");
        } else require(!d.calls && !d.wall_ns && !d.cpu_ns,"excluded work counted");
        require(svc::add(bin,d),"cell sum overflow");
    }
    for (unsigned i=0;i<before.threads.size();++i) {
        svc::Counts d;
        require(svc::subtract(after.threads[i].total,before.threads[i].total,d) &&
                svc::add(thread,d),"thread sum");
        if (d.calls) require(after.threads[i].tid!=0,"missing thread identity");
    }
    require(bin.calls==thread.calls && bin.rows==thread.rows && bin.token_rows==thread.token_rows &&
            bin.wall_ns==thread.wall_ns && bin.cpu_ns==thread.cpu_ns,"bin/thread conservation");
}

void run(cpu::ExpertPool& pool,bool enabled,uint64_t& checksum) {
    for (int fmt=0;fmt<7;++fmt) {
        require(svc::kTypes[fmt][0]==kPairs[fmt][0] && svc::kTypes[fmt][1]==kPairs[fmt][1],"format labels");
        cpu::NativeFmt f;std::string error;
        const bool supported=cpu::native_fmt(kPairs[fmt][0],kPairs[fmt][1],kH,kF,f,error);
        require(supported,error.c_str());
        const Inputs input(f);std::array<std::vector<uint8_t>,kBlobs> blobs;
        for (int b=0;b<kBlobs;++b) blobs[b]=weights(f,19u+uint32_t(7*fmt+b));
        for (const auto phase:{cpu::NativeGuPhase::Prefill,cpu::NativeGuPhase::Decode,
                               cpu::NativeGuPhase::Prefill,cpu::NativeGuPhase::Decode}) {
            References refs;
            for (int b=0;b<kBlobs;++b) for (int nt=1;nt<=kT;++nt)
                refs[b][nt-1]=reference(f,blobs[b],input,nt,phase);
            // One case exceeds the pool's96-expert scratch batch, with mixed NT.
            const int n=fmt==5 ? cpu::ExpertPool::kMaxSplitMulti+1 : 7;
            std::vector<cpu::ExpertJobMulti> jobs(n);std::vector<Output> output(n);
            for (int e=0;e<n;++e) {
                jobs[e].blob=blobs[e%kBlobs].data();jobs[e].nt=e%kT+1;
                for (int t=0;t<kT;++t) {
                    output[e][t].assign(kH+2,kGuard);
                    jobs[e].out[t]=output[e][t].data()+1;jobs[e].nact[t]=input.quant[t].data();
                }
            }
            const auto before=pool.service_snapshot();
            pool.run_split_multi_native(f,jobs.data(),n,phase);
            const auto after=pool.service_snapshot();
            counters(before,after,fmt,n,enabled && phase==cpu::NativeGuPhase::Decode);
            for (int e=0;e<n;++e) for (int t=0;t<kT;++t) {
                const auto& got=output[e][t];
                require(got.front()==kGuard && got.back()==kGuard,"output row canary");
                for (int r=0;r<kH;++r) {
                    const float want=t<jobs[e].nt ? refs[e%kBlobs][jobs[e].nt-1][t][r] : kGuard;
                    require(std::isfinite(got[r+1]) && std::bit_cast<uint32_t>(got[r+1])==
                            std::bit_cast<uint32_t>(want),"pool output/unused token differs");
                    const uint32_t bits=std::bit_cast<uint32_t>(got[r+1]);
                    for (unsigned shift=0;shift<32;shift+=8) {
                        checksum^=(bits>>shift)&255u;checksum*=1099511628211ull;
                    }
                }
            }
            pool.run_split_multi_native(f,nullptr,0,phase);
            counters(after,pool.service_snapshot(),fmt,0,false);
        }
    }
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2 && (!std::strcmp(argv[1],"0") || !std::strcmp(argv[1],"1")),"usage: cpu_service_pool_test 0|1");
        const bool enabled=argv[1][0]=='1';const char* setting=std::getenv("STRATA_CPU_SERVICE");
        require(setting && !std::strcmp(setting,argv[1]),"set observer before exec;no environment mutation");
        require(cpu::cpu_avx2_ok(),"production pool requires AVX2/FMA/F16C");
        cpu::NativeFmt initial;std::string format_error;
        const bool available=cpu::native_fmt(18,20,kH,kF,initial,format_error);
        require(available,format_error.c_str()); // initialize GGML before creating workers
        cpu::ExpertPool pool(5,false,true);
        require(pool.service_snapshot().enabled==enabled,"pool observer mode");
        uint64_t checksum=14695981039346656037ull;std::exception_ptr error;
        // One persistent dispatch host owns slot64; the generation thread reads
        // the final snapshot after its join. The separate host test covers a
        // generation reader blocked on a concurrent batch's publication mutex.
        std::thread dispatch([&] {try {run(pool,enabled,checksum);} catch (...) {error=std::current_exception();}});
        dispatch.join();if (error) std::rethrow_exception(error);
        const auto final=pool.service_snapshot();
        require(final.enabled==enabled && final.error_mask==0 && (!enabled || final.valid),"generation reader");
        std::printf("cpu_service_pool PASS observer=%d formats=7 nt=1..4 cases=28 checksum_fnv1a64_le=%016llx\n",
                    int(enabled),static_cast<unsigned long long>(checksum));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"cpu_service_pool: %s\n",error.what());return 1;
    }
}
