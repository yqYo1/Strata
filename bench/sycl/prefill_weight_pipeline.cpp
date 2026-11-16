// Main-authored production-adapter source comparison; repository LICENSE.
// Source unit S only. Queue/order/math fixed; no production selector changes.
#include "strata/prefill/expert_upload_staging.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/prefill/moe_window_routes.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
namespace sb=strata::sycl_backend;
namespace ops=strata::prefill::native16;
using Routes=strata::prefill::detail::MoeWindowRoutes;
using Staging=strata::prefill::detail::ExpertUploadStaging;
constexpr std::size_t k=2560,ff=640,epochs=512;
constexpr std::size_t gate_bytes=524800,down_bytes=921600;
constexpr std::size_t payload=2*gate_bytes+down_bytes,edge=256;
constexpr std::size_t source_experts=512,source_bytes=source_experts*payload;
using Clock=std::chrono::steady_clock;
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
double elapsed(Clock::time_point at) {return std::chrono::duration<double>(Clock::now()-at).count();}
void gate(const char* name) {
    std::cout<<"{\"kind\":\"gate\",\"name\":\""<<name<<"\"}"<<std::endl;
    std::string line;need(bool(std::getline(std::cin,line)) && line=="continue","Main gate refused");
}
struct ReadOnlyWeights {
    void* pointer=MAP_FAILED;
    explicit ReadOnlyWeights(const std::string& path) {
        const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
        need(fd>=0,"actual-weight source open");
        struct stat st{};
        const bool valid=::fstat(fd,&st)==0 && S_ISREG(st.st_mode) &&
            st.st_size>=0 && std::uint64_t(st.st_size)==source_bytes;
        if(valid) pointer=::mmap(nullptr,source_bytes,PROT_READ,MAP_PRIVATE,fd,0);
        ::close(fd);
        need(valid && pointer!=MAP_FAILED,"exact read-only actual-weight mapping required");
    }
    ~ReadOnlyWeights() {if(pointer!=MAP_FAILED) ::munmap(pointer,source_bytes);}
    ReadOnlyWeights(const ReadOnlyWeights&)=delete;
    ReadOnlyWeights& operator=(const ReadOnlyWeights&)=delete;
    const std::uint8_t* data() const {return static_cast<const std::uint8_t*>(pointer);}
};
template<class T> struct Buffer {
    sb::Allocation owner;
    std::size_t count,pad;
    Buffer(std::shared_ptr<sb::Runtime> r,std::size_t n,std::size_t p=64)
        :owner(r,(n+2*p)*sizeof(T),sb::MemoryKind::Device),count(n),pad(p) {}
    T* data() {return owner.as<T>()+pad;}
    std::size_t bytes() const {return (count+2*pad)*sizeof(T);}
};
struct PrefillSourceDrain {
    std::shared_ptr<sb::Runtime> runtime;
    ~PrefillSourceDrain() noexcept {
        sb::checked_cleanup([&] {runtime->wait();runtime->check();},[] {},
                            "prefill source fixture retirement");
    }
};
int run(bool staged,bool metadata,std::size_t rows,const std::string& path) {
    // POSIX file mapping matches the production ordinary-RAM source category.
    // All512real layer2experts form a1GBsource footprint,avoiding a two-expert
    // CPU-cache-hot copy loop. The frozen extraction recipe pins every role.
    ReadOnlyWeights weights(path);
    const auto primary_rows=rows/2,tail_rows=rows-primary_rows;
    gate("before-runtime");
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    {
        sb::Allocation baseline(runtime,source_bytes,sb::MemoryKind::Host);
        Buffer<std::uint8_t> bank(runtime,payload,edge);
        Buffer<std::uint16_t> x(runtime,rows*k),gu_weights(runtime,2*ff*k),
            down_weights(runtime,k*ff),gu(runtime,rows*2*ff),hidden(runtime,rows*ff),
            down(runtime,rows*k);
        Buffer<float> accumulator(runtime,primary_rows*k),tail(runtime,tail_rows*k),job_weights(runtime,rows*10);
        Buffer<std::int32_t> source_rows(runtime,rows*10),original(runtime,rows*10);
        Buffer<std::uint32_t> status(runtime,1);
        const auto largest=std::max(gu_weights.bytes(),rows*k*4+2*edge);
        sb::Allocation host(runtime,largest,sb::MemoryKind::Host),
            golden(runtime,rows*k*4,sb::MemoryKind::Host);
        strata::prefill::Gemm dense;
        std::unique_ptr<Staging> staging;
        // Last owner: drains before staging,math/workspace and mapping retire,
        // including any exception after an accepted asynchronous operation.
        PrefillSourceDrain drain{runtime};
        std::string error;
        if(!dense.init_external(&q,gu_weights.data(),2*ff*k,nullptr,0,error))
            throw std::runtime_error(error);
        std::memcpy(baseline.data(),weights.data(),source_bytes);
        const auto prep=Clock::now();
        if(staged) staging=std::make_unique<Staging>(bank.data(),payload,payload,&q);
        const double preparation_s=elapsed(prep);
        gate("runtime-ready");
        need(sycl::get_pointer_type(weights.data(),q.get_context())==sycl::usm::alloc::unknown,
             "source must remain an ordinary host file mapping");
        std::cout<<"{\"kind\":\"source-proof\",\"ordinary_readonly_mapping\":true,\"staged\":"
                 <<(staged?"true":"false")<<",\"host_slots\":"<<(staged?4:0)<<"}"<<std::endl;
        if(!metadata) {
            auto initialise=[&](auto& b) {runtime->wait(q.memset(b.owner.data(),0xa5,b.bytes()));};
            initialise(bank);initialise(x);initialise(gu_weights);initialise(down_weights);
            initialise(gu);initialise(hidden);initialise(down);initialise(accumulator);
            initialise(job_weights);initialise(source_rows);initialise(original);initialise(status);initialise(tail);
            for(std::size_t i=0;i<rows*k;++i) {
                const sycl::half value(float(int((i*13+i/k)%17)-8)/64);
                std::memcpy(host.as<std::uint16_t>()+i,&value,2);
            }
            runtime->wait(q.memcpy(x.data(),host.data(),rows*k*2));
            std::vector<std::int32_t> ids(rows*10);
            for(std::size_t i=0;i<ids.size();++i) ids[i]=std::int32_t(i%10);
            // Actual split-mode entrypoints with both planes exercised. This
            // bounded fixture is not physical262144/model-state qualification.
            const auto routes=Routes::build_split_accum(rows,ids.data(),ids.size());
            need(routes->jobs().size()==10 && routes->jobs()[0].rows==int64_t(rows)
                && routes->jobs()[1].rows==int64_t(rows),"frozen production row mapping");
            std::memcpy(host.data(),routes->source_rows().data(),rows*10*4);
            runtime->wait(q.memcpy(source_rows.data(),host.data(),rows*10*4));
            std::memcpy(host.data(),routes->original_slots().data(),rows*10*4);
            runtime->wait(q.memcpy(original.data(),host.data(),rows*10*4));
            for(std::size_t i=0;i<rows*10;++i) host.as<float>()[i]=i<rows?.125f:.1875f;
            runtime->wait(q.memcpy(job_weights.data(),host.data(),rows*10*4));
            auto reset=[&] {
                runtime->wait(q.memset(accumulator.data(),0,primary_rows*k*4));
                runtime->wait(q.memset(tail.data(),0,tail_rows*k*4));
                runtime->wait(q.memset(status.data(),0,4));
            };
            std::size_t gu_fused=0,down_fused=0,uploads=0;
            auto prefill_consume=[&](std::size_t e) {
                strata::kernels::iq_dequant_gu_f16(22,bank.data(),bank.data()+gate_bytes,ff,k,gu_weights.data(),&q);
                strata::kernels::iq_dequant_f16(20,bank.data()+2*gate_bytes,k*ff,down_weights.data(),&q);
                const bool fused_gu=dense.try_gu_swiglu16(x.data(),gu_weights.data(),hidden.data(),rows);
                if(!fused_gu) {
                    dense.f16_out16(x.data(),gu_weights.data(),gu.data(),rows,2*ff,k,true);
                    ops::swiglu(gu.data(),hidden.data(),rows,&q);
                } else ++gu_fused;
                const bool fused_down=dense.try_down_scatter16_split(*routes,e,hidden.data(),down_weights.data(),
                    source_rows.data(),original.data(),job_weights.data(),accumulator.data(),tail.data(),primary_rows,status.data());
                if(!fused_down) {
                    dense.f16_out16(hidden.data(),down_weights.data(),down.data(),rows,k,ff,true);
                    ops::moe_accumulate_split(*routes,e,down.data(),source_rows.data(),original.data(),
                        job_weights.data(),accumulator.data(),tail.data(),primary_rows,status.data(),&q);
                } else ++down_fused;
            };
            reset();
            for(std::size_t i=0;i<epochs;++i) {
                q.memcpy(bank.data(),baseline.as<std::uint8_t>()+i*payload,payload);
                prefill_consume(i%2);
            }
            runtime->wait(q.memcpy(golden.data(),accumulator.data(),primary_rows*k*4));
            runtime->wait(q.memcpy(golden.as<float>()+primary_rows*k,tail.data(),tail_rows*k*4));
            auto prefill_upload=[&](std::size_t e) {
                const auto* input=weights.data()+e*payload;
                if(staged) need(staging->enqueue(bank.data(),input,payload,&q),"staged upload rejected");
                else need(cudaMemcpyAsync(bank.data(),input,payload,cudaMemcpyHostToDevice,&q)==cudaSuccess,
                          "ordinary mapped upload rejected");
                ++uploads;
            };
            reset();
            for(std::size_t i=0;i<2;++i) {prefill_upload(i);prefill_consume(i);}
            runtime->wait();gate("prewarm-ready");
            reset();runtime->wait(q.memset(bank.data(),0xe7,payload));gate("before-timing");
            gu_fused=down_fused=uploads=0;
            double clock_max=0;
            for(int i=0;i<128;++i) {const auto at=Clock::now();clock_max=std::max(clock_max,elapsed(at));}
            const auto at=Clock::now();
            for(std::size_t i=0;i<epochs;++i) {prefill_upload(i);prefill_consume(i%2);}
            runtime->wait();const double wall_s=elapsed(at);
            need(uploads==epochs && (!staged || (staging->copies()==epochs+2 &&
                staging->copied_bytes()==(epochs+2)*payload)),"actual source-copy accounting");
            runtime->wait(q.memcpy(host.data(),accumulator.data(),primary_rows*k*4));
            runtime->wait(q.memcpy(host.as<float>()+primary_rows*k,tail.data(),tail_rows*k*4));
            need(std::memcmp(host.data(),golden.data(),rows*k*4)==0,"source adapter output differs");
            for(std::size_t i=0;i<rows*k;++i) need(std::isfinite(host.as<float>()[i]),"nonfinite source output");
            runtime->wait(q.memcpy(host.data(),status.data(),4));need(*host.as<std::uint32_t>()==0,"route status");
            auto guards=[&](auto& b) {
                runtime->wait(q.memcpy(host.data(),b.owner.data(),b.bytes()));
                const auto* raw=host.as<std::uint8_t>();const auto n=b.pad*sizeof(*b.data());
                for(std::size_t i=0;i<n;++i) need(raw[i]==0xa5 && raw[b.bytes()-n+i]==0xa5,"device edge changed");
            };
            guards(bank);guards(x);guards(gu_weights);guards(down_weights);guards(gu);
            guards(hidden);guards(down);guards(accumulator);guards(job_weights);
            guards(source_rows);guards(original);guards(status);guards(tail);
            runtime->wait(q.memcpy(host.data(),bank.data(),payload));
            need(std::memcmp(host.data(),weights.data()+(source_experts-1)*payload,payload)==0,"final bank expert differs");
            need(std::memcmp(baseline.data(),weights.data(),source_bytes)==0,"source mapping changed");
            if(staged) staging->check_guards();
            std::cout.precision(17);
            std::cout<<"{\"kind\":\"batch\",\"arm\":\""<<(staged?"staged":"direct")
                     <<"\",\"rows\":"<<rows<<",\"epochs\":512,\"requested_H2D_bytes\":"<<epochs*payload
                     <<",\"copies\":"<<uploads<<",\"gu_fused\":"<<gu_fused<<",\"down_fused\":"<<down_fused
                     <<",\"wall_s\":"<<wall_s<<",\"timer_pair_max_s\":"<<clock_max
                     <<",\"production_gemm_dispatch\":true,\"split_primary_rows\":"<<primary_rows
                     <<",\"actual_source_experts\":"<<source_experts<<",\"source_footprint_bytes\":"<<source_bytes
                     <<",\"split_tail_checked\":true"
                     <<",\"full_bitwise_reference_and_guards\":true}"<<std::endl;
        }
        const auto retire=Clock::now();staging.reset();const double retirement_s=elapsed(retire);
        std::cout.precision(17);
        std::cout<<"{\"kind\":\"source\",\"preparation_s\":"<<preparation_s
                 <<",\"retirement_s\":"<<retirement_s<<"}"<<std::endl;
    }
    runtime->wait();runtime->check();gate("before-exit");
    std::cout<<"{\"kind\":\"done\",\"passed\":true}"<<std::endl;return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {
        need(argc==5,"use metadata|batch direct|staged ROWS ACTUAL_WEIGHTS");
        const std::string mode=argv[1],source=argv[2];
        need(mode=="metadata" || mode=="batch","explicit source comparison mode required");
        need(source=="direct" || source=="staged","explicit source adapter required");
        const std::string value=argv[3];std::size_t consumed=0;
        const auto rows=std::stoull(value,&consumed);
        need(consumed==value.size() && (rows==16 || rows==128 || rows==640 || rows==1024 ||
             rows==1280 || rows==4096 || rows==8192),"finite source comparison shape required");
        return run(source=="staged",mode=="metadata",rows,argv[4]);
    } catch(const std::exception& e) {std::cerr<<"prefill source adapter failure: "<<e.what()<<std::endl;return 1;}
}
