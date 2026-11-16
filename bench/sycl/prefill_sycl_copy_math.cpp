// Main-authored fixed-layout marker comparison; repository LICENSE.
// Based only on owned e5cd358 fixture and695565fbd grouped-copy proposal.
// No selected engine path. Independent controls precede the2x2marker matrix.
#include "strata/prefill/sycl_copy_groups.hpp"
#include <array>
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
#include <optional>
#include <span>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
namespace sb=strata::sycl_backend;
namespace ops=strata::prefill::native16;
using Routes=strata::prefill::detail::MoeWindowRoutes;
using Groups=strata::prefill::detail::PrefillSyclCopyGroups;
struct PrefillPolicy {
    bool adapter,separate,shortcut;
    std::size_t slots,window,ready,readers;
};
PrefillPolicy prefill_policy(const std::string& name) {
    if(name=="direct") return {false,false,false,1,1,1,1};
    if(name=="shortcut") return {false,false,true,1,1,1,1};
    if(name=="layout") return {false,false,true,128,1,1,1};
    if(name=="bridge") return {true,false,false,128,1,1,1};
    if(name=="queue") return {true,true,false,128,1,1,1};
    if(name=="window") return {true,true,false,128,128,1,1};
    if(name=="ready64") return {true,true,false,128,128,64,1};
    if(name=="readers64") return {true,true,false,128,128,1,64};
    if(name=="both64") return {true,true,false,128,128,64,64};
    throw std::invalid_argument("explicit finite prefill copy policy required");
}
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
int run(const std::string& policy,bool metadata,std::size_t rows,const std::string& path) {
    const auto config=prefill_policy(policy);
    // POSIX file mapping matches the production ordinary-RAM source category.
    // All512real layer2experts form a1GBsource footprint,avoiding a two-expert
    // CPU-cache-hot copy loop. The frozen extraction recipe pins every role.
    ReadOnlyWeights weights(path);
    const auto primary_rows=rows/2,tail_rows=rows-primary_rows;
    gate("before-runtime");
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    {
        sb::Allocation baseline(runtime,source_bytes,sb::MemoryKind::Host);
        Buffer<std::uint8_t> bank(runtime,payload*config.slots,edge);
        Buffer<std::uint16_t> x(runtime,rows*k),gu_weights(runtime,2*ff*k),
            down_weights(runtime,k*ff),gu(runtime,rows*2*ff),hidden(runtime,rows*ff),
            down(runtime,rows*k);
        Buffer<float> accumulator(runtime,primary_rows*k),tail(runtime,tail_rows*k),job_weights(runtime,rows*10);
        Buffer<std::int32_t> source_rows(runtime,rows*10),original(runtime,rows*10);
        Buffer<std::uint32_t> status(runtime,1);
        const auto largest=std::max({gu_weights.bytes(),rows*k*4+2*edge,bank.bytes()});
        sb::Allocation host(runtime,largest,sb::MemoryKind::Host),
            golden(runtime,rows*k*4,sb::MemoryKind::Host);
        strata::prefill::Gemm dense;
        std::unique_ptr<Groups> copy_groups;
        // Last owner: drains before copy lane,math/weight buffers and mapping retire,
        // including any exception after an accepted asynchronous operation.
        PrefillSourceDrain drain{runtime};
        std::string error;
        if(!dense.init_external(&q,gu_weights.data(),2*ff*k,nullptr,0,error))
            throw std::runtime_error(error);
        std::memcpy(baseline.data(),weights.data(),source_bytes);
        const auto prep=Clock::now();
        if(config.adapter) copy_groups=std::make_unique<Groups>(runtime,bank.data(),
            Groups::Layout{payload,1,config.slots},
            config.separate?Groups::Placement::Separate:Groups::Placement::Compute,epochs+2);
        const double preparation_s=elapsed(prep);
        gate("runtime-ready");
        need(sycl::get_pointer_type(weights.data(),q.get_context())==sycl::usm::alloc::unknown,
             "source must remain an ordinary host file mapping");
        sb::validate_device_span(bank.data(),payload*config.slots,q);
        need(q.get_context()==runtime->context(),"copy queue must retain its Runtime context");
        std::cout<<"{\"kind\":\"source-proof\",\"ordinary_readonly_mapping\":true,\"host_slots\":0"
                 <<",\"policy\":\""<<policy<<"\",\"dependency_bridge\":"<<(config.adapter?"true":"false")
                 <<",\"separate_queue\":"<<(config.separate?"true":"false")
                 <<",\"physical_slots\":"<<config.slots<<",\"prefetch_window\":"<<config.window
                 <<",\"ready_width\":"<<config.ready<<",\"readers_width\":"<<config.readers
                 <<",\"reader_marker\":\""<<(config.adapter?"after-math":"none")
                 <<"\",\"runtime_check_and_observer_retained\":true}"<<std::endl;
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
            auto prefill_consume=[&](std::size_t e,const std::uint8_t* quantized) {
                strata::kernels::iq_dequant_gu_f16(22,quantized,quantized+gate_bytes,ff,k,gu_weights.data(),&q);
                strata::kernels::iq_dequant_f16(20,quantized+2*gate_bytes,k*ff,down_weights.data(),&q);
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
                prefill_consume(i%2,bank.data());
            }
            runtime->wait(q.memcpy(golden.data(),accumulator.data(),primary_rows*k*4));
            runtime->wait(q.memcpy(golden.as<float>()+primary_rows*k,tail.data(),tail_rows*k*4));
            std::vector<Groups::Ready> tickets; tickets.reserve(epochs);
            std::size_t upload_ordinal=0;
            std::size_t compat_calls=0,shortcut_calls=0,adapter_calls=0,ready_calls=0,reader_calls=0;
            auto prefill_upload=[&](std::size_t e) {
                const auto* input=weights.data()+e*payload;
                auto* destination=bank.data()+(upload_ordinal%config.slots)*payload;
                if(copy_groups) {
                    const std::array<std::span<const std::byte>,1> source{{{
                        reinterpret_cast<const std::byte*>(input),payload}}};
                    tickets.push_back(copy_groups->prefill_copy_group(source));
                    ++adapter_calls;
                } else if(config.shortcut) {
                    runtime->check();
                    auto event=q.memcpy(destination,input,payload);
                    sb::command_profile::observe(&q,event);
                    ++shortcut_calls;
                } else {
                    need(cudaMemcpyAsync(destination,input,payload,cudaMemcpyHostToDevice,&q)==cudaSuccess,
                         "ordinary mapped upload rejected");
                    ++compat_calls;
                }
                ++upload_ordinal; ++uploads;
            };
            auto prefill_schedule=[&](std::size_t count) {
                tickets.clear();
                const auto first_ordinal=upload_ordinal;
                const auto window=std::min(config.window,count);
                for(std::size_t i=0;i<window;++i) prefill_upload(i);
                for(std::size_t i=0;i<count;++i) {
                    if(copy_groups && i%config.ready==0) {
                        copy_groups->prefill_wait_ready(std::span<const Groups::Ready>(tickets)
                            .subspan(i,std::min(config.ready,count-i)));
                        ++ready_calls;
                    }
                    const auto* quantized=copy_groups
                        ?reinterpret_cast<const std::uint8_t*>(copy_groups->prefill_expert(tickets[i],0).data())
                        :bank.data()+((first_ordinal+i)%config.slots)*payload;
                    prefill_consume(i%2,quantized);
                    if((i+1)%config.readers==0 || i+1==count) {
                        const auto first=i-(i%config.readers);
                        if(copy_groups) {
                            copy_groups->prefill_record_readers(
                                std::span<const Groups::Ready>(tickets).subspan(first,i+1-first));
                            ++reader_calls;
                        }
                        for(std::size_t j=first;j<=i;++j)
                            if(j+window<count) prefill_upload(j+window);
                    }
                }
            };
            reset();prefill_schedule(2);
            runtime->wait();gate("prewarm-ready");
            reset();runtime->wait(q.memset(bank.data(),0xe7,payload*config.slots));gate("before-timing");
            gu_fused=down_fused=uploads=0;
            compat_calls=shortcut_calls=adapter_calls=ready_calls=reader_calls=0;
            double clock_max=0;
            for(int i=0;i<128;++i) {const auto at=Clock::now();clock_max=std::max(clock_max,elapsed(at));}
            const auto at=Clock::now();
            prefill_schedule(epochs);
            if(copy_groups) copy_groups->drain(); else runtime->wait();
            const double wall_s=elapsed(at);
            need(uploads==epochs,"actual source-copy accounting");
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
            runtime->wait(q.memcpy(host.data(),bank.data(),payload*config.slots));
            std::vector<std::size_t> final_expert(config.slots);
            for(std::size_t e=0;e<epochs;++e) final_expert[(e+2)%config.slots]=e;
            for(std::size_t slot=0;slot<config.slots;++slot)
                need(std::memcmp(host.as<std::uint8_t>()+slot*payload,
                    weights.data()+final_expert[slot]*payload,payload)==0,"final slot expert differs");
            need(std::memcmp(baseline.data(),weights.data(),source_bytes)==0,"source mapping changed");
            std::cout.precision(17);
            std::cout<<"{\"kind\":\"batch\",\"arm\":\""<<policy
                     <<"\",\"rows\":"<<rows<<",\"epochs\":512,\"requested_H2D_bytes\":"<<epochs*payload
                     <<",\"copies\":"<<uploads<<",\"gu_fused\":"<<gu_fused<<",\"down_fused\":"<<down_fused
                     <<",\"copy_compat_calls\":"<<compat_calls<<",\"copy_shortcut_calls\":"<<shortcut_calls
                     <<",\"copy_adapter_calls\":"<<adapter_calls<<",\"ready_barrier_calls\":"<<ready_calls
                     <<",\"reader_barrier_calls\":"<<reader_calls
                     <<",\"wall_s\":"<<wall_s<<",\"timer_pair_max_s\":"<<clock_max
                     <<",\"production_gemm_dispatch\":true,\"split_primary_rows\":"<<primary_rows
                     <<",\"actual_source_experts\":"<<source_experts<<",\"source_footprint_bytes\":"<<source_bytes
                     <<",\"split_tail_checked\":true"
                     <<",\"full_bitwise_reference_and_guards\":true}"<<std::endl;
        }
        const auto retire=Clock::now();copy_groups.reset();const double retirement_s=elapsed(retire);
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
        need(argc==5,"use metadata|batch direct|shortcut|layout|bridge|queue|window|ready64|readers64|both64 ROWS ACTUAL_WEIGHTS");
        const std::string mode=argv[1],source=argv[2];
        need(mode=="metadata" || mode=="batch","explicit source comparison mode required");
        (void)prefill_policy(source);
        const std::string value=argv[3];std::size_t consumed=0;
        const auto rows=std::stoull(value,&consumed);
        need(consumed==value.size() && (rows==16 || rows==128 || rows==640 || rows==1024 ||
             rows==1280 || rows==4096 || rows==8192),"finite source comparison shape required");
        return run(source,mode=="metadata",rows,argv[4]);
    } catch(const std::exception& e) {std::cerr<<"prefill fixed-slot math fixture failure: "<<e.what()<<std::endl;return 1;}
}
