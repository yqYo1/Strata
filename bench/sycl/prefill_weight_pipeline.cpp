// Main-authored actual-weight scheduling component; repository LICENSE.
// Host USM/math/placement are fixed; production selectors are unchanged.
#include "strata/prefill/upload_event_lane.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "prefill/sycl/vllm_xe2/native16.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
namespace sb=strata::sycl_backend;
namespace xe=strata::experimental::vllm_xe2_native16;
namespace ops=strata::prefill::native16;
using Lane=strata::prefill::detail::PrefillUploadEventLane;
using Routes=strata::prefill::detail::MoeWindowRoutes;
constexpr std::size_t rows=1024,k=2560,ff=640,epochs=512;
constexpr std::size_t gate_bytes=524800,down_bytes=921600;
constexpr std::size_t payload=2*gate_bytes+down_bytes,edge=256;
using Clock=std::chrono::steady_clock;
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
double elapsed(Clock::time_point at) {return std::chrono::duration<double>(Clock::now()-at).count();}
void gate(const char* name) {
    std::cout<<"{\"kind\":\"gate\",\"name\":\""<<name<<"\"}"<<std::endl;
    std::string line;need(bool(std::getline(std::cin,line)) && line=="continue","Main gate refused");
}
template<class T> struct Buffer {
    sb::Allocation owner;
    std::size_t count,pad;
    Buffer(std::shared_ptr<sb::Runtime> r,std::size_t n,std::size_t p=64)
        :owner(r,(n+2*p)*sizeof(T),sb::MemoryKind::Device),count(n),pad(p) {}
    T* data() {return owner.as<T>()+pad;}
    std::size_t bytes() const {return (count+2*pad)*sizeof(T);}
};

int run(bool lookahead,bool metadata,const std::string& path) {
    // Exact two real layer2expert blobs; the committed extraction manifest pins
    // source roles/offsets and hashes. No synthetic packed weight implementation.
    std::vector<std::uint8_t> weights(2*payload);
    std::ifstream file(path,std::ios::binary);
    need(bool(file) && bool(file.read(reinterpret_cast<char*>(weights.data()),weights.size()))
         && file.peek()==std::char_traits<char>::eof(),"exact actual-weight bundle required");
    gate("before-runtime");
    auto runtime=sb::runtime_for();auto& q=runtime->compute();
    {
        // All owners precede the lane. Its checked destructor drains even when
        // an enqueue throws; no constructor below this line submits GPU work.
        std::array<sb::Allocation,2> source;
        sb::Allocation baseline(runtime,2*payload,sb::MemoryKind::Host);
        Buffer<std::uint8_t> bank(runtime,payload,edge);
        Buffer<std::uint16_t> x(runtime,rows*k),gu_weights(runtime,2*ff*k),
            down_weights(runtime,k*ff),gu(runtime,rows*2*ff),hidden(runtime,rows*ff),
            down(runtime,rows*k);
        Buffer<float> accumulator(runtime,rows*k),job_weights(runtime,rows*10);
        Buffer<std::int32_t> source_rows(runtime,rows*10),original(runtime,rows*10);
        Buffer<std::uint32_t> status(runtime,1);
        const auto largest=std::max(gu_weights.bytes(),accumulator.bytes());
        sb::Allocation host(runtime,largest,sb::MemoryKind::Host),
            golden(runtime,rows*k*4,sb::MemoryKind::Host);
        xe::Gemm dense(*runtime,q);
        Lane lane(q,[runtime] {runtime->check();},1024);
        std::array<Lane::Source,2> views{Lane::Source{nullptr,0},Lane::Source{nullptr,0}};
        std::memcpy(baseline.data(),weights.data(),weights.size());
        const auto prep=Clock::now();
        for(std::size_t e=0;e<2;++e) {
            source[e]=sb::Allocation(runtime,payload+2*edge,sb::MemoryKind::Host);
            auto* base=source[e].as<std::uint8_t>();
            std::memset(base,0xa5,payload+2*edge);
            std::memcpy(base+edge,weights.data()+e*payload,payload);
            views[e]=Lane::Source{base+edge,payload};
        }
        const double preparation_s=elapsed(prep);
        gate("runtime-ready");
        std::cout<<"{\"kind\":\"mapping-proof\",\"source\":\"usm\",\"property_type\":0,\"allocation_ids\":[0,0]}"<<std::endl;
        if(!metadata) {
            auto initialise=[&](auto& b) {runtime->wait(q.memset(b.owner.data(),0xa5,b.bytes()));};
            initialise(bank);initialise(x);initialise(gu_weights);initialise(down_weights);
            initialise(gu);initialise(hidden);initialise(down);initialise(accumulator);
            initialise(job_weights);initialise(source_rows);initialise(original);initialise(status);
            auto* hx=host.as<std::uint16_t>();
            for(std::size_t i=0;i<rows*k;++i) {
                const sycl::half value(float(int((i*13+i/k)%17)-8)/64);
                std::memcpy(hx+i,&value,2);
            }
            runtime->wait(q.memcpy(x.data(),hx,rows*k*2));
            std::vector<std::int32_t> ids(rows*10);
            for(std::size_t i=0;i<ids.size();++i) ids[i]=std::int32_t(i%10);
            const auto routes=Routes::build_accum(rows,ids.data(),ids.size());
            need(routes->jobs().size()==10 && routes->jobs()[0].rows==rows
                && routes->jobs()[1].rows==rows,"frozen production row mapping");
            std::memcpy(host.data(),routes->source_rows().data(),rows*10*4);
            runtime->wait(q.memcpy(source_rows.data(),host.data(),rows*10*4));
            std::memcpy(host.data(),routes->original_slots().data(),rows*10*4);
            runtime->wait(q.memcpy(original.data(),host.data(),rows*10*4));
            for(std::size_t i=0;i<rows*10;++i) host.as<float>()[i]=i<rows?.125f:.1875f;
            runtime->wait(q.memcpy(job_weights.data(),host.data(),rows*10*4));
            auto reset=[&] {
                runtime->wait(q.memset(accumulator.data(),0,rows*k*4));
                runtime->wait(q.memset(status.data(),0,4));
            };
            auto prefill_dequant=[&] {
                // Actual original production fallback for M1024 (<2334 fused
                // threshold): FP16 dequant, BF16 GU, FP16 hidden, BF16 Down,
                // FP32 routed accumulation. No storage/rounding/kernel change.
                strata::kernels::iq_dequant_gu_f16(22,bank.data(),bank.data()+gate_bytes,ff,k,gu_weights.data(),&q);
                strata::kernels::iq_dequant_f16(20,bank.data()+2*gate_bytes,k*ff,down_weights.data(),&q);
            };
            auto prefill_multiply=[&](std::size_t e) {
                // These operators read only the dequantized workspace,never
                // the quantized scratch bank. The next IQ pair stays behind
                // the current math on the same in-order compute queue.
                xe::Problem a{x.data(),gu_weights.data(),gu.data(),rows,2*ff,k,0,xe::Storage::fp16,xe::Storage::bf16,0};
                dense.beta0(a,xe::Tile::wide_128x128);
                ops::swiglu(gu.data(),hidden.data(),rows,&q);
                xe::Problem b{hidden.data(),down_weights.data(),down.data(),rows,k,ff,0,xe::Storage::fp16,xe::Storage::bf16,0};
                dense.beta0(b,xe::Tile::wide_128x128);
                ops::moe_accumulate(*routes,e,down.data(),source_rows.data(),original.data(),
                    job_weights.data(),accumulator.data(),status.data(),&q);
            };
            // Independent existing host-USM/SYCL-copy reference, outside timing.
            // The same per-expert math/order is compared; no FP32-oracle policy.
            reset();
            for(std::size_t i=0;i<epochs;++i) {
                q.memcpy(bank.data(),baseline.as<std::uint8_t>()+(i%2)*payload,payload);
                prefill_dequant();prefill_multiply(i%2);
            }
            runtime->wait(q.memcpy(golden.data(),accumulator.data(),rows*k*4));
            struct UploadCounts {std::size_t after_iq=0,after_math=0;};
            auto prefill_batch=[&](std::size_t count) {
                UploadCounts counts;
                const auto first=lane.copy_after({bank.data(),payload},views[0],payload,lane.readers_done());
                ++counts.after_math;lane.wait_ready(first);
                for(std::size_t i=0;i<count;++i) {
                    prefill_dequant();
                    std::optional<Lane::Ready> next;
                    if(lookahead && i+1<count) {
                        // Both IQ kernels are the last readers of this blob.
                        // This fence does not retire GU/Down workspace reads.
                        next=lane.copy_after({bank.data(),payload},views[(i+1)%2],payload,lane.readers_done());
                        ++counts.after_iq;
                    }
                    prefill_multiply(i%2);
                    if(!lookahead && i+1<count) {
                        next=lane.copy_after({bank.data(),payload},views[(i+1)%2],payload,lane.readers_done());
                        ++counts.after_math;
                    }
                    // Even lookahead cannot start the next IQ overwrite until
                    // current GU/Down/accumulate have consumed their workspace.
                    if(next) lane.wait_ready(*next);
                }
                return counts;
            };
            reset();prefill_batch(2);lane.drain(); // Source/event prewarm.
            gate("prewarm-ready");
            reset();runtime->wait(q.memset(bank.data(),0xe7,payload));
            gate("before-timing");
            double clock_max=0;
            for(int i=0;i<128;++i) {const auto a=Clock::now();clock_max=std::max(clock_max,elapsed(a));}
            const auto at=Clock::now();
            const auto counts=prefill_batch(epochs);
            lane.drain();const double wall_s=elapsed(at);
            runtime->wait(q.memcpy(host.data(),accumulator.data(),rows*k*4));
            need(std::memcmp(host.data(),golden.data(),rows*k*4)==0,"actual weight pipeline differs from serialized reference");
            for(std::size_t i=0;i<rows*k;++i) need(std::isfinite(host.as<float>()[i]),"nonfinite pipeline output");
            runtime->wait(q.memcpy(host.data(),status.data(),4));
            need(*host.as<std::uint32_t>()==0,"routed accumulator status");
            auto guards=[&](auto& b) {
                runtime->wait(q.memcpy(host.data(),b.owner.data(),b.bytes()));
                const auto* raw=host.as<std::uint8_t>();const auto p=b.pad*sizeof(*b.data());
                for(std::size_t i=0;i<p;++i) need(raw[i]==0xa5 && raw[b.bytes()-p+i]==0xa5,"device buffer edge changed");
            };
            guards(bank);guards(x);guards(gu_weights);guards(down_weights);guards(gu);
            guards(hidden);guards(down);guards(accumulator);guards(job_weights);
            guards(source_rows);guards(original);guards(status);
            runtime->wait(q.memcpy(host.data(),bank.data(),payload));
            need(std::memcmp(host.data(),weights.data()+payload,payload)==0,"final bank holds wrong expert");
            for(std::size_t e=0;e<2;++e) {
                need(std::memcmp(views[e].data,weights.data()+e*payload,payload)==0,"source weight payload changed");
                const auto* raw=static_cast<const std::uint8_t*>(views[e].data)-edge;
                for(std::size_t i=0;i<edge;++i) need(raw[i]==0xa5 && raw[edge+payload+i]==0xa5,"source weight edge changed");
            }
            std::cout.precision(17);
            std::cout<<"{\"kind\":\"batch\",\"arm\":\""<<(lookahead?"lookahead":"serial")<<"\",\"source\":\"usm\""
                     <<",\"rows\":1024,\"epochs\":512,\"requested_H2D_bytes\":"<<epochs*payload
                     <<",\"copies_after_iq\":"<<counts.after_iq<<",\"copies_after_math\":"<<counts.after_math
                     <<",\"wall_s\":"<<wall_s<<",\"timer_pair_max_s\":"<<clock_max
                     <<",\"actual_weight_reader\":true,\"full_bitwise_reference_and_guards\":true}"<<std::endl;
        }
        const auto retire=Clock::now();
        for(auto& s:source) s=sb::Allocation();
        const double retirement_s=elapsed(retire);
        std::cout.precision(17);
        std::cout<<"{\"kind\":\"source\",\"preparation_s\":"<<preparation_s<<",\"import_s\":0"
                 <<",\"retirement_s\":"<<retirement_s<<"}"<<std::endl;
    }
    runtime->wait();runtime->check();gate("before-exit");
    std::cout<<"{\"kind\":\"done\",\"passed\":true}"<<std::endl;return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {
        need(argc==3,"use metadata|serial|lookahead ACTUAL_WEIGHTS");
        const std::string mode=argv[1];
        need(mode=="metadata" || mode=="serial" || mode=="lookahead","explicit scheduling mode required");
        return run(mode=="lookahead",mode=="metadata",argv[2]);
    } catch(const std::exception& error) {std::cerr<<"prefill actual weight pipeline failure: "<<error.what()<<std::endl;return 1;}
}
