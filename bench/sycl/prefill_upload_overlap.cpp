// Main-authored E2proposal; repository LICENSE. Not built/admitted until E1.
// Isolates one-expert lookahead with the existing production Xe2 dense kernel.
// Copy primitive, payload, consumer math and allocation are identical in arms.
#include "strata/prefill/mapped_host_source.hpp"
#include "prefill_host_pages.hpp"
#include "strata/sycl/runtime.hpp"
#include "prefill/sycl/vllm_xe2/native16.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace {
using Lane = strata::prefill::detail::PrefillUploadEventLane;
using Mapped = strata::prefill::detail::PrefillMappedHostSource;
using Pages = strata::prefill::test_detail::PrefillHostPages;
namespace gemm = strata::experimental::vllm_xe2_native16;
namespace backend = strata::sycl_backend;
constexpr std::size_t k = 2560, n = 1280, payload = 1971200, guard = 256;
void need(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
void gate(const char* name) {
    std::cout << "{\"kind\":\"gate\",\"name\":\"" << name << "\"}" << std::endl;
    std::string line;need(bool(std::getline(std::cin,line)) && line=="continue","Main gate refused");
}
std::uint16_t half_bits(float value) {
    const sycl::half h(value);std::uint16_t result;std::memcpy(&result,&h,2);return result;
}
int run(bool overlap,std::size_t rows,std::size_t iterations,bool mapped) {
    need((rows==1024 || rows==4096) && iterations==512,"frozen E2shape/batch required");
    gate("before-runtime");
    auto runtime=backend::runtime_for();
    auto& queue=runtime->compute();
    {
        std::unique_ptr<Pages> ordinary_source;
        const auto xb=rows*k*2,wb=n*k*2,yb=rows*n*2;
        // Registered production Allocation owners; their original Runtime and
        // compute queue survive both the transfer lane and dense adapter.
        backend::Allocation hx(runtime,xb,backend::MemoryKind::Host),
            hw(runtime,wb,backend::MemoryKind::Host),hy(runtime,yb,backend::MemoryKind::Host),
            hb,
            readback(runtime,payload+2*guard,backend::MemoryKind::Host),
            x(runtime,xb,backend::MemoryKind::Device),w(runtime,wb,backend::MemoryKind::Device),
            y(runtime,yb,backend::MemoryKind::Device),bank(runtime,payload+2*guard,backend::MemoryKind::Device);
        Lane lane(queue,[runtime] {runtime->check();},1024);
        gemm::Gemm dense(*runtime,queue);
        gate("runtime-ready");
        for(std::size_t i=0;i<rows;++i)
            for(std::size_t j=0;j<k;++j) hx.as<std::uint16_t>()[i*k+j]=half_bits(float(i%5+1)/32);
        for(std::size_t i=0;i<n;++i)
            for(std::size_t j=0;j<k;++j) hw.as<std::uint16_t>()[i*k+j]=half_bits(float(int(i%7)-3)/64);
        const auto preparation_begin=std::chrono::steady_clock::now();
        if(mapped) ordinary_source.reset(new Pages(payload+2*guard));
        else hb=backend::Allocation(runtime,payload+2*guard,backend::MemoryKind::Host);
        auto* source=static_cast<std::uint8_t*>(mapped ? ordinary_source->data() : hb.data());
        for(std::size_t i=0;i<payload+2*guard;++i) {
            const bool edge=i<guard || i>=guard+payload;
            const auto at=edge ? 0 : i-guard;
            source[i]=edge ? 0xa5 : std::uint8_t((at*131u)^(at>>8)^(at>>16)^17u);
            readback.as<std::uint8_t>()[i]=edge ? 0xa5 : 0xe7;
        }
        std::unique_ptr<Mapped> imported;
        double import_s=0;
        if(mapped) {
            ordinary_source->read_only();
            const auto begin=std::chrono::steady_clock::now();
            imported.reset(new Mapped(lane,ordinary_source->data(),ordinary_source->bytes(),
                Mapped::PropertyContract::PinnedNeoHost));
            import_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        }
        const auto import_proof=mapped ? imported->property_proof() : Mapped::PropertyProof{};
        const auto source_view=mapped ? imported->view(guard,payload) : Lane::Source{source+guard,payload};
        const double preparation_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-preparation_begin).count();
        runtime->wait(queue.memcpy(x.data(),hx.data(),xb));
        runtime->wait(queue.memcpy(w.data(),hw.data(),wb));
        runtime->wait(queue.memcpy(bank.data(),readback.data(),payload+2*guard));
        gemm::Problem problem{x.as<std::uint16_t>(),w.as<std::uint16_t>(),y.as<std::uint16_t>(),
            std::int64_t(rows),std::int64_t(n),std::int64_t(k),0,gemm::Storage::fp16,gemm::Storage::fp16,0};
        runtime->wait(dense.beta0(problem,gemm::Tile::wide_128x128)); // Exact prewarm, outside timing.
        auto verify=[&](bool copied) {
            runtime->wait(queue.memcpy(hy.data(),y.data(),yb));
            for(std::size_t i=0;i<rows;++i)
                for(std::size_t j=0;j<n;++j)
                    need(hy.as<std::uint16_t>()[i*n+j]==half_bits(1.25f*float(i%5+1)*float(int(j%7)-3)),
                        "production dense output differs from exact fixture oracle");
            runtime->wait(queue.memcpy(readback.data(),bank.data(),payload+2*guard));
            for(std::size_t i=0;i<payload+2*guard;++i) {
                const bool edge=i<guard || i>=guard+payload;
                const auto expected=edge || copied ? source[i] : std::uint8_t(0xe7);
                need(readback.as<std::uint8_t>()[i]==expected,"copied bank/payload/guards changed");
                const auto at=edge ? 0 : i-guard;
                need(source[i]==(edge ? 0xa5 : std::uint8_t((at*131u)^(at>>8)^(at>>16)^17u)),
                     "read-only source payload/guards changed");
            }
        };
        verify(false);gate("prewarm-ready");
        std::size_t accepted_copies=0;
        auto epoch=[&] {
            // Next epoch cannot copy until this epoch's compute and copy have
            // both retired on queue0. At most ONE transfer can look ahead.
            const auto prior=lane.readers_done();
            const auto copied=lane.copy_after(
                {bank.as<std::uint8_t>()+guard,payload},
                source_view,payload,prior);
            need(copied.index==accepted_copies++,"timed copy generation/count differs");
            if(!overlap) lane.wait_ready(copied);
            dense.beta0(problem,gemm::Tile::wide_128x128);
            if(overlap) lane.wait_ready(copied);
        };
        epoch();lane.drain();verify(true); // Event path warmup, excluded.
        runtime->wait(queue.memset(bank.as<std::uint8_t>()+guard,0xe7,payload));
        gate("before-timing");
        double timer_pair_max_s=0;
        for(int i=0;i<128;++i) {
            const auto a=std::chrono::steady_clock::now();
            const auto b=std::chrono::steady_clock::now();
            timer_pair_max_s=std::max(timer_pair_max_s,std::chrono::duration<double>(b-a).count());
        }
        const auto begin=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<iterations;++i) epoch();
        lane.drain();
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        need(accepted_copies==iterations+1,"timed upload count mismatch");
        verify(true);
        const auto retirement_begin=std::chrono::steady_clock::now();
        imported.reset(); // Completion and unimport belong to this source unit.
        if(mapped) ordinary_source.reset();
        else hb=backend::Allocation();
        const double retirement_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-retirement_begin).count();
        std::cout.precision(17);
        std::cout << "{\"kind\":\"batch\",\"wall_s\":" << seconds << ",\"iterations\":" << iterations
                  << ",\"arm\":\"" << (overlap ? "overlap" : "serial") << "\",\"timer_pair_max_s\":" << timer_pair_max_s
                  << ",\"requested_H2D_bytes\":" << iterations*payload << ",\"gemm_rows\":" << rows
                  << ",\"gemm_n\":" << n << ",\"gemm_k\":" << k
                  << ",\"lookahead\":1,\"oracle_and_guards\":true}" << std::endl;
        std::cout << "{\"kind\":\"source\",\"source\":\"" << (mapped ? "mapped-neo" : "usm")
                  << "\",\"preparation_s\":" << preparation_s << ",\"import_s\":" << import_s
                  << ",\"retirement_s\":" << retirement_s
                  << ",\"import_property_type\":" << unsigned(import_proof.type)
                  << ",\"import_allocation_id\":" << import_proof.allocation_id << "}" << std::endl;
    }
    runtime->wait();runtime->check();gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true}" << std::endl;
    return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {
        need(argc==4 || argc==5,"use serial|overlap ROWS1024or4096 ITERATIONS512 [usm|mapped-neo]");
        const std::string arm=argv[1];need(arm=="serial" || arm=="overlap","explicit arm required");
        const std::string source=argc==5 ? argv[4] : "usm";
        need(source=="usm" || source=="mapped-neo","explicit source kind required");
        return run(arm=="overlap",std::stoull(argv[2]),std::stoull(argv[3]),source=="mapped-neo");
    } catch(const std::exception& error) {
        std::cerr << "upload overlap failure: " << error.what() << std::endl;return 1;
    }
}
