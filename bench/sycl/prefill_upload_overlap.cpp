// Main-authored E2proposal; repository LICENSE. Not built/admitted until E1.
// Isolates one-expert lookahead with the existing production Xe2 dense kernel.
// Copy primitive, payload, consumer math and allocation are identical in arms.
#include "strata/prefill/upload_event_lane.hpp"
#include "strata/sycl/runtime.hpp"
#include "prefill/sycl/vllm_xe2/native16.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

namespace {
using Lane = strata::prefill::detail::PrefillUploadEventLane;
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
int run(bool overlap,std::size_t rows,std::size_t iterations) {
    need((rows==1024 || rows==4096) && iterations==512,"frozen E2shape/batch required");
    gate("before-runtime");
    auto runtime=backend::runtime_for();
    auto& queue=runtime->compute();
    {
        const auto xb=rows*k*2,wb=n*k*2,yb=rows*n*2;
        // Registered production Allocation owners; their original Runtime and
        // compute queue survive both the transfer lane and dense adapter.
        backend::Allocation hx(runtime,xb,backend::MemoryKind::Host),
            hw(runtime,wb,backend::MemoryKind::Host),hy(runtime,yb,backend::MemoryKind::Host),
            hb(runtime,payload+2*guard,backend::MemoryKind::Host),
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
        for(std::size_t i=0;i<payload+2*guard;++i)
            hb.as<std::uint8_t>()[i]=std::uint8_t(i*131u+17u);
        runtime->wait(queue.memcpy(x.data(),hx.data(),xb));
        runtime->wait(queue.memcpy(w.data(),hw.data(),wb));
        runtime->wait(queue.memcpy(bank.data(),hb.data(),payload+2*guard));
        gemm::Problem problem{x.as<std::uint16_t>(),w.as<std::uint16_t>(),y.as<std::uint16_t>(),
            std::int64_t(rows),std::int64_t(n),std::int64_t(k),0,gemm::Storage::fp16,gemm::Storage::fp16,0};
        runtime->wait(dense.beta0(problem,gemm::Tile::wide_128x128)); // Exact prewarm, outside timing.
        auto verify=[&] {
            runtime->wait(queue.memcpy(hy.data(),y.data(),yb));
            for(std::size_t i=0;i<rows;++i)
                for(std::size_t j=0;j<n;++j)
                    need(hy.as<std::uint16_t>()[i*n+j]==half_bits(1.25f*float(i%5+1)*float(int(j%7)-3)),
                        "production dense output differs from exact fixture oracle");
            runtime->wait(queue.memcpy(readback.data(),bank.data(),payload+2*guard));
            need(std::memcmp(hb.data(),readback.data(),payload+2*guard)==0,"copied bank/payload/guards changed");
        };
        verify();gate("prewarm-ready");
        auto epoch=[&] {
            // Next epoch cannot copy until this epoch's compute and copy have
            // both retired on queue0. At most ONE transfer can look ahead.
            const auto prior=lane.readers_done();
            const auto copied=lane.copy_after(
                {bank.as<std::uint8_t>()+guard,payload},
                {hb.as<std::uint8_t>()+guard,payload},payload,prior);
            if(!overlap) lane.wait_ready(copied);
            dense.beta0(problem,gemm::Tile::wide_128x128);
            if(overlap) lane.wait_ready(copied);
        };
        epoch();lane.drain();verify(); // Event path warmup, excluded.
        gate("before-timing");
        const auto begin=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<iterations;++i) epoch();
        lane.drain();
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        verify();
        std::cout.precision(17);
        std::cout << "{\"kind\":\"batch\",\"wall_s\":" << seconds << ",\"iterations\":" << iterations
                  << ",\"requested_H2D_bytes\":" << iterations*payload << ",\"gemm_rows\":" << rows
                  << ",\"gemm_n\":" << n << ",\"gemm_k\":" << k
                  << ",\"lookahead\":1,\"oracle_and_guards\":true}" << std::endl;
    }
    runtime->wait();runtime->check();gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true}" << std::endl;
    return 0;
}
} // namespace
int main(int argc,char** argv) {
    try {
        need(argc==4,"use serial|overlap ROWS1024or4096 ITERATIONS512");
        const std::string arm=argv[1];need(arm=="serial" || arm=="overlap","explicit arm required");
        return run(arm=="overlap",std::stoull(argv[2]),std::stoull(argv[3]));
    } catch(const std::exception& error) {
        std::cerr << "upload overlap failure: " << error.what() << std::endl;return 1;
    }
}
