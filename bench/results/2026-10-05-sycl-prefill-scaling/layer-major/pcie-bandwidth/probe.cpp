#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr size_t GiB = size_t(1) << 30;
struct Result { double wall_ms, dma_ms; };
double median(std::vector<double> x) {
    std::sort(x.begin(),x.end()); return x[x.size()/2];
}
int main() try {
    sycl::queue q(sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
    const auto d=q.get_device();
    std::fprintf(stderr,"device=%s driver=%s global_memory_bytes=%llu\n",
        d.get_info<sycl::info::device::name>().c_str(),
        d.get_info<sycl::info::device::driver_version>().c_str(),
        (unsigned long long)d.get_info<sycl::info::device::global_mem_size>());
    std::vector<uint8_t> host(GiB),check(GiB);
    for(size_t i=0;i<host.size();++i)host[i]=uint8_t((i*37)^(i>>8)^(i>>19));
    auto free_usm=[&q](uint8_t*p){if(p)sycl::free(p,q);};
    std::unique_ptr<uint8_t,decltype(free_usm)> pinned(sycl::malloc_host<uint8_t>(GiB,q),free_usm);
    std::unique_ptr<uint8_t,decltype(free_usm)> device(sycl::malloc_device<uint8_t>(GiB,q),free_usm);
    if(!pinned || !device)throw std::runtime_error("USM allocation failed");
    std::memcpy(pinned.get(),host.data(),GiB);
    const std::vector<size_t> sizes={1510400,2176000,2662400,5242880,167772160,335544320,GiB};
    for(const size_t bytes:sizes) {
        const size_t iterations=std::max<size_t>(3,std::min<size_t>(512,(GiB+bytes-1)/bytes));
        const size_t offsets=std::max<size_t>(1,GiB/bytes);
        for(const char*mode:{"pageable-h2d","host-usm-h2d","pageable-d2h","host-usm-d2h","staged-ring8-h2d"}) {
            const std::string kind=mode;
            const bool d2h=kind.find("d2h")!=std::string::npos;
            const bool usm=kind.find("host-usm")!=std::string::npos;
            const bool staged=kind.find("staged")!=std::string::npos;
            std::vector<std::unique_ptr<uint8_t,decltype(free_usm)>> ring;
            if(staged)for(size_t i=0;i<std::min<size_t>(8,iterations);++i) {
                ring.emplace_back(sycl::malloc_host<uint8_t>(bytes,q),free_usm);
                if(!ring.back())throw std::runtime_error("ring allocation failed");
            }
            std::memcpy(pinned.get(),host.data(),GiB);
            if(d2h)q.memcpy(device.get(),host.data(),GiB).wait_and_throw();
            auto run=[&](size_t count)->Result {
                std::vector<sycl::event> events;
                events.reserve(count);
                std::vector<sycl::event> done(8);
                size_t last_offset=0;
                const auto begin=Clock::now();
                for(size_t i=0;i<count;++i) {
                    last_offset=(i%offsets)*bytes;
                    uint8_t* source=usm?pinned.get()+last_offset
                                       :(d2h?check.data()+last_offset:host.data()+last_offset);
                    if(staged) {
                        if(i>=8)done[i%8].wait_and_throw();
                        std::memcpy(ring[i%8].get(),host.data()+last_offset,bytes);
                        source=ring[i%8].get();
                    }
                    events.push_back(d2h ? q.memcpy(source,device.get(),bytes)
                                         : q.memcpy(device.get(),source,bytes));
                    if(staged)done[i%8]=events.back();
                }
                q.wait_and_throw();
                const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
                double dma=0;
                for(auto&e:events) {
                    const auto a=e.get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto b=e.get_profiling_info<sycl::info::event_profiling::command_end>();
                    if(b<=a)throw std::runtime_error("invalid event timestamps");
                    dma+=double(b-a)/1e6;
                }
                if(d2h) {
                    const uint8_t* result=usm?pinned.get()+last_offset:check.data()+last_offset;
                    if(std::memcmp(result,host.data(),bytes)!=0)
                        throw std::runtime_error("D2H complete-byte validation failed");
                } else {
                    q.memcpy(check.data(),device.get(),bytes).wait_and_throw();
                    const uint8_t* expected=host.data()+last_offset;
                    if(std::memcmp(check.data(),expected,bytes)!=0)
                        throw std::runtime_error("H2D complete-byte validation failed");
                }
                return {elapsed,dma};
            };
            // Destination verification and source initialization are outside all transfer timers.
            // Sources stay unchanged throughout the independent rounds. H2D offsets rotate within 1 GiB.
            run(std::min<size_t>(3,iterations));
            std::vector<double> walls,dmas;
            for(int round=0;round<5;++round) {
                const auto r=run(iterations);walls.push_back(r.wall_ms);dmas.push_back(r.dma_ms);
                std::printf("{\"mode\":\"%s\",\"bytes_per_copy\":%zu,\"iterations\":%zu,\"round\":%d,"
                    "\"wall_ms\":%.9f,\"dma_ms\":%.9f,\"final_copy_all_bytes_identical\":true}\n",
                    mode,bytes,iterations,round+1,r.wall_ms,r.dma_ms);std::fflush(stdout);
            }
            const double wall=median(walls),dma=median(dmas),total=double(bytes)*iterations;
            std::printf("{\"mode\":\"%s\",\"bytes_per_copy\":%zu,\"iterations\":%zu,\"rounds\":5,"
                "\"median_wall_ms\":%.9f,\"median_dma_ms\":%.9f,\"wall_GB_s\":%.9f,\"dma_GB_s\":%.9f,"
                "\"final_copy_all_bytes_identical\":true}\n",mode,bytes,iterations,wall,dma,total/wall/1e6,total/dma/1e6);
            std::fflush(stdout);
        }
    }
    return 0;
} catch(const std::exception&e) {
    std::fprintf(stderr,"bandwidth probe failed: %s\n",e.what());return 1;
}
