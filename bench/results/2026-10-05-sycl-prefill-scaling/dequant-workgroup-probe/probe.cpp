#include "iq_kernels.dp.cpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
template<int Blocks, bool Root, bool Gu> class DqWorkgroupProbe;
template<int Blocks, bool Root, bool Gu>
sycl::event candidate(sycl::queue& q, int ty, const void* gate, const void* up,
                      int64_t n_ff, int64_t n_embd, sycl::half* y) {
    const int64_t per_row = n_embd / 256, blocks = n_ff * per_row;
    const auto global = sycl::range<3>(1, Gu ? 2 : 1, ((blocks+Blocks-1)/Blocks)*Blocks*32);
    const auto local = sycl::range<3>(1,1,Blocks*32);
    auto body = [=](sycl::nd_item<3> item) {
        const int64_t i = item.get_global_id(2)/32;
        if (i >= blocks) return;
        const int lane = item.get_local_id(2)%32;
        if constexpr (Gu) {
            const int parity = item.get_group(1);
            const int64_t r = i/per_row, c = i%per_row;
            dq_dispatch<sycl::half>(ty,parity ? up : gate,i,
                y+((2*r+parity)*per_row+c)*256,lane);
        } else dq_dispatch<sycl::half>(ty,gate,i,y+i*256,lane);
    };
    if constexpr (Root) return q.parallel_for<DqWorkgroupProbe<Blocks,Root,Gu>>(
        sycl::nd_range<3>(global,local),
        sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync},body);
    else return q.parallel_for<DqWorkgroupProbe<Blocks,Root,Gu>>(sycl::nd_range<3>(global,local),body);
}
}

int main() try {
    using namespace strata::kernels;
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    std::fprintf(stderr,"device=%s driver=%s\n",q.get_device().get_info<sycl::info::device::name>().c_str(),
        q.get_device().get_info<sycl::info::device::driver_version>().c_str());
    std::mt19937 rng(7);
    constexpr int64_t K=2560;
    constexpr int reps=128;
    for (int ty : {16,17,18,20,21,22,23,11,42}) {
        for (int64_t n_ff : {1,640,641}) {
            const int qk = ty==20 ? 32 : ty==42 ? 64 : 256;
            const size_t nb = iq_row_bytes(ty,qk);
            const size_t bytes = iq_row_bytes(ty,K)*n_ff;
            auto random_blob = [&]() {
                std::vector<uint8_t> h(bytes);
                for(auto& v:h) v=uint8_t(rng());
                for(size_t i=0;i<bytes;i+=nb) {
                    const size_t scale=i+(ty==11 ? nb-2 : 0);
                    h[scale]=0;h[scale+1]=0x30;
                }
                return h;
            };
            auto hg=random_blob(),hu=random_blob();
            auto* g=sycl::malloc_device<uint8_t>(bytes,q);
            auto* u=sycl::malloc_device<uint8_t>(bytes,q);
            auto* ref=sycl::malloc_device<sycl::half>(2*n_ff*K,q);
            auto* got=sycl::malloc_device<sycl::half>(2*n_ff*K,q);
            if(!g||!u||!ref||!got)throw std::runtime_error("allocation failed");
            q.memcpy(g,hg.data(),bytes);q.memcpy(u,hu.data(),bytes);q.wait_and_throw();
            for (bool gu : {false,true}) {
                const size_t elements=(gu ? 2 : 1)*n_ff*K;
                std::vector<uint16_t> hr(elements),hc(elements);
                auto original=[&]() {
                    if(gu) iq_dequant_gu_f16(ty,g,u,n_ff,K,reinterpret_cast<uint16_t*>(ref),&q);
                    else iq_dequant_f16(ty,g,n_ff*K,reinterpret_cast<uint16_t*>(ref),&q);
                };
                original();q.wait_and_throw();q.memcpy(hr.data(),ref,elements*2).wait_and_throw();
                for(auto h:hr)if(!std::isfinite(float(sycl::bit_cast<sycl::half>(h))))
                    throw std::runtime_error("nonfinite reference value");
                for(int variant=0;variant<8;++variant) {
                    const int blocks=1<<(variant%4);
                    const bool root=variant<4;
                    auto launch=[&]() {
#define LAUNCH(B,R,G) candidate<B,R,G>(q,ty,g,u,n_ff,K,got)
#define MODE(B,R) do { if(gu) LAUNCH(B,R,true); else LAUNCH(B,R,false); } while(0)
                        switch(variant) {
                            case 0:MODE(1,true);break;case 1:MODE(2,true);break;
                            case 2:MODE(4,true);break;case 3:MODE(8,true);break;
                            case 4:MODE(1,false);break;case 5:MODE(2,false);break;
                            case 6:MODE(4,false);break;case 7:MODE(8,false);break;
                        }
#undef MODE
#undef LAUNCH
                    };
                    launch();q.wait_and_throw();q.memcpy(hc.data(),got,elements*2).wait_and_throw();
                    const bool equal=std::memcmp(hr.data(),hc.data(),elements*2)==0;
                    if(!equal)throw std::runtime_error("complete FP16-byte mismatch");
                    std::vector<double> base_times,new_times;
                    if(n_ff==640) {
                        for(int w=0;w<10;++w){original();launch();}q.wait_and_throw();
                        auto measure=[&](auto fn) {
                            auto start=std::chrono::steady_clock::now();
                            for(int i=0;i<reps;++i)fn();q.wait_and_throw();
                            return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/reps;
                        };
                        for(int round=0;round<5;++round) {
                            double a,b;
                            if(round%2){b=measure(launch);a=measure(original);}else{a=measure(original);b=measure(launch);}
                            base_times.push_back(a);new_times.push_back(b);
                            std::printf("{\"type\":%d,\"gu\":%s,\"n_ff\":%lld,\"blocks_per_group\":%d,\"root\":%s,\"round\":%d,\"reps\":%d,\"original_ms\":%.9f,\"candidate_ms\":%.9f,\"all_half_bytes_identical\":true}\n",
                                ty,gu?"true":"false",(long long)n_ff,blocks,root?"true":"false",round+1,reps,a,b);
                        }
                        std::sort(base_times.begin(),base_times.end());std::sort(new_times.begin(),new_times.end());
                    }
                    std::printf("{\"type\":%d,\"gu\":%s,\"n_ff\":%lld,\"blocks_per_group\":%d,\"root\":%s,\"elements\":%zu,\"all_half_bytes_identical\":true,\"rounds\":%d,\"original_ms\":%.9f,\"candidate_ms\":%.9f}\n",
                        ty,gu?"true":"false",(long long)n_ff,blocks,root?"true":"false",elements,n_ff==640?5:0,n_ff==640?base_times[2]:0,n_ff==640?new_times[2]:0);
                    std::fflush(stdout);
                }
            }
            sycl::free(g,q);sycl::free(u,q);sycl::free(ref,q);sycl::free(got,q);
        }
    }
    return 0;
} catch(const std::exception& e) {
    std::fprintf(stderr,"dequant probe failed: %s\n",e.what());return 1;
}
