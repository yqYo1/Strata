#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Standalone probe: every grouped product retains the original row count,
// transpose flags, alpha, beta, FP16 inputs and FP32 result. Compare all bits
// with the engine's exact DPCT oneMKL call before considering integration.
struct Args {
    std::array<oneapi::mkl::transpose,8> ta,tb;
    std::array<int64_t,8> m,n,k,lda,ldb,ldc,size;
    std::array<float,8> alpha,beta;
    std::array<const sycl::half*,8> a,b;
    std::array<float*,8> c;
};
int main() try {
    auto* q=&dpct::get_in_order_queue();
    std::fprintf(stderr,"device=%s driver=%s\n",q->get_device().get_info<sycl::info::device::name>().c_str(),
                 q->get_device().get_info<sycl::info::device::driver_version>().c_str());
    dpct::blas::descriptor original; original.set_queue(q);
    Args* args=sycl::malloc_shared<Args>(1,*q);
    if (!args) return 2;
    std::mt19937 rng(731);
    std::uniform_real_distribution<float> input(-1.f,1.f);
    const std::array<std::array<int64_t,8>,3> row_sets={{{1,2,3,5,7,9,17,31},
                                                      {32,63,64,65,79,81,97,129},
                                                      {128,129,255,256,257,384,511,513}}};
    for (bool gu : {false,true}) {
        const int64_t N=gu?1280:2560,K=gu?2560:640;
        std::array<sycl::half*,8> w{},x{};
        std::array<float*,8> ref{},group{};
        for (int e=0;e<8;++e) {
            w[e]=sycl::malloc_device<sycl::half>(N*K,*q);
            x[e]=sycl::malloc_device<sycl::half>(513*K,*q);
            ref[e]=sycl::malloc_device<float>(513*N,*q);
            group[e]=sycl::malloc_device<float>(513*N,*q);
            if (!w[e]||!x[e]||!ref[e]||!group[e]) return 2;
            std::vector<sycl::half> hw(N*K),hx(513*K);
            for (auto& v:hw) v=sycl::half(input(rng));
            for (auto& v:hx) v=sycl::half(input(rng));
            q->memcpy(w[e],hw.data(),hw.size()*sizeof(sycl::half));
            q->memcpy(x[e],hx.data(),hx.size()*sizeof(sycl::half));q->wait_and_throw();
            args->ta[e]=oneapi::mkl::transpose::trans;args->tb[e]=oneapi::mkl::transpose::nontrans;
            args->m[e]=N;args->k[e]=K;args->lda[e]=K;args->ldb[e]=K;args->ldc[e]=N;args->size[e]=1;
            args->alpha[e]=1.f;args->beta[e]=0.f;args->a[e]=w[e];args->b[e]=x[e];args->c[e]=group[e];
        }
        for (int count:{2,4,8}) for (size_t set=0;set<row_sets.size();++set) {
            for (int e=0;e<count;++e) args->n[e]=row_sets[set][e];
            auto individual=[&] {
                const float alpha=1.f,beta=0.f;
                for (int e=0;e<count;++e)
                    dpct::blas::gemm(&original,oneapi::mkl::transpose::trans,oneapi::mkl::transpose::nontrans,
                                    N,args->n[e],K,&alpha,w[e],dpct::library_data_t::real_half,K,
                                    x[e],dpct::library_data_t::real_half,K,&beta,ref[e],
                                    dpct::library_data_t::real_float,N,dpct::compute_type::f32);
            };
            auto grouped=[&] {
                oneapi::mkl::blas::column_major::gemm_batch(*q,args->ta.data(),args->tb.data(),
                    args->m.data(),args->n.data(),args->k.data(),args->alpha.data(),args->a.data(),args->lda.data(),
                    args->b.data(),args->ldb.data(),args->beta.data(),args->c.data(),args->ldc.data(),
                    count,args->size.data(),oneapi::mkl::blas::compute_mode::unset);
            };
            individual();grouped();q->wait_and_throw();
            size_t compared=0,bad=0,nonfinite=0;double maxerr=0;
            for (int e=0;e<count;++e) {
                std::vector<float> a(args->n[e]*N),b(a.size());
                q->memcpy(a.data(),ref[e],a.size()*4);q->memcpy(b.data(),group[e],b.size()*4);q->wait_and_throw();
                for (size_t i=0;i<a.size();++i) {
                    uint32_t u,v;std::memcpy(&u,&a[i],4);std::memcpy(&v,&b[i],4);
                    ++compared;bad+=u!=v;nonfinite+=!std::isfinite(a[i])||!std::isfinite(b[i]);
                    maxerr=std::max(maxerr,std::abs(double(a[i])-double(b[i])));
                }
            }
            std::array<double,3> original_ms{},grouped_ms{};
            for (size_t round=0;round<3;++round) for (int order=0;order<2;++order) {
                const bool batch=(order!=0)^(round%2!=0);
                q->wait_and_throw();const auto t0=std::chrono::steady_clock::now();
                for (int rep=0;rep<10;++rep) { if (batch) grouped();else individual(); }
                q->wait_and_throw();
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/10;
                (batch?grouped_ms:original_ms)[round]=ms;
            }
            std::sort(original_ms.begin(),original_ms.end());std::sort(grouped_ms.begin(),grouped_ms.end());
            std::printf("{\"mode\":\"%s\",\"group\":%d,\"row_set\":%zu,\"elements\":%zu,\"bit_mismatches\":%zu,"
                        "\"nonfinite\":%zu,\"max_error\":%.9g,\"original_median_ms\":%.6f,\"grouped_median_ms\":%.6f}\n",
                        gu?"gate-up":"down",count,set,compared,bad,nonfinite,maxerr,original_ms[1],grouped_ms[1]);
            std::fflush(stdout);
            if (bad||nonfinite) return 1;
        }
        q->wait_and_throw();for (int e=0;e<8;++e) {sycl::free(w[e],*q);sycl::free(x[e],*q);sycl::free(ref[e],*q);sycl::free(group[e],*q);}
    }
    sycl::free(args,*q);return 0;
} catch (const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
