#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include "strata/prefill/gemm.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
namespace blas = oneapi::mkl::blas::column_major;
using I = int64_t;
using tr = oneapi::mkl::transpose;
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    return v[v.size()/2];
}
bool run(sycl::queue &q,I N,I K,I T,I multiple) {
    const I padded=(T+multiple-1)/multiple*multiple;
    strata::prefill::Gemm engine;
    std::string err;
    if(!engine.init_external(&q,nullptr,0,nullptr,0,err)) {std::fprintf(stderr,"%s\n",err.c_str());return false;}
    constexpr I guard=32;
    constexpr float sentinel=-98765.f;
    auto W=sycl::malloc_device<sycl::half>(N*K,q);
    auto X=sycl::malloc_device<sycl::half>(padded*K,q);
    auto P=sycl::malloc_device<sycl::half>(padded*K,q);
    auto Ya=sycl::malloc_device<float>(padded*N+2*guard,q);
    auto Za=sycl::malloc_device<float>(padded*N+2*guard,q);
    auto Ra=sycl::malloc_device<float>(T*N+2*guard,q);
    float *Y=Ya+guard,*Z=Za+guard,*R=Ra+guard;
    std::mt19937 rng(7123+T+K);
    std::uniform_real_distribution<float> d(-1,1);
    std::vector<sycl::half> w(N*K),x(padded*K,sycl::half(0));
    for(auto &v:w)v=sycl::half(d(rng));
    for(I i=0;i<T*K;i++)x[i]=sycl::half(d(rng));
    q.memcpy(W,w.data(),w.size()*2);
    q.memcpy(X,x.data(),x.size()*2);
    q.fill(Ya,sentinel,padded*N+2*guard);
    q.fill(Za,sentinel,padded*N+2*guard);
    q.fill(Ra,sentinel,T*N+2*guard);
    q.wait_and_throw();
    auto gemm=[&](const sycl::half *a,I rows,float *out) {
        return blas::gemm(q,tr::trans,tr::nontrans,N,rows,K,1.f,W,K,a,K,0.f,out,N,
                          oneapi::mkl::blas::compute_mode::unset);
    };
    auto original=[&] {gemm(X,T,Y);};
    auto pad_only=[&] {engine.f16(reinterpret_cast<const uint16_t*>(X),reinterpret_cast<const uint16_t*>(W),Z,T,N,K);};
    auto pad_with_copies=[&] {engine.f16(reinterpret_cast<const uint16_t*>(X),reinterpret_cast<const uint16_t*>(W),R,T,N,K);};
    original();pad_only();pad_with_copies();q.wait_and_throw();
    std::vector<float> y(padded*N+2*guard),z(y.size()),r(T*N+2*guard);
    q.memcpy(y.data(),Ya,y.size()*4);q.memcpy(z.data(),Za,z.size()*4);
    q.memcpy(r.data(),Ra,r.size()*4);q.wait_and_throw();
    size_t unequal=0,copy_unequal=0,guards=0,padding_nonzero=0;
    bool finite=true;
    for(I i=0;i<padded*N+2*guard;i++) {
        if(i<guard || i>=padded*N+guard) {
            guards+=y[i]!=sentinel || z[i]!=sentinel;
        } else if(i<guard+T*N) {
            unequal+=std::bit_cast<uint32_t>(y[i])!=std::bit_cast<uint32_t>(z[i]);
            copy_unequal+=std::bit_cast<uint32_t>(y[i])!=std::bit_cast<uint32_t>(r[i]);
            finite=finite&&std::isfinite(y[i])&&std::isfinite(z[i])&&std::isfinite(r[i]);
        } else {
            guards+=y[i]!=sentinel;
            padding_nonzero+=z[i]!=0;
            finite=finite&&std::isfinite(z[i]);
        }
    }
    for(I i=0;i<guard;i++)guards+=r[i]!=sentinel || r[T*N+guard+i]!=sentinel;
    double referror=0,newerror=0;
    for(I t:{I(0),T/2,T-1})for(I row=0;row<N;row+=std::max<I>(1,N/64)) {
        double ref=0,l1=0;
        for(I k=0;k<K;k++) {
            double product=double(float(x[t*K+k]))*float(w[row*K+k]);
            ref+=product;l1+=std::abs(product);
        }
        referror=std::max(referror,std::abs(double(y[guard+t*N+row])-ref)/std::max(1e-30,l1));
        newerror=std::max(newerror,std::abs(double(z[guard+t*N+row])-ref)/std::max(1e-30,l1));
    }
    auto time=[&](auto fn) {
        q.wait_and_throw();auto start=std::chrono::steady_clock::now();
        for(int i=0;i<10;i++)fn();
        q.wait_and_throw();
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/10;
    };
    std::vector<double> old_times,pad_times,copy_times;
    for(int round=0;round<3;round++) {
        if(round%2) {
            copy_times.push_back(time(pad_with_copies));pad_times.push_back(time(pad_only));
            old_times.push_back(time(original));
        } else {
            old_times.push_back(time(original));pad_times.push_back(time(pad_only));
            copy_times.push_back(time(pad_with_copies));
        }
    }
    bool okay=finite && guards==0 && padding_nonzero==0 && referror<1e-6 && newerror<1e-6;
    std::printf("%s N=%lld K=%lld T=%lld padded=%lld multiple=%lld unequal=%zu copied_unequal=%zu guards=%zu padding_nonzero=%zu finite=%d ref_L1=%.9g pad_L1=%.9g direct=%.6fms wrapper=%.6fms wrapper_repeat=%.6fms\n",
        okay?"PASS":"FAIL",(long long)N,(long long)K,(long long)T,(long long)padded,(long long)multiple,
        unequal,copy_unequal,guards,padding_nonzero,finite,referror,newerror,
        median(old_times),median(pad_times),median(copy_times));
    std::fflush(stdout);
    for(void *p:std::vector<void*>{W,X,P,Ya,Za,Ra})sycl::free(p,q);
    return okay;
}
int main() {
    const char *value=std::getenv("STRATA_PROBE_QUEUE_PROFILING");
    bool profiling=value&&std::atoi(value)!=0;
    const auto props=profiling?sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}:
        sycl::property_list{sycl::property::queue::in_order{}};
    sycl::queue q(sycl::gpu_selector_v,props);
    std::printf("device=%s queue_profiling=%d\n",q.get_device().get_info<sycl::info::device::name>().c_str(),profiling);
    bool okay=true;
    for(I T:{I(1),I(3),I(8),I(17),I(32),I(64),I(80),I(128),I(160),I(256)})
        for(I multiple:{I(1)}) {
            okay=run(q,2560,640,T,multiple)&&okay;
            okay=run(q,1280,2560,T,multiple)&&okay;
        }
    return okay?0:2;
}
