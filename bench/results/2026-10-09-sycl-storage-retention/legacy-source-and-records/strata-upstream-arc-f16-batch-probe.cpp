#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using tr = oneapi::mkl::transpose;
using mode = oneapi::mkl::blas::compute_mode;
namespace blas = oneapi::mkl::blas::column_major;
using I = int64_t;
double median(std::vector<double> v) { std::sort(v.begin(),v.end()); return v[v.size()/2]; }
bool run(sycl::queue &q, int group, I N, I K, I T, bool varying) {
  auto transa=sycl::malloc_shared<tr>(group,q),transb=sycl::malloc_shared<tr>(group,q);
  auto m=sycl::malloc_shared<I>(group,q),n=sycl::malloc_shared<I>(group,q),k=sycl::malloc_shared<I>(group,q);
  auto lda=sycl::malloc_shared<I>(group,q),ldb=sycl::malloc_shared<I>(group,q),ldc=sycl::malloc_shared<I>(group,q);
  auto sizes=sycl::malloc_shared<I>(group,q);
  auto alpha=sycl::malloc_shared<float>(group,q),beta=sycl::malloc_shared<float>(group,q);
  auto aa=sycl::malloc_shared<const sycl::half*>(group,q),bb=sycl::malloc_shared<const sycl::half*>(group,q);
  auto cc=sycl::malloc_shared<float*>(group,q);
  const I strideY=T*(N+3),strideX=T*K,strideW=N*K;
  auto W=sycl::malloc_device<sycl::half>(strideW*group,q),X=sycl::malloc_device<sycl::half>(strideX*group,q);
  auto Y=sycl::malloc_device<float>(strideY*group,q),Z=sycl::malloc_device<float>(strideY*group,q);
  std::vector<sycl::half> w(strideW*group),x(strideX*group);
  std::mt19937 rng(6239+T+group);std::uniform_real_distribution<float> d(-1,1);
  for(auto &v:w)v=sycl::half(d(rng)); for(auto &v:x)v=sycl::half(d(rng));
  q.memcpy(W,w.data(),w.size()*2);q.memcpy(X,x.data(),x.size()*2);
  q.fill(Y,-98765.f,strideY*group);q.fill(Z,-98765.f,strideY*group);q.wait_and_throw();
  for(int g=0;g<group;g++) {
    transa[g]=tr::trans;transb[g]=tr::nontrans;m[g]=N;n[g]=varying?std::max<I>(1,T-g*(T/8+1)):T;k[g]=K;
    lda[g]=K;ldb[g]=K;ldc[g]=N+3;alpha[g]=1;beta[g]=0;sizes[g]=1;
    aa[g]=W+g*strideW;bb[g]=X+g*strideX;cc[g]=Z+g*strideY;
  }
  auto original=[&] { for(int g=0;g<group;g++)blas::gemm(q,tr::trans,tr::nontrans,N,n[g],K,1.f,aa[g],K,bb[g],K,0.f,Y+g*strideY,N+3,mode::unset); };
  auto batched=[&] { blas::gemm_batch(q,transa,transb,m,n,k,alpha,aa,lda,bb,ldb,beta,cc,ldc,group,sizes,mode::unset); };
  original();batched();q.wait_and_throw();
  std::vector<float> y(strideY*group),z(y.size());
  q.memcpy(y.data(),Y,y.size()*4);q.memcpy(z.data(),Z,z.size()*4);q.wait_and_throw();
  size_t different=0,values=0,guard_errors=0;double error=0,original_error=0,max_delta=0;bool finite=true;
  for(int g=0;g<group;g++) {
    for(I t=0;t<T;t++)for(I o=0;o<N+3;o++) {
      I i=g*strideY+t*(N+3)+o;
      if(t>=n[g] || o>=N) {if(y[i]!=-98765.f || z[i]!=-98765.f)guard_errors++;continue;}
      values++;finite=finite&&std::isfinite(y[i])&&std::isfinite(z[i]);
      different+=std::bit_cast<uint32_t>(y[i])!=std::bit_cast<uint32_t>(z[i]);
      max_delta=std::max(max_delta,std::abs(double(y[i])-double(z[i])));
    }
    // Independent FP64 reference sampled across first/last/interior output rows.
    for(I t:std::array<I,3>{0,n[g]/2,n[g]-1})for(I o=0;o<N;o+=std::max<I>(1,N/64)) {
      double sum=0,l1=0;for(I r=0;r<K;r++){double v=double(float(x[g*strideX+t*K+r]))*float(w[g*strideW+o*K+r]);sum+=v;l1+=std::abs(v);}
      const I i=g*strideY+t*(N+3)+o;
      error=std::max(error,std::abs(z[i]-sum)/std::max(l1,1e-30));
      original_error=std::max(original_error,std::abs(y[i]-sum)/std::max(l1,1e-30));
    }
  }
  auto time=[&](auto fn){q.wait_and_throw();auto a=std::chrono::steady_clock::now();for(int r=0;r<10;r++)fn();q.wait_and_throw();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count()/10;};
  std::vector<double> before,after;for(int r=0;r<3;r++) {if(r%2){after.push_back(time(batched));before.push_back(time(original));}else{before.push_back(time(original));after.push_back(time(batched));}}
  bool ok=finite&&guard_errors==0&&error<1e-6&&original_error<1e-6;
  printf("%s G=%d N=%lld K=%lld T=%lld varied=%d elements=%zu unequal=%zu guards=%zu maxdelta=%.9g batch_L1error=%.9g original_L1error=%.9g original=%.6fms batch=%.6fms\n",ok?"PASS":"FAIL",group,(long long)N,(long long)K,(long long)T,varying,values,different,guard_errors,max_delta,error,original_error,median(before),median(after));fflush(stdout);
  for(void *p:std::vector<void*>{transa,transb,m,n,k,lda,ldb,ldc,sizes,alpha,beta,aa,bb,cc,W,X,Y,Z})sycl::free(p,q);
  return ok;
}
int main(){sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});printf("device=%s\n",q.get_device().get_info<sycl::info::device::name>().c_str());bool ok=true;
  for(int g:{2,4,8})for(I t:{I(1),I(8),I(32),I(128)}) {
    ok=run(q,g,2560,640,t,true)&&ok;
    ok=run(q,g,1280,2560,t,true)&&ok;
  }
  return ok?0:2;
}
