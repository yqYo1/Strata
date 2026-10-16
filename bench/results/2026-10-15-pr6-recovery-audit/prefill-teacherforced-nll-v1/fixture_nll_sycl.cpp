// SOURCE ONLY; links the proposed sequence.cpp and existing SYCL runtime.
// Run only under Main's serialized GPU supervisor. No model quality inference.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <cassert>
#include <climits>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
int main() {
  auto rt=strata::sycl_backend::runtime_for(); auto &q=rt->compute();
  constexpr int rows=10,vocab=259;
  strata::sycl_backend::Allocation l(rt,rows*vocab*4,strata::sycl_backend::MemoryKind::Host);
  strata::sycl_backend::Allocation ids(rt,rows*4,strata::sycl_backend::MemoryKind::Host);
  strata::sycl_backend::Allocation result(rt,rows*4,strata::sycl_backend::MemoryKind::Host);
  auto x=l.as<float>(); auto gold=ids.as<int32_t>(); auto out=result.as<float>();
  for(int r=0;r<rows;++r) { gold[r]=r==3?vocab-1:7; out[r]=-123;
    for(int i=0;i<vocab;++i) x[r*vocab+i]=float((i*13)%29-14)/3; }
  // ordinary, large common offset, tiny gold probability, non-power-of-two tail,
  // extreme finite range, NaN/+Inf/-Inf, negative and upper-bound invalid IDs.
  for(int i=0;i<vocab;++i) x[vocab+i]=10000.f+float(i%13);
  x[2*vocab+gold[2]]=-1000; x[2*vocab+1]=1000;
  x[3*vocab+vocab-1]=15;
  x[4*vocab+1]=1e20f; x[4*vocab+gold[4]]=-1e20f;
  x[5*vocab+123]=NAN; x[6*vocab+1]=INFINITY; x[7*vocab+1]=-INFINITY;
  gold[8]=-1; gold[9]=vocab;
  auto expect_throw=[&](auto fn) { bool threw=false; try { fn(); } catch(const std::invalid_argument&) { threw=true; } assert(threw); };
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,vocab,gold,out,nullptr); });
  expect_throw([&] { strata::kernels::row_target_nll(x,-1,vocab,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,0,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,vocab,gold,x,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(reinterpret_cast<float*>(reinterpret_cast<char*>(x)+1),rows,vocab,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(nullptr,rows,vocab,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,-1,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,INT_MAX,INT_MAX,gold,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,vocab,nullptr,out,&q); });
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,vocab,gold,nullptr,&q); });
  // All rejections above occur before enqueue. No host/device fixture env switch.
  sycl::queue unordered(q.get_context(),q.get_device());
  expect_throw([&] { strata::kernels::row_target_nll(x,rows,vocab,gold,out,&unordered); });
  try { strata::kernels::row_target_nll(x,rows,vocab,gold,out,&q); rt->wait(); }
  catch (...) { strata::sycl_backend::cleanup_fail_stop("NLL fixture completion unknown"); }
  for(int r=0;r<rows;++r) {
    if(r>=5) { assert(std::isnan(out[r])); std::cout<<r<<" invalid NaN\n"; continue; }
    double m=-INFINITY; for(int i=0;i<vocab;++i) m=std::max(m,double(x[r*vocab+i]));
    double sum=0; for(int i=0;i<vocab;++i) sum+=std::exp(double(x[r*vocab+i])-m);
    const double oracle=(m-double(x[r*vocab+gold[r]]))+std::log(sum);
    assert(std::isfinite(out[r]) && out[r]>=0);
    assert(std::abs(double(out[r])-oracle)<=2e-5+2e-6*std::abs(oracle));
    std::cout<<r<<" "<<out[r]<<" "<<oracle<<"\n"; // every output, including invalid rows above
  }
  // Smallest vocabulary: gold 0 has exactly NLL 0; zero rows preserves sentinel.
  x[0]=123; gold[0]=0; out[0]=-99;
  try {
    strata::kernels::row_target_nll(x,0,1,gold,out,&q); rt->wait(); assert(out[0]==-99);
    strata::kernels::row_target_nll(x,1,1,gold,out,&q); rt->wait(); assert(out[0]==0);
  } catch (...) { strata::sycl_backend::cleanup_fail_stop("minimal NLL fixture completion unknown"); }
  std::cout<<"vocab1 "<<out[0]<<" oracle 0\n";
}
