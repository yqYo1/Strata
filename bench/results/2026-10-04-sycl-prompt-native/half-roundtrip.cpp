#include <sycl/sycl.hpp>
#include "strata/kernels/f16_bits.hpp"
#include <iostream>
int main() {
 sycl::queue q;
 auto out=sycl::malloc_shared<uint16_t>(2*65536,q);
 q.parallel_for(sycl::range<1>(65536),[=](sycl::id<1> i) {
  uint16_t h=uint16_t(i[0]);
  sycl::half old=sycl::half(strata::kernels::f32_from_f16(h));
  sycl::half native=(h&0x7fff)<=0x7c00 ? sycl::bit_cast<sycl::half>(h) : old;
  out[2*i[0]]=sycl::bit_cast<uint16_t>(old);
  out[2*i[0]+1]=sycl::bit_cast<uint16_t>(native);
 }).wait_and_throw();
 int mismatches=0,finite_mismatches=0,nan_mismatches=0;
 for(int i=0;i<65536;++i) if(out[2*i]!=out[2*i+1]){++mismatches;if((i&0x7fff)<0x7c00)++finite_mismatches;else if((i&0x7fff)>0x7c00)++nan_mismatches;}
 std::cout<<"patterns=65536 mismatches="<<mismatches<<" finite_mismatches="<<finite_mismatches<<" nan_mismatches="<<nan_mismatches<<"\n";
 sycl::free(out,q);return mismatches?1:0;
}
