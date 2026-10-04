#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
extern "C" __m128 __svml_expf4(__m128);
extern "C" __m128 __svml_expf4_ha(__m128);
extern "C" __m256 __svml_expf8(__m256);
extern "C" __m256 __svml_expf8_ha(__m256);
int main(){
 constexpr size_t N=1<<21;
 std::vector<float> input(N),ref(N),got(N);uint32_t state=12345;
 float (*volatile scalar)(float)=std::exp;
 for(size_t i=0;i<N;++i){state=state*1664525u+1013904223u;
  if(i<N/2)input[i]=float(int32_t(state))*(80.f/2147483648.f);
  else {uint32_t bits=(state&0x80000000u)|((state&0x7fffffffu)%0x42a00000u);std::memcpy(&input[i],&bits,4);}
  ref[i]=scalar(input[i]);
 }
 for(int variant=0;variant<4;++variant){
  if(variant<2)for(size_t i=0;i<N;i+=4){auto x=_mm_loadu_ps(input.data()+i);auto y=variant==0?__svml_expf4(x):__svml_expf4_ha(x);_mm_storeu_ps(got.data()+i,y);}
  else for(size_t i=0;i<N;i+=8){auto x=_mm256_loadu_ps(input.data()+i);auto y=variant==2?__svml_expf8(x):__svml_expf8_ha(x);_mm256_storeu_ps(got.data()+i,y);}
  size_t unequal=0;for(size_t i=0;i<N;++i)if(memcmp(&ref[i],&got[i],4)){if(unequal<3)printf("example v%d x=%a scalar=%a vector=%a\n",variant,input[i],ref[i],got[i]);++unequal;}
  printf("variant=%d total=%zu unequal=%zu\n",variant,N,unequal);
 }
}
