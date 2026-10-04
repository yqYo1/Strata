from pathlib import Path
src=Path('../ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c').read_text()
prefix='''// Experimental copies of GGML AVX2 kernels (MIT), for isolated comparison only.
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT restrict
#define UNUSED(x) (void)(x)
#define GGML_CPU_FP16_TO_FP32(h) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(h))))
'''
for start,end in [('static inline float hsum_float_8(', '\n}\n'),('static inline __m256i get_scale_shuffle_k4(', '\n}\n'),('static const int8_t keven_signs_q2xs[1024]', '\n};')]:
 p=src.index(start);q=src.index(end,p)+len(end);prefix+=src[p:q]+'\n'
out=prefix
for ty,name in [(21,'iq3_s')]:
 p=src.index('void ggml_vec_dot_'+name+'_q8_K(');q=src.index('#elif defined(__AVX__)',p)
 raw=src[p:q].replace('#if defined(__AVX2__)','')+'}\n'
 for variant in [0,1,2,4]:
  body=raw.replace('ggml_vec_dot_'+name+'_q8_K','single_'+str(ty)+'_'+str(variant))
  if variant:
   body=body.replace('        for (int ib32',f'        #pragma clang loop unroll_count({variant})\n        for (int ib32')
  out+=body
Path('/tmp/strata-sycl-goal-single-iq3s-kernels.c').write_text(out)
