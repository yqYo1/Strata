from pathlib import Path
import re
s=Path('src/kernels/cpu/iq_single_avx2.cpp').read_text()
s=s.replace('#include "ggml-common.h"','#include "ggml-common.h"\n#include "/tmp/strata-sycl-goal-index-format.hpp"')
s=s.replace('block_iq3_s','preindexed_iq3s').replace('block_iq2_s','preindexed_iq2s')
s=s.replace('const uint8_t * GGML_RESTRICT qs = x[i].qs;','const uint16_t * GGML_RESTRICT qs = x[i].indices;')
s=s.replace('        const uint8_t * GGML_RESTRICT qh = x[i].qh;\n','')
s=s.replace('(const uint16_t *)(x[i].qs + QK_K/8)','(const uint16_t *)x[i].signs')
for k in range(16):s=s.replace(f'qs[{k}] | (((qh[ib32+{k//8}] >> {k%8}) & 1) << 8)',f'qs[{k}]')
for k in range(8):s=s.replace(f'qs[{k}] | ((qh[ib32+{k//4}] << {8-2*(k%4)}) & 0x300)',f'qs[{k}]')
assert 'qh[' not in s
s=s.replace('if (type != 18 && type != 21 && type != 22)', 'if (type != 21 && type != 22)').replace('type == 18 ? iq3xxs_dot : type == 21 ? iq3s_dot : iq2s_dot','type == 21 ? iq3s_dot : iq2s_dot')
for v in range(1,6):
 new=s
 for fn,ty in [('iq2s_dot',22),('iq3s_dot',21)]:
  a=new.index('__attribute__((noinline)) void '+fn+'(')
  b=new.index('\n}\n',a)+3;part=new[a:b]
  if v in [2,3,4]:
   part=re.sub(r'        #pragma clang loop unroll_count\(\d+\)\n','',part)
   part=part.replace('        for (int ib32',f'        #pragma clang loop unroll_count({[1,2,4][v-2]})\n        for (int ib32')
  if v==5:
   for k in [1,2]:
    if ty==21:expr=f'_mm256_i32gather_epi32((const int*)iq3s_grid, _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(qs+{8*(k-1)}))), 4)'
    else:expr=f'_mm256_i64gather_epi64((const long long*)iq2s_grid, _mm256_cvtepu16_epi64(_mm_loadl_epi64((const __m128i*)(qs+{4*(k-1)}))), 8)'
    part,count=re.subn(r'const __m256i q2_'+str(k)+r' = .*?;', 'const __m256i q2_'+str(k)+' = '+expr+';',part,count=1,flags=re.S);assert count==1
  new=new[:a]+part+new[b:]
 new=new.replace('iq256_single_gu_rows(',f'iq256_single_gu_rows_index_v{v}(')
 Path(f'/tmp/strata-sycl-goal-index-v{v}.cpp').write_text(new)

# Combined candidate: scalar indices, IQ3_S unroll2 and IQ2_S unroll4.
new=s
a=new.index('__attribute__((noinline)) void iq3s_dot(');b=new.index('\n}\n',a)+3
part=new[a:b].replace('        for (int ib32','        #pragma clang loop unroll_count(2)\n        for (int ib32')
new=new[:a]+part+new[b:]
new=new.replace('iq256_single_gu_rows(', 'iq256_single_gu_rows_index_v6(')
Path('/tmp/strata-sycl-goal-index-v6.cpp').write_text(new)
