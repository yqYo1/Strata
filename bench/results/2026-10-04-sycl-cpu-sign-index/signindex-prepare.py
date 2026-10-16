from pathlib import Path
import re
Path('/tmp/strata-sycl-goal-signindex-format.hpp').write_text('#pragma once\n#include <cstdint>\nstruct signindexed_iq3s { uint16_t d; uint16_t indices[64]; uint8_t scales[4]; };\nstatic_assert(sizeof(signindexed_iq3s)==134);\n')
s=Path('/tmp/strata-sycl-goal-index-v6.cpp').read_text().replace('#include "/tmp/strata-sycl-goal-index-format.hpp"','#include "/tmp/strata-sycl-goal-signindex-format.hpp"')
for fn in ('iq3xxs_dot','iq2s_dot'):
 a=s.index('__attribute__((noinline)) void '+fn+'(');b=s.index('\n}\n',a)+3;s=s[:a]+s[b:]
s=s.replace('preindexed_iq3s','signindexed_iq3s')
s=s.replace('namespace {\n','''namespace {
struct SignGrid {
    alignas(64) int32_t data[8192];
    SignGrid() {
        for (int signs=0; signs<16; ++signs) for (int idx=0; idx<512; ++idx) {
            const uint32_t raw=iq3s_grid[idx]; uint32_t word=0;
            for (int j=0; j<4; ++j) {
                const int v=(raw>>(8*j))&255;
                word |= uint32_t(uint8_t((signs>>j)&1 ? -v : v)) << (8*j);
            }
            data[idx | (signs<<9)]=int32_t(word);
        }
    }
};
const SignGrid signed_grid;
''',1)
s=s.replace('        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;\n','')
s=s.replace('iq3s_grid[qs[','signed_grid.data[qs[')
a=s.index('            __m256i aux256 =');b=s.index('            const uint16_t ls1',a)
s=s[:a]+'''            // Q8_K quantization produces [-127,127], allowing sign transfer.
            const __m256i dot1 = _mm256_maddubs_epi16(_mm256_abs_epi8(q8_1), _mm256_sign_epi8(q2_1, q8_1));
            const __m256i dot2 = _mm256_maddubs_epi16(_mm256_abs_epi8(q8_2), _mm256_sign_epi8(q2_2, q8_2));
'''+s[b:]
s=s.replace('if (type != 21 && type != 22)','if (type != 21)').replace('const auto dot = type == 21 ? iq3s_dot : iq2s_dot;','const auto dot = iq3s_dot;')
for v,u in ((1,1),(2,2),(3,4)):
 t=s.replace('unroll_count(2)',f'unroll_count({u})').replace('iq256_single_gu_rows_index_v6(',f'iq256_single_gu_rows_signindex_v{v}(')
 Path(f'/tmp/strata-sycl-goal-signindex-v{v}.cpp').write_text(t)
h=Path('/tmp/strata-sycl-goal-index-thread-bench.cpp').read_text().replace('/tmp/strata-sycl-goal-index-format.hpp','/tmp/strata-sycl-goal-signindex-format.hpp').replace('iq256_single_gu_rows_index_v','iq256_single_gu_rows_signindex_v')
h=h.replace('(g->type!=21&&g->type!=22)','g->type!=21').replace('type==21?sizeof(block_iq3_s):sizeof(block_iq2_s)','sizeof(block_iq3_s)').replace('type==21?sizeof(preindexed_iq3s):sizeof(preindexed_iq2s)','sizeof(signindexed_iq3s)')
a=h.index('  auto pack=[&]');b=h.index('  auto p0=',a)
h=h[:a]+'''  auto pack=[&]{
   auto*src=reinterpret_cast<const block_iq3_s*>(raw.data());auto*dst=reinterpret_cast<signindexed_iq3s*>(packed.data());
   for(size_t b=0;b<blocks;++b){dst[b].d=src[b].d;for(int k=0;k<64;++k)dst[b].indices[k]=src[b].qs[k]|(((src[b].qh[k/8]>>(k%8))&1)<<8)|(((src[b].signs[k/2]>>(4*(k%2)))&15)<<9);memcpy(dst[b].scales,src[b].scales,4);}
  };
'''+h[b:]
h=h.replace('Fn fns[]={iq256_single_gu_rows,iq256_single_gu_rows_signindex_v6};','Fn fns[]={iq256_single_gu_rows,iq256_single_gu_rows_signindex_v1,iq256_single_gu_rows_signindex_v2,iq256_single_gu_rows_signindex_v3};').replace('v<=1','v<=3').replace('seen.size()!=2','seen.size()!=1')
Path('/tmp/strata-sycl-goal-signindex-thread-bench.cpp').write_text(h)
