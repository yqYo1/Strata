from pathlib import Path
src=Path('../ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c').read_text()
base=Path('/tmp/strata-sycl-goal-single-iq3s-kernels.c').read_text()
prefix=base[:base.index('void single_21_0(')]
p=src.index('void ggml_vec_dot_iq3_s_q8_K(');q=src.index('#elif defined(__AVX__)',p)
raw=src[p:q].replace('#if defined(__AVX2__)','')+'}\n'
start=raw.index('            const __m256i idx_l')
end=raw.index('            __m256i aux256',start)
old=raw[start:end]
gather=old[:old.index('            // At leat')]+'''            const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
'''
scalar=''
for half in range(2):
 scalar+=f'            const __m256i q2_{half+1} = _mm256_set_epi32(\n                '
 scalar+=', '.join(f'iq3s_grid[qs[{half*8+k}] | (((qh[ib32+{half}] >> {k}) & 1) << 8)]' for k in range(7,-1,-1))
 scalar+=');\n'
scalar+='            qs += 16;\n'
out=prefix
for variant in [0,10,11,12,14]:
 body=raw.replace('ggml_vec_dot_iq3_s_q8_K',f'single_21_{variant}')
 if variant in [10,12]:body=body.replace(old,scalar)
 if variant in [11,14]:body=body.replace(old,gather)
 if variant in [12,14]:body=body.replace('        for (int ib32','        #pragma clang loop unroll_count(1)\n        for (int ib32')
 out+=body
Path('/tmp/strata-sycl-goal-single-iq3s-index-kernels.c').write_text(out)
s=Path('/tmp/strata-sycl-goal-single-iq3s-bench.cpp').read_text()
s=s.replace('DECL(21,0) DECL(21,1) DECL(21,2) DECL(21,4)','DECL(21,0) DECL(21,10) DECL(21,11) DECL(21,12) DECL(21,14)')
s=s.replace('single_21_0,single_21_1,single_21_2,single_21_4','single_21_0,single_21_10,single_21_11,single_21_12,single_21_14').replace('variants[]={0,1,2,4}','variants[]={0,10,11,12,14}').replace('vi<4','vi<5')
Path('/tmp/strata-sycl-goal-single-iq3s-index-bench.cpp').write_text(s)
for kind in ['build','run']:
 s=Path(f'/tmp/strata-sycl-goal-single-iq3s-{kind}.sh').read_text().replace('single-iq3s-','single-iq3s-index-')
 Path(f'/tmp/strata-sycl-goal-single-iq3s-index-{kind}.sh').write_text(s)
