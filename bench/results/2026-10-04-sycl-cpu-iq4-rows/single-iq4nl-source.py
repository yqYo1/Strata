from pathlib import Path
src=Path('../ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c').read_text()
base=Path('/tmp/strata-sycl-goal-single-kernels.c').read_text()
prefix=base[:base.index('static inline float hsum_float_8')]
prefix+='#define MM256_SET_M128I(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(b),(a),1)\n'
for start in ['static inline float hsum_float_8(', 'static inline __m256i mul_add_epi8(']:
 p=src.index(start);q=src.index('\n}',p)+2;prefix+=src[p:q]+'\n'
p=src.index('void ggml_vec_dot_iq4_nl_q8_0(');q=src.index('\nvoid ggml_vec_dot_iq4_xs_q8_K(',p)
raw=src[p:q];p=raw.index('#elif defined __AVX__');q=raw.index('#endif',p)+len('#endif')
raw=raw[:p]+raw[q:];raw=raw.replace('#if defined __AVX2__','')
a=raw.index('        const __m256i q4b_1');b=raw.index('        const __m256i p16_1',a);old=raw[a:b]
wide=''
for idx in [1,2]:
 wide+=f'        const __m256i q4b_{idx} = _mm256_shuffle_epi8(values256, _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bits_{idx},4),q4bits_{idx}), mask256));\n'
out=prefix
for v in [0,1,2,4,10,11,12,14]:
 body=raw.replace('ggml_vec_dot_iq4_nl_q8_0', f'single_20_{v}')
 if v>=10:
  body=body.replace(old,wide).replace('    const __m128i m4b  = _mm_set1_epi8(0x0f);','    const __m256i values256 = _mm256_broadcastsi128_si256(values128);\n    const __m256i mask256 = _mm256_set1_epi8(0x0f);')
 if v%10:
  body=body.replace('    for (; ib + 1', f'    #pragma clang loop unroll_count({v%10})\n    for (; ib + 1')
 out+=body
Path('/tmp/strata-sycl-goal-single-iq4nl-kernels.c').write_text(out)
s=Path('/tmp/strata-sycl-goal-single-iq3s-bench.cpp').read_text()
s=s.replace('DECL(21,0) DECL(21,1) DECL(21,2) DECL(21,4)', 'DECL(20,0) DECL(20,1) DECL(20,2) DECL(20,4) DECL(20,10) DECL(20,11) DECL(20,12) DECL(20,14)')
a=s.index('  auto*g=model.find(');b=s.index('  auto ref_fn=',a)
s=s[:a]+'''  auto*g=model.find(pre+"ffn_down_exps.weight",&gs);
  if(!g||g->type!=20||!seen.insert(g->type).second)continue;
  if(g->shape.size()!=3||g->shape[2]<512||!model.in_bounds(*g,gs))throw std::runtime_error("unexpected geometry");
  const int n=g->shape[0],rows=g->shape[1],nb=n/32;const size_t row=nb*sizeof(block_iq4_nl),up=rows*row;
  std::vector<uint8_t> blobs(E*up+4096);
  for(int e=0;e<E;++e)memcpy(blobs.data()+e*up,model.shard(gs).tensor_data(*g)+e*7*up,up);
  std::vector<block_q8_0> acts(nb);uint32_t state=12345;
  for(auto&x:acts){x.d=0x211f;for(int k=0;k<32;++k){state=state*1664525u+1013904223u;x.qs[k]=(int)(state%255)-127;}}
''' +s[b:]
s=s.replace('single_21_0,single_21_1,single_21_2,single_21_4','single_20_0,single_20_1,single_20_2,single_20_4,single_20_10,single_20_11,single_20_12,single_20_14').replace('variants[]={0,1,2,4}','variants[]={0,1,2,4,10,11,12,14}').replace('vi<4','vi<8').replace('ref(E*rows*3)','ref(E*rows)')
a=s.index('   auto run=');b=s.index('   run(false,ref)',a)
s=s[:a]+'''   auto run=[&](bool candidate,std::vector<float>&out){auto fn=candidate?cand_fn:ref_fn;for(int e=0;e<E;++e)for(int r=0;r<rows;++r){const auto*p=blobs.data()+e*up+r*row;fn(n,&out[e*rows+r],0,p,0,acts.data(),0,1);}};
'''+s[b:]
Path('/tmp/strata-sycl-goal-single-iq4nl-bench.cpp').write_text(s)
for kind in ['build','run']:
 s=Path(f'/tmp/strata-sycl-goal-single-iq3s-{kind}.sh').read_text().replace('single-iq3s-', 'single-iq4nl-')
 Path(f'/tmp/strata-sycl-goal-single-iq4nl-{kind}.sh').write_text(s)
