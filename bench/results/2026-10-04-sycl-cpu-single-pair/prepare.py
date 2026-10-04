from pathlib import Path
import re
s=Path('src/kernels/cpu/iq_single_avx2.cpp').read_text()
def close(text,start):
 level=0
 for i in range(start,len(text)):
  if text[i]=='{':level+=1
  if text[i]=='}':
   level-=1
   if not level:return i
 raise ValueError('unclosed')
def rename(t,lane):
 # Persistent per-dot state. Inner temporaries have their own braced scope.
 names=['x','d','q3','gas','qs','qh','signs','scales8','scales16','sumi1','sumi2','accumf']
 return re.sub(r'(?<!\.)\b('+'|'.join(names)+r')\b',lambda m:m[0]+str(lane),t)
for variant,unroll in [(6,1),(7,2),(8,4)]:
 new=s
 for fn in ['iq3xxs_dot','iq2s_dot','iq3s_dot']:
  a=new.index('__attribute__((noinline)) void '+fn+'(');brace=new.index('{',a);end=close(new,brace)
  body=new[brace+1:end]
  ptr=re.search(r'    const (block_\w+) \* GGML_RESTRICT x = .*?;\n',body);assert ptr
  block=ptr[1]
  head=body[ptr.end():body.index('    __m256 accumf')]
  loop=body.index('    for (int i = 0; i < nb; ++i)');ob=body.index('{',loop);oe=close(body,ob)
  outer=body[ob+1:oe]
  ib=outer.index('        for (int ib32');ibb=outer.index('{',ib);ibe=close(outer,ibb)
  pre=outer[:ib];pre=re.sub(r'\s*#pragma clang loop unroll_count\(\d+\)\n','\n',pre)
  pre=re.sub(r'        const int8_t  \* GGML_RESTRICT q8 = y\[i\]\.qs;\n','',pre)
  inner=outer[ibb+1:ibe]
  qloads=''
  for suffix in [1,2]:
   m=re.search(r'            const __m256i q8_'+str(suffix)+r' = .*?; q8 \+= 32;\n',inner);assert m
   qloads+=m[0];inner=inner[:m.start()]+inner[m.end():]
  post=outer[ibe+1:]
  tail=body[oe+1:]
  final=''
  for lane in [0,1]:final+=rename(tail,lane).replace('*s =',f'*s{lane} =')
  result=f'''__attribute__((noinline)) void {fn}(int n, float* s0, float* s1, const void* vx0, const void* vx1, const void* vy) {{
    assert(n % QK_K == 0);
    const {block} * GGML_RESTRICT x0 = static_cast<const {block} *>(vx0);
    const {block} * GGML_RESTRICT x1 = static_cast<const {block} *>(vx1);
'''+head+'''    __m256 accumf0 = _mm256_setzero_ps(), accumf1 = _mm256_setzero_ps();
    for (int i=0; i<nb; ++i) {
'''+rename(pre,0)+rename(pre,1)+'''        const int8_t* q8 = y[i].qs;
'''+f'        #pragma clang loop unroll_count({unroll})\n'+outer[ib:ibb+1]+'\n'+qloads+'            {\n'+rename(inner,0)+'            }\n            {\n'+rename(inner,1)+'            }\n        }\n'+rename(post,0)+rename(post,1)+'    }\n'+final+'}'
  new=new[:a]+result+new[end+1:]
 new=new.replace('        dot(n, &g, 0, gate, 0, act, 0, 1);\n        dot(n, &u, 0, gate + up_off, 0, act, 0, 1);','        dot(n, &g, &u, gate, gate + up_off, act);')
 new=new.replace('iq256_single_gu_rows(',f'iq256_single_gu_rows_v{variant}(')
 Path(f'/tmp/strata-sycl-goal-single-pair-v{variant}.cpp').write_text(new)
p=Path('/tmp/strata-sycl-goal-single-dispatch-bench.cpp').read_text()
p=p.replace('namespace strata::kernels::cpu {','namespace strata::kernels::cpu {\n'+''.join(f'bool iq256_single_gu_rows_v{v}(int,const uint8_t*,size_t,size_t,int,const void*,float*,int,int);\n' for v in [6,7,8]))
p=p.replace('funcs[]={iq256_single_gu_rows,iq256_single_gu_rows_v1,iq256_single_gu_rows_v2,iq256_single_gu_rows_v3}', 'funcs[]={iq256_single_gu_rows,iq256_single_gu_rows_v6,iq256_single_gu_rows_v7,iq256_single_gu_rows_v8}')
Path('/tmp/strata-sycl-goal-single-pair-bench.cpp').write_text(p)
