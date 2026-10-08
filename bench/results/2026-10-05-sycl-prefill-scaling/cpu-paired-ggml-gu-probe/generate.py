"""Pair original ggml AVX2 gate/up loops, retaining each dot's exact order."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root=Path(__file__).resolve().parents[4]
out=Path.home()/'.local/state/strata-sycl/cpu-paired-ggml-gu-probe'
out.mkdir(exist_ok=True)
source=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/src/ggml-cpu/arch/x86/quants.c')
text=source.read_text()
original=root/'build-sycl-upstream-jit/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o'
symbols=subprocess.check_output(['nm','-g','--defined-only',str(original)],text=True)
exports=[line.split()[-1] for line in symbols.splitlines()]
header=''.join(f'#define {x} unused_{x}\n' for x in exports)+f'#include "{source}"\n'

def braces(text,start):
    assert text[start]=='{'
    level=0
    for p in range(start,len(text)):
        if text[p]=='{':level+=1
        elif text[p]=='}':
            level-=1
            if level==0:return p
    raise ValueError('unclosed block')

names=('x','accumf','sumi1','sumi2','d','qs','qh','signs','q3','gas','idx','aux32','aux64',
       'scales8','scales16','idx_l','q2_1','q2_2','aux256','s2_1','s2_2','q8s_1','q8s_2',
       'dot1','dot2','ls1','ls2','p1','p2')
def lane(text,suffix):
    # Structure field names such as x[i].d and x[i].qs stay unchanged.
    return re.sub(r'(?<![.>])\b('+'|'.join(names)+r')\b',lambda m:m[0]+'_'+suffix,text)

functions=[]
for name in ('iq3_xxs','iq3_s','iq2_s'):
    start=text.index('void ggml_vec_dot_'+name+'_q8_K(')
    end=text.index('\nvoid ',start+1)
    function=text[start:end]
    section=function.split('#if defined(__AVX2__)\n',1)[1].split('#elif defined(__AVX__)\n',1)[0]
    outer=section.index('for (int i = 0; i < nb; ++i)')
    pre=section[:outer]
    declarations=[]
    for declaration in ('index_t idx;','uint32_t aux32[2];','uint64_t aux64;'):
        if declaration in pre:
            pre=pre.replace(declaration,'')
            declarations += [lane(declaration,k) for k in ('g','u')]
    accum='__m256 accumf = _mm256_setzero_ps();'
    assert pre.count(accum)==1
    pre=pre.replace(accum,'\n'.join(lane(accum,k) for k in ('g','u')))
    ob=section.index('{',outer);oe=braces(section,ob)
    body=section[ob+1:oe]
    inner=body.index('for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2)')
    before=body[:inner]
    q8=re.search(r'^\s*const int8_t\s*\*.*\bq8 = y\[i\]\.qs;\s*$',before,re.M)
    assert q8
    shared=q8[0].strip()
    before=before[:q8.start()]+before[q8.end():]
    ib=body.index('{',inner);ie=braces(body,ib)
    inside=body[ib+1:ie]
    loads=[]
    for which in (1,2):
        load=re.search(r'^\s*const __m256i q8_'+str(which)+r' = [^\n]+$',inside,re.M)
        assert load and 'q8 += 32;' in load[0]
        loads.append(load[0].strip())
        inside=inside[:load.start()]+inside[load.end():]
    after=body[ie+1:]
    final=section[oe+1:]
    assert final.count('*s =')==1
    blocktype='block_'+name
    paired=f'''void paired_{name}(int n,float* gate,float* up,const void* vg,const void* vu,const void* vy) {{
    assert(n % QK_K == 0);
    const {blocktype}* GGML_RESTRICT x_g=vg;
    const {blocktype}* GGML_RESTRICT x_u=vu;
    const block_q8_K* GGML_RESTRICT y=vy;
    const int nb=n/QK_K;
'''+pre+'\n'+'\n'.join(declarations)+'''
    for(int i=0;i<nb;++i) {
'''+shared+'\n'+lane(before,'g')+lane(before,'u')+'''
        for(int ib32=0;ib32<QK_K/32;ib32+=2) {
'''+ '\n'.join(loads)+'\n'+lane(inside,'g')+lane(inside,'u')+'''
        }
'''+lane(after,'g')+lane(after,'u')+'''
    }
'''+lane(final,'g').replace('*s =','*gate =')+lane(final,'u').replace('*s =','*up =')+'}\n'
    functions.append(paired)
generated=header+'\n'.join(functions)
(out/'paired.c').write_text('\n'.join(line.rstrip() for line in generated.splitlines())+'\n')
for f in ('probe.cpp','rows.cpp'):shutil.copy2(Path(__file__).with_name(f),out/f)
manifest={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'production_object_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
          'paired_source_sha256':hashlib.sha256((out/'paired.c').read_bytes()).hexdigest(),
          'source_exports_renamed':exports,'formats':[18,21,22],
          'math':'Each original integer partial sum, per-block FMA and final hsum is retained; each q8_1/q8_2 load is shared'}
(out/'generation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Generated paired original-ggml loops:',out/'paired.c')
