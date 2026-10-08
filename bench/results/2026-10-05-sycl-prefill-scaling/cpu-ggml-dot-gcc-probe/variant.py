"""Measure a contracted GCC build or an icx F16C-scale build, without a GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('variant',choices=['gcc-contract','icx-f16c'])
a=p.parse_args()
root=Path(__file__).resolve().parents[4]
base=Path.home()/'.local/state/strata-sycl/cpu-ggml-dot-gcc-probe'
out=base/a.variant;out.mkdir(exist_ok=True)
ggml=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml')
source=ggml/'src/ggml-cpu/arch/x86/quants.c'
icx='/opt/intel/oneapi/compiler/2026.1/bin/icx'
icpx='/opt/intel/oneapi/compiler/2026.1/bin/icpx'
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
commands=[]
def run(cmd,**kwargs):
    commands.append(list(map(str,cmd)))
    return subprocess.run(cmd,env=env,check=True,**kwargs)
shutil.copy2(base/'reference_quants.o',out/'reference_quants.o')
flags=['-O3','-DNDEBUG','-std=gnu11','-march=native','-fno-associative-math',
       '-DGGML_SCHED_MAX_COPIES=4','-DGGML_USE_CPU_REPACK','-DSTRATA_VERSION="0.1.39-sycl"',
       '-D_GNU_SOURCE','-D_XOPEN_SOURCE=600']
includes=['-I'+str(x) for x in (ggml,ggml/'src',ggml/'src/ggml-cpu',ggml/'include')]
wrapper='''#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include <immintrin.h>
#undef GGML_CPU_FP16_TO_FP32
#define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
'''+f'#include "{source}"\n'
with (out/'build.log').open('w') as log:
    if a.variant=='icx-f16c':
        (out/'f16c.c').write_text(wrapper)
        run([icx,*flags,'-fp-model=precise',*includes,'-c',str(out/'f16c.c'),'-o',str(out/'alternate.o')],stdout=log,stderr=subprocess.STDOUT)
    else:
        run(['gcc',*flags,'-ffp-contract=fast',*includes,'-c',str(source),'-o',str(out/'alternate.o')],stdout=log,stderr=subprocess.STDOUT)
    symbols=subprocess.check_output(['nm','-g','--defined-only',str(out/'alternate.o')],text=True)
    pairs=[(x.split()[-1],'alternate_'+x.split()[-1]) for x in symbols.splitlines()]
    (out/'symbols.txt').write_text(''.join(f'{x} {y}\n' for x,y in pairs))
    run(['objcopy','--redefine-syms='+str(out/'symbols.txt'),str(out/'alternate.o'),str(out/'alternate_quants.o')],stdout=log,stderr=subprocess.STDOUT)
    harness=(base/'probe.cpp').read_text().replace('gcc_','alternate_')
    harness=harness.replace('using Dot =', '#include <immintrin.h>\nextern "C" float ggml_table_f32_f16[65536];\nusing Dot =')
    proof='''ggml_cpu_init();
    unsigned checked_half=0;
    for(unsigned h=0;h<65536;++h) if((h&0x7c00)!=0x7c00) {
        float original=ggml_table_f32_f16[h],converted=_cvtsh_ss((uint16_t)h);
        require(std::memcmp(&original,&converted,4)==0,"finite half conversion differs");
        ++checked_half;
    }
    std::printf("{\\"kind\\":\\"half-conversion\\",\\"checked_finite_halves\\":%u,\\"all_bits_equal\\":true}\\n",checked_half);
    std::mt19937 rng(1913);'''
    assert 'ggml_cpu_init();std::mt19937 rng(1913);' in harness
    harness=harness.replace('ggml_cpu_init();std::mt19937 rng(1913);',proof)
    (out/'probe.cpp').write_text(harness)
    run([icpx,'-O3','-DNDEBUG','-std=c++20','-fp-model=precise','-mavx2','-mfma','-mf16c',
         '-I'+str(root/'include'),'-I'+str(ggml/'include'),'-I'+str(ggml/'src'),
         str(out/'probe.cpp'),str(out/'reference_quants.o'),str(out/'alternate_quants.o'),
         str(root/'build-sycl-upstream-jit/ggml/src/libggml-cpu.a'),
         str(root/'build-sycl-upstream-jit/ggml/src/libggml-base.a'),'-lpthread','-ldl','-o',str(out/'probe')],stdout=log,stderr=subprocess.STDOUT)
manifest={'variant':a.variant,'commands':commands,'symbol_mappings':pairs,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'strata_variables':{k:v for k,v in env.items() if k.startswith('STRATA_')},
          'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (source,out/'reference_quants.o',out/'alternate.o',out/'alternate_quants.o',out/'probe.cpp',out/'probe')}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
model=Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
validation=[];timings={}
for i in range(1,4):
    with (out/f'round{i}.jsonl').open('w') as stdout,(out/f'round{i}.stderr').open('w') as stderr:
        run([str(out/'probe'),str(model)],stdout=stdout,stderr=stderr)
    rows=[json.loads(x) for x in (out/f'round{i}.jsonl').read_text().splitlines()]
    assert rows[-1]['kind']=='completed'
    for r in rows:
        if r['kind']=='validation':validation.append(dict(r,run=i))
        if r['kind']=='timing':timings.setdefault((r['type'],r['experts']),[]).append(dict(r,run=i))
summary={'variant':a.variant,'scope':'CPU single-token dots only; not TG','validation':validation,'timings':[]}
for (type_,experts),rows in sorted(timings.items()):
    summary['timings'].append({'type':type_,'experts':experts,'pairs':len(rows),
                              'median_speed_ratio':statistics.median(r['speed_ratio'] for r in rows),
                              'process_medians':[statistics.median(r['speed_ratio'] for r in rows if r['run']==i) for i in range(1,4)],
                              'min_speed_ratio':min(r['speed_ratio'] for r in rows),'max_speed_ratio':max(r['speed_ratio'] for r in rows)})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['timings'],indent=2))
print('Mismatches:',[(r['run'],r['type'],r['mismatched_floats']) for r in validation if not r['all_bits_equal']])
