import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[4]
probe = Path.home()/'.local/state/strata-sycl/cpu-ggml-dot-gcc-probe'
probe.mkdir(exist_ok=True)
ggml = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml')
source = ggml/'src/ggml-cpu/arch/x86/quants.c'
original = root/'build-sycl-upstream-jit/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o'
commands = []
def run(args):
    commands.append(list(map(str,args)))
    subprocess.run(args,check=True)
flags = ['-O3','-DNDEBUG','-std=gnu11','-march=native','-ffp-contract=off','-fno-associative-math',
         '-DGGML_SCHED_MAX_COPIES=4','-DGGML_USE_CPU_REPACK','-DSTRATA_VERSION="0.1.39-sycl"',
         '-D_GNU_SOURCE','-D_XOPEN_SOURCE=600']
includes = ['-I'+str(p) for p in (ggml,ggml/'src',ggml/'src/ggml-cpu',ggml/'include')]
shutil.copy2(original,probe/'production.o')
shutil.copy2(Path(__file__).with_name('probe.cpp'),probe/'probe.cpp')
run(['gcc',*flags,*includes,'-c',str(source),'-o',str(probe/'gcc.o')])
mappings = {}
for filename,prefix in (('production.o','reference_'),('gcc.o','gcc_')):
    obj = probe/filename
    symbols = subprocess.check_output(['nm','-g','--defined-only',str(obj)],text=True)
    pairs = [(line.split()[-1],prefix+line.split()[-1]) for line in symbols.splitlines()]
    mapping = probe/(prefix+'symbols.txt')
    mapping.write_text(''.join(f'{old} {new}\n' for old,new in pairs))
    renamed = probe/(prefix+'quants.o')
    run(['objcopy','--redefine-syms='+str(mapping),str(obj),str(renamed)])
    mappings[filename] = pairs
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tracked = [source,ggml/'src/ggml-common.h',original,probe/'gcc.o',probe/'reference_quants.o',
           probe/'gcc_quants.o',probe/'probe.cpp',root/'build-sycl-upstream-jit/ggml/src/libggml-cpu.a',
           root/'build-sycl-upstream-jit/ggml/src/libggml-base.a',root/'build-sycl-upstream-jit/strata']
manifest = {'commands':commands,'symbol_mappings':mappings,
            'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
            'ggml_revision':subprocess.check_output(['git','-C',str(ggml),'rev-parse','HEAD'],text=True).strip(),
            'ggml_status':subprocess.check_output(['git','-C',str(ggml),'status','--short'],text=True),
            'sha256':{str(p):digest(p) for p in tracked}}
(probe/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared actual baseline and renamed GCC objects:',probe)
