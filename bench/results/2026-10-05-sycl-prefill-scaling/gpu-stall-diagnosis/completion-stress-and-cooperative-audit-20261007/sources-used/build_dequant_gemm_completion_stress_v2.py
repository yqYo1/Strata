"""Offline actual-wrapper stress fixture build, regular kernels only."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'dequant-gemm-completion-stress-build-v2';out.mkdir(mode=0o700)
receipt=json.loads((base/'dequant-no-root-refresh-build/record.json').read_text())
assert receipt['passed'] and receipt['production_inputs_unchanged']
obj=base/'dequant-no-root-refresh-build/iq_kernels.dp.cpp.o'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=digest(obj)
argv=receipt['original_compile_argv'].copy()
for flag,value in [('-c',base/'dequant_gemm_completion_stress.cpp'),('-o',out/'probe.o'),('-MT',out/'probe.o'),('-MF',out/'probe.d')]:
    argv[argv.index(flag)+1]=str(value)
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu',MKLROOT='/opt/intel/oneapi/mkl/2026.1')
for k in ['LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(k,None)
r={'scope':'Offline actual regular IQ wrappers plus unchanged production GEMM/SwiGLU fixture build; no GPU execution', 'passed':False,'steps':[],
   'source_sha256':digest(base/'dequant_gemm_completion_stress.cpp'),'controller_sha256':digest(Path(__file__)),
   'kernel_object_sha256':before,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    link=[argv[0],'-O3','-DNDEBUG','-fsycl','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-fsycl-device-code-split=per_kernel',str(out/'probe.o'),str(obj),'-o',str(out/'probe')]
    link.insert(-2,'-qmkl=sequential')
    for path in ['libstrata_prefill.a','libstrata_kernels.a']:
        link.insert(-2,str(root/'build-sycl-refresh-20261007'/path))
    for label,args in [('compile',argv),('link',link)]:
        start=time.monotonic()
        with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:
            p=subprocess.run(args,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
        r['steps'].append(dict(label=label,argv=args,exit_code=p.returncode,seconds=time.monotonic()-start))
        assert p.returncode==0,label
    assert digest(obj)==before
    r['passed']=True;r['binary_sha256']=digest(out/'probe')
except BaseException as e:r['error']=repr(e);raise
finally:
    r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
print('completion stress offline build PASS')
