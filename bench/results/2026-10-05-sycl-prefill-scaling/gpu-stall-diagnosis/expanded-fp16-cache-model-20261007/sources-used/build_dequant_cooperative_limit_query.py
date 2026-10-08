"""Build a query of the actual dequant kernels' cooperative limits; no GPU submission."""
from pathlib import Path
import json,subprocess,time,os,hashlib,datetime
base=Path(__file__).parent
out=base/'dequant-cooperative-limit-query'
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
receipt=json.loads((base/'dequant-no-root-refresh-build/record.json').read_text());assert receipt['passed'] and receipt['production_inputs_unchanged']
obj=root/'build-sycl-refresh-20261007/CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o'
assert hashlib.sha256(obj.read_bytes()).hexdigest()==receipt['production_object_sha256']
av=receipt['original_compile_argv'].copy()
for flag,value in [('-c',out/'query.cpp'),('-o',out/'query.o'),('-MT',out/'query.o'),('-MF',out/'query.d')]:av[av.index(flag)+1]=str(value)
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
r=dict(passed=False,steps=[],scope='Actual dequant object offline query build; no kernel executed',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 for name,args in [('compile',av),('link',[av[0],'-O3','-DNDEBUG','-fsycl','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-fsycl-device-code-split=per_kernel',str(out/'query.o'),str(obj),'-o',str(out/'query')])]:
  t=time.monotonic()
  with (out/(name+'.stdout')).open('wb') as stdout,(out/(name+'.stderr')).open('wb') as stderr:p=subprocess.run(args,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
  r['steps'].append(dict(label=name,argv=args,exit_code=p.returncode,elapsed_seconds=time.monotonic()-t));assert p.returncode==0,name
 r['passed']=True;r['binary_sha256']=hashlib.sha256((out/'query').read_bytes()).hexdigest();r['iq_object_sha256']=hashlib.sha256(obj.read_bytes()).hexdigest()
except BaseException as e:r['error']=repr(e);raise
finally:
 r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'build-record.json').write_text(json.dumps(r,indent=2)+'\n')
print('cooperative-limit query build PASS')
