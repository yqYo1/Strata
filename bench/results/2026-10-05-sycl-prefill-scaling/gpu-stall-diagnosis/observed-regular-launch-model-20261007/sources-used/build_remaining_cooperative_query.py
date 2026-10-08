"""Offline resource query build with unchanged production IQ/prefill objects."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'remaining-cooperative-query';out.mkdir(mode=0o700)
observed=json.loads((base/'owned-observed-regular-launch-short-launch-flags.json').read_text())
rows=[r for r in observed['rows'] if r['cooperative']]
assert len(rows)==7
assert rows and all(not r['kernel'].startswith('unknown:') for r in rows)
(out/'targets.txt').write_text(''.join(r['kernel']+' '+ ' '.join(map(str,r['sycl_local']))+' '+str(r['max_total_groups'])+'\n' for r in rows))
receipt=json.loads((base/'dequant-no-root-refresh-build/record.json').read_text());assert receipt['passed']
argv=receipt['original_compile_argv'].copy()
for flag,value in [('-c',base/'prefill_cooperative_query.cpp'),('-o',out/'query.o'),('-MT',out/'query.o'),('-MF',out/'query.d')]:argv[argv.index(flag)+1]=str(value)
build=root/'build-sycl-refresh-20261007'
objects=[build/'CMakeFiles/strata_prefill.dir/src/prefill/kernels.dp.cpp.o']+[build/('CMakeFiles/strata_kernels.dir/src/kernels/cuda/'+name+'.dp.cpp.o') for name in ['qsa_decode_attn','qsa_select','verify_kernels']]+[build/'libstrata_kernels.a']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):digest(p) for p in objects}
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
for k in ['LD_PRELOAD','LD_LIBRARY_PATH']:env.pop(k,None)
r={'scope':'Offline actual unchanged remaining observed object resource query build; no GPU execution', 'passed':False,'steps':[],
   'source_sha256':digest(base/'prefill_cooperative_query.cpp'),'controller_sha256':digest(Path(__file__)),
   'targets_sha256':digest(out/'targets.txt'),'target_count':len(rows),'production_inputs_sha256':before,
   'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    link=[argv[0],'-O3','-DNDEBUG','-fsycl','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-fsycl-device-code-split=per_kernel',str(out/'query.o')]+list(map(str,objects))+['-o',str(out/'query')]
    for label,args in [('compile',argv),('link',link)]:
        start=time.monotonic()
        with (out/(label+'.stdout')).open('wb') as stdout,(out/(label+'.stderr')).open('wb') as stderr:p=subprocess.run(args,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
        r['steps'].append(dict(label=label,argv=args,exit_code=p.returncode,seconds=time.monotonic()-start));assert p.returncode==0,label
    assert before=={str(p):digest(p) for p in objects}
    r.update(passed=True,binary_sha256=digest(out/'query'),production_inputs_unchanged=True)
except BaseException as e:r['error']=repr(e);raise
finally:
    r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'build-record.json').write_text(json.dumps(r,indent=2)+'\n')
print('actual prefill resource query offline build PASS',len(rows),'targets')
