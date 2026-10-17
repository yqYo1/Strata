#!/usr/bin/env python3
"""Bounded nice19 syntax checks of the real candidate TUs; no codegen/GPU."""
import fcntl, hashlib, importlib.util, json, os, shlex, shutil, sys, datetime
from pathlib import Path
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=A/'moe-window-engine-syntax-v1.json'; root=B/'main-moe-window-engine-syntax-v1'
 assert not out.exists() and not root.exists()
 helper=A/'direct_owner_transient_artifacts_v1.py'
 assert sha(helper)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
 spec=importlib.util.spec_from_file_location('owned',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 database=B/'main-bf16-prefix-tilepack-engine-build-v1/build/compile_commands.json'
 entries=json.loads(database.read_text()); selected=['src/prefill/prefill.cpp','src/program/generate.cpp']
 sources={str(p.relative_to(P)):sha(p) for p in [Path(__file__).resolve(),P/'include/strata/prefill/moe_window_routes.hpp',P/'include/strata/prefill/prefill.hpp',P/'src/prefill/moe_window.inc',P/'src/prefill/prefill.cpp',P/'src/prefill/layer_major.inc',P/'src/program/generate.cpp']}
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir();(root/'tmp').mkdir()
  env={k:os.environ[k] for k in ('PATH','LD_LIBRARY_PATH') if k in os.environ}
  env.update(LANG='C',LC_ALL='C',TMPDIR=str(root/'tmp'))
  r=dict(schema='strata-moe-window-real-engine-syntax-v1',passed=False,complete=False,GPU_executed=False,code_generation=False,sources=sources,owner_sha256=sha(helper),database_sha256=sha(database),base_commit='6c923e05e69cc65ba9a0bae13fba5a54890945c9',scope='Real SYCL engine translation-unit syntax only; not a linked binary, GPU, numerical, lifetime, fit, RESTORE or speed proof.',limits=dict(priority=19,wall_seconds_per_TU=45,CPU_seconds_per_process=20,session_RSS_bytes=2<<30,total_artifact_bytes=8<<20),retention=dict(root=str(root),budget_bytes=8<<20,consumer='Main real engine syntax diagnostics',review_point='After qualified build or distinct syntax failure fixed; keep compact receipt and distinct error excerpt.'))
  def save():out.write_text(json.dumps(r,indent=2)+'\n')
  owner=mod.Owner(root/'owned',save);r['commands']=owner.commands;save()
  try:
   for target in selected:
    entry=next(e for e in entries if e['file'].endswith('/'+target))
    argv=shlex.split(entry['command']); compiler=argv[0]
    r.setdefault('compiler',dict(path=compiler,sha256=sha(compiler)))
    final=[str(Path(shutil.which('nice'))),'-n','19',compiler]
    i=1
    while i<len(argv):
     arg=argv[i];i+=1
     if arg=='-o':i+=1;continue
     if arg=='-c' or arg.startswith('-O'):continue
     if arg.startswith('-DSTRATA_VERSION='):arg='-DSTRATA_VERSION="0.1.38"'
     arg=arg.replace('/home/yayoi/orca/workspaces/Strata/sycl',str(P))
     final.append(arg)
    final+=['-O0','-fsyntax-only']
    assert all(sha(P/n)==h for n,h in sources.items())
    c,so,se=owner.run(target.replace('/','_'),final,env,P,wall=45,cpu=20,rss_cap=2<<30,text_cap=64<<10,total_cap=8<<20,file_cap=1<<20)
    if not mod.closed(c) or c['exit_code']!=0:raise RuntimeError(target+' syntax did not pass')
   assert all(sha(P/n)==h for n,h in sources.items())
   r.update(passed=True,complete=True,ownership_closed=all(mod.closed(c) for c in owner.commands))
  except BaseException as e:r['error']=repr(e)
  finally:r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
  print(json.dumps(dict(passed=r['passed'],complete=r['complete'],receipt=str(out))),flush=True)
  return 0 if r['passed'] else 1
if __name__=='__main__':sys.exit(main())
