"""Compile and run existing upstream standalone CPU tests; no GPU invocation."""
from pathlib import Path
import json,subprocess,datetime,time,hashlib
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=Path(__file__).parent/'upstream-refresh-20261007'/'cpu-tests';out.mkdir()
cases=[('message_boundary',['src/program/message_boundary_test.cpp']),
 ('exchange_storage',['tests/core/exchange_storage_test.cpp']),
 ('draft_policy',['src/spec/draft_policy_test.cpp','src/spec/draft_policy.cpp']),
 ('suffix_drafter',['src/spec/suffix_drafter_test.cpp','src/spec/suffix_drafter.cpp'])]
record=dict(active=True,passed=False,scope='Existing standalone CPU tests only',cases=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
try:
 for name,files in cases:
  item=dict(name=name,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},steps=[])
  record['cases'].append(item);save()
  argv=['/usr/bin/g++','-std=c++20','-O2','-pthread','-Iinclude']+files+['-o',str(out/name)]
  for phase,args in [('build',argv),('run',[str(out/name)])]:
   start=time.monotonic()
   with (out/(name+'-'+phase+'.stdout')).open('wb') as stdout,(out/(name+'-'+phase+'.stderr')).open('wb') as stderr:
    p=subprocess.run(args,cwd=root,stdout=stdout,stderr=stderr,timeout=120)
   item['steps'].append(dict(phase=phase,argv=args,exit_code=p.returncode,elapsed_seconds=time.monotonic()-start));save()
   print(name,phase,'exit',p.returncode,flush=True)
   if p.returncode: raise RuntimeError(name+' '+phase+' failed')
 record['passed']=True
except BaseException as e: record['error']=repr(e)
finally:
 record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if not record['passed']:raise SystemExit(1)
