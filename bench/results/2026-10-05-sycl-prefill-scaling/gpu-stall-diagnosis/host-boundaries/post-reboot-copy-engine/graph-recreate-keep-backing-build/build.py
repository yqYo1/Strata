import json,os,subprocess,shutil,hashlib,datetime
from pathlib import Path
out=Path(__file__).parent;rec=json.loads((out/'record.json').read_text());build=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/build-sycl-upstream-jit')
env=dict(os.environ,**rec['environment'])
try:
 for name,cmd in [('compile',rec['compile_argv']),('archive',['/usr/bin/ar','r',str(out/'libstrata_engine.a'),str(out/'mtp.cpp.o')]),('link',rec['link_argv'])]:
  if name=='archive':
   shutil.copy2(build/'libstrata_engine.a',out/'libstrata_engine.a')
   members=subprocess.check_output(['/usr/bin/ar','t',str(out/'libstrata_engine.a')],text=True).splitlines();assert members.count('mtp.cpp.o')==1
  with (out/(name+'.stdout')).open('wb') as so,(out/(name+'.stderr')).open('wb') as se:r=subprocess.run(cmd,cwd=build,env=env,stdout=so,stderr=se,timeout=300)
  rec['steps'].append(dict(name=name,exit_code=r.returncode));assert r.returncode==0,name
 exe=Path(rec['link_argv'][rec['link_argv'].index('-o')+1]);rec['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();rec['passed']=True
except BaseException as e:rec['error']=repr(e)
finally:
 rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'record.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k not in ['compile_argv','link_argv']},indent=2))
if not rec['passed']:raise SystemExit(1)
