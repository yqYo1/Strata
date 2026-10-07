from pathlib import Path
import datetime, hashlib, json, os, subprocess, time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-upstream-jit'
out=base/'workspace-reclaim-production-build'
out.mkdir()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=root/'sycl/src/prefill/prefill.cpp'
assert source.read_bytes()==(base/'prefill-workspace-reclaim-build/prefill.cpp').read_bytes()
assert sha(source)=='bdc0dea41753d345f5b5c329d807d3f9013280e2488e39df1d615a3b84506b92'
assert sha(build/'strata')=='619c830b581504072cbc2ba64782887ab847509fa2449f4680bd2b24af4cb659'
env=os.environ.copy()
settings=json.loads((base/'qsa-vector-load-production-build/record.json').read_text())['environment']
for k,v in settings.items():
 if v is None:env.pop(k,None)
 else:env[k]=v
r={'scope':'Normal production source build of short-tested workspace-reclaim candidate; no independent GPU execution', 'passed':False,'source_sha256':sha(source),'binary_before_sha256':sha(build/'strata'),'environment':settings,'argv':['/usr/bin/ninja','-v','strata'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
t=time.monotonic()
with (out/'build.stdout').open('wb') as a,(out/'build.stderr').open('wb') as c:
 proc=subprocess.run(r['argv'],cwd=build,env=env,stdout=a,stderr=c,timeout=300)
r.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-t,binary_sha256=sha(build/'strata'),object_sha256=sha(build/'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
r['passed']=proc.returncode==0 and r['binary_sha256']==sha(base/'strata-prefill-workspace-reclaim-candidate')
(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
assert r['passed']
