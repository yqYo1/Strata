"""Link the exact cache and no-root dequant objects together; no GPU submission."""
from pathlib import Path
import json,hashlib,subprocess,os,datetime,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-refresh-20261007'
cache=json.loads((base/'expanded-fp16-cache-refresh-build/record.json').read_text())
iq=json.loads((base/'dequant-no-root-refresh-build/record.json').read_text())
assert cache['passed'] and iq['passed'] and cache['production_inputs_unchanged'] and iq['production_inputs_unchanged']
assert cache['production_binary_sha256']==iq['production_binary_sha256']
out=base/'expanded-fp16-cache-no-root-refresh-build';out.mkdir()
binary=base/'strata-expanded-fp16-cache-no-root-refresh-candidate';assert not binary.exists()
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
av=cache['original_link_argv'].copy();inputs={k:v for k,v in cache['link_input_sha256'].items()}
assert all(digest(p)==s for p,s in inputs.items())
for name,path,expected in [('libstrata_prefill.a',base/'expanded-fp16-cache-refresh-build/libstrata_prefill.a',cache['candidate_archive_sha256']),('libstrata_kernels.a',base/'dequant-no-root-refresh-build/libstrata_kernels.a',iq['candidate_archive_sha256'])]:
 assert av.count(name)==1 and digest(path)==expected;av[av.index(name)]=str(path)
av[av.index('-o')+1]=str(binary)
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
r=dict(passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline two-object combination: optional unchanged-FP16 memo plus regular two-dequant kernel launch; math,ND-ranges and all other modules/phasewaits retained',cache_receipt_sha256=digest(base/'expanded-fp16-cache-refresh-build/record.json'),iq_receipt_sha256=digest(base/'dequant-no-root-refresh-build/record.json'),link_argv=av,production_inputs_sha256=inputs)
t=time.monotonic()
try:
 with (out/'link.stdout').open('wb') as stdout,(out/'link.stderr').open('wb') as stderr:p=subprocess.run(av,cwd=build,env=env,stdout=stdout,stderr=stderr,timeout=180)
 r['exit_code']=p.returncode;assert p.returncode==0
 assert all(digest(p)==s for p,s in inputs.items());r.update(passed=True,production_inputs_unchanged=True,candidate_binary_sha256=digest(binary))
except BaseException as e:r['error']=repr(e);raise
finally:
 r.update(elapsed_seconds=time.monotonic()-t,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:r[k] for k in ['passed','candidate_binary_sha256','elapsed_seconds']})
