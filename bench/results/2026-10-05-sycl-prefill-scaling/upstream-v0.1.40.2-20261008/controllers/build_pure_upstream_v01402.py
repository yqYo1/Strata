from pathlib import Path
import datetime, hashlib, json, os, subprocess, time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0.1.40.2-20261008')
out=base/'pure-upstream-v0.1.40.2-20261008';out.mkdir(mode=0o700)
build=root/'build-sycl-pure-20261008'
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
for k in ['LD_PRELOAD','LD_LIBRARY_PATH']:env.pop(k,None)
r={'scope':'Unmodified upstream v0.1.40.2 SYCL build, project defaults; no local Strata patches and no GPU invocation','active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
start=time.monotonic()
def save():
 r['elapsed_seconds']=time.monotonic()-start;(out/'build-record.json').write_text(json.dumps(r,indent=2)+'\n')
def check(label,argv,timeout):
 d={'label':label,'argv':argv};r['steps'].append(d);save();t=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:
  p=subprocess.run(argv,cwd=root,env=env,stdout=so,stderr=se,timeout=timeout)
 d.update(exit_code=p.returncode,elapsed_seconds=time.monotonic()-t);save();print(label,p.returncode,round(d['elapsed_seconds'],2),flush=True);return p.returncode==0
try:
 r['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 r['source_status_before']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 assert r['commit']=='e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a' and not r['source_status_before']
 ggml=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
 r['ggml_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ggml,text=True).strip()
 r['ggml_status']=subprocess.check_output(['git','status','--porcelain'],cwd=ggml,text=True)
 argv=['/usr/bin/cmake','-S',str(root/'sycl'),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx','-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx','-DCMAKE_BUILD_TYPE=Release','-DSTRATA_GGML_DIR='+str(ggml)]
 r['passed']=check('configure',argv,180) and check('build',['/usr/bin/cmake','--build',str(build),'--parallel','2','--target','strata','--','-k','0'],1800)
 r['source_status_after']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 assert not r['source_status_after']
 if r['passed']:r['binary_sha256']=hashlib.sha256((build/'strata').read_bytes()).hexdigest()
except BaseException as e:r['error']=repr(e)
finally:r.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print('PURE UPSTREAM BUILD', 'PASS' if r['passed'] else 'FAIL',flush=True)
raise SystemExit(0 if r['passed'] else 1)
