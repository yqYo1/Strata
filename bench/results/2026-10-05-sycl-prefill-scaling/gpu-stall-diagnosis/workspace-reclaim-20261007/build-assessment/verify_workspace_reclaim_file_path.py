from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-upstream-jit'
out=base/'workspace-reclaim-file-path-audit'
out.mkdir()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
private=json.loads((base/'prefill-workspace-reclaim-build/record.json').read_text())
production=json.loads((base/'workspace-reclaim-production-build/record.json').read_text())
assert production['exit_code']==0
assert sha(build/'strata')==production['binary_sha256']
source=base/'prefill-workspace-reclaim-build/prefill.cpp'
prodsource=root/'sycl/src/prefill/prefill.cpp'
assert source.read_bytes()==prodsource.read_bytes()
env=os.environ.copy()
for k,v in private['environment'].items():
 if v is None:env.pop(k,None)
 else:env[k]=v
argv=private['steps'][0]['argv'].copy()
for key,new in [('-MT',out/'prefill.cpp.o'),('-MF',out/'prefill.cpp.o.d'),('-o',out/'prefill.cpp.o')]:argv[argv.index(key)+1]=str(new)
argv.append('-ffile-prefix-map='+str(source)+'='+str(prodsource))
r={'scope':'Offline compile and relink with only __FILE__/file metadata path remapping; establishes exact reproduction of normal-build object and executable, not an independent GPU check','passed':False,'environment':private['environment'],'source_sha256':sha(source),'private_binary_sha256':private['candidate_binary_sha256'],'production_binary_sha256':production['binary_sha256'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
started=time.monotonic()
def run(label,args,cwd=None):
 t=time.monotonic()
 with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
  p=subprocess.run(args,cwd=cwd,env=env,stdout=a,stderr=b,timeout=120)
 r['steps'].append({'label':label,'argv':args,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t})
 assert p.returncode==0
run('compile',argv)
r['mapped_object_sha256']=sha(out/'prefill.cpp.o')
r['production_object_sha256']=sha(build/'CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o')
shutil.copyfile(base/'prefill-workspace-reclaim-build/libstrata_prefill.a',out/'libstrata_prefill.a')
run('archive-replace',['/usr/bin/ar','r',str(out/'libstrata_prefill.a'),str(out/'prefill.cpp.o')])
run('archive-index',['/usr/bin/ranlib',str(out/'libstrata_prefill.a')])
link=private['steps'][-1]['argv'].copy()
link[link.index('-o')+1]=str(out/'strata')
old=str(base/'prefill-workspace-reclaim-build/libstrata_prefill.a')
link[link.index(old)]=str(out/'libstrata_prefill.a')
run('link',link,build)
r.update(mapped_binary_sha256=sha(out/'strata'),object_equal=r['mapped_object_sha256']==r['production_object_sha256'],elapsed_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
r['binary_equal']=r['mapped_binary_sha256']==r['production_binary_sha256']
r['passed']=r['object_equal'] and r['binary_equal']
(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
assert sha(build/'strata')==production['binary_sha256']
shutil.copy2(build/'strata',base/'strata-workspace-reclaim-production')
print(json.dumps(r,indent=2))
assert r['passed']
